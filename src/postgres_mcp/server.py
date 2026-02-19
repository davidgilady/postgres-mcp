# ruff: noqa: B008
import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Any
from typing import List
from typing import Literal
from typing import Optional

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pydantic import validate_call
from starlette.requests import Request
from starlette.responses import Response

from postgres_mcp.models import AccessMode

from .database_health import HealthType
from .database_service import DatabaseService

# Initialize FastMCP with default settings
mcp = FastMCP("postgres-mcp")

# Constants
PG_STAT_STATEMENTS = "pg_stat_statements"
HYPOPG_EXTENSION = "hypopg"

ResponseType = List[types.TextContent |
                    types.ImageContent | types.EmbeddedResource]

logger = logging.getLogger(__name__)


# Global variables
db_services: dict[str, DatabaseService] = {}
current_access_mode = AccessMode.UNRESTRICTED
database_host: Optional[str] = None
database_port: Optional[str] = None
database_username: Optional[str] = None
database_password: Optional[str] = None
shutdown_in_progress = False


async def get_service(database_name: str) -> DatabaseService:
    if database_name not in db_services:
        database_url = create_database_url(database_name)
        db_services[database_name] = DatabaseService(
            database_url, current_access_mode)
    return db_services[database_name]


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    return Response({"status": "ok"})


@mcp.tool(description="List all schemas in the database")
async def list_schemas(database_name: str = Field(description="Database name")) -> ResponseType:
    service = await get_service(database_name)
    return await service.list_schemas()


@mcp.tool(description="List objects in a schema")
async def list_objects(
    database_name: str = Field(description="Database name"),
    schema_name: str = Field(description="Schema name"),
    object_type: str = Field(
        description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.list_objects(schema_name, object_type)


@mcp.tool(description="Show detailed information about a database object")
async def get_object_details(
    database_name: str = Field(description="Database name"),
    schema_name: str = Field(description="Schema name"),
    object_name: str = Field(description="Object name"),
    object_type: str = Field(
        description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.get_object_details(schema_name, object_name, object_type)


@mcp.tool(description="Explains the execution plan for a SQL query, showing how the database will execute it and provides detailed cost estimates.")
async def explain_query(
    database_name: str = Field(description="Database name"),
    sql: str = Field(description="SQL query to explain"),
    analyze: bool = Field(
        description="When True, actually runs the query to show real execution statistics instead of estimates. "
        "Takes longer but provides more accurate information.",
        default=False,
    ),
    hypothetical_indexes: list[dict[str, Any]] = Field(
        description="""A list of hypothetical indexes to simulate. Each index must be a dictionary with these keys:
    - 'table': The table name to add the index to (e.g., 'users')
    - 'columns': List of column names to include in the index (e.g., ['email'] or ['last_name', 'first_name'])
    - 'using': Optional index method (default: 'btree', other options include 'hash', 'gist', etc.)

Examples: [
    {"table": "users", "columns": ["email"], "using": "btree"},
    {"table": "orders", "columns": ["user_id", "created_at"]}
]
If there is no hypothetical index, you can pass an empty list.""",
        default=[],
    ),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.explain_query(sql, analyze, hypothetical_indexes)


# Query function declaration without the decorator - we'll add it dynamically based on access mode
async def execute_sql(
    database_name: str = Field(description="Database name"),
    sql: str = Field(description="SQL to run", default="all"),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.execute_sql(sql)


@mcp.tool(description="Analyze frequently executed queries in the database and recommend optimal indexes")
@validate_call
async def analyze_workload_indexes(
    database_name: str = Field(description="Database name"),
    max_index_size_mb: int = Field(
        description="Max index size in MB", default=10000),
    method: Literal["dta", "llm"] = Field(
        description="Method to use for analysis", default="dta"),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.analyze_workload_indexes(max_index_size_mb, method)


@mcp.tool(description="Analyze a list of (up to 10) SQL queries and recommend optimal indexes")
@validate_call
async def analyze_query_indexes(
    database_name: str = Field(description="Database name"),
    queries: list[str] = Field(description="List of Query strings to analyze"),
    max_index_size_mb: int = Field(
        description="Max index size in MB", default=10000),
    method: Literal["dta", "llm"] = Field(
        description="Method to use for analysis", default="dta"),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.analyze_query_indexes(queries, max_index_size_mb, method)


@mcp.tool(
    description="Analyzes database health. Here are the available health checks:\n"
    "- index - checks for invalid, duplicate, and bloated indexes\n"
    "- connection - checks the number of connection and their utilization\n"
    "- vacuum - checks vacuum health for transaction id wraparound\n"
    "- sequence - checks sequences at risk of exceeding their maximum value\n"
    "- replication - checks replication health including lag and slots\n"
    "- buffer - checks for buffer cache hit rates for indexes and tables\n"
    "- constraint - checks for invalid constraints\n"
    "- all - runs all checks\n"
    "You can optionally specify a single health check or a comma-separated list of health checks. The default is 'all' checks."
)
async def analyze_db_health(
    database_name: str = Field(description="Database name"),
    health_type: str = Field(
        description=f"Optional. Valid values are: {', '.join(sorted([t.value for t in HealthType]))}.",
        default="all",
    ),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.analyze_db_health(health_type)


@mcp.tool(
    name="get_top_queries",
    description=f"Reports the slowest or most resource-intensive queries using data from the '{PG_STAT_STATEMENTS}' extension.",
)
async def get_top_queries(
    database_name: str = Field(description="Database name"),
    sort_by: str = Field(
        description="Ranking criteria: 'total_time' for total execution time or 'mean_time' for mean execution time per call, or 'resources' "
        "for resource-intensive queries",
        default="resources",
    ),
    limit: int = Field(
        description="Number of queries to return when ranking based on mean_time or total_time", default=10),
) -> ResponseType:
    service = await get_service(database_name)
    return await service.get_top_queries(sort_by, limit)


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PostgreSQL MCP Server")
    parser.add_argument(
        "database_host", help="Database host: e.g database.example.com", nargs="?")
    parser.add_argument(
        "--database-port",
        type=int,
        default=5432,
        help="Database port number (default: 5432)",
    )
    parser.add_argument(
        "--database-username",
        type=str,
        default=None,
        help="Database username (default: postgres)",
    )
    parser.add_argument(
        "--database-password",
        type=str,
        default=None,
        help="Database password (default: empty, but recommended to set via environment variable or credentials file for security)"
    )
    parser.add_argument(
        "--database-creds-file",
        type=str,
        default=None,
        help="Database credentials file (optional, alternative to environment variables or command line args for username & password)",
    )
    parser.add_argument(
        "--access-mode",
        type=str,
        choices=[mode.value for mode in AccessMode],
        default=AccessMode.UNRESTRICTED.value,
        help="Set SQL access mode: unrestricted (unrestricted) or restricted (read-only with protections)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Select MCP transport: stdio (default) or sse",
    )
    parser.add_argument(
        "--sse-host",
        type=str,
        default="localhost",
        help="Host to bind SSE server to (default: localhost)",
    )
    parser.add_argument(
        "--sse-port",
        type=int,
        default=8000,
        help="Port for SSE server (default: 8000)",
    )

    args = parser.parse_args()

    # Store the access mode in the global variable
    global current_access_mode
    global database_host
    global database_port
    global database_username
    global database_password

    current_access_mode = AccessMode(args.access_mode)
    database_host = os.environ.get("DATABASE_HOST", args.database_host)
    database_port = os.environ.get("DATABASE_PORT", args.database_port)

    if args.database_creds_file:
        database_username, database_password = read_database_creds(
            args.database_creds_file)
    else:
        database_username = os.environ.get(
            "DATABASE_USERNAME", args.database_username)
        database_password = os.environ.get(
            "DATABASE_PASSWORD", args.database_password)

    # Add the query tool with a description appropriate to the access mode
    if current_access_mode == AccessMode.UNRESTRICTED:
        mcp.add_tool(execute_sql, description="Execute any SQL query")
    else:
        mcp.add_tool(execute_sql, description="Execute a read-only SQL query")

    logger.info(
        f"Starting PostgreSQL MCP Server in {current_access_mode.upper()} mode. transport={args.transport}, database_host={database_host}, "
        f"database_port={database_port}, database_username={database_username}, sse_host={args.sse_host}, sse_port={args.sse_port}")

    # Set up proper shutdown handling
    try:
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(s)))
    except NotImplementedError:
        # Windows doesn't support signals properly
        logger.warning("Signal handling not supported on Windows")
        pass

    # Run the server with the selected transport (always async)
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    else:
        # Update FastMCP settings based on command line arguments
        mcp.settings.host = args.sse_host
        mcp.settings.port = args.sse_port
        await mcp.run_sse_async()


def read_database_creds(creds_file: str) -> tuple[str, str]:
    """Reads database credentials from a file. The file should contain the username on the first line and the password on the second line."""
    try:
        with open(creds_file) as f:
            lines = f.read().splitlines()
            if len(lines) < 2:
                raise ValueError(
                    "Credentials file must contain at least two lines: username and password")
            return lines[0], lines[1]
    except Exception as e:
        logger.error(f"Error reading database credentials from file: {e}")
        raise


def create_database_url(database_name: str) -> str:
    if not database_host:
        raise ValueError(
            "Database host must be specified via command line argument or DATABASE_HOST environment variable"
        )
    if not database_username:
        raise ValueError(
            "Database username must be specified via command line argument, DATABASE_USERNAME environment variable or a database credentials file"
        )
    if not database_password:
        raise ValueError(
            "Database password must be specified via command line argument, DATABASE_PASSWORD environment variable or a database credentials file"
        )

    return f"postgresql://{database_username}:{database_password}@{database_host}:{database_port}/{database_name}"


async def shutdown(sig=None):
    """Clean shutdown of the server."""
    global shutdown_in_progress

    if shutdown_in_progress:
        logger.warning("Forcing immediate exit")
        # Use sys.exit instead of os._exit to allow for proper cleanup
        sys.exit(1)

    shutdown_in_progress = True

    if sig:
        logger.info(f"Received exit signal {sig.name}")

    # Close database connections
    try:
        for service in db_services.values():
            await service.close()
        logger.info("Closed database connections")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

    # Exit with appropriate status code
    sys.exit(128 + sig if sig is not None else 0)
