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
from postgres_mcp.models import HostConfig

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
host_configs: dict[str, HostConfig] = {}
query_timeout: Optional[float] = None
shutdown_in_progress = False


def parse_host_configs_from_env() -> dict[str, HostConfig]:
    """Parse DATABASES__N__HOST/PORT/USERNAME/PASSWORD environment variables into HostConfigs."""
    configs: dict[str, HostConfig] = {}
    indices: set[str] = set()
    for key in os.environ:
        if key.startswith("DATABASES__") and key.endswith("__HOST"):
            parts = key.split("__")
            if len(parts) == 3:
                indices.add(parts[1])

    for idx in sorted(indices):
        host = os.environ.get(f"DATABASES__{idx}__HOST")
        if not host:
            continue
        port = int(os.environ.get(f"DATABASES__{idx}__PORT", "5432"))
        username = os.environ.get(f"DATABASES__{idx}__USERNAME")
        password = os.environ.get(f"DATABASES__{idx}__PASSWORD")
        if not username or not password:
            raise ValueError(
                f"DATABASES__{idx}__USERNAME and DATABASES__{idx}__PASSWORD must both be set when DATABASES__{idx}__HOST is configured")
        configs[host] = HostConfig(
            host=host, port=port, username=username, password=password)

    return configs


def resolve_host_config(host: Optional[str] = None) -> HostConfig:
    """Resolve the HostConfig for the given host name.

    When host is None, returns the single configured host or raises if multiple are configured.
    """
    if host is not None:
        if host not in host_configs:
            available = ", ".join(sorted(host_configs.keys()))
            raise ValueError(
                f"No configuration found for host '{host}'. Available hosts: {available}")
        return host_configs[host]

    if len(host_configs) == 1:
        return next(iter(host_configs.values()))

    if len(host_configs) == 0:
        raise ValueError(
            "No database host configured. Set DATABASE_HOST or DATABASES__N__HOST environment variables.")

    available = ", ".join(sorted(host_configs.keys()))
    raise ValueError(
        f"Multiple database hosts are configured ({available}). The 'host' parameter is required when multiple hosts are configured.")


def create_database_url_from_config(config: HostConfig, database_name: str) -> str:
    return f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{database_name}"


async def get_service(database_name: str, host: Optional[str] = None) -> DatabaseService:
    config = resolve_host_config(host)
    service_key = f"{config.host}:{config.port}/{database_name}"
    if service_key not in db_services:
        database_url = create_database_url_from_config(config, database_name)
        db_services[service_key] = DatabaseService(
            database_url, current_access_mode, query_timeout)
    return db_services[service_key]


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    return Response({"status": "ok"})


@mcp.tool(description="List all schemas in the database")
async def list_schemas(
    database_name: str = Field(description="Database name"),
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
    return await service.list_schemas()


@mcp.tool(description="List objects in a schema")
async def list_objects(
    database_name: str = Field(description="Database name"),
    schema_name: str = Field(description="Schema name"),
    object_type: str = Field(
        description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
    return await service.list_objects(schema_name, object_type)


@mcp.tool(description="Show detailed information about a database object")
async def get_object_details(
    database_name: str = Field(description="Database name"),
    schema_name: str = Field(description="Schema name"),
    object_name: str = Field(description="Object name"),
    object_type: str = Field(
        description="Object type: 'table', 'view', 'sequence', or 'extension'", default="table"),
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
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
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
    return await service.explain_query(sql, analyze, hypothetical_indexes)


# Query function declaration without the decorator - we'll add it dynamically based on access mode
async def execute_sql(
    database_name: str = Field(description="Database name"),
    sql: str = Field(description="SQL to run", default="all"),
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
    return await service.execute_sql(sql)


@mcp.tool(description="Analyze frequently executed queries in the database and recommend optimal indexes")
@validate_call
async def analyze_workload_indexes(
    database_name: str = Field(description="Database name"),
    max_index_size_mb: int = Field(
        description="Max index size in MB", default=10000),
    method: Literal["dta", "llm"] = Field(
        description="Method to use for analysis", default="dta"),
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
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
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
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
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
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
    host: Optional[str] = Field(
        description="Database host name or address. Only provide when explicitly requested.", default=None),
) -> ResponseType:
    service = await get_service(database_name, host)
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
        help="Database password (default: empty, but recommended to set via environment variable or credentials file for security)",
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
    global host_configs
    global query_timeout

    current_access_mode = AccessMode(args.access_mode)

    raw_query_timeout = os.environ.get("QUERY_TIMEOUT")
    query_timeout = float(
        raw_query_timeout) if raw_query_timeout is not None else None

    # Build host configs from multi-host env vars (DATABASES__N__*)
    host_configs = parse_host_configs_from_env()

    # Also support the legacy single-host configuration (CLI args + single env vars)
    legacy_host = os.environ.get("DATABASE_HOST", args.database_host)
    if legacy_host:
        legacy_port = int(os.environ.get(
            "DATABASE_PORT", str(args.database_port)))
        if args.database_creds_file:
            legacy_username, legacy_password = read_database_creds(
                args.database_creds_file)
        else:
            legacy_username = os.environ.get(
                "DATABASE_USERNAME", args.database_username)
            legacy_password = os.environ.get(
                "DATABASE_PASSWORD", args.database_password)

        if legacy_username and legacy_password and legacy_host not in host_configs:
            host_configs[legacy_host] = HostConfig(
                host=legacy_host,
                port=legacy_port,
                username=legacy_username,
                password=legacy_password,
            )

    # Add the query tool with a description appropriate to the access mode
    if current_access_mode == AccessMode.UNRESTRICTED:
        mcp.add_tool(execute_sql, description="Execute any SQL query")
    else:
        mcp.add_tool(execute_sql, description="Execute a read-only SQL query")

    configured_hosts = ", ".join(sorted(host_configs.keys())) or "(none)"
    logger.info(
        f"Starting PostgreSQL MCP Server in {current_access_mode.upper()} mode. "
        f"transport={args.transport}, configured_hosts=[{configured_hosts}], "
        f"sse_host={args.sse_host}, sse_port={args.sse_port}"
    )

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
