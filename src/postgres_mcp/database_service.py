# ruff: noqa: B008
import asyncio
import logging
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import mcp.types as types

from postgres_mcp import models
from postgres_mcp.sql.sql_driver import obfuscate_password

from .artifacts import ErrorResult
from .artifacts import ExplainPlanArtifact
from .database_health import DatabaseHealthTool
from .explain import ExplainPlanTool
from .index.dta_calc import DatabaseTuningAdvisor
from .index.index_opt_base import MAX_NUM_INDEX_TUNING_QUERIES
from .index.index_opt_base import IndexTuningBase
from .index.llm_opt import LLMOptimizerTool
from .index.presentation import TextPresentation
from .sql import DbConnPool
from .sql import SafeSqlDriver
from .sql import SqlDriver
from .sql import check_hypopg_installation_status
from .top_queries import TopQueriesCalc

# Constants
PG_STAT_STATEMENTS = "pg_stat_statements"
HYPOPG_EXTENSION = "hypopg"

ResponseType = List[types.TextContent | types.ImageContent | types.EmbeddedResource]

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self, database_url: str, current_access_mode: models.AccessMode, query_timeout: float | None = None):
        self.database_url = database_url
        self.current_access_mode = current_access_mode
        self.query_timeout = query_timeout
        self._connect_lock = asyncio.Lock()

    db_connection: Optional[DbConnPool] = None

    async def get_sql_driver(self) -> Union[SqlDriver, SafeSqlDriver]:
        if not self.db_connection or not self.db_connection.is_valid:
            async with self._connect_lock:
                # Re-check after acquiring lock
                if not self.db_connection or not self.db_connection.is_valid:
                    self.db_connection = await self.create_db_connection()

        base_driver = SqlDriver(conn=self.db_connection)

        if self.current_access_mode == models.AccessMode.RESTRICTED:
            logger.debug("Using SafeSqlDriver with restrictions (RESTRICTED mode)")
            return SafeSqlDriver(sql_driver=base_driver)
        else:
            logger.debug("Using unrestricted SqlDriver (UNRESTRICTED mode)")
            return base_driver

    async def create_db_connection(self) -> DbConnPool:
        logger.info(
            f"Creating new database connection pool for URL: {obfuscate_password(self.database_url)}, statement_timeout={self.query_timeout}s"
        )
        self.db_connection = DbConnPool(
            connection_url=self.database_url,
            statement_timeout_seconds=self.query_timeout,
        )
        try:
            await self.db_connection.pool_connect(self.database_url)
            logger.info("Successfully connected to database and initialized connection pool")
            return self.db_connection
        except Exception as e:
            self.db_connection = None
            logger.warning(
                f"Could not connect to database: {obfuscate_password(str(e))}",
            )
            logger.warning(
                "The MCP server will start but database operations will fail until a valid connection is established.",
            )
            raise e

    def format_text_response(self, text: Any) -> ResponseType:
        """Format a text response."""
        import mcp.types as types

        return [types.TextContent(type="text", text=str(text))]

    def format_error_response(self, error: str) -> ResponseType:
        """Format an error response."""
        return self.format_text_response(f"Error: {error}")

    async def list_schemas(self) -> ResponseType:
        """List all schemas in the database."""
        try:
            sql_driver = await self.get_sql_driver()
            rows = await sql_driver.execute_query(
                """
                SELECT
                    schema_name,
                    schema_owner,
                    CASE
                        WHEN schema_name LIKE 'pg_%' THEN 'System Schema'
                        WHEN schema_name = 'information_schema' THEN 'System Information Schema'
                        ELSE 'User Schema'
                    END as schema_type
                FROM information_schema.schemata
                ORDER BY schema_type, schema_name
                """
            )
            schemas = [row.cells for row in rows] if rows else []
            return self.format_text_response(schemas)
        except Exception as e:
            logger.error(f"Error listing schemas: {e}")
            return self.format_error_response(str(e))

    async def list_objects(
        self,
        schema_name: str,
        object_type: str = "table",
    ) -> ResponseType:
        """List objects of a given type in a schema."""
        try:
            sql_driver = await self.get_sql_driver()

            if object_type in ("table", "view"):
                table_type = "BASE TABLE" if object_type == "table" else "VIEW"
                rows = await SafeSqlDriver.execute_param_query(
                    sql_driver,
                    """
                    SELECT table_schema, table_name, table_type
                    FROM information_schema.tables
                    WHERE table_schema = {} AND table_type = {}
                    ORDER BY table_name
                    """,
                    [schema_name, table_type],
                )
                objects = (
                    [{"schema": row.cells["table_schema"], "name": row.cells["table_name"], "type": row.cells["table_type"]} for row in rows]
                    if rows
                    else []
                )

            elif object_type == "sequence":
                rows = await SafeSqlDriver.execute_param_query(
                    sql_driver,
                    """
                    SELECT sequence_schema, sequence_name, data_type
                    FROM information_schema.sequences
                    WHERE sequence_schema = {}
                    ORDER BY sequence_name
                    """,
                    [schema_name],
                )
                objects = (
                    [
                        {"schema": row.cells["sequence_schema"], "name": row.cells["sequence_name"], "data_type": row.cells["data_type"]}
                        for row in rows
                    ]
                    if rows
                    else []
                )

            elif object_type == "extension":
                # Extensions are not schema-specific
                rows = await sql_driver.execute_query(
                    """
                    SELECT extname, extversion, extrelocatable
                    FROM pg_extension
                    ORDER BY extname
                    """
                )
                objects = (
                    [{"name": row.cells["extname"], "version": row.cells["extversion"], "relocatable": row.cells["extrelocatable"]} for row in rows]
                    if rows
                    else []
                )

            else:
                return self.format_error_response(f"Unsupported object type: {object_type}")

            return self.format_text_response(objects)
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return self.format_error_response(str(e))

    async def get_object_details(
        self,
        schema_name: str,
        object_name: str,
        object_type: str = "table",
    ) -> ResponseType:
        """Get detailed information about a database object."""
        try:
            sql_driver = await self.get_sql_driver()

            if object_type in ("table", "view"):
                # Get columns
                col_rows = await SafeSqlDriver.execute_param_query(
                    sql_driver,
                    """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = {} AND table_name = {}
                    ORDER BY ordinal_position
                    """,
                    [schema_name, object_name],
                )
                columns = (
                    [
                        {
                            "column": r.cells["column_name"],
                            "data_type": r.cells["data_type"],
                            "is_nullable": r.cells["is_nullable"],
                            "default": r.cells["column_default"],
                        }
                        for r in col_rows
                    ]
                    if col_rows
                    else []
                )

                # Get constraints
                con_rows = await SafeSqlDriver.execute_param_query(
                    sql_driver,
                    """
                    SELECT tc.constraint_name, tc.constraint_type, kcu.column_name
                    FROM information_schema.table_constraints AS tc
                    LEFT JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = {} AND tc.table_name = {}
                    """,
                    [schema_name, object_name],
                )

                constraints = {}
                if con_rows:
                    for row in con_rows:
                        cname = row.cells["constraint_name"]
                        ctype = row.cells["constraint_type"]
                        col = row.cells["column_name"]

                        if cname not in constraints:
                            constraints[cname] = {"type": ctype, "columns": []}
                        if col:
                            constraints[cname]["columns"].append(col)

                constraints_list = [{"name": name, **data} for name, data in constraints.items()]

                # Get indexes
                idx_rows = await SafeSqlDriver.execute_param_query(
                    sql_driver,
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = {} AND tablename = {}
                    """,
                    [schema_name, object_name],
                )

                indexes = [{"name": r.cells["indexname"], "definition": r.cells["indexdef"]} for r in idx_rows] if idx_rows else []

                result = {
                    "basic": {"schema": schema_name, "name": object_name, "type": object_type},
                    "columns": columns,
                    "constraints": constraints_list,
                    "indexes": indexes,
                }

            elif object_type == "sequence":
                rows = await SafeSqlDriver.execute_param_query(
                    sql_driver,
                    """
                    SELECT sequence_schema, sequence_name, data_type, start_value, increment
                    FROM information_schema.sequences
                    WHERE sequence_schema = {} AND sequence_name = {}
                    """,
                    [schema_name, object_name],
                )

                if rows and rows[0]:
                    row = rows[0]
                    result = {
                        "schema": row.cells["sequence_schema"],
                        "name": row.cells["sequence_name"],
                        "data_type": row.cells["data_type"],
                        "start_value": row.cells["start_value"],
                        "increment": row.cells["increment"],
                    }
                else:
                    result = {}

            elif object_type == "extension":
                rows = await SafeSqlDriver.execute_param_query(
                    sql_driver,
                    """
                    SELECT extname, extversion, extrelocatable
                    FROM pg_extension
                    WHERE extname = {}
                    """,
                    [object_name],
                )

                if rows and rows[0]:
                    row = rows[0]
                    result = {"name": row.cells["extname"], "version": row.cells["extversion"], "relocatable": row.cells["extrelocatable"]}
                else:
                    result = {}

            else:
                return self.format_error_response(f"Unsupported object type: {object_type}")

            return self.format_text_response(result)
        except Exception as e:
            logger.error(f"Error getting object details: {e}")
            return self.format_error_response(str(e))

    async def explain_query(
        self,
        sql: str,
        analyze: bool = False,
        hypothetical_indexes: Optional[list[dict[str, Any]]] = None,
    ) -> ResponseType:
        """
        Explains the execution plan for a SQL query.

        Args:
            sql: The SQL query to explain
            analyze: When True, actually runs the query for real statistics
            hypothetical_indexes: Optional list of indexes to simulate
        """
        if hypothetical_indexes is None:
            hypothetical_indexes = []

        try:
            sql_driver = await self.get_sql_driver()
            explain_tool = ExplainPlanTool(sql_driver=sql_driver)
            result: ExplainPlanArtifact | ErrorResult | None = None

            # If hypothetical indexes are specified, check for HypoPG extension
            if hypothetical_indexes and len(hypothetical_indexes) > 0:
                if analyze:
                    return self.format_error_response("Cannot use analyze and hypothetical indexes together")
                try:
                    # Use the common utility function to check if hypopg is installed
                    (
                        is_hypopg_installed,
                        hypopg_message,
                    ) = await check_hypopg_installation_status(sql_driver)

                    # If hypopg is not installed, return the message
                    if not is_hypopg_installed:
                        return self.format_text_response(hypopg_message)

                    # HypoPG is installed, proceed with explaining with hypothetical indexes
                    result = await explain_tool.explain_with_hypothetical_indexes(sql, hypothetical_indexes)
                except Exception:
                    raise  # Re-raise the original exception
            elif analyze:
                try:
                    # Use EXPLAIN ANALYZE
                    result = await explain_tool.explain_analyze(sql)
                except Exception:
                    raise  # Re-raise the original exception
            else:
                try:
                    # Use basic EXPLAIN
                    result = await explain_tool.explain(sql)
                except Exception:
                    raise  # Re-raise the original exception

            if result and isinstance(result, ExplainPlanArtifact):
                return self.format_text_response(result.to_text())
            else:
                error_message = "Error processing explain plan"
                if isinstance(result, ErrorResult):
                    error_message = result.to_text()
                return self.format_error_response(error_message)
        except Exception as e:
            logger.error(f"Error explaining query: {e}")
            return self.format_error_response(str(e))

    async def execute_sql(
        self,
        sql: str = "all",
    ) -> ResponseType:
        """Executes a SQL query against the database."""
        try:
            sql_driver = await self.get_sql_driver()
            rows = await sql_driver.execute_query(sql)  # type: ignore
            if rows is None:
                return self.format_text_response("No results")
            return self.format_text_response(list([r.cells for r in rows]))
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return self.format_error_response(str(e))

    async def analyze_workload_indexes(
        self,
        max_index_size_mb: int = 10000,
        method: str = "dta",
    ) -> ResponseType:
        """Analyze frequently executed queries in the database and recommend optimal indexes."""
        try:
            sql_driver = await self.get_sql_driver()
            if method == "dta":
                index_tuning: IndexTuningBase = DatabaseTuningAdvisor(sql_driver)
            else:
                index_tuning = LLMOptimizerTool(sql_driver)
            dta_tool = TextPresentation(sql_driver, index_tuning)
            result = await dta_tool.analyze_workload(max_index_size_mb=max_index_size_mb)
            return self.format_text_response(result)
        except Exception as e:
            logger.error(f"Error analyzing workload: {e}")
            return self.format_error_response(str(e))

    async def analyze_query_indexes(
        self,
        queries: list[str],
        max_index_size_mb: int = 10000,
        method: str = "dta",
    ) -> ResponseType:
        """Analyze a list of SQL queries and recommend optimal indexes."""
        if len(queries) == 0:
            return self.format_error_response("Please provide a non-empty list of queries to analyze.")
        if len(queries) > MAX_NUM_INDEX_TUNING_QUERIES:
            return self.format_error_response(f"Please provide a list of up to {MAX_NUM_INDEX_TUNING_QUERIES} queries to analyze.")

        try:
            sql_driver = await self.get_sql_driver()
            if method == "dta":
                index_tuning: IndexTuningBase = DatabaseTuningAdvisor(sql_driver)
            else:
                index_tuning = LLMOptimizerTool(sql_driver)
            dta_tool = TextPresentation(sql_driver, index_tuning)
            result = await dta_tool.analyze_queries(queries=queries, max_index_size_mb=max_index_size_mb)
            return self.format_text_response(result)
        except Exception as e:
            logger.error(f"Error analyzing queries: {e}")
            return self.format_error_response(str(e))

    async def analyze_db_health(
        self,
        health_type: str = "all",
    ) -> ResponseType:
        """Analyze database health for specified components.

        Args:
            health_type: Comma-separated list of health check types to perform.
                        Valid values: index, connection, vacuum, sequence, replication, buffer, constraint, all
        """
        health_tool = DatabaseHealthTool(await self.get_sql_driver())
        result = await health_tool.health(health_type=health_type)
        return self.format_text_response(result)

    async def get_top_queries(
        self,
        sort_by: str = "resources",
        limit: int = 10,
    ) -> ResponseType:
        try:
            sql_driver = await self.get_sql_driver()
            top_queries_tool = TopQueriesCalc(sql_driver=sql_driver)

            if sort_by == "resources":
                result = await top_queries_tool.get_top_resource_queries()
                return self.format_text_response(result)
            elif sort_by == "mean_time" or sort_by == "total_time":
                # Map the sort_by values to what get_top_queries_by_time expects
                result = await top_queries_tool.get_top_queries_by_time(limit=limit, sort_by="mean" if sort_by == "mean_time" else "total")
            else:
                return self.format_error_response("Invalid sort criteria. Please use 'resources' or 'mean_time' or 'total_time'.")
            return self.format_text_response(result)
        except Exception as e:
            logger.error(f"Error getting slow queries: {e}")
            return self.format_error_response(str(e))

    async def close(self):
        """Close database connections."""
        if self.db_connection:
            await self.db_connection.close()
            logger.info("Closed database connection pool")
