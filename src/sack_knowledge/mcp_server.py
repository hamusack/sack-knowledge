import logging

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .query import Query, encode


def server(config):
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    app = FastMCP("sack-knowledge", log_level="CRITICAL")
    read_only = ToolAnnotations(readOnlyHint=True)
    tools = Query(config)

    @app.tool(annotations=read_only, structured_output=False)
    def search(query: str = "", start: str | None = None, end: str | None = None,
               person: str | None = None, project: str | None = None, source: str | None = None,
               target: str = "summary", position: int = 0) -> str:
        """Find summaries or body excerpts."""
        return encode(tools.search(query, start, end, person, project, source, target, position))

    @app.tool(annotations=read_only, structured_output=False)
    def get(ids: list[str], neighbors: int = 0) -> str:
        """Read up to five records or chunks."""
        return encode(tools.get(ids, neighbors))

    @app.tool(annotations=read_only, structured_output=False)
    def count(group_by: str, start: str | None = None, end: str | None = None,
              person: str | None = None, project: str | None = None, source: str | None = None) -> str:
        """Count records by month, person, project or source."""
        return encode(tools.count(group_by, start, end, person, project, source))

    return app
