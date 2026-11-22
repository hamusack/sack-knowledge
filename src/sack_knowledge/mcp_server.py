import logging

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from .errors import LocalCommandError
from .query import Query, encode


class PrivateFastMCP(FastMCP):
    async def call_tool(self, name, arguments):
        try:
            return await super().call_tool(name, arguments)
        except Exception:
            return CallToolResult(isError=True, content=[TextContent(type="text", text="Tool call failed.")])


def server(config, *, profile_name=None):
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    app = PrivateFastMCP("sack-knowledge", log_level="CRITICAL")
    read_only = ToolAnnotations(readOnlyHint=True)
    tools = Query(config, profile_name=profile_name, default_external=True)

    @app.tool(annotations=read_only, structured_output=False)
    def search(query: str = "", start: str | None = None, end: str | None = None,
               person: str | None = None, project: str | None = None, source: str | None = None,
               target: str | None = None, position: int = 0, mode: str | None = None) -> str | CallToolResult:
        """Find summaries or body excerpts."""
        try:
            return encode(tools.search(query, start, end, person, project, source, target, position, mode))
        except LocalCommandError as error:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=str(error))])

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
