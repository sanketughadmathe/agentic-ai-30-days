from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("utility-mcp")


@mcp.tool()
def get_utc_time() -> str:
    """
    Return current UTC time.
    Read-only, safe utility tool.
    """
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    mcp.run()
