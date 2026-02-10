from datetime import datetime, timezone

from fastmcp import FastMCP

mcp = FastMCP("dockerized-mcp")


@mcp.tool()
def get_utc_time() -> str:
    """
    Read-only utility tool.
    Safe to expose from a container.
    """
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    # Run with HTTP transport for network access
    mcp.run(transport="http", host="0.0.0.0", port=3333)
