from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time-mcp-server")


@mcp.tool()
def get_current_time() -> str:
    """
    Returns the current UTC time.
    Safe, read-only example tool.
    """
    from datetime import datetime

    return datetime.utcnow().isoformat()


if __name__ == "__main__":
    mcp.run()
