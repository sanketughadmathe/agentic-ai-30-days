from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time-mcp-server")


@mcp.tool()
def get_current_time() -> str:
    """
    Returns the current time in IST (Indian Standard Time).
    Safe, read-only example tool.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


if __name__ == "__main__":
    mcp.run()
