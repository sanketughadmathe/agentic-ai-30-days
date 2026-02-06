from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("utility-mcp")


@mcp.tool()
def get_utc_time() -> str:
    """
    Return current UTC time.
    Read-only, safe utility tool.
    """
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


if __name__ == "__main__":
    mcp.run()
