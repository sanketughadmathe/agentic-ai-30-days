from mcp.server.fastmcp import FastMCP
from datetime import datetime, timezone
from pathlib import Path
import json

mcp = FastMCP("audited-mcp-server")

AUDIT_LOG = Path("audit.log")


def audit(event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(event) + "\n")


@mcp.tool()
def get_utc_time() -> str:
    """
    Safe, read-only tool.
    """
    audit({"tool": "get_utc_time", "decision": "allowed"})
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def read_file(path: str) -> str:
    """
    Unsafe tool — intentionally denied.
    Demonstrates enforcement + logging.
    """
    audit({"tool": "read_file", "path": path, "decision": "denied"})
    raise PermissionError("File access is not permitted")


if __name__ == "__main__":
    mcp.run()
