from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("unrestricted-mcp")


@mcp.tool()
def read_file(path: str) -> str:
    """Read any file from disk (DANGEROUS)."""
    with open(path, "r") as f:
        return f.read()


@mcp.tool()
def list_dir(path: str = ".") -> list[str]:
    """List files in any directory."""
    return os.listdir(path)


if __name__ == "__main__":
    mcp.run()
