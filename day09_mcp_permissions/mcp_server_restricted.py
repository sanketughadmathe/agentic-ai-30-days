from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("restricted-mcp")

# Allow access ONLY inside this directory
ALLOWED_DIR = Path("./safe_data").resolve()
ALLOWED_DIR.mkdir(exist_ok=True)


def _validate_path(path: str) -> Path:
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(ALLOWED_DIR)):
        raise ValueError("Access denied: path outside allowed directory")
    return resolved


@mcp.tool()
def read_safe_file(filename: str) -> str:
    """
    Read a file ONLY from the safe_data directory.
    """
    file_path = _validate_path(ALLOWED_DIR / filename)
    return file_path.read_text()


@mcp.tool()
def list_safe_files() -> list[str]:
    """
    List files in the safe_data directory.
    """
    return [p.name for p in ALLOWED_DIR.iterdir() if p.is_file()]


if __name__ == "__main__":
    mcp.run()
