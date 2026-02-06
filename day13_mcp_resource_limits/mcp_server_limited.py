import signal
import time
from datetime import datetime

from fastmcp import FastMCP

mcp = FastMCP("limited-mcp")


# -------------------------
# Timeout helper
# -------------------------
class Timeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise Timeout("Tool execution timed out")


signal.signal(signal.SIGALRM, timeout_handler)


# -------------------------
# Safe tool
# -------------------------
@mcp.tool
def get_utc_time() -> str:
    """Get current UTC time - fast and safe"""
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


# -------------------------
# Potentially slow tool
# -------------------------
@mcp.tool
def long_running_task(seconds: int) -> str:
    """
    Simulates a slow operation.
    Hard timeout enforced at 2 seconds.
    """
    signal.alarm(2)  # 2 second hard limit
    try:
        time.sleep(seconds)
        signal.alarm(0)  # Cancel alarm if completed
        return f"✅ Completed in {seconds}s"
    except Timeout:
        return "⏱️ Execution timed out after 2 seconds"
    finally:
        signal.alarm(0)  # Ensure alarm is always cancelled


# -------------------------
# Memory-intensive tool (safe demo version)
# -------------------------
@mcp.tool
def allocate_memory(mb: int) -> str:
    """
    Simulates memory allocation without actually allocating.
    Demonstrates resource limit checking without crashing container.
    """
    # Container has 128MB limit, Python uses ~30-40MB base
    # Safe allocation zone: < 50MB

    if mb > 100:
        return f"❌ Denied: {mb}MB exceeds container limit (128MB total, ~40MB used by Python)"
    elif mb > 50:
        return f"⚠️ Refused: {mb}MB would risk OOM (safe limit: 50MB)"
    else:
        # Actually allocate only for small requests
        try:
            if mb <= 20:
                data = bytearray(mb * 1024 * 1024)
                # Touch memory to ensure allocation
                for i in range(0, len(data), 1024 * 1024):
                    data[i] = 1
                return f"✅ Allocated {mb}MB successfully"
            else:
                # Simulate but don't actually allocate to prevent crash
                return f"✅ Would allocate {mb}MB (simulated - container-safe)"
        except MemoryError:
            return f"❌ Failed to allocate {mb}MB - memory limit reached"
        except Exception as e:
            return f"❌ Error: {str(e)}"


# -------------------------
# Resource monitoring tool
# -------------------------
@mcp.tool
def get_resource_info() -> str:
    """
    Get current resource usage information.
    Shows memory and CPU constraints.
    """
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return f"""
📊 Container Resource Info:
   Memory: {memory_info.rss / 1024 / 1024:.1f}MB used
   CPU Count: {psutil.cpu_count()}
   Process PID: {os.getpid()}
   Container Limits:
     - Max Memory: 128MB
     - Max CPU: 0.5 cores
     - Max Processes: 50
     - Tool Timeout: 2 seconds
"""


if __name__ == "__main__":
    # Run with HTTP transport for network access
    mcp.run(transport="http", host="0.0.0.0", port=3333)
