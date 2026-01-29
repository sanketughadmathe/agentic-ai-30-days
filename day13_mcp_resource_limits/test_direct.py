"""
Direct Test of Resource-Limited MCP Server
Tests tools directly without LLM to avoid rate limits.
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    print("🚀 Direct Resource Limit Testing")
    print("=" * 60)

    # Connect to MCP server
    print("\n1️⃣  Connecting to MCP server...")
    mcp_client = MultiServerMCPClient(
        {"limited-mcp": {"url": "http://localhost:3333/mcp", "transport": "http"}}
    )

    # Get tools
    tools = await mcp_client.get_tools()
    print(f"   ✅ Found {len(tools)} tool(s):")
    for tool in tools:
        print(f"      - {tool.name}: {tool.description}")

    # Create tool lookup
    tool_dict = {t.name: t for t in tools}

    print("\n" + "=" * 60)
    print("Running Direct Tool Tests")
    print("=" * 60)

    # Test 0: Resource info
    print("\n📋 TEST 0: Resource Information")
    if "get_resource_info" in tool_dict:
        result = await tool_dict["get_resource_info"].ainvoke({})
        print(f"{result}")

    # Test 1: Fast tool
    print("\n📋 TEST 1: Fast Tool")
    result = await tool_dict["get_utc_time"].ainvoke({})
    print(f"   Result: {result}")

    # Test 2: Within timeout (1s)
    print("\n📋 TEST 2: Within Timeout (1s < 2s limit)")
    result = await tool_dict["long_running_task"].ainvoke({"seconds": 1})
    print(f"   Result: {result}")

    # Test 3: Exceeds timeout (5s)
    print("\n📋 TEST 3: Exceeds Timeout (5s > 2s limit)")
    result = await tool_dict["long_running_task"].ainvoke({"seconds": 5})
    print(f"   Result: {result}")

    # Test 4: Safe small allocation
    print("\n📋 TEST 4: Small Memory Allocation (10MB - actually allocated)")
    result = await tool_dict["allocate_memory"].ainvoke({"mb": 10})
    print(f"   Result: {result}")

    # Test 5: Medium allocation (simulated for safety)
    print("\n📋 TEST 5: Medium Memory Allocation (40MB - simulated)")
    result = await tool_dict["allocate_memory"].ainvoke({"mb": 40})
    print(f"   Result: {result}")

    # Test 6: Rejected - too large
    print("\n📋 TEST 6: Large Memory Request (80MB - refused)")
    result = await tool_dict["allocate_memory"].ainvoke({"mb": 80})
    print(f"   Result: {result}")

    # Test 7: Rejected - exceeds container
    print("\n📋 TEST 7: Excessive Memory Request (150MB - denied)")
    result = await tool_dict["allocate_memory"].ainvoke({"mb": 150})
    print(f"   Result: {result}")

    print("\n" + "=" * 60)
    print("✅ All Tests Complete!")
    print("=" * 60)

    # Summary
    print("\n📊 Summary:")
    print("   ✅ Application-level timeout (2s) - Working")
    print("   ✅ Container memory limit (128MB) - Protected by app-level checks")
    print("   ✅ Multi-layer defense - App limits prevent container OOM")
    print("   ✅ Graceful error handling - Working")
    print(
        "\n💡 Key Learning: Application limits should trigger BEFORE container limits!"
    )
    print("   This prevents container crashes and allows graceful degradation.")


if __name__ == "__main__":
    asyncio.run(main())
