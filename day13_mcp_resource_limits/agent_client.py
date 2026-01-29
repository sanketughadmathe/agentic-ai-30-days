"""
Agent Client for Testing Resource-Limited MCP Server
Tests timeout handling and resource constraints.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


async def test_tool(llm, tools, query, description):
    """Test a specific tool with a query"""
    print(f"\n{'=' * 60}")
    print(f"TEST: {description}")
    print(f"{'=' * 60}")
    print(f"Query: {query}\n")

    messages = [HumanMessage(content=query)]

    # Get LLM response with rate limit handling
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"\n\n❌ Error during LLM invocation: {e}\n\n")
        if "rate" in str(e).lower() or "quota" in str(e).lower():
            print(f"⚠️  Rate limit hit. Waiting 60 seconds...")
            await asyncio.sleep(60)
            response = llm.invoke(messages)
        else:
            raise

    messages.append(response)

    # Execute tool calls if any
    if response.tool_calls:
        print("🔧 Tool Calls:")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"  - {tool_name}({tool_args})")

            try:
                # Find and invoke the tool
                tool_to_call = next((t for t in tools if t.name == tool_name), None)
                if tool_to_call:
                    result = await tool_to_call.ainvoke(tool_args)
                    print(f"    Result: {result}")

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_id)
                    )
            except Exception as e:
                print(f"    ❌ Error: {e}")
                messages.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=tool_id)
                )

        # Get final response
        final = llm.invoke(messages)
        print(f"\n🤖 Assistant: {final.content}")
    else:
        print(f"🤖 Assistant: {response.content}")


async def main():
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not set")
        return

    print("🚀 Starting Resource-Limited MCP Test Suite")
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
        print(f"      - {tool.name}")

    # Create LLM
    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0,
    ).bind_tools(tools)

    # Test Suite (with delays to avoid rate limits)
    await test_tool(llm, tools, "What is the current UTC time?", "Safe Tool (Fast)")
    await asyncio.sleep(2)  # Delay between tests

    await test_tool(
        llm,
        tools,
        "Run long_running_task for 1 second",
        "Within Timeout Limit (1s < 2s limit)",
    )
    await asyncio.sleep(2)

    await test_tool(
        llm,
        tools,
        "Run long_running_task for 5 seconds",
        "Exceeds Timeout Limit (5s > 2s limit)",
    )
    await asyncio.sleep(2)

    await test_tool(
        llm,
        tools,
        "Allocate 50MB of memory",
        "Within Memory Limit (50MB < 128MB limit)",
    )
    await asyncio.sleep(2)

    await test_tool(
        llm,
        tools,
        "Allocate 200MB of memory",
        "Exceeds Memory Limit (200MB > 128MB limit)",
    )

    print(f"\n{'=' * 60}")
    print("✅ Test Suite Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
