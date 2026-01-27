"""
Agent Client for Dockerized MCP Server
This script demonstrates how to use an LLM with tools from a dockerized MCP server.
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import os

from dotenv import load_dotenv

load_dotenv()


async def main():
    # Check for OpenAI API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        return

    print("🚀 Starting Agent Client...")
    print("=" * 60)

    # 1. Connect to MCP server running in Docker
    print("\n1️⃣  Connecting to MCP server at http://localhost:3333/mcp")
    try:
        mcp_client = MultiServerMCPClient(
            {
                "dockerized-mcp": {
                    "url": "http://localhost:3333/mcp",
                    "transport": "http",
                }
            }
        )
        print("   ✅ Connected successfully")
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        return

    # 2. Get available tools from MCP server
    print("\n2️⃣  Fetching available tools...")
    try:
        tools = await mcp_client.get_tools()
        print(f"   ✅ Found {len(tools)} tool(s):")
        for tool in tools:
            print(f"      - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"   ❌ Failed to fetch tools: {e}")
        return

    # 3. Create LLM with tools
    print("\n3️⃣  Initializing LLM with tools...")
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0,
    ).bind_tools(tools)
    print("   ✅ LLM initialized")

    # 4. Run agent loop
    print("\n4️⃣  Running agent conversation...")
    print("=" * 60)

    # Initial query
    user_query = "What is the current UTC time?"
    print(f"\n👤 User: {user_query}")

    messages = [HumanMessage(content=user_query)]

    # Agent loop - allows multiple tool calls if needed
    max_iterations = 5
    for iteration in range(max_iterations):
        # Get LLM response
        response = llm.invoke(messages)
        messages.append(response)

        # Check if LLM wants to use tools
        if not response.tool_calls:
            # No tool calls - LLM has final answer
            print(f"\n🤖 Assistant: {response.content}")
            break

        # Execute tool calls
        print(f"\n🔧 Tool Calls (iteration {iteration + 1}):")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"   - Calling: {tool_name}")
            print(f"     Args: {tool_args}")

            try:
                # Find and invoke the tool
                tool_to_call = next((t for t in tools if t.name == tool_name), None)
                if tool_to_call:
                    result = await tool_to_call.ainvoke(tool_args)
                    print(f"     Result: {result}")

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_id)
                    )
                else:
                    print(f"     ❌ Tool not found: {tool_name}")
            except Exception as e:
                print(f"     ❌ Error: {e}")
                messages.append(
                    ToolMessage(
                        content=f"Error calling tool: {e}", tool_call_id=tool_id
                    )
                )

    print("\n" + "=" * 60)
    print("✅ Agent conversation complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
