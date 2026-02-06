import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

# -----------------------------
# 1. Connect to MCP server (MultiServerMCPClient API)
# -----------------------------
mcp_client = MultiServerMCPClient(
    {
        "time-mcp": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["mcp_server.py"],
            "cwd": os.path.dirname(__file__),
        }
    }
)

tools = asyncio.run(mcp_client.get_tools())


# -----------------------------
# 2. LLM bound to MCP tools
# -----------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
).bind_tools(tools)


# -----------------------------
# 3. Simple agent loop (invoke → execute tools → invoke until done)
# MCP tools are async-only, so we use ainvoke throughout.
# -----------------------------
async def _run_tool(name: str, args: dict):
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return f"Unknown tool: {name}"
    return await tool.ainvoke(args)


async def _main():
    messages = [HumanMessage(content="What is the current time?")]

    while True:
        response = await llm.ainvoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for tc in response.tool_calls:
            result = await _run_tool(tc["name"], tc.get("args") or {})
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    print(response.content or "(no content)")


if __name__ == "__main__":
    asyncio.run(_main())
