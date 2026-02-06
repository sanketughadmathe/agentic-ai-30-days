import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

# -----------------------------
# Choose which MCP server to use
# -----------------------------
mode = (sys.argv[1] if len(sys.argv) > 1 else "restricted").lower()

if mode not in {"restricted", "unrestricted"}:
    raise SystemExit("Usage: python mcp_client_test.py [restricted|unrestricted]")

script = (
    "mcp_server_unrestricted.py"
    if mode == "unrestricted"
    else "mcp_server_restricted.py"
)
server_name = f"{mode}-mcp"

# Switch between servers to see behavior change
mcp_client = MultiServerMCPClient(
    {
        server_name: {
            "transport": "stdio",
            "command": sys.executable,
            "args": [script],
            "cwd": os.path.dirname(__file__),
        }
    }
)

tools = asyncio.run(mcp_client.get_tools())

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
).bind_tools(tools)


async def _run_tool(name: str, args: dict):
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return f"Unknown tool: {name}"
    return await tool.ainvoke(args)


async def _main():
    messages = [HumanMessage(content="List available files and read one.")]

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
