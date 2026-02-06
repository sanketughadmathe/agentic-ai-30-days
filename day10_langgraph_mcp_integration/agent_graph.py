import asyncio
import json
import os
import sys
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


# -----------------------------
# 1. State
# -----------------------------
class AgentState(TypedDict):
    messages: List


# -----------------------------
# 2. MCP client (tool boundary)
# -----------------------------
# Use the new MultiServerMCPClient API; one stdio server for this demo.
mcp_client = MultiServerMCPClient(
    {
        "utility-mcp": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["mcp_server.py"],
            "cwd": os.path.dirname(__file__),
        }
    }
)

# Fetch tools synchronously at startup for simplicity.
tools = asyncio.run(mcp_client.get_tools())
tool_node = ToolNode(tools)


# -----------------------------
# 3. LLM (tool-aware)
# -----------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
).bind_tools(tools)


# -----------------------------
# 4. Agent node (decides)
# -----------------------------
async def agent(state: AgentState) -> AgentState:
    # Use async LLM to keep the whole graph async-friendly.
    # Some providers (like Mistral via OpenRouter) are strict about message roles
    # and tool content formats, so we convert the conversation into a simple
    # user-only history before sending it to the model.

    last_human: HumanMessage | None = None
    last_tool: ToolMessage | None = None

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            last_human = msg
        elif isinstance(msg, ToolMessage):
            last_tool = msg

    prompt_messages: List = []
    if last_human is not None:
        prompt_messages.append(last_human)

    if last_tool is not None:
        tool_content = last_tool.content
        if not isinstance(tool_content, str):
            tool_content = json.dumps(tool_content)
        prompt_messages.append(
            HumanMessage(content=f"Tool '{last_tool.name}' returned: {tool_content}")
        )

    response = await llm.ainvoke(prompt_messages or state["messages"])
    return {"messages": state["messages"] + [response]}


# -----------------------------
# 5. Routing logic
# -----------------------------
def should_use_tools(state: AgentState) -> Literal["tools", END]:
    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"

    return END


# -----------------------------
# 6. Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")

builder.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})

builder.add_edge("tools", "agent")

graph = builder.compile()


# -----------------------------
# 7. Run (async)
# -----------------------------
async def _main() -> None:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What time is it right now?")]}
    )

    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(_main())
