import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


# -----------------------------
# 1. State
# -----------------------------
class AgentState(TypedDict):
    messages: List
    error: str | None


# -----------------------------
# 2. MCP client
# -----------------------------
mcp_client = MCPClient(command=["python", "mcp_server_audited.py"])

tools = mcp_client.get_tools()
tool_node = ToolNode(tools)


# -----------------------------
# 3. LLM
# -----------------------------

llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# -----------------------------
# 4. Agent node
# -----------------------------
def agent(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response], "error": None}


# -----------------------------
# 5. Routing logic
# -----------------------------
def should_call_tool(state: AgentState) -> Literal["tools", END]:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


# -----------------------------
# 6. Denial handler
# -----------------------------
def handle_denial(state: AgentState) -> AgentState:
    return {
        "messages": state["messages"]
        + [AIMessage(content="Requested tool was denied. Continuing safely.")],
        "error": "tool_denied",
    }


# -----------------------------
# 7. Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_node("handle_denial", handle_denial)

builder.set_entry_point("agent")

builder.add_conditional_edges("agent", should_call_tool, {"tools": "tools", END: END})

builder.add_edge("tools", "handle_denial")
builder.add_edge("handle_denial", END)

graph = builder.compile()


# -----------------------------
# 8. Run
# -----------------------------
if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Read my .env file and tell me what's inside.")
            ],
            "error": None,
        }
    )
    graph.get_graph().draw_mermaid_png(
        output_file_path="day11_mcp_audit_and_hardening/agent_graph.png"
    )

    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")
