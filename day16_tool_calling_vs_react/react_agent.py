import os
from datetime import datetime, timezone
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    # Use add_messages reducer to handle message appending automatically
    messages: Annotated[List, add_messages]


# -----------------------------
# Tool
# -----------------------------
@tool
def get_utc_time() -> str:
    """Return the current UTC time."""
    return datetime.now(timezone.utc).isoformat()


tools = [get_utc_time]
tool_node = ToolNode(tools)


# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
).bind_tools(tools)


# -----------------------------
# Agent node
# -----------------------------
def agent(state: AgentState) -> AgentState:
    """Call the LLM with the current message history."""
    response = llm.invoke(state["messages"])
    # With add_messages reducer, just return the new message
    return {"messages": [response]}


# -----------------------------
# Routing
# -----------------------------
def route(state: AgentState) -> Literal["tools", END]:
    """Route to tools if the agent called them, otherwise end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# -----------------------------
# Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")

builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the current time?")]}
    )
    graph.get_graph().draw_mermaid_png(
        output_file_path="day16_tool_calling_vs_react/react_agent.png"
    )

    # Print conversation
    print("=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)
    for i, msg in enumerate(result["messages"], 1):
        msg_type = msg.__class__.__name__
        content = msg.content

        # Show tool calls if present
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"\n{i}. {msg_type}:")
            print(f"   Tool Calls: {msg.tool_calls}")
        else:
            print(f"\n{i}. {msg_type}: {content}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["messages"][-1].content)
