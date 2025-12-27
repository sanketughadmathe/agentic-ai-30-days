import os
from typing import Annotated, List, Literal, TypedDict
from operator import add

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


# -----------------------------
# 1. Define state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add]


# -----------------------------
# 2. Define a simple tool
# -----------------------------
@tool
def get_current_time() -> str:
    """Returns the current time in UTC."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


tools = [get_current_time]
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
# 4. Agent (think + act)
# -----------------------------
def agent(state: AgentState) -> AgentState:
    # Bind tools to the LLM so it knows what tools are available
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    # Return just the new message - the reducer will append it
    return {"messages": [response]}


# -----------------------------
# 5. Decide next step
# -----------------------------
def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]

    # If the LLM requested a tool → execute it
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Otherwise stop
    return END


# -----------------------------
# 6. Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("tools", "agent")

graph = builder.compile()


# -----------------------------
# 7. Run
# -----------------------------
if __name__ == "__main__":
    initial_state = {"messages": [HumanMessage(content="What is the current time?")]}

    result = graph.invoke(initial_state)
    graph.get_graph().draw_mermaid_png(
        output_file_path="day03_react_agent/react_agent.png"
    )

    for msg in result["messages"]:
        role = msg.__class__.__name__
        content = msg.content if msg.content else ""

        # Show tool calls if present
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_info = f" [Tool Calls: {[tc['name'] for tc in msg.tool_calls]}]"
            print(f"{role}: {content}{tool_info}")
        else:
            print(f"{role}: {content}")

        # HumanMessage: What is the current time?
        # AIMessage:  [Tool Calls: ['get_current_time']]
        # ToolMessage: 2025-12-27T13:28:41.256365+00:00
        # AIMessage: The current time is **13:28 UTC** on **December 27, 2025**.
