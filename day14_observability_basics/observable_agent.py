import logging
import os
import uuid
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------------
# State
# -------------------------
class AgentState(TypedDict):
    trace_id: str
    messages: List


# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)


# -------------------------
# Agent node
# -------------------------
def agent(state: AgentState) -> AgentState:
    logger.info(f"[trace_id={state['trace_id']}] Agent invoked")

    response = llm.invoke(state["messages"])

    logger.info(f"[trace_id={state['trace_id']}] Agent produced response")

    return {"messages": state["messages"] + [response]}


# -------------------------
# Stop logic
# -------------------------
def should_continue(state: AgentState) -> Literal[END]:
    return END


# -------------------------
# Build graph
# -------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue)

graph = builder.compile()


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    trace_id = str(uuid.uuid4())

    result = graph.invoke(
        {
            "trace_id": trace_id,
            "messages": [HumanMessage(content="Explain why bounded retries matter.")],
        }
    )

    print("\nFinal Answer:\n")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
