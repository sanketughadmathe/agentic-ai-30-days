import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables from .env file
load_dotenv()


# -----------------------------
# 1. Define the state
# -----------------------------
class GraphState(TypedDict):
    message: str
    response: str


# -----------------------------
# 2. Initialize LLM
# -----------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)


# -----------------------------
# 3. Define a node
# -----------------------------
def hello_agent(state: GraphState) -> GraphState:
    prompt = f"Reply politely to this message: {state['message']}"
    result = llm.invoke(prompt)

    return {"response": result.content}


# -----------------------------
# 4. Build the graph
# -----------------------------
builder = StateGraph(GraphState)

builder.add_node("hello_agent", hello_agent)

builder.set_entry_point("hello_agent")
builder.add_edge("hello_agent", END)

graph = builder.compile()


# -----------------------------
# 5. Run the graph
# -----------------------------
if __name__ == "__main__":
    input_state = {"message": "Hello, LangGraph!"}

    output = graph.invoke(input_state)

    print("Final Output:")
    print(output["response"])
