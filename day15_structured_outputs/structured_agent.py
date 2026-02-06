import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

load_dotenv()


# -----------------------------
# 1. Output schema
# -----------------------------
class AnswerSchema(BaseModel):
    answer: str
    confidence: float


# -----------------------------
# 2. State
# -----------------------------
class AgentState(TypedDict):
    messages: List
    parsed: AnswerSchema | None
    error: str | None


# -----------------------------
# 3. LLM
# -----------------------------
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# -----------------------------
# 4. Generation node
# -----------------------------
def generate(state: AgentState) -> AgentState:
    prompt = (
        "Answer the question and return JSON with:\n"
        "{ answer: string, confidence: number between 0 and 1 }\n\n"
        f"Question: {state['messages'][0].content}"
    )

    response = llm.invoke(prompt)

    return {"messages": state["messages"] + [response], "parsed": None, "error": None}


# -----------------------------
# 5. Parse + validate
# -----------------------------
def parse(state: AgentState) -> AgentState:
    last = state["messages"][-1]

    try:
        data = AnswerSchema.model_validate_json(last.content)
        return {"parsed": data, "error": None}
    except ValidationError as e:
        return {"parsed": None, "error": str(e)}


# -----------------------------
# 6. Decide next step
# -----------------------------
def decide(state: AgentState) -> Literal["regenerate", END]:  # type: ignore
    if state["parsed"] is not None:
        return END
    return "regenerate"


# -----------------------------
# 7. Regenerate
# -----------------------------
def regenerate(state: AgentState) -> AgentState:
    correction = (
        "Return ONLY valid JSON matching this schema:\n"
        "{ answer: string, confidence: number }\n"
    )

    response = llm.invoke(state["messages"] + [HumanMessage(content=correction)])

    return {"messages": state["messages"] + [response]}


# -----------------------------
# 8. Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("generate", generate)
builder.add_node("parse", parse)
builder.add_node("regenerate", regenerate)

builder.set_entry_point("generate")
builder.add_edge("generate", "parse")

builder.add_conditional_edges("parse", decide, {"regenerate": "regenerate", END: END})

builder.add_edge("regenerate", "parse")

graph = builder.compile()


# -----------------------------
# 9. Run
# -----------------------------
if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Why are bounded retries important?")],
            "parsed": None,
            "error": None,
        }
    )
    # Save as PNG
    graph.get_graph().draw_mermaid_png(
        output_file_path="day15_structured_outputs/structured_agent.png"
    )

    print("\nValidated Output:\n")
    print(result["parsed"])
