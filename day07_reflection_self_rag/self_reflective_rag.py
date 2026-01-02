import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()


# -----------------------------
# 1. Graph state
# -----------------------------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    grounded: bool
    iterations: int


MAX_ITERATIONS = 2


# -----------------------------
# 2. LLM
# -----------------------------
llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# -----------------------------
# 3. Retrieve node (stub)
# -----------------------------
def retrieve(state: GraphState) -> GraphState:
    docs = [
        Document(
            page_content="Agent memory allows LLM agents to store and recall intermediate information across steps."
        )
    ]

    return {"documents": docs}


# -----------------------------
# 4. Generate answer
# -----------------------------
def generate(state: GraphState) -> GraphState:
    context = "\n".join(doc.page_content for doc in state["documents"])

    prompt = (
        "Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{state['question']}"
    )

    answer = llm.invoke(prompt).content

    return {"answer": answer}


# -----------------------------
# 5. Reflection / grounding check
# -----------------------------
def reflect(state: GraphState) -> GraphState:
    prompt = (
        "Check whether the answer is fully grounded in the provided context.\n\n"
        f"Context:\n{[d.page_content for d in state['documents']]}\n\n"
        f"Answer:\n{state['answer']}\n\n"
        "Respond with YES or NO."
    )

    result = llm.invoke(prompt).content.strip().upper()

    grounded = result == "YES"

    return {"grounded": grounded}


# -----------------------------
# 6. Decide next step
# -----------------------------
def decide_next_step(state: GraphState) -> Literal["regenerate", END]:
    if state["grounded"]:
        return END

    if state["iterations"] >= MAX_ITERATIONS:
        return END

    return "regenerate"


# -----------------------------
# 7. Regeneration node
# -----------------------------
def regenerate(state: GraphState) -> GraphState:
    critique = (
        "The previous answer was not fully grounded in the context. "
        "Regenerate a grounded answer using only the provided documents."
    )

    revised_prompt = (
        f"{critique}\n\n"
        f"Context:\n{[d.page_content for d in state['documents']]}\n\n"
        f"Question:\n{state['question']}"
    )

    answer = llm.invoke(revised_prompt).content

    return {"answer": answer, "iterations": state["iterations"] + 1}


# -----------------------------
# 8. Build graph
# -----------------------------
builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_node("regenerate", regenerate)

builder.set_entry_point("retrieve")

builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "reflect")

builder.add_conditional_edges(
    "reflect", decide_next_step, {"regenerate": "regenerate", END: END}
)

builder.add_edge("regenerate", "reflect")

graph = builder.compile()


# -----------------------------
# 9. Run
# -----------------------------
if __name__ == "__main__":
    initial_state = {
        "question": "What is agent memory?",
        "documents": [],
        "answer": "",
        "grounded": False,
        "iterations": 0,
    }

    result = graph.invoke(initial_state)
    graph.get_graph().draw_mermaid_png(
        output_file_path="day07_reflection_self_rag/self_reflective_rag.png"
    )

    print("\nFinal Answer:\n")
    print(result["answer"])
