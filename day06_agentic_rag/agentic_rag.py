import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
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
    needs_web_search: bool


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
# 3. Retrieve node (vector store stub)
# -----------------------------
def retrieve(state: GraphState) -> GraphState:
    # Simulated vector store retrieval
    docs = [
        Document(
            page_content="Agent memory allows LLM agents to retain context across steps."
        )
    ]

    return {"documents": docs, "needs_web_search": False}


# -----------------------------
# 4. Relevance grading node
# -----------------------------
def grade_documents(state: GraphState) -> GraphState:
    question = state["question"]
    docs = state["documents"]

    grading_prompt = (
        "Determine if the following documents are relevant "
        "to answering the question.\n\n"
        f"Question: {question}\n\n"
        f"Documents: {[d.page_content for d in docs]}\n\n"
        "Respond with YES or NO."
    )

    result = llm.invoke(grading_prompt)

    needs_web_search = result.content.strip().upper() != "YES"

    return {"needs_web_search": needs_web_search}


# -----------------------------
# 5. Web search fallback node (stub)
# -----------------------------
def web_search(state: GraphState) -> GraphState:
    docs = state["documents"] + [
        Document(
            page_content="Web search: Agent memory is a mechanism to store and recall intermediate reasoning steps."
        )
    ]

    return {"documents": docs}


# -----------------------------
# 6. Generate answer node
# -----------------------------
def generate(state: GraphState) -> GraphState:
    context = "\n".join(doc.page_content for doc in state["documents"])

    prompt = (
        f"Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{state['question']}"
    )

    answer = llm.invoke(prompt).content

    return {"answer": answer}


# -----------------------------
# 7. Routing logic
# -----------------------------
def decide_next_step(state: GraphState) -> Literal["web_search", "generate"]:
    if state["needs_web_search"]:
        return "web_search"
    return "generate"


# -----------------------------
# 8. Build graph
# -----------------------------
builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("web_search", web_search)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")

builder.add_edge("retrieve", "grade_documents")

builder.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {"web_search": "web_search", "generate": "generate"},
)

builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


# -----------------------------
# 9. Run
# -----------------------------
if __name__ == "__main__":
    initial_state = {
        "question": "What is agent memory?",
        "documents": [],
        "answer": "",
        "needs_web_search": False,
    }

    result = graph.invoke(initial_state)

    graph.get_graph().draw_mermaid_png(
        output_file_path="day06_agentic_rag/agentic_rag.png"
    )

    print("\nFinal Answer:\n")
    print(result["answer"])
