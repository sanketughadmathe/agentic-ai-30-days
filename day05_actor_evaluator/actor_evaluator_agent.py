import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()


# -----------------------------
# 1. Define state
# -----------------------------
class AgentState(TypedDict):
    messages: List
    score: int
    iterations: int


MAX_ITERATIONS = 2
QUALITY_THRESHOLD = 7


# -----------------------------
# 2. LLMs (separate roles)
# -----------------------------
actor_llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
)

evaluator_llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# -----------------------------
# 3. Actor node (generation)
# -----------------------------
def actor(state: AgentState) -> AgentState:
    prompt = state["messages"]

    response = actor_llm.invoke(prompt)

    return {"messages": state["messages"] + [response]}


# -----------------------------
# 4. Evaluator node (judgment)
# -----------------------------
def evaluator(state: AgentState) -> AgentState:
    last_answer = state["messages"][-1].content

    eval_prompt = [
        HumanMessage(
            content=(
                "You are an evaluator. Score the answer from 1–10 "
                "based on correctness, clarity, and completeness.\n\n"
                f"Answer:\n{last_answer}\n\n"
                "Respond with only a number."
            )
        )
    ]

    score_msg = evaluator_llm.invoke(eval_prompt)
    score = int(score_msg.content.strip())

    return {"score": score}


# -----------------------------
# 5. Decide whether to revise
# -----------------------------
def should_continue(state: AgentState) -> Literal["revise", END]:
    if state["score"] >= QUALITY_THRESHOLD:
        return END

    if state["iterations"] >= MAX_ITERATIONS:
        return END

    return "revise"


# -----------------------------
# 6. Revision node
# -----------------------------
def revise(state: AgentState) -> AgentState:
    critique = (
        f"The previous answer scored {state['score']}/10. "
        "Improve clarity, correctness, and completeness."
    )

    return {
        "messages": state["messages"] + [HumanMessage(content=critique)],
        "iterations": state["iterations"] + 1,
    }


# -----------------------------
# 7. Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("actor", actor)
builder.add_node("evaluator", evaluator)
builder.add_node("revise", revise)

builder.set_entry_point("actor")

builder.add_edge("actor", "evaluator")

builder.add_conditional_edges(
    "evaluator", should_continue, {"revise": "revise", END: END}
)

builder.add_edge("revise", "actor")

graph = builder.compile()


# -----------------------------
# 8. Run
# -----------------------------
if __name__ == "__main__":
    initial_state = {
        "messages": [
            HumanMessage(content="Explain why retries are dangerous in agent systems.")
        ],
        "score": 0,
        "iterations": 0,
    }

    result = graph.invoke(initial_state)
    graph.get_graph().draw_mermaid_png(
        output_file_path="day05_actor_evaluator/actor_evaluator_agent.png"
    )

    print(f"\nFinal score: {result['score']}\n")

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
