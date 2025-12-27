from typing import TypedDict

from langgraph.graph import StateGraph, END


# -----------------------------
# 1. Define Graph State
# -----------------------------
class GraphState(TypedDict):
    input_text: str
    step_1: str
    step_2: str


# -----------------------------
# 2. Define nodes
# -----------------------------
def step_one(state: GraphState) -> GraphState:
    return {"step_1": f"Processed step 1: {state['input_text']}"}


def step_two(state: GraphState) -> GraphState:
    return {"step_2": f"Processed step 2 after -> {state['step_1']}"}


# -----------------------------
# 3. Build graph
# -----------------------------
builder = StateGraph(GraphState)

builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)

builder.set_entry_point("step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)

graph = builder.compile()


# -----------------------------
# 4. Run graph
# -----------------------------
if __name__ == "__main__":
    result = graph.invoke({"input_text": "Hello LangGraph state!"})
    # {'input_text': 'Hello LangGraph state!', 'step_1': 'Processed step 1: Hello LangGraph state!', 'step_2': 'Processed step 2 after -> Processed step 1: Hello LangGraph state!'}
    # Save as PNG
    graph.get_graph().draw_mermaid_png(
        output_file_path="day02_graph_state/graph_state_basics.png"
    )
    print(result)
