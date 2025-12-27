from typing import TypedDict, List, Annotated
from operator import add

from langgraph.graph import StateGraph, END


# -----------------------------
# 1. State with reducer
# -----------------------------
class GraphState(TypedDict):
    events: Annotated[List[str], add]


# -----------------------------
# 2. Nodes
# -----------------------------
def node_a(state: GraphState) -> GraphState:
    return {"events": ["event from node A"]}


def node_b(state: GraphState) -> GraphState:
    return {"events": ["event from node B"]}


# -----------------------------
# 3. Graph with reducer
# -----------------------------
builder = StateGraph(GraphState)

builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

builder.set_entry_point("node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile()


# -----------------------------
# 4. Run
# -----------------------------
if __name__ == "__main__":
    result = graph.invoke({"events": []})
    # {"events": ["event from node A", "event from node B"]}
    # Save as PNG
    graph.get_graph().draw_mermaid_png(
        output_file_path="day02_graph_state/reducers_example.png"
    )
    print(result)
