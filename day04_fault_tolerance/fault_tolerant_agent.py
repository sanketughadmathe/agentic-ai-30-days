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
    retries: int
    error: str | None


MAX_RETRIES = 2


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
# 3. Agent node (can fail)
# -----------------------------
def agent(state: AgentState) -> AgentState:
    try:
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response], "error": None}
    except Exception as e:
        return {"messages": state["messages"], "error": str(e)}


# -----------------------------
# 4. Retry decision logic
# -----------------------------
def decide_next_step(state: AgentState) -> Literal["retry", END]:
    if state["error"] is None:
        return END

    if state["retries"] >= MAX_RETRIES:
        return END

    return "retry"


# -----------------------------
# 5. Retry node
# -----------------------------
def retry(state: AgentState) -> AgentState:
    return {"retries": state["retries"] + 1}


# -----------------------------
# 6. Build graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("retry", retry)

builder.set_entry_point("agent")

builder.add_conditional_edges("agent", decide_next_step, {"retry": "retry", END: END})

builder.add_edge("retry", "agent")

graph = builder.compile()


# -----------------------------
# 7. Run
# -----------------------------
if __name__ == "__main__":
    initial_state = {
        "messages": [
            HumanMessage(content="Explain why retries are dangerous in agent systems.")
        ],
        "retries": 0,
        "error": None,
    }

    result = graph.invoke(initial_state)
    graph.get_graph().draw_mermaid_png(
        output_file_path="day04_fault_tolerance/fault_tolerant_agent.png"
    )

    print("Retries:", result["retries"])

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print("\nFinal Output:\n")
            print(msg.content)
            # Retries: 0

            # Final Output:

            # Retries in agent systems can be dangerous for several reasons, particularly when agents interact with external systems, shared resources, or other agents. Here’s why:

            # ### 1. **Amplification of Failures (Retry Storms)**
            # - If multiple agents retry failed operations simultaneously (e.g., due to a temporary outage), they can overwhelm the target system when it recovers, causing a **thundering herd problem**.
            # - Example: A database under load may recover briefly, only to be crushed again by a flood of retries,
            # leading to cascading failures.

            # ### 2. **Non-Idempotent Operations**
            # - Retrying non-idempotent operations (e.g., financial transactions, state changes) can lead to **duplicate side effects**, such as:
            #     - Double-charging a customer.
            #     - Creating duplicate records.
            #     - Overwriting data incorrectly.
            # - Example: An agent retries a "send payment" API call, resulting in multiple charges.

            # ### 3. **State Inconsistency**
            # - Agents often rely on shared state (e.g., databases, caches). Retries can cause **race conditions** or **stale data issues** if the system’s state changes between attempts.
            # - Example: An agent retries a "reserve inventory" call after the item is already sold, leading to overselling.

            # ### 4. **Resource Exhaustion**
            # - Persistent retries consume **CPU, memory, network bandwidth**, and other resources, degrading system performance.
            # - Example: A misconfigured retry loop in a microservice could exhaust connection pools or thread resources.

            # ### 5. **Deadlocks and Livelocks**
            # - If agents depend on each other (e.g., distributed transactions), retries can create **circular dependencies**, leading to deadlocks.
            # - Livelocks occur when agents keep retrying in a way that perpetually blocks progress (e.g., two agents retrying conflicting updates).

            # ### 6. **Delayed Error Handling**
            # - Retries can **mask underlying problems** (e.g., misconfigurations, bugs, or permanent failures), delaying proper error handling or alerts.
            # - Example: An agent retries a failed authentication indefinitely instead of notifying an admin.

            # ### 7. **Violation of SLA/QoS Guarantees**
            # - Excessive retries can cause **latency spikes** or **timeout violations**, breaking service-level agreements (SLAs).
            # - Example: A real-time system (e.g., trading) may miss critical deadlines due to retry delays.

            # ### 8. **Costly External Dependencies**
            # - Retrying calls to **paid APIs** (e.g., cloud services, third-party APIs) can incur **unexpected costs**.
            # - Example: A cloud function retrying a failed AI API call 100 times, racking up charges.

            # ### 9. **Feedback Loops in Distributed Systems**
            # - In **multi-agent systems**, retries can create **positive feedback loops** where failures in one agent trigger retries that cause failures in others.
            # - Example: Agent A retries a call to Agent B, which is now overloaded and fails, causing Agent C to also retry.

            # ### 10. **Security Risks**
            # - Retries can be exploited in **denial-of-service (DoS) attacks** or **brute-force attacks** (e.g., retrying password guesses).
            # - Example: An attacker triggers retries to exhaust system resources.

            # ---

            # ### **Mitigation Strategies**
            # To safely use retries in agent systems:
            # 1. **Use Exponential Backoff** – Space out retries to reduce load (e.g., 1s, 2s, 4s delays).
            # 2. **Limit Retry Attempts** – Cap the number of retries to avoid infinite loops.
            # 3. **Idempotency Keys** – Ensure retries of non-idempotent operations are safe (e.g., deduplication IDs).4. **Circuit Breakers** – Temporarily stop retries if failures exceed a threshold (e.g., using **Hystrix** or **Resilience4j**).
            # 5. **Jitter** – Add randomness to retry delays to prevent synchronized retries.
            # 6. **Fail Fast** – For permanent errors (e.g., "404 Not Found"), fail immediately instead of retrying.
            # 7. **Monitor & Alert** – Track retry metrics to detect abnormal behavior early.
            # 8. **Distributed Coordination** – Use locks or consensus (e.g., **etcd**, **ZooKeeper**) to prevent conflicting retries.

            # ### **When to Avoid Retries**
            # - For **non-recoverable errors** (e.g., invalid input, authentication failures).
            # - In **real-time systems** where latency is critical.
            # - When the **cost of retries outweighs the benefit** (e.g., high API costs).

            # ### **Conclusion**
            # While retries can improve resilience, they must be **carefully designed** to avoid introducing new failure modes. The key is to balance persistence with safety, ensuring retries don’t turn a minor issue into a
            # system-wide outage.
