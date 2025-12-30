# Day 6 – Agentic RAG with LangGraph

Focus:
- Retrieval as a step, not an assumption
- Relevance grading before generation
- Explicit fallback to web search
- Deterministic routing via graph edges

Key Concepts:
- RAG is a control-flow problem
- Not all retrieved docs are useful
- Routing decisions should be explicit
- Fallbacks must be intentional

Why this matters:
Naive RAG fails silently when retrieval
is weak or misleading.

Agentic RAG makes those failures visible
and correctable.
