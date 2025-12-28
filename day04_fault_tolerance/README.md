# Day 4 – Fault Tolerance & Retries in LangGraph

Focus:
- Failure-aware agent design
- Explicit retry logic
- Bounded loops
- Deterministic stopping conditions

Key Concepts:
- Failures are part of normal execution
- Retries must be bounded
- Retry decisions belong in orchestration, not prompts
- Graphs make failure paths explicit

Why this matters:
Most agent failures in production come from
uncontrolled retries and hidden loops.

LangGraph forces retry logic to be modeled explicitly.
