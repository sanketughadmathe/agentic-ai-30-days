# Day 20 – RAG Guardrails & Failure Handling

Focus:
- Grounding validation
- Confidence scoring
- Explicit fallback handling
- Detecting hallucination risk

Key Concepts:
- Retrieval quality does not guarantee correctness
- Answers must be checked for grounding
- Low-confidence responses should not be returned
- Safe systems must know when to abstain

Why this matters:
In production, the most dangerous answer
is a confident but unsupported one.
