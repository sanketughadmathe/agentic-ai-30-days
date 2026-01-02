# Day 7 – Reflection & Self-Corrective RAG

Focus:
- Reflect on generated answers
- Check grounding against context
- Regenerate when answers hallucinate
- Bound self-correction loops

Key Concepts:
- Reflection is a system step, not a prompt trick
- Grounding must be checked explicitly
- Regeneration should be controlled
- Stopping conditions are mandatory

Why this matters:
Most RAG failures happen *after* generation.
Self-reflection makes those failures visible
and correctable.
