# Day 15 – Structured Outputs & Contracts

Focus:
- Enforcing schemas on model outputs
- Validating with Pydantic
- Regenerating on schema failure

Key Concepts:
- Text is not a contract
- Schemas are contracts
- Validation belongs in the system layer
- Regeneration should be controlled

Why this matters:
Most production failures come from
unexpected model output shapes.

Schemas turn probabilistic text
into deterministic interfaces.
