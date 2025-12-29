# Day 5 – Actor–Evaluator Pattern

Focus:
- Separate generation from evaluation
- Treat judgment as a first-class step
- Avoid infinite self-reflection loops

Key Concepts:
- Actor generates content
- Evaluator scores quality independently
- Conditional edge controls revision
- Iterations are explicitly bounded

Why this matters:
Most agents fail because they try to
generate and judge in the same step.

Separating concerns improves reliability,
debuggability, and control.
