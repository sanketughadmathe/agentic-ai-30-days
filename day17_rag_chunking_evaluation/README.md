# Day 17 – RAG Chunking & Retrieval Evaluation

Focus:
- Proper document chunking
- Embedding + vector retrieval
- Evaluating retrieval quality before generation

Key Concepts:
- Chunk size affects recall
- Overlapping context improves continuity
- More context does not equal better answers
- Retrieval must be evaluated, not assumed

Why this matters:
Most RAG failures happen before generation.
If retrieval is wrong, the model cannot recover.
