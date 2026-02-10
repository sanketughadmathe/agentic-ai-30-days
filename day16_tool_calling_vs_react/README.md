# Day 16 – Tool Calling vs ReAct

Focus:
- Compare direct tool calling and ReAct
- Understand orchestration trade-offs

Tool Calling:
- Simple
- Deterministic
- Minimal overhead
- Best for known actions

ReAct:
- Explicit reasoning
- Inspectable control flow
- Handles uncertainty
- Higher orchestration cost

Rule of Thumb:
Use tool calling when the decision is obvious.
Use ReAct when the decision itself is part of the problem.
