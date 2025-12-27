
# Day 3 – ReAct Agent in LangGraph

Focus:
- Implement ReAct explicitly using LangGraph
- No hidden chains or executors
- Tool execution modeled as a node
- Control flow via conditional edges

Key Concepts:
- Agent node = think + decide
- ToolNode = execute tools
- Conditional edge determines continuation
- Loop is explicit and debuggable

Why this matters:
ReAct fails in production not because of prompts,
but because orchestration is hidden.

LangGraph forces explicit control flow.
