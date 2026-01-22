# Day 11 – MCP Hardening & Auditability

Focus:
- Audit logging for MCP tool calls
- Explicit allow / deny decisions
- Server-side security enforcement
- Safe agent behavior on denial

Key Concepts:
- Prompts are not security boundaries
- MCP servers own execution authority
- Denials must be logged and visible
- Agents must handle failure paths deliberately

Why this matters:
In production, the question is not
"did something fail?"

It's:
"who attempted what, and why was it blocked?"
