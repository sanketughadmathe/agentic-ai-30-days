# Day 13 – MCP Resource Limits & Kill Switches

Focus:
- Tool execution timeouts
- Container CPU & memory limits
- Controlled failure instead of hangs

Key Concepts:
- Isolation != safety
- Every tool must have an upper bound
- Timeouts belong in execution layer
- Agents must see failures explicitly

Why this matters:
Without limits, an agent can create
denial-of-service against its own system.

Limits turn infinite behavior into
bounded behavior.
