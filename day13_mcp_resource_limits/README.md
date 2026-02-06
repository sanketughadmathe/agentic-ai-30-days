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

# Code
```bash
docker build -t mcp-server-limited .

docker run -p 3333:3333 \
  --name mcp-limited \
  --memory="128m" \
  --memory-swap="128m" \
  --cpus="0.5" \
  --pids-limit=50 \
  --restart=unless-stopped \
  mcp-server-limited

```
