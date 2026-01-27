# Day 12 – Dockerized MCP Server

Focus:
- Isolate MCP execution from the host
- Make tool execution reproducible
- Prepare MCP for real deployment

Key Concepts:
- MCP servers should not run on the host
- Containers define the security boundary
- Agents connect over a stable interface
- Execution environment becomes predictable

Architecture Visualization:
```
┌────────────────────────────────────┐
│   Host Machine (agent_client.py)   │
│                                    │
│  ┌──────────────────────────────┐  │
│  │   LangChain Agent (LLM)      │  │
│  │   - Decides to use tools     │  │
│  │   - Formulates response      │  │
│  └──────────┬───────────────────┘  │
│             │                      │
│             │ HTTP Request         │
│             ↓                      │
│  ┌──────────────────────────────┐  │
│  │ MultiServerMCPClient         │  │
│  │ (langchain-mcp-adapters)     │  │
│  └──────────┬───────────────────┘  │
└─────────────┼──────────────────────┘
              │
              │ http://localhost:3333/mcp
              │
┌─────────────▼──────────────────────┐
│   Docker Container (mcp-server)    │
│                                    │
│  ┌──────────────────────────────┐  │
│  │   FastMCP Server             │  │
│  │   - Exposes tools            │  │
│  │   - get_utc_time()           │  │
│  │   - Runs as non-root user    │  │
│  └──────────────────────────────┘  │
│                                    │
│  Security: Isolated, Read-only     │
└────────────────────────────────────┘
```

Why this matters:
Without isolation, MCP turns agents into
host-level processes.

Docker is the minimum viable boundary
for safe MCP deployment.
