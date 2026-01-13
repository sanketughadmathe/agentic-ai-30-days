# Day 9 – MCP Permissions & Tool Boundaries

Focus:
- Understanding MCP security failure modes
- Comparing unrestricted vs restricted MCP servers
- Enforcing allowlists and path validation

Key Concepts:
- MCP security lives on the server side
- Agents should never own execution power
- Tool surface area must be intentionally small

Why this matters:
An unrestricted MCP server turns agents
into remote shells.

Safe MCP usage requires strict boundaries,
validation, and explicit contracts.
