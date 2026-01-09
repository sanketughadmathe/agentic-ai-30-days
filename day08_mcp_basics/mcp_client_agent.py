from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MCPClient


# -----------------------------
# 1. Connect to MCP server
# -----------------------------
mcp_client = MCPClient(command=["python", "mcp_server.py"])

tools = mcp_client.get_tools()


# -----------------------------
# 2. LLM bound to MCP tools
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# -----------------------------
# 3. Simple agent invocation
# -----------------------------
if __name__ == "__main__":
    response = llm.invoke([HumanMessage(content="What is the current time?")])

    print(response.content)
