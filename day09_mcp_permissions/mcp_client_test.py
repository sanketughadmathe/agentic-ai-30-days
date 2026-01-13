from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MCPClient


# Switch between servers to see behavior change
mcp_client = MCPClient(command=["python", "mcp_server_restricted.py"])

tools = mcp_client.get_tools()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


if __name__ == "__main__":
    response = llm.invoke([HumanMessage(content="List available files and read one.")])

    print(response.content)
