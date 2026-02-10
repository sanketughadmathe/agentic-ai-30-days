import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


# -----------------------------
# Tool
# -----------------------------
@tool
def get_utc_time() -> str:
    """Return the current UTC time."""

    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# LLM bound to tool
# -----------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([get_utc_time])
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
).bind_tools([get_utc_time])


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    response = llm.invoke([HumanMessage(content="What time is it right now?")])

    # Check if the model wants to call a tool
    if response.tool_calls:
        print("Tool calls requested:")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call['name']}: {tool_call['args']}")

            # Execute the tool
            if tool_call["name"] == "get_utc_time":
                result = get_utc_time.invoke(tool_call["args"])
                print(f"  Result: {result}")

    # Print any text response
    if response.content:
        print(f"\nText response: {response.content}")
