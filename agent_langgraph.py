"""Task 2 - same task as agent.py, but with LangGraph's prebuilt ReAct agent."""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()


# Tools (same mocked values as agent.py)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is sunny, 25°C with light wind."


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains characters that are not allowed."
    try:
        # don't allow dangerous chars
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def get_current_time() -> str:
    """Get the current date and time in ISO 8601 format."""
    return datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc).isoformat()


# Model + agent
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# create_react_agent builds the whole loop/router/state graph for us
agent = create_react_agent(model, tools=[get_weather, calculator, get_current_time])


def main() -> None:
    # same prompt as agent.py: forces sequential tool calls
    prompt = (
        "First get the current time. Then take the hour from that time and "
        "calculate hour + 10. Also tell me the weather in Haifa."
    )
    result = agent.invoke({"messages": [("user", prompt)]})

    # the final message content can be a string or a list of text blocks
    final = result["messages"][-1]
    text = getattr(final, "text", None) or final.content

    print("=== FINAL ANSWER ===")
    print(text)
    print("\nLangGraph hid the manual loop, the routing decision (when to stop), "
          "and the state/history management I wrote by hand in agent.py.")


if __name__ == "__main__":
    main()


# What LangGraph hid vs agent.py: the loop (re-calling the model), the routing
# (deciding when to stop vs call another tool), and the state (the growing
# conversation history) — all managed by create_react_agent instead of by hand.
