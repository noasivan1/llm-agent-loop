# From an LLM Call to an Agent Loop

A small project that shows how to grow a single LLM call into a real
agent loop with tools and structured output, using Google Gemini.

It has two implementations of the same task:

- **`agent.py`** – a manual tool-calling loop (Task 1).
- **`agent_langgraph.py`** – the same task using the LangGraph framework (bonus / Task 2).

## What it does

The agent is given three mocked tools:

- `get_weather(city)` – returns a fake weather string
- `calculator(expression)` – evaluates a simple math expression
- `get_current_time()` – returns a fixed mocked time (`2026-06-08 14:30`)

The example prompt is designed to force **sequential** tool calls:

> First get the current time. Then take the hour from that time and calculate
> hour + 10. Also tell me the weather in Haifa.

So the model must call `get_current_time` first (hour = 14), then
`calculator("14 + 10")`, and also `get_weather("Haifa")`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## API key

Create a file called `.env` in the project root with your Gemini key:

```
GEMINI_API_KEY=your_key_here
```

`.env` is in `.gitignore`, so it is never committed.

## Run

Manual loop (Task 1):

```bash
python agent.py
```

LangGraph version (bonus):

```bash
python agent_langgraph.py
```

## How the manual loop works (`agent.py`)

1. Send the prompt + tool definitions to the model.
2. If the model asks for tools, run them in Python and send the results back.
3. Repeat until the model stops asking for tools.
4. Make one final call that returns a short summary, then build the final JSON
   from the **real** tool log in Python (not from the model's memory).

Every iteration, every tool request, and every tool result is logged.

The final answer is validated against a Pydantic schema and printed as JSON:

```json
{
  "summary": "The current time is 14:30, 14 + 10 = 24, and Haifa is sunny, 25°C.",
  "tools_used": ["get_current_time", "calculator", "get_weather"],
  "tool_calls": [
    { "tool": "get_current_time", "result": "2026-06-08T14:30:00+00:00" },
    { "tool": "calculator", "result": "14 + 10 = 24" },
    { "tool": "get_weather", "result": "The weather in Haifa is sunny, 25°C with light wind." }
  ]
}
```

## What LangGraph handles for you (`agent_langgraph.py`)

In the manual version I write the loop, decide when to stop, and pass the
conversation history around myself. LangGraph's `create_react_agent` hides all
three: the **loop**, the **routing** (when to stop vs. call another tool), and
the **state** (conversation history).
