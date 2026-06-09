"""Tool calling loop + structured output with Google Gemini."""

import os
import json
import time
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.genai import errors

# Setup
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 8  # safety cap


def generate(**kwargs):
    """Call the model, retrying a few times on rate-limit / busy errors."""
    for attempt in range(5):
        try:
            return client.models.generate_content(model=MODEL, **kwargs)
        except errors.APIError as exc:
            if exc.code in (429, 503) and attempt < 4:
                log.warning("Model busy (%s), waiting before retry...", exc.code)
                time.sleep(5)
            else:
                raise


# Tools
def get_weather(city: str) -> str:
    """Mocked weather lookup."""
    return f"The weather in {city} is sunny, 25°C with light wind."


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


def get_current_time() -> str:
    """Mocked current time (fixed value)."""
    return datetime(2026, 6, 8, 14, 30, tzinfo=timezone.utc).isoformat()


# map tool name -> function
tools = {
    "get_weather": get_weather,
    "calculator": calculator,
    "get_current_time": get_current_time,
}

# tool schemas the model sees
TOOL_DECLARATIONS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a given city.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="City name, e.g. 'Haifa'.",
                    )
                },
                required=["city"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculator",
            description="Evaluate a simple arithmetic expression and return the result.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="Arithmetic expression, e.g. '(25 * 2) + 10'.",
                    )
                },
                required=["expression"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_current_time",
            description="Get the current date and time in ISO 8601 format.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]
)


# Response schema
class ToolCallRecord(BaseModel):
    tool: str = Field(description="Name of the tool that was called.")
    result: str = Field(description="The value the tool returned.")


class FinalAnswer(BaseModel):
    summary: str = Field(description="Natural-language answer to the user's request.")
    tools_used: list[str] = Field(description="Names of every tool that was called.")
    tool_calls: list[ToolCallRecord] = Field(
        description="One record per tool call, with its result."
    )


# small schema for the model's summary (the rest we fill from the real log)
class Summary(BaseModel):
    summary: str = Field(description="Short natural-language answer to the user.")


# Tool-calling loop
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools whenever they are needed "
    "to answer the user. You may call several tools, one after another, before "
    "giving your final answer."
)


def run_tool_loop(user_prompt: str) -> tuple[list, list]:
    """Drive the model/tool conversation. Returns (contents, log_records)."""
    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]
    log_records: list[ToolCallRecord] = []

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[TOOL_DECLARATIONS],
        # drive the loop ourselves so we can log each step
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info("Iteration %d → calling the model...", iteration)
        response = generate(contents=contents, config=config)

        calls = response.function_calls
        if not calls:
            log.info("Iteration %d ← no tool calls, model produced final text.", iteration)
            return contents, log_records

        # keep the model's tool requests in history
        contents.append(response.candidates[0].content)

        # run each requested tool and feed results back
        response_parts = []
        for call in calls:
            args = dict(call.args or {})
            log.info("Iteration %d   tool requested: %s(%s)", iteration, call.name, args)

            impl = tools.get(call.name)
            if impl is None:
                result = f"Error: unknown tool '{call.name}'."
            else:
                result = impl(**args)

            log.info("Iteration %d   tool result:    %s", iteration, result)
            log_records.append(ToolCallRecord(tool=call.name, result=str(result)))
            response_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))

    log.warning("Reached MAX_ITERATIONS (%d) without a final answer.", MAX_ITERATIONS)
    return contents, log_records


# Final structured-output call
def get_structured_answer(contents: list, tool_log: list[ToolCallRecord]) -> FinalAnswer:
    """Build the final JSON answer.

    The summary comes from the model (structured output), but tools_used and
    tool_calls come from our real Python log, not from the model's memory.
    Gemini can't mix tools + response_schema, so this is a separate tool-free call.
    """
    log.info("Requesting final structured (JSON) answer...")
    contents = contents + [
        types.Content(
            role="user",
            parts=[types.Part(text="Now write a short summary of the result.")],
        )
    ]
    response = generate(
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Summary,
        ),
    )

    return FinalAnswer(
        summary=response.parsed.summary,
        tools_used=[r.tool for r in tool_log],
        tool_calls=tool_log,
    )


# Main
def main() -> None:
    # this prompt forces sequential calls: time first, then hour + 10
    user_prompt = (
        "First get the current time. Then take the hour from that time and "
        "calculate hour + 10. Also tell me the weather in Haifa."
    )
    log.info("USER PROMPT: %s", user_prompt)

    contents, tool_log = run_tool_loop(user_prompt)
    answer = get_structured_answer(contents, tool_log)

    print("\n=== FINAL STRUCTURED ANSWER (validated JSON) ===")
    print(json.dumps(answer.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
