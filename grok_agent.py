"""Grok function-calling loop over computer.py primitives."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import computer

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)

MODEL = "grok-4.6"
MAX_ROUNDS = 12

INSTRUCTIONS = """You control this Mac through tools only.
Break every request into keyboard, mouse, and app steps.
Prefer shortcuts such as command+t, command+l, command+f, command+tab.
If the user wants a new Apple Note, call make_new_note. Do not use Spotlight.
Call get_context when you are unsure what is focused.
If they ask to simplify this, make this easier to read, or remake the current
page, call simplify_page and do not click around the site.
Do not invent site-specific tools. After actions, say briefly what you did.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_context",
        "description": "Read the frontmost app, window title, cursor position, and display size.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "make_new_note",
        "description": "Create a new note in Apple Notes with this body. Use this instead of command+n when the user wants a new note.",
        "parameters": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "Text for the new note"},
            },
            "required": ["body"],
        },
    },
    {
        "type": "function",
        "name": "open_app",
        "description": "Open or focus an application by macOS name, such as Notes, Safari, or Google Chrome.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Application name"},
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "key_combo",
        "description": "Press a keyboard shortcut. Example: command+n is keys [\"command\", \"n\"].",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key names in order, including modifiers",
                },
            },
            "required": ["keys"],
        },
    },
    {
        "type": "function",
        "name": "type_text",
        "description": "Type text into the currently focused field.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "type": "function",
        "name": "mouse_move",
        "description": "Move the cursor to a screen point. Origin is the top-left of the main display.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "type": "function",
        "name": "click",
        "description": "Click the mouse. Optional x and y move first.",
        "parameters": {
            "type": "object",
            "properties": {
                "button": {"type": "string", "enum": ["left", "right"], "default": "left"},
                "count": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
        },
    },
    {
        "type": "function",
        "name": "drag",
        "description": "Drag from the current cursor position to x,y.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "button": {"type": "string", "enum": ["left", "right"], "default": "left"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "type": "function",
        "name": "scroll",
        "description": "Scroll the wheel. Positive lines scroll up in Quartz units.",
        "parameters": {
            "type": "object",
            "properties": {
                "lines": {"type": "integer"},
            },
            "required": ["lines"],
        },
    },
    {
        "type": "function",
        "name": "simplify_page",
        "description": (
            "Read the current Zen, Safari, or Chrome page and generate a Grok Imagine "
            "diagram that simplifies it (recipe steps, key facts, and similar)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ask": {
                    "type": "string",
                    "description": "What to emphasize, such as simplify this recipe.",
                },
            },
        },
    },
]


def _client() -> OpenAI:
    key = os.getenv("XAI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("No XAI_API_KEY found. Paste your key into .env.")
    return OpenAI(api_key=key, base_url="https://api.x.ai/v1")


def _run_tool(name: str, arguments: dict) -> dict:
    if name == "get_context":
        return computer.get_context()
    if name == "open_app":
        return computer.open_app(str(arguments.get("name", "")))
    if name == "make_new_note":
        return computer.make_new_note(str(arguments.get("body", "")))
    if name == "key_combo":
        keys = arguments.get("keys") or []
        if not isinstance(keys, list):
            return {"ok": False, "error": "keys must be a list"}
        return computer.key_combo([str(k) for k in keys])
    if name == "type_text":
        return computer.type_text(str(arguments.get("text", "")))
    if name == "mouse_move":
        return computer.mouse_move(float(arguments["x"]), float(arguments["y"]))
    if name == "click":
        x = arguments.get("x")
        y = arguments.get("y")
        return computer.click(
            button=str(arguments.get("button", "left")),
            count=int(arguments.get("count", 1)),
            x=None if x is None else float(x),
            y=None if y is None else float(y),
        )
    if name == "drag":
        return computer.drag(
            float(arguments["x"]),
            float(arguments["y"]),
            button=str(arguments.get("button", "left")),
        )
    if name == "scroll":
        return computer.scroll(int(arguments["lines"]))
    if name == "simplify_page":
        from simplify import simplify_current_page

        return simplify_current_page(str(arguments.get("ask") or ""))
    return {"ok": False, "error": f"Unknown tool: {name}"}


def _function_calls(response) -> list:
    calls = []
    for item in response.output or []:
        if getattr(item, "type", None) == "function_call":
            calls.append(item)
    return calls


def run_instruction(prompt: str) -> str:
    """Send a text request to Grok and execute any computer tools it asks for."""
    client = _client()
    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": prompt}],
        tools=TOOLS,
        tool_choice="auto",
    )

    for _ in range(MAX_ROUNDS):
        calls = _function_calls(response)
        if not calls:
            return (response.output_text or "").strip()

        outputs = []
        for call in calls:
            raw = getattr(call, "arguments", "{}") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
                result = {"ok": False, "error": "Could not parse tool arguments"}
            else:
                if not isinstance(args, dict):
                    args = {}
                print(f"Tool: {call.name} {json.dumps(args)}")
                result = _run_tool(call.name, args)
                print(f"Result: {json.dumps(result)}")
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        response = client.responses.create(
            model=MODEL,
            input=outputs,
            tools=TOOLS,
            previous_response_id=response.id,
        )

    return (response.output_text or "Stopped after too many tool rounds.").strip()
