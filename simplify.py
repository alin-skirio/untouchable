"""Turn the front browser page into one Grok Imagine diagram."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv
from openai import OpenAI

import browser

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)

OUT_DIR = ROOT / "simplified"
OUT_FILE = OUT_DIR / "latest.png"
TEXT_MODEL = "grok-4.6"
IMAGE_MODEL = "grok-imagine-image-2.0"

PLAN_INSTRUCTIONS = """You choose the best single infographic for a web page, then write the image prompt.

Return ONLY JSON:
{
  "kind": "recipe_steps" | "key_facts" | "how_to" | "timeline" | "comparison" | "overview",
  "aspect_ratio": "4:3" | "3:4" | "16:9" | "1:1",
  "prompt": "full image-generation prompt"
}

Choose:
- recipe_steps: cooking or making something — numbered steps, simple drawings of food/tools
- how_to: other procedures — numbered steps
- key_facts: Wikipedia, explainers, bios — short bullets of the most important facts
- timeline: history or sequences over time
- comparison: two or more options
- overview: everything else — a clean one-page map of the main ideas

The prompt must:
- Describe an infographic or diagram, not a photograph
- Put the real facts and step text from the page into the prompt so they appear as large readable type
- Use a cream or white background, high contrast, bold sans-serif labels
- Stay faithful to the page; do not invent steps, numbers, or names
- No watermark, no URL chrome, no fake browser UI
"""


def _client() -> OpenAI:
    key = os.getenv("XAI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("No XAI_API_KEY found. Paste your key into .env.")
    return OpenAI(api_key=key, base_url="https://api.x.ai/v1")


def _parse_plan(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0)) if match else {}
    if not isinstance(data, dict):
        data = {}
    kind = str(data.get("kind") or "overview")
    ratio = str(data.get("aspect_ratio") or "4:3")
    if ratio not in {"4:3", "3:4", "16:9", "1:1", "3:2", "2:3"}:
        ratio = "4:3"
    prompt = str(data.get("prompt") or "").strip()
    return {"kind": kind, "aspect_ratio": ratio, "prompt": prompt}


def _plan(title: str, url: str, text: str, ask: str) -> dict:
    clipped = text[:7000]
    user = (
        f"User ask: {ask or 'Simplify this page as a diagram.'}\n"
        f"Title: {title}\nURL: {url}\n\nPage:\n{clipped}"
    )
    print("Choosing a diagram with Grok…", flush=True)
    response = _client().responses.create(
        model=TEXT_MODEL,
        instructions=PLAN_INSTRUCTIONS,
        input=[{"role": "user", "content": user}],
        max_output_tokens=1800,
    )
    plan = _parse_plan(response.output_text or "")
    if len(plan["prompt"]) < 40:
        plan["prompt"] = (
            "Clean infographic on a cream background, large readable sans-serif text, "
            f"summarizing this page titled {title}. Include the key points as bullets. "
            "No watermark."
        )
        plan["kind"] = "overview"
    return plan


def _generate_image(prompt: str, aspect_ratio: str) -> bytes:
    print("Drawing the diagram with Grok Imagine…", flush=True)
    kwargs = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "response_format": "b64_json",
        "extra_body": {
            "aspect_ratio": aspect_ratio,
            "resolution": "2k",
            "quality": "medium",
        },
    }
    try:
        response = _client().images.generate(**kwargs)
    except TypeError:
        kwargs.pop("extra_body", None)
        response = _client().images.generate(**kwargs)
    item = (response.data or [None])[0]
    if item is None:
        raise RuntimeError("Imagine returned no image")
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if not url:
        raise RuntimeError("Imagine returned neither base64 nor a URL")
    with urlopen(url, timeout=60) as resp:
        return resp.read()


def simplify_current_page(ask: str = "") -> dict:
    """Read the current browser page and open a Grok Imagine diagram of it."""
    page = browser.get_browser_page()
    if not page.get("ok"):
        return page
    url = str(page.get("url") or "")
    title = str(page.get("title") or "Simplified page")
    text = str(page.get("text") or page.get("html") or "").strip()
    if len(text) < 40:
        return {
            "ok": False,
            "error": "Could not read the page. Keep the tab visible and try again.",
            "url": url,
            "title": title,
        }

    plan = _plan(title, url, text, ask)
    image = _generate_image(plan["prompt"], plan["aspect_ratio"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_bytes(image)
    opened = browser.open_file(str(OUT_FILE))
    if not opened.get("ok"):
        return opened
    return {
        "ok": True,
        "url": url,
        "title": title,
        "kind": plan["kind"],
        "path": str(OUT_FILE),
        "app": page.get("app"),
    }
