"""Turn the front browser page into a step-by-step Grok Imagine flyer."""

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

EXTRACT_INSTRUCTIONS = """You extract a printable how-to flyer from a web page.

Return ONLY JSON:
{
  "kind": "recipe_steps" | "how_to" | "key_facts" | "timeline" | "comparison",
  "title": "short dish or topic name, no site branding",
  "kicker": "one line: yield, time, or source",
  "items": [
    {"label": "1", "text": "one concrete instruction, 8-18 words"}
  ]
}

Rules:
- Prefer a procedure. For a recipe, items are METHOD STEPS in order, not a title, not a vibe, not a history blurb.
- Pull real quantities, times, temperatures, and ingredient names from the page.
- 6 to 12 items. Each item is something a person can do or a fact they can use.
- Skip ads, comments, nutrition tables, and "about the author".
- Do not invent steps. If the page is an article, items are the most important facts as short bullets.
- Never return a single hero caption. A cover with only a name is a failure.
"""


def _client() -> OpenAI:
    key = os.getenv("XAI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("No XAI_API_KEY found. Paste your key into .env.")
    return OpenAI(api_key=key, base_url="https://api.x.ai/v1")


def _parse_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0)) if match else {}
    return data if isinstance(data, dict) else {}


def _clean_item(raw, index: int) -> dict | None:
    if isinstance(raw, str):
        text = re.sub(r"\s+", " ", raw).strip()
        label = str(index)
    elif isinstance(raw, dict):
        text = re.sub(r"\s+", " ", str(raw.get("text") or raw.get("step") or "")).strip()
        label = str(raw.get("label") or raw.get("n") or index).strip()
    else:
        return None
    if len(text) < 8:
        return None
    if len(text) > 140:
        text = text[:137].rsplit(" ", 1)[0] + "…"
    return {"label": label or str(index), "text": text}


def _extract(title: str, url: str, text: str, ask: str) -> dict:
    clipped = text[:9000]
    user = (
        f"User ask: {ask or 'Make a step-by-step flyer from this page.'}\n"
        f"Title: {title}\nURL: {url}\n\nPage:\n{clipped}"
    )
    print("Extracting steps with Grok…", flush=True)
    response = _client().responses.create(
        model=TEXT_MODEL,
        instructions=EXTRACT_INSTRUCTIONS,
        input=[{"role": "user", "content": user}],
        max_output_tokens=2200,
    )
    data = _parse_json(response.output_text or "")
    items = []
    for i, raw in enumerate(data.get("items") or [], start=1):
        item = _clean_item(raw, i)
        if item:
            items.append(item)
    kind = str(data.get("kind") or "how_to")
    flyer_title = str(data.get("title") or title or "How to").strip()[:72]
    kicker = str(data.get("kicker") or "").strip()[:90]
    if len(items) < 4:
        raise RuntimeError("Page did not yield enough steps to build a flyer.")
    items = items[:12]
    print(f"Flyer: {kind}, {len(items)} steps — {flyer_title}", flush=True)
    return {
        "kind": kind,
        "title": flyer_title,
        "kicker": kicker,
        "items": items,
        "aspect_ratio": "3:4",
    }


def _flyer_prompt(extract: dict) -> str:
    lines = []
    for item in extract["items"]:
        lines.append(f"{item['label']}. {item['text']}")
    body = "\n".join(lines)
    kicker = extract["kicker"] or "Follow each numbered step in order."
    title = extract["title"]
    return f"""Kitchen / how-to FLYER infographic, portrait poster that someone could cook from.

NOT a cover. NOT a title page. NOT a single hero illustration with a name underneath.
The numbered instructions must fill at least three quarters of the page.

Layout:
- Thin top banner with the title only
- One small kicker line under the title
- Then a dense vertical list or 2-column grid of numbered steps that runs to the bottom
- Each step: bold number in a circle, then the instruction in large readable sans-serif
- Tiny simple icons beside steps are ok; they must not replace the words
- Cream or off-white paper, dark ink, high contrast, generous margins, print-ready
- No watermark, no website chrome, no QR code, no fake browser, no giant empty cabbage in the center

Letter this exact text, spelled correctly:

TITLE: {title}
KICKER: {kicker}
STEPS:
{body}
"""


def _generate_image(prompt: str, aspect_ratio: str) -> bytes:
    print("Drawing the flyer with Grok Imagine…", flush=True)
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
    """Read the current browser page and open a step-by-step Imagine flyer."""
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

    extract = _extract(title, url, text, ask)
    prompt = _flyer_prompt(extract)
    image = _generate_image(prompt, extract["aspect_ratio"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_bytes(image)
    opened = browser.open_file(str(OUT_FILE))
    if not opened.get("ok"):
        return opened
    return {
        "ok": True,
        "url": url,
        "title": extract["title"],
        "kind": extract["kind"],
        "steps": len(extract["items"]),
        "path": str(OUT_FILE),
        "app": page.get("app"),
    }
