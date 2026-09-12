"""Rewrite the front browser page into a large-target local HTML view."""

from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path
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

import computer
import browser

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)

OUT_DIR = ROOT / "simplified"
OUT_FILE = OUT_DIR / "latest.html"
MODEL = "grok-4.6"

REWRITE_INSTRUCTIONS = """You rewrite a web page for someone who clicks with a coarse hand pointer.

You receive stripped HTML from the original page (headings, lists, links, images).
Use that structure. Do not invent sections that are not in the HTML.

Return ONLY inner HTML for <main>. No markdown fences, scripts, or extra chrome.
Use one h1, h2 with ids, p/ul/ol. Keep prose short: under 500 words besides images.

You will also receive a numbered list of image URLs from the original page.
You MUST include each of those images exactly once using
<img src="THE_EXACT_URL" alt="..."> (optionally wrap in <figure> with a short figcaption).
Do not invent, rewrite, or omit image URLs.

Placement:
- First image: hero, directly under the h1.
- Recipes: put each step photo immediately after that step's <li>. Use alt/caption to match the right step.
- Wikipedia / articles: put each photo under the heading it illustrates.
If unsure, put leftover photos under an h2 id="photos".
"""

PAGE_CSS = """
:root {
  --bg: #f4efe6;
  --ink: #1c1916;
  --muted: #5c564e;
  --card: #fffdf8;
  --line: #d9d0c4;
  --hit: #2a2118;
  --hit-ink: #fff8ee;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 22px/1.45 "Iowan Old Style", "Palatino Linotype", Palatino, serif;
}
.bar {
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 16px 20px;
  background: var(--bg);
  border-bottom: 1px solid var(--line);
}
.bar a.hit {
  min-height: 64px;
  min-width: 140px;
  padding: 16px 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: var(--hit);
  color: var(--hit-ink);
  text-decoration: none;
  font-size: 20px;
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-weight: 600;
}
.wrap { max-width: 42rem; margin: 0 auto; padding: 28px 22px 120px; }
.source {
  color: var(--muted);
  font-size: 16px;
  font-family: ui-sans-serif, system-ui, sans-serif;
  word-break: break-all;
}
h1 { font-size: 2.4rem; line-height: 1.15; margin: 0.4em 0 0.8em; }
h2 {
  font-size: 1.7rem;
  margin: 1.6em 0 0.6em;
  padding-top: 0.4em;
  scroll-margin-top: 96px;
}
p, li { font-size: 1.25rem; }
li { margin: 0.55em 0; }
ol, ul { padding-left: 1.2em; }
main a.hit {
  display: inline-flex;
  min-height: 56px;
  padding: 12px 18px;
  margin: 8px 8px 8px 0;
  border-radius: 16px;
  background: var(--card);
  border: 2px solid var(--ink);
  color: var(--ink);
  text-decoration: none;
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-weight: 600;
}
main img {
  display: block;
  width: 100%;
  max-height: 420px;
  object-fit: cover;
  border-radius: 18px;
  margin: 0.55em 0 1em;
  background: var(--line);
}
figure { margin: 0.7em 0 1.2em; padding: 0; }
figcaption {
  margin-top: 0.4em;
  font-size: 16px;
  color: var(--muted);
  font-family: ui-sans-serif, system-ui, sans-serif;
}
li img, li figure { margin-top: 0.5em; }
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


def _strip_unsafe(html: str) -> str:
    cleaned = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    cleaned = re.sub(r"(?is)<iframe[^>]*>.*?</iframe>", "", cleaned)
    cleaned = re.sub(r"(?is)<object[^>]*>.*?</object>", "", cleaned)
    cleaned = re.sub(r'(?is) on\w+\s*=\s*("[^"]*"|\'[^\']*\'|[^\s>]+)', "", cleaned)
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:html)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _localize_images(images: list[dict], page_url: str) -> list[dict]:
    """Save photos next to the simplified page so file:// tabs can show them."""
    if not images:
        return []
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen

    media = OUT_DIR / "media"
    media.mkdir(parents=True, exist_ok=True)
    for old in media.glob("*"):
        if old.is_file():
            old.unlink()
    localized: list[dict] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
        ),
        "Referer": page_url,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    for index, item in enumerate(images[:12], start=1):
        remote = str(item.get("src") or "")
        suffix = Path(urlparse(remote).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}:
            suffix = ".jpg"
        dest = media / f"img-{index:02d}{suffix}"
        try:
            req = Request(remote, headers=headers)
            with urlopen(req, timeout=4) as resp:
                dest.write_bytes(resp.read(2_000_000))
            if dest.stat().st_size < 400:
                dest.unlink(missing_ok=True)
                localized.append(item)
                continue
            localized.append({**item, "src": f"media/{dest.name}"})
        except Exception:
            localized.append(item)
    return localized


def _img_tag(item: dict) -> str:
    src = escape(str(item.get("src") or ""), quote=True)
    alt = escape(str(item.get("alt") or ""))
    cap = escape(str(item.get("cap") or ""))
    photo = f'<img src="{src}" alt="{alt}">'
    if cap:
        return f"<figure>{photo}<figcaption>{cap}</figcaption></figure>"
    return photo


def _ensure_images(body: str, images: list[dict]) -> str:
    if not images:
        return body
    used = set(re.findall(r'<img\b[^>]*src="([^"]+)"', body, re.I))
    missing = [item for item in images if str(item.get("src") or "") not in used]
    if not missing:
        return body
    if not used:
        hero = _img_tag(images[0])
        if "</h1>" in body:
            body = body.replace("</h1>", "</h1>\n" + hero, 1)
        else:
            body = hero + body
        missing = images[1:]
    leftover = "".join(_img_tag(item) for item in missing)
    if leftover:
        body += '<h2 id="photos">Photos</h2>\n' + leftover
    return body


def _rewrite(title: str, url: str, html: str, ask: str, images: list[dict]) -> str:
    clipped = html[:8000]
    lines = []
    for index, item in enumerate(images, start=1):
        alt = item.get("alt") or item.get("cap") or "photo"
        cap = item.get("cap") or ""
        extra = f" | {cap}" if cap else ""
        lines.append(f"{index}. {item.get('src')} | {alt}{extra}")
    image_block = "\n".join(lines) if lines else "(no images found)"
    user = (
        f"User ask: {ask or 'Simplify this page.'}\n"
        f"Title: {title}\nURL: {url}\n\n"
        f"Images to place:\n{image_block}\n\nPage HTML:\n{clipped}"
    )
    print(
        f"Simplifying with Grok ({len(clipped)} html chars, {len(images)} images)…",
        flush=True,
    )
    response = _client().responses.create(
        model=MODEL,
        instructions=REWRITE_INSTRUCTIONS,
        input=[{"role": "user", "content": user}],
        max_output_tokens=2600,
    )
    return _ensure_images(_strip_unsafe(response.output_text or ""), images)


def _wrap(title: str, url: str, body: str) -> str:
    safe_title = escape(title or "Simplified page")
    safe_url = escape(url or "")
    inner = body.strip() or "<p>Nothing to simplify on that page.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <nav class="bar">
    <a class="hit" href="#top">Top</a>
    <a class="hit" href="#start">Start</a>
  </nav>
  <div class="wrap" id="top">
    <p class="source">From {safe_url}</p>
    <main id="start">
      {inner}
    </main>
  </div>
</body>
</html>
"""


def simplify_current_page(ask: str = "") -> dict:
    """Read the current browser page, rewrite it, and open the local HTML."""
    page = computer.get_browser_page()
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
    html = str(page.get("html") or "").strip()
    if len(html) < 40:
        html = str(page.get("text") or "").strip()
    if len(html) < 40:
        err = page.get("js_hint") or "Could not read the page HTML."
        return {"ok": False, "error": err, "url": url, "title": title}

    images = page.get("images") if isinstance(page.get("images"), list) else []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = _localize_images(images, url)
    body = _rewrite(title, url, html, ask, images)
    html = _wrap(title, url, body)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8")
    opened = computer.open_html_file(str(OUT_FILE), app=str(page.get("app") or "") or None)
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
        "path": str(OUT_FILE),
        "app": page.get("app"),
        "chars": len(html),
        "images": len(images),
        "kind": plan["kind"],
        "path": str(OUT_FILE),
        "app": page.get("app"),
    }
