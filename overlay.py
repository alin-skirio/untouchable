"""Desk overlay: ephemeral intro frost, then optional HUD chrome.

A thin localhost page is shown in a transparent WKWebView (macOS). Python owns
the state machine so H / Esc / click / timeout stay consistent with track.py.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from chrome import display_name

ROOT = Path(__file__).resolve().parent
HOST = ROOT / "overlay_host.jxa"
PAGE = ROOT / "overlay_hud.html"
VOICE_PATH = ROOT / ".voice.json"
STATE_PATH = ROOT / ".overlay.json"
EVENT_PATH = ROOT / ".overlay.event"
LOG_PATH = ROOT / ".overlay.host.log"
INTRO_SEC = 2.5
MAX_NAME = 28
STALE_VOICE_SEC = 4.0

_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None
_thread: threading.Thread | None = None
_proc: subprocess.Popen | None = None
_port = 0

_state = {
    "mode": "desk",
    "intro": False,
    "hud": False,
    "name": "",
    "display_name": "",
    "unlocked": True,
    "voice": False,
    "pointer": False,
    "lock_in": None,
    "lock_reason": "",
    "quit": False,
    "capture": False,
}

_intro_started = 0.0
_started = False


def _voice_listening() -> bool:
    try:
        data = json.loads(VOICE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        age = time.time() - float(data.get("at") or 0)
    except (TypeError, ValueError):
        return False
    if age > STALE_VOICE_SEC:
        return False
    return bool(data.get("listening"))


def snapshot() -> dict:
    with _lock:
        payload = dict(_state)
        payload["voice"] = bool(_state["voice"] or _voice_listening())
        return payload


def intro_active() -> bool:
    with _lock:
        return bool(_state["intro"])


def desk_hud() -> bool:
    with _lock:
        return bool(_state["hud"])


def _write_state() -> None:
    payload = snapshot()
    try:
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except OSError:
        pass


def _drain_events() -> None:
    if not EVENT_PATH.is_file():
        return
    try:
        payload = json.loads(EVENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    try:
        EVENT_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    event = str(payload.get("type") or "")
    if event:
        apply_event(event)


def _set(**kwargs) -> None:
    with _lock:
        _state.update(kwargs)
        _state["capture"] = bool(_state["intro"])
        if _state["intro"]:
            _state["mode"] = "intro"
        elif not _state["unlocked"]:
            _state["mode"] = "locked"
        else:
            _state["mode"] = "desk"
    _write_state()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, _fmt: str, *_args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/chrome", "/toggle"):
            html = PAGE.read_bytes() if PAGE.is_file() else b"<html></html>"
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/state.json":
            tick()
            body = json.dumps(snapshot()).encode("utf-8")
            self._send(200, body, "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(max(0, length)) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        event = str(payload.get("type") or "")
        if path == "/event":
            apply_event(event)
            self._send(200, json.dumps(snapshot()).encode("utf-8"), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def apply_event(event: str) -> None:
    if event == "dismiss_intro":
        dismiss_intro()
    elif event == "toggle_hud":
        toggle_hud()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start() -> None:
    """Launch the overlay server (and the native host on macOS)."""
    global _server, _thread, _port, _started
    if _started and _server is not None:
        _ensure_host()
        return
    _started = True
    _port = _free_port()
    _server = ThreadingHTTPServer(("127.0.0.1", _port), _Handler)
    _thread = threading.Thread(target=_server.serve_forever, name="overlay-hud", daemon=True)
    _thread.start()
    _write_state()
    if sys.platform == "darwin":
        _start_host()


def _ensure_host() -> None:
    if sys.platform != "darwin":
        return
    if _proc is not None and _proc.poll() is None:
        return
    _start_host()


def _start_host() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return
    if not HOST.is_file():
        print("Overlay host script is missing; desk HUD will not draw.")
        return
    log = None
    try:
        log = LOG_PATH.open("w", encoding="utf-8")
        _proc = subprocess.Popen(
            [
                "osascript",
                "-l",
                "JavaScript",
                str(HOST),
                str(STATE_PATH),
                str(EVENT_PATH),
                str(LOG_PATH),
            ],
            stdout=log,
            stderr=log,
        )
    except OSError as exc:
        print(f"Could not start overlay host: {exc}")
        _proc = None
        if log is not None:
            log.close()
        return
    time.sleep(0.25)
    if _proc is not None and _proc.poll() is not None:
        detail = ""
        try:
            detail = LOG_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            detail = f"exit {_proc.returncode}"
        print(f"Overlay host exited immediately ({detail or 'no log'}). Desk HUD will not draw.")
        _proc = None


def server_url() -> str:
    if not _port:
        return ""
    return f"http://127.0.0.1:{_port}/"


def show_welcome(name: str) -> None:
    """Full-viewport frost intro. Replaces the old top-of-screen pill."""
    who = display_name(name, MAX_NAME)
    if not who:
        return
    start()
    global _intro_started
    _intro_started = time.monotonic()
    _set(
        intro=True,
        hud=False,
        name=who,
        display_name=who,
        unlocked=True,
        lock_reason="",
        lock_in=None,
        capture=True,
    )


def dismiss_intro() -> None:
    global _intro_started
    if not intro_active():
        return
    _intro_started = 0.0
    _set(intro=False, hud=False, capture=False)


def toggle_hud() -> None:
    start()
    if intro_active():
        dismiss_intro()
    with _lock:
        _state["hud"] = not bool(_state["hud"])
        visible = bool(_state["hud"])
    _write_state()
    print(f"HUD {'on' if visible else 'off'}.")


def set_hud(visible: bool) -> None:
    start()
    _set(hud=bool(visible))


def sync(
    *,
    name: str = "",
    unlocked: bool = False,
    pointer: bool = False,
    voice: bool | None = None,
    lock_in: float | None = None,
    lock_reason: str = "",
) -> None:
    """Push live desk status into the overlay (no-op until the host is up)."""
    if not _started:
        return
    _ensure_host()
    _drain_events()
    who = display_name(name, MAX_NAME) if name else snapshot()["display_name"]
    listening = _voice_listening() if voice is None else bool(voice)
    reason = lock_reason if not unlocked else ""
    _set(
        name=who or snapshot()["name"],
        display_name=who or snapshot()["display_name"],
        unlocked=bool(unlocked),
        pointer=bool(pointer),
        voice=listening,
        lock_in=None if lock_in is None else round(float(lock_in), 1),
        lock_reason=reason or ("unfamiliar face" if not unlocked else ""),
    )
    tick()


def set_status(_title: str, _detail: str = "", _locked: bool = False) -> None:
    """Older call site: map a toast into live overlay status."""
    if _locked:
        sync(unlocked=False, lock_reason=_detail or "unfamiliar face")
    elif _title:
        sync(name=_title, unlocked=True)


def tick() -> None:
    """Auto-dismiss the intro after INTRO_SEC and apply overlay clicks."""
    _drain_events()
    if not intro_active():
        return
    if time.monotonic() - _intro_started >= INTRO_SEC:
        dismiss_intro()


def close_welcome() -> None:
    dismiss_intro()


def stop() -> None:
    global _server, _thread, _proc, _started, _port, _intro_started
    _set(quit=True, intro=False, capture=False)
    if _proc is not None:
        if _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=0.6)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None
    if _server is not None:
        try:
            _server.shutdown()
        except Exception:
            pass
        try:
            _server.server_close()
        except Exception:
            pass
        _server = None
    _thread = None
    _started = False
    _port = 0
    _intro_started = 0.0
    for path in (STATE_PATH, EVENT_PATH):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    _set(
        mode="desk",
        intro=False,
        hud=False,
        quit=False,
        capture=False,
        unlocked=False,
        pointer=False,
        voice=False,
        lock_in=None,
        lock_reason="",
    )
