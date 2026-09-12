"""Start the webcam tracker and Grok voice listener together."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    python = sys.executable
    track = subprocess.Popen([python, "-u", str(ROOT / "track.py"), *sys.argv[1:]])
    voice = subprocess.Popen([python, "-u", str(ROOT / "voice.py")])
    children = (track, voice)
    shutting_down = False

    def shutdown(_signum=None, _frame=None) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        for proc in children:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    exit_code = 0
    print("Running tracker and voice together. Ctrl+C stops both.")
    try:
        while True:
            living = 0
            for proc in children:
                code = proc.poll()
                if code is None:
                    living += 1
                elif code != 0 and exit_code == 0:
                    exit_code = code
            if living == 0:
                break
            if living < len(children):
                shutdown()
            time.sleep(0.15)
    except KeyboardInterrupt:
        shutdown()

    for proc in children:
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
