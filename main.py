"""
main.py - start the HackathonShop operations console.

    python main.py                 start fresh (empty playbook library)
    python main.py --keep-memory   keep playbooks learned in earlier runs
    python main.py --port 8001     use another port

Then open http://127.0.0.1:8000 in your browser.
Needs  python detector.py  to have been run once.
"""
import argparse
import sys
import threading
import webbrowser

import uvicorn

from detector import MODEL_PATH
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--keep-memory", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not Path(MODEL_PATH).exists():
        print("Diagnosis model not found. Run  python detector.py  first.")
        sys.exit(1)

    import server
    server.start_session(keep_memory=args.keep_memory)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nHackathonShop console running at {url}   (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(server.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
