"""Start the local FastAPI backend and Streamlit frontend together.

Run from repository root:
    uv run python code/run_app.py
"""
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
processes = []

def stop_all(*_):
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    raise SystemExit(0)

def main():
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT
    )
    processes.append(backend)
    time.sleep(2)
    if backend.poll() is not None:
        raise RuntimeError("Backend failed to start; check the error above.")

    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
        cwd=ROOT
    )
    processes.append(frontend)
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    try:
        while any(p.poll() is None for p in processes):
            time.sleep(1)
    finally:
        stop_all()

if __name__ == "__main__":
    main()
