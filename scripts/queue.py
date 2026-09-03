"""Run commands from a text file sequentially; skip DONE runs; retry once on failure.

Usage: python scripts/queue.py configs/queue/core.txt
Each line: a full command containing --name <run_name>. Lines starting with # are ignored.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path


def _name(line: str) -> str:
    m = re.search(r"--name\s+(\S+)", line)
    if not m:
        raise ValueError(f"queue line has no --name: {line}")
    return m.group(1)


def run_queue(queue_file: Path, runs_root: Path = Path("runs")) -> dict[str, str]:
    runs_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, str] = {}
    with open(runs_root / "queue.log", "a", encoding="utf-8") as logf:

        def log(msg: str) -> None:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
            print(line, flush=True)
            logf.write(line + "\n"); logf.flush()

        for raw in Path(queue_file).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name = _name(line)
            if (runs_root / name / "DONE").exists() or (runs_root / name / "results.json").exists():
                log(f"skip {name}: DONE exists"); summary[name] = "skipped"; continue
            status = "failed"
            for attempt in (1, 2):
                log(f"start {name} (attempt {attempt}): {line}")
                rc = subprocess.call(line, shell=True)
                if rc == 0:
                    status = "ok"; log(f"ok {name}"); break
                log(f"exit {rc} for {name}")
            summary[name] = status
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("queue_file"); ap.add_argument("--runs-root", default="runs")
    a = ap.parse_args()
    print(run_queue(Path(a.queue_file), Path(a.runs_root)))
