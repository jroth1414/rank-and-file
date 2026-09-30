"""Terminal dashboard for training runs. Stdlib only; reads runs/*/metrics.jsonl.

    .venv\\Scripts\\python.exe scripts/watch_runs.py                      # newest 4 runs, refresh 10 s
    .venv\\Scripts\\python.exe scripts/watch_runs.py sweep_adamw_lr4e-3_wu10 sweep_muon_lr4e-3_wu10 \\
        --ref sweep_adamw_lr4e-3_wu10=sweep_adamw_lr4e-3 --ref sweep_muon_lr4e-3_wu10=sweep_muon_lr4e-3
    ... --once                                                         # print one snapshot and exit

`--ref RUN=BASELINE` shows BASELINE's val_loss at the same step next to RUN's (Δ < 0 is better).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read_metrics(run: Path) -> tuple[dict, list[dict], list[dict]]:
    start, train, val = {}, [], []
    f = run / "metrics.jsonl"
    if not f.exists():
        return start, train, val
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # partially written last line
        if r.get("event") == "start":
            start = r
        elif "val_loss" in r:
            val.append(r)
        elif "loss" in r:
            train.append(r)
    return start, train, val


def bar(frac: float, width: int = 24) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "·" * (width - n)


def fmt_dur(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        util, used, total, temp, pw = [x.strip() for x in out.split(",")]
        return f"GPU {util:>3}%  mem {float(used) / 1024:.1f}/{float(total) / 1024:.1f} GiB  {temp}°C  {float(pw):.0f} W"
    except Exception as e:  # noqa: BLE001 — dashboard must never crash on a telemetry hiccup
        return f"GPU: n/a ({type(e).__name__})"


def render_run(root: Path, name: str, ref: str | None) -> list[str]:
    run = root / name
    start, train, val = read_metrics(run)
    done = (run / "DONE").exists()
    if not train:
        return [f"{name}: waiting ({'no dir' if not run.exists() else 'no steps yet'})"]
    last = train[-1]
    total = start.get("steps_total") or 0
    step = last["step"]
    frac = step / total if total else 0.0
    age = time.time() - last.get("time", time.time())
    tps = last.get("tok_per_s", 0.0)
    remaining = (total - step) * (last["tokens"] / step) / tps if (tps and step and not done) else 0
    state = "DONE" if done else ("STALE?" if age > 300 else "running")
    lines = [
        f"{name}  [{state}]",
        f"  {bar(frac)} {step}/{total} steps  {last['tokens'] / 1e6:.0f}M tok  {frac:5.1%}"
        + ("" if done else f"  ETA {fmt_dur(remaining)}  (last log {fmt_dur(age)} ago)"),
        f"  loss {last['loss']:.3f}  lr {last.get('lr', 0):.2e}  gnorm {last.get('grad_norm', 0):.2f}"
        f"  {tps / 1e3:.1f}k tok/s  peak {last.get('mem_gib', 0):.2f} GiB",
    ]
    ref_val = {}
    if ref:
        _, _, rv = read_metrics(root / ref)
        ref_val = {r["step"]: r["val_loss"] for r in rv}
    if val:
        hdr = "  val:  " + "  ".join(f"{r['step']:>5d}" for r in val)
        row = "        " + "  ".join(f"{r['val_loss']:5.3f}" for r in val)
        lines += [hdr, row]
        if ref_val:
            deltas = [
                f"{r['val_loss'] - ref_val[r['step']]:+5.2f}" if r["step"] in ref_val else "  n/a"
                for r in val
            ]
            lines.append(f"  Δ vs {ref}: " + "  ".join(deltas))
    return lines


def newest_runs(root: Path, n: int) -> list[str]:
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()]
    dirs.sort(key=lambda d: (d / "metrics.jsonl").stat().st_mtime, reverse=True)
    return [d.name for d in dirs[:n]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run names under --root (default: newest 4)")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--ref", action="append", default=[], help="RUN=BASELINE val_loss comparison")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    refs = dict(r.split("=", 1) for r in a.ref)
    if os.name == "nt":
        os.system("")  # enable ANSI escape handling in the Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    while True:
        names = a.runs or newest_runs(root, 4)
        out = [f"rank-and-file runs  {time.strftime('%Y-%m-%d %H:%M:%S')}   {gpu_line()}", ""]
        for n in names:
            out += render_run(root, n, refs.get(n)) + [""]
        q = root / "queue.log"
        if q.exists():
            out.append("queue.log:")
            width = max(40, os.get_terminal_size().columns - 3) if sys.stdout.isatty() else 150
            out += ["  " + ln[:width] for ln in q.read_text(encoding="utf-8").splitlines()[-4:]]
        if a.once:
            print("\n".join(out))
            return
        print("\x1b[2J\x1b[H" + "\n".join(out), flush=True)
        try:
            time.sleep(a.interval)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
