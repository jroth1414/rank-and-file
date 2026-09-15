"""Scripts must stay runnable as ``python scripts/<name>.py``.

That invocation puts ``scripts/`` first on ``sys.path``, so a script whose
name matches a standard-library module (``queue.py`` once did) shadows it
for every import that follows; torch imports ``queue`` internally, so every
script that reaches torch failed at import time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_no_script_shadows_a_stdlib_module():
    names = {p.stem for p in SCRIPTS.glob("*.py")} - {"__init__"}
    clashes = sorted(names & sys.stdlib_module_names)
    assert clashes == [], f"scripts/ shadows standard-library modules: {clashes}"


def test_script_style_invocation_reaches_torch():
    # prepare_data imports rankfile.data -> torch at module import time.
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "prepare_data.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--max-tokens" in r.stdout
