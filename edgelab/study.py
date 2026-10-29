"""Study plumbing shared by the study scripts (from SKILL-06 on; earlier scripts stay as they ran).

- Pre-registration lock: `code_lock` fingerprints `edgelab/` and the study script (sha256 of their bytes) plus the
  git commit and numpy / scipy / python versions. It is stored in the prereg; `check_lock` refuses to run when the
  fingerprint or versions differ, or when those files have uncommitted changes (`state/` is not checked: the run
  writes there). The commit is recorded for traceability, not compared (registering itself needs a new commit).
- Uncertainty: `mean_se` (mean and standard error across seeds) and `paired_diff` (per-seed differences with a
  95 % Student-t interval), so criteria are written on paired differences, not on bare means.
- `smoke_spec`: a tiny copy of a SPEC for `--smoke` runs (no registry writes; catches bugs before a long run).
"""
from __future__ import annotations

import copy
import glob
import hashlib
import os
import platform
import subprocess

import numpy as np
import scipy
from scipy import stats as _st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args, cwd=ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def locked_files(script: str, root: str = ROOT) -> list:
    """Repo-relative paths covered by the lock: every edgelab/*.py and the study script."""
    files = sorted(glob.glob(os.path.join(root, "edgelab", "*.py"))) + [os.path.abspath(script)]
    return sorted({os.path.relpath(f, root) for f in files})


def code_lock(script: str, root: str = ROOT) -> dict:
    h = hashlib.sha256()
    for rel in locked_files(script, root):
        h.update(rel.encode() + b"\0")
        with open(os.path.join(root, rel), "rb") as f:
            h.update(f.read() + b"\0")
    return {"files_sha256": h.hexdigest(), "files": locked_files(script, root), "commit": _git("rev-parse", "HEAD", cwd=root),
            "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()}


def uncommitted(script: str, root: str = ROOT) -> list:
    """Locked files with uncommitted changes (modified, staged or untracked)."""
    out = _git("status", "--porcelain", "--untracked-files=all", "--", *locked_files(script, root), cwd=root)
    return [line[3:] for line in out.splitlines() if line.strip()]


def check_lock(pre: dict, script: str, root: str = ROOT) -> dict:
    """Raise SystemExit unless the code and versions match the prereg's lock and are committed.
    Returns the current lock (to store in the result)."""
    dirty = uncommitted(script, root)
    if dirty:
        raise SystemExit(f"uncommitted changes in locked files: {dirty}: refusing to run")
    if "lock" not in pre:
        raise SystemExit("pre-registration has no code lock: refusing to run")
    now = code_lock(script, root)
    diff = [k for k in ("files_sha256", "files", "numpy", "scipy", "python") if pre["lock"][k] != now[k]]
    if diff:
        raise SystemExit(f"code or versions differ from the pre-registration ({', '.join(diff)}): refusing to run")
    return now


def mean_se(xs) -> dict:
    x = np.asarray([v for v in xs if v is not None], dtype=float)
    if x.size == 0:
        return {"mean": None, "se": None, "n": 0}
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else None
    return {"mean": round(float(x.mean()), 4), "se": None if se is None else round(se, 4), "n": int(x.size)}


def paired_diff(a, b) -> dict:
    """Per-seed a - b: mean, standard error and 95 % Student-t interval."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m = mean_se(d)
    if m["se"] is None:
        return {**m, "ci95": None}
    half = float(_st.t.ppf(0.975, d.size - 1)) * float(d.std(ddof=1) / np.sqrt(d.size))
    return {**m, "ci95": [round(float(d.mean()) - half, 4), round(float(d.mean()) + half, 4)]}


def smoke_spec(spec: dict, n_traders: int = 300, n_seeds: int = 2) -> dict:
    """Tiny copy of a SPEC for --smoke runs: few traders, few seeds, exploration seed range."""
    s = copy.deepcopy(spec)
    s["population"]["n_traders"] = n_traders
    s["seeds"] = list(range(9990, 9990 + n_seeds))
    return s


def pooled_ratio_ci(num, den, iters: int = 2000, seed: int = 0) -> dict:
    """sum(num) / sum(den) over seeds (e.g. true flags / flags), with a 95 % percentile interval from resampling
    seeds. Seeds with den = 0 stay in (they are part of the population), unlike a mean of per-seed ratios."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if den.sum() == 0:
        return {"ratio": None, "ci95": None, "num": 0, "den": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, (iters, num.size))
    d = den[idx].sum(axis=1)
    r = num[idx].sum(axis=1)[d > 0] / d[d > 0]
    lo, hi = np.quantile(r, [0.025, 0.975])
    return {"ratio": round(float(num.sum() / den.sum()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "num": int(num.sum()), "den": int(den.sum())}
