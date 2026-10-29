"""Concurrency-safe atomic file writes (copied from the Research Engine).

Writing to a fixed ``<file>.tmp`` and then ``os.replace`` breaks when two processes build the same entry at once:
one process renames the shared tmp away and the other's ``os.replace`` raises FileNotFoundError. ``atomic_open``
writes to a unique temporary file in the destination directory (so the final rename stays atomic on the same
filesystem), gives it the permissions a plain ``open()`` would have, and removes it if the write fails. The
destination bytes are exactly what the caller writes, so outputs are unchanged.
"""
from __future__ import annotations

import contextlib
import os
import tempfile


def _umask() -> int:
    m = os.umask(0)
    os.umask(m)
    return m


@contextlib.contextmanager
def atomic_open(path: str, mode: str = "wb", **kw):
    """``with atomic_open(path, "wb") as fh: ...`` -> ``path`` replaced atomically when the block succeeds."""
    if mode not in ("w", "wb"):
        raise ValueError("atomic_open supports mode 'w' or 'wb'")
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        os.chmod(tmp, 0o666 & ~_umask())
        with os.fdopen(fd, mode, **kw) as fh:
            yield fh
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
