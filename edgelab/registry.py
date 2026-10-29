"""Append-only, hash-chained research log (simplified from the Research Engine registry).

Kinds: prereg (written BEFORE a study runs), result, note. A result must cite an existing prereg id;
records are never edited or removed. verify() checks the chain.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

GENESIS = "0" * 64
KINDS = {"genesis", "prereg", "result", "note"}


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class Registry:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._write({"kind": "genesis", "payload": {"note": "Edge Audit lab registry"}})

    def records(self) -> list:
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _write(self, rec: dict) -> dict:
        if rec["kind"] not in KINDS:
            raise ValueError(f"unknown kind {rec['kind']}")
        rs = self.records() if os.path.exists(self.path) else []
        last = rs[-1] if rs else None
        rec = dict(rec)
        rec["seq"] = 0 if last is None else last["seq"] + 1
        rec["ts_utc"] = datetime.now(timezone.utc).isoformat()
        rec["prev_hash"] = GENESIS if last is None else last["hash"]
        rec["hash"] = hashlib.sha256(canon(rec).encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canon(rec) + "\n")
        return rec

    def prereg(self, study_id: str, spec: dict) -> dict:
        if any(r["kind"] == "prereg" and r["payload"]["id"] == study_id for r in self.records()):
            raise PermissionError(f"{study_id} already pre-registered (specs are never changed)")
        return self._write({"kind": "prereg", "payload": {"id": study_id, "spec": spec,
                                                          "spec_hash": hashlib.sha256(canon(spec).encode()).hexdigest()}})

    def spec_of(self, study_id: str) -> dict:
        for r in self.records():
            if r["kind"] == "prereg" and r["payload"]["id"] == study_id:
                return r["payload"]
        raise KeyError(f"{study_id} is not pre-registered")

    def result(self, study_id: str, payload: dict) -> dict:
        pre = self.spec_of(study_id)
        return self._write({"kind": "result", "payload": {"id": study_id, "spec_hash": pre["spec_hash"], **payload}})

    def note(self, payload: dict) -> dict:
        return self._write({"kind": "note", "payload": payload})

    def verify(self) -> dict:
        prev, problems = GENESIS, []
        rows = self.records()
        for i, r in enumerate(rows):
            if r.get("seq") != i:
                problems.append(f"seq gap at line {i}")
            if r.get("prev_hash") != prev:
                problems.append(f"broken chain at seq {r.get('seq')}")
            body = {k: v for k, v in r.items() if k != "hash"}
            if hashlib.sha256(canon(body).encode()).hexdigest() != r.get("hash"):
                problems.append(f"hash mismatch at seq {r.get('seq')}")
            prev = r.get("hash")
        return {"ok": not problems, "records": len(rows), "problems": problems}
