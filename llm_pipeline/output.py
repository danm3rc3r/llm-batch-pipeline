"""Run state persistence for resume (state.json, D7/D8)."""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("llm_pipeline")


class StateStore:
    """Per-chunk progress in <out_dir>/state.json, written atomically (tmp + os.replace)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._meta: dict = {}
        self._files: dict[str, dict] = {}

    def load(self, meta: dict) -> None:
        """Load the state; start fresh when the file is missing, corrupt or stale."""
        self._meta = dict(meta)
        self._files = {}
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("state file is corrupt; starting fresh")
            return
        if data.get("meta") != meta:
            log.warning("state file was created with different parameters; ignoring it")
            return
        files = data.get("files")
        if isinstance(files, dict):
            self._files = files

    def is_file_done(self, relpath: str, sha256: str, expected: dict[str, int]) -> bool:
        """True when the file sha matches and every task has all expected chunks."""
        entry = self._files.get(relpath)
        if entry is None or entry.get("sha256") != sha256:
            return False
        done = entry.get("done", {})
        return all(len(done.get(task, [])) >= count for task, count in expected.items())

    def is_chunk_done(self, relpath: str, sha256: str, task: str, chunk: int) -> bool:
        """True when the file sha matches and the (task, chunk) pair is recorded."""
        entry = self._files.get(relpath)
        if entry is None or entry.get("sha256") != sha256:
            return False
        return chunk in entry.get("done", {}).get(task, [])

    def mark_chunk_done(self, relpath: str, sha256: str, task: str, chunk: int) -> None:
        """Record a finished chunk; a changed file sha resets the recorded chunks (AC-5)."""
        entry = self._files.setdefault(relpath, {"sha256": sha256, "done": {}})
        if entry.get("sha256") != sha256:
            entry["done"] = {}
        entry["sha256"] = sha256
        chunks = entry["done"].setdefault(task, [])
        if chunk not in chunks:
            chunks.append(chunk)
            chunks.sort()

    def save(self) -> None:
        """Write the state atomically: state.json.tmp + os.replace (D8)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.parent / (self._path.name + ".tmp")
        payload = {"meta": self._meta, "files": self._files}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        os.replace(tmp, self._path)
