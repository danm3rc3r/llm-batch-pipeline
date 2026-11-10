"""Offline smoke test for llm-batch-pipeline (no Ollama required).

Covers chunking, prompt rendering, task loading, the state store, retry logic,
the JSONL record shape and base-url resolution. Prints SMOKE OK and exits 0
when all checks pass.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import itertools

from llm_pipeline.chunker import chunk_text
from llm_pipeline.cli import build_config
from llm_pipeline.client import (
    GenerationResult,
    OllamaBadOutput,
    OllamaClient,
    OllamaError,
)
from llm_pipeline.config import RunConfig, RunStats, TaskSpec
from llm_pipeline.output import JsonlWriter, StateStore
from llm_pipeline.runner import discover_files, process_chunk, run_batch
from llm_pipeline.tasks import TaskConfigError, load_tasks, render_prompt

KEYS = ["file", "sha256", "task", "chunk", "chunk_count", "output", "model",
        "elapsed_ms", "tokens_in", "tokens_out"]


class FakeClient(OllamaClient):
    """OllamaClient stub: generate() replays a scripted outcome list."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def is_available(self):
        return True

    def list_models(self):
        return ["llama3:8b"]

    def generate(self, model, prompt, *, response_json=True):
        self.calls += 1
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if outcome == "bad":
            raise OllamaBadOutput("model returned invalid JSON: 'oops'")
        if outcome == "server":
            raise OllamaError("Ollama server error: HTTP 500")
        return GenerationResult(output={"summary": "- ok"}, tokens_in=10, tokens_out=5)

    def close(self):
        pass


def check_chunking():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []
    assert chunk_text("short text") == ["short text"]
    try:
        chunk_text("x", 10, 10)
        raise AssertionError("expected ValueError for overlap >= size")
    except ValueError as exc:
        assert "chunk_overlap must be in [0, size)" in str(exc)
    long = " ".join(f"word{i}" for i in range(800))
    chunks = chunk_text(long, 500, 100)
    assert len(chunks) >= 8
    assert all(len(c) <= 500 for c in chunks), "chunk longer than size"
    covered = set()
    for c in chunks:
        covered |= set(c.split())
    assert set(long.split()) <= covered, "words lost across chunks"
    for prev, nxt in itertools.pairwise(chunks):
        assert set(prev.split()) & set(nxt.split()), "consecutive chunks do not overlap"


def check_prompt():
    task = TaskSpec(
        name="summarize",
        prompt="Summarize the following text in 3 bullet points. "
               "Return JSON: {\"summary\": string}\n{text}",
        output_fields=("summary",),
    )
    out = render_prompt(task, "CUSTOMER TEXT")
    assert "CUSTOMER TEXT" in out
    assert "{\"summary\": string}" in out
    assert "{text}" not in out


def check_tasks(tmp):
    valid = tmp / "valid.yaml"
    valid.write_text(
        "- name: summarize\n"
        "  prompt: 'Summarize. Return JSON: {\"summary\": string} {text}'\n"
        "  output_fields: [summary]\n"
        "- name: classify\n"
        "  prompt: 'Classify. Return JSON: {\"category\": string} {text}'\n"
        "  output_fields: [category]\n",
        encoding="utf-8",
    )
    tasks = load_tasks(valid)
    assert [t.name for t in tasks] == ["summarize", "classify"]
    assert tasks[0].output_fields == ("summary",)

    bad = tmp / "bad.yaml"
    bad.write_text("- name: x\n  prompt: 'no placeholder here'\n", encoding="utf-8")
    try:
        load_tasks(bad)
        raise AssertionError("expected TaskConfigError")
    except TaskConfigError as exc:
        assert "must contain the {text} placeholder" in str(exc)

    dup = tmp / "dup.yaml"
    dup.write_text(
        "- name: x\n  prompt: 'a {text}'\n"
        "- name: x\n  prompt: 'b {text}'\n",
        encoding="utf-8",
    )
    try:
        load_tasks(dup)
        raise AssertionError("expected TaskConfigError")
    except TaskConfigError as exc:
        assert "duplicate task name" in str(exc)

    not_list = tmp / "not_list.yaml"
    not_list.write_text("name: x\n", encoding="utf-8")
    try:
        load_tasks(not_list)
        raise AssertionError("expected TaskConfigError")
    except TaskConfigError:
        pass

    bad_fields = tmp / "bad_fields.yaml"
    bad_fields.write_text(
        "- name: x\n  prompt: 'a {text}'\n  output_fields: [ok, ok]\n", encoding="utf-8")
    try:
        load_tasks(bad_fields)
        raise AssertionError("expected TaskConfigError")
    except TaskConfigError as exc:
        assert "output_fields" in str(exc)


def check_state(tmp):
    path = tmp / "state.json"
    meta = {"chunk_size": 500, "chunk_overlap": 100, "model": "llama3:8b",
            "tasks": ["classify", "summarize"]}
    digest = "ab" * 32
    store = StateStore(path)
    store.load(meta)
    store.mark_chunk_done("a.txt", digest, "summarize", 0)
    store.mark_chunk_done("a.txt", digest, "classify", 0)
    store.save()
    reloaded = StateStore(path)
    reloaded.load(meta)
    assert reloaded.is_file_done("a.txt", digest, {"summarize": 1, "classify": 1})
    assert not reloaded.is_file_done("a.txt", digest, {"summarize": 2})
    assert not reloaded.is_file_done("a.txt", "cd" * 32, {"summarize": 1, "classify": 1})
    assert reloaded.is_chunk_done("a.txt", digest, "summarize", 0)
    other_meta = {"chunk_size": 800, "chunk_overlap": 100, "model": "llama3:8b",
                  "tasks": ["classify", "summarize"]}
    stale = StateStore(path)
    stale.load(other_meta)
    assert not stale.is_file_done("a.txt", digest, {"summarize": 1, "classify": 1})
    path.write_text("{ not json", encoding="utf-8")
    corrupt = StateStore(path)
    corrupt.load(meta)
    assert not corrupt.is_file_done("a.txt", digest, {"summarize": 1, "classify": 1})


def check_retries(tmp):
    cfg = RunConfig(input_dir=tmp, out_dir=tmp / "out", model="llama3:8b")
    task = TaskSpec(name="summarize", prompt="Summarize {text}",
                    output_fields=("summary",))
    digest = "ab" * 32

    writer = JsonlWriter(tmp / "recover.jsonl")
    state = StateStore(tmp / "s_recover.json")
    state.load({"model": "llama3:8b"})
    stats = RunStats()
    client = FakeClient(["bad", "bad", "good"])
    process_chunk(cfg, task, "chunk text", "a.txt", digest, 0, 1, client, state, writer, stats)
    writer.close()
    recs = [json.loads(l) for l in
            (tmp / "recover.jsonl").read_text(encoding="utf-8").splitlines()]
    assert client.calls == 3 and stats.chunks_ok == 1 and stats.chunks_failed == 0
    assert list(recs[0].keys()) == KEYS
    assert recs[0]["output"] == {"summary": "- ok"}
    assert recs[0]["tokens_in"] == 10 and recs[0]["tokens_out"] == 5

    writer2 = JsonlWriter(tmp / "exhausted.jsonl")
    state2 = StateStore(tmp / "s_exhausted.json")
    state2.load({"model": "llama3:8b"})
    stats2 = RunStats()
    client2 = FakeClient(["bad"])
    process_chunk(cfg, task, "chunk text", "b.txt", digest, 0, 1, client2, state2, writer2, stats2)
    writer2.close()
    recs2 = [json.loads(l) for l in
             (tmp / "exhausted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert client2.calls == 3 and stats2.chunks_failed == 1 and stats2.chunks_ok == 0
    assert recs2[0]["output"]["error"].startswith("invalid output after 3 attempts")

    writer3 = JsonlWriter(tmp / "server.jsonl")
    state3 = StateStore(tmp / "s_server.json")
    state3.load({"model": "llama3:8b"})
    stats3 = RunStats()
    client3 = FakeClient(["server"])
    process_chunk(cfg, task, "chunk text", "c.txt", digest, 0, 1, client3, state3, writer3, stats3)
    writer3.close()
    recs3 = [json.loads(l) for l in
             (tmp / "server.jsonl").read_text(encoding="utf-8").splitlines()]
    assert client3.calls == 1 and stats3.chunks_failed == 1
    assert recs3[0]["output"]["error"] == "server error: Ollama server error: HTTP 500"


def check_base_url(tmp):
    (tmp / "t.yaml").write_text("- name: x\n  prompt: 'a {text}'\n", encoding="utf-8")
    args = ["run", "--input", str(tmp), "--tasks", str(tmp / "t.yaml")]
    saved = os.environ.pop("OLLAMA_BASE_URL", None)
    try:
        cfg = build_config(args)
        assert cfg.base_url == "http://localhost:11434", cfg.base_url
        assert cfg.timeout_s == 300.0, cfg.timeout_s
        os.environ["OLLAMA_BASE_URL"] = "http://env-host:1234"
        cfg = build_config(args)
        assert cfg.base_url == "http://env-host:1234", cfg.base_url
        cfg = build_config(args + ["--base-url", "http://flag-host:1234"])
        assert cfg.base_url == "http://flag-host:1234", cfg.base_url
    finally:
        if saved is not None:
            os.environ["OLLAMA_BASE_URL"] = saved
        else:
            os.environ.pop("OLLAMA_BASE_URL", None)


def check_run_batch(tmp):
    in_dir = tmp / "data"
    in_dir.mkdir()
    (in_dir / "a.txt").write_text("alpha text one", encoding="utf-8")
    (in_dir / "b.md").write_text("beta text two", encoding="utf-8")
    tasks = [
        TaskSpec(name="summarize", prompt="Summarize {text}", output_fields=("summary",)),
        TaskSpec(name="classify", prompt="Classify {text}", output_fields=("category",)),
    ]

    class FakeTwoTask(FakeClient):
        def generate(self, model, prompt, *, response_json=True):
            self.calls += 1
            if "Classify" in prompt:
                return GenerationResult(output={"category": "c"},
                                        tokens_in=10, tokens_out=5)
            return GenerationResult(output={"summary": "- ok"},
                                    tokens_in=10, tokens_out=5)

    cfg = RunConfig(input_dir=in_dir, out_dir=tmp / "out", model="llama3:8b")
    stats = run_batch(cfg, tasks, discover_files(in_dir), FakeTwoTask(["good"]))
    records = (tmp / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert stats.chunks_ok == 4 and len(records) == 4
    assert all(list(json.loads(l).keys()) == KEYS for l in records)
    stats2 = run_batch(cfg, tasks, discover_files(in_dir), FakeTwoTask(["good"]))
    records2 = (tmp / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert stats2.chunks_ok == 0 and stats2.files_skipped_resume == 2
    assert len(records2) == 4, "resume must not append to results.jsonl"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_chunking()
        check_prompt()
        check_tasks(tmp)
        check_state(tmp)
        check_retries(tmp)
        check_base_url(tmp)
        check_run_batch(tmp)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
