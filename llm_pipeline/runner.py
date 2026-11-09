"""Batch processing loop: files -> tasks -> chunks, with retry, state and stats."""

import hashlib
import logging
import time
from pathlib import Path

from .chunker import chunk_text
from .client import OllamaBadOutput, OllamaClient, OllamaError, OllamaUnavailable
from .config import EXTENSIONS, RETRY_DELAYS_S, RunConfig, RunStats, TaskSpec
from .output import StateStore
from .tasks import render_prompt

log = logging.getLogger("llm_pipeline")


def discover_files(input_dir: Path) -> list[Path]:
    """Return all .txt/.md files under input_dir, sorted by path."""
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def run_batch(
    cfg: RunConfig, tasks: list[TaskSpec], files: list[Path], client: OllamaClient
) -> RunStats:
    """Process every file with every task, resuming from state.json (stage S4)."""
    start = time.monotonic()
    stats = RunStats()
    meta = {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap,
            "model": cfg.model, "tasks": sorted(t.name for t in tasks)}
    state = StateStore(cfg.out_dir / "state.json")
    state.load(meta)
    try:
        for path in files:
            stats.files_total += 1
            relpath = path.relative_to(cfg.input_dir).as_posix()
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                stats.files_skipped_other += 1
                log.warning("skip (%s): %s", type(exc).__name__, relpath)
                continue
            digest = hashlib.sha256(raw).hexdigest()
            chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
            if not chunks:
                stats.files_skipped_other += 1
                log.info("skip (empty): %s", relpath)
                continue
            if state.is_file_done(relpath, digest, {t.name: len(chunks) for t in tasks}):
                stats.files_skipped_resume += 1
                log.info("skip (resume): %s", relpath)
                continue
            stats.files_processed += 1
            log.info("processing: %s (%d chunks, %d bytes)", relpath, len(chunks), len(raw))
            for task in tasks:
                for i, chunk in enumerate(chunks):
                    if state.is_chunk_done(relpath, digest, task.name, i):
                        stats.chunks_skipped_resume += 1
                        log.info("skip (resume): %s [%s chunk %d/%d]",
                                 relpath, task.name, i + 1, len(chunks))
                        continue
                    process_chunk(cfg, task, chunk, relpath, digest, i, len(chunks),
                                  client, state, stats)
    finally:
        state.save()
    stats.elapsed_s = time.monotonic() - start
    if stats.files_total > 0 and stats.files_processed == 0:
        log.info("nothing to process: %d/%d files already done",
                 stats.files_skipped_resume, stats.files_total)
    return stats


def process_chunk(
    cfg: RunConfig,
    task: TaskSpec,
    chunk: str,
    relpath: str,
    digest: str,
    i: int,
    count: int,
    client: OllamaClient,
    state: StateStore,
    stats: RunStats,
) -> None:
    """Call the model for one chunk with up to two retries on bad output (D4)."""
    prompt = render_prompt(task, chunk)
    last_error = ""
    for attempt in range(1, 4):  # 1 + 2 retries (D4)
        t0 = time.monotonic()
        try:
            result = client.generate(cfg.model, prompt, response_json=True)
            missing = [f for f in task.output_fields if f not in result.output]
            if missing:
                raise OllamaBadOutput(f"missing fields: {', '.join(missing)}")
        except OllamaBadOutput as exc:
            last_error = str(exc)
            if attempt < 3:
                log.warning(
                    "retry %d for %s [%s chunk %d/%d]: %s",
                    attempt, relpath, task.name, i + 1, count, exc,
                )
                time.sleep(RETRY_DELAYS_S[attempt - 1])
                continue
        except OllamaUnavailable:
            raise  # connection-level: propagate to main (exit 2)
        except OllamaError as exc:  # 404/5xx - no retries (D4)
            stats.chunks_failed += 1
            log.error("failed: %s [%s chunk %d/%d]: %s", relpath, task.name, i + 1, count, exc)
            return
        else:
            elapsed = int((time.monotonic() - t0) * 1000)
            state.mark_chunk_done(relpath, digest, task.name, i)
            state.save()
            stats.chunks_ok += 1
            stats.tokens_in += result.tokens_in
            stats.tokens_out += result.tokens_out
            log.info("chunk %d/%d ok: %s (%d ms)", i + 1, count, task.name, elapsed)
            return
    # retries exhausted
    stats.chunks_failed += 1
    log.error("failed: %s [%s chunk %d/%d]: %s", relpath, task.name, i + 1, count, last_error)
