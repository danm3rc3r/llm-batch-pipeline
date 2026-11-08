"""Command-line interface: argument parsing, validation and entry point."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .client import OllamaClient, OllamaUnavailable
from .config import (
    BUILTIN_SUMMARIZE,
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    RunConfig,
)
from .runner import discover_files, run_batch

log = logging.getLogger("llm_pipeline")

_DESCRIPTION = "Batch-process a folder of text files with a local LLM (Ollama)."


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (flags appear as their stages land)."""
    parser = argparse.ArgumentParser(
        prog="llm_pipeline",
        description=_DESCRIPTION,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="Run the batch pipeline: --input DIR --out DIR [--model NAME] "
             "[--chunk-size N] [--chunk-overlap N] [--verbose]",
        description="Run the batch pipeline over a directory of .txt/.md files.",
    )
    run.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory with .txt/.md files to process.",
    )
    run.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Output directory for results.jsonl and state.json (default: %(default)s).",
    )
    run.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name (default: %(default)s).",
    )
    run.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size in characters (default: %(default)s).",
    )
    run.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlap between consecutive chunks in characters (default: %(default)s).",
    )
    run.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def build_config(argv: list[str] | None = None) -> RunConfig:
    """Parse arguments and return a validated RunConfig (exit code 2 on errors)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"--input must be an existing directory: {args.input}")
    if not args.model:
        parser.error("--model must not be empty")
    if args.chunk_size < 100:
        parser.error("--chunk-size must be >= 100")
    if not (0 <= args.chunk_overlap < args.chunk_size):
        parser.error("--chunk-overlap must be in [0, chunk-size)")
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or DEFAULT_BASE_URL
    return RunConfig(
        input_dir=args.input,
        out_dir=args.out,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        base_url=base_url,
        verbose=args.verbose,
    )


def configure_logging(cfg: RunConfig) -> None:
    """Configure the llm_pipeline logger (DEBUG with --verbose)."""
    logger = logging.getLogger("llm_pipeline")
    logger.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (stage S2: built-in summarize task, no state/writer yet)."""
    cfg = build_config(argv)
    configure_logging(cfg)
    files = discover_files(cfg.input_dir)
    if not files:
        log.error("no .txt/.md files found in %s", cfg.input_dir)
        return 2
    client = OllamaClient(cfg.base_url, cfg.timeout_s)
    if not client.is_available():
        log.error("Ollama not reachable at %s (start with: ollama serve)", cfg.base_url)
        return 2
    if cfg.model not in client.list_models():
        log.error("model '%s' not found; run: ollama pull %s", cfg.model, cfg.model)
        return 2
    log.info("ollama ok: %s (model: %s)", cfg.base_url, cfg.model)
    try:
        stats = run_batch(cfg, [BUILTIN_SUMMARIZE], files, client)
    except OllamaUnavailable as exc:
        log.error("Ollama became unreachable: %s", exc)
        return 2
    finally:
        client.close()
    print(json.dumps(stats.as_dict()), file=sys.stderr)
    return 0 if stats.chunks_failed == 0 else 1
