"""Command-line interface: argument parsing, validation and entry point."""

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_MODEL, RunConfig

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
        help="Run the batch pipeline: --input DIR --out DIR [--model NAME] [--verbose]",
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
    return RunConfig(
        input_dir=args.input,
        out_dir=args.out,
        model=args.model,
        verbose=args.verbose,
    )


def configure_logging(cfg: RunConfig) -> None:
    """Configure the llm_pipeline logger (DEBUG with --verbose)."""
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (stage S1 stub; the batch loop lands in stage S2)."""
    cfg = build_config(argv)
    configure_logging(cfg)
    log.info("scaffold: pipeline not implemented yet")
    return 1
