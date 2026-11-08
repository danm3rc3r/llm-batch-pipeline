"""Runtime configuration and shared constants for the batch pipeline."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "llama3:8b"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 300.0  # one generate() call
PREFLIGHT_TIMEOUT_S = 5.0  # is_available / list_models
RETRY_DELAYS_S = (1.0, 2.0)  # delays of retry 1 and 2 (D4)
EXTENSIONS = (".txt", ".md")

# Built-in task (stage S2 only; replaced by --tasks from stage S3 on)
BUILTIN_SUMMARIZE_PROMPT = (
    "Summarize the following text in 3 bullet points. "
    "Return JSON: {\"summary\": string}\n{text}"
)


@dataclass(frozen=True)
class TaskSpec:
    """A named task: prompt template plus the expected JSON output fields."""

    name: str
    prompt: str  # template with the {text} placeholder
    output_fields: tuple[str, ...] = ()  # expected JSON keys (validation)


BUILTIN_SUMMARIZE = TaskSpec(
    name="summarize",
    prompt=BUILTIN_SUMMARIZE_PROMPT,
    output_fields=("summary",),
)


@dataclass(frozen=True)
class RunConfig:
    """All runtime options for one batch run (CLI + OLLAMA_BASE_URL env)."""

    input_dir: Path  # required, existing directory
    tasks_file: Path | None = None  # S1-S2: None (built-in task); S3+: required
    out_dir: Path = Path("results")
    model: str = DEFAULT_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    base_url: str = DEFAULT_BASE_URL  # --base-url > env OLLAMA_BASE_URL > default
    timeout_s: float = DEFAULT_TIMEOUT_S
    verbose: bool = False


@dataclass
class RunStats:
    """Counters collected during one run, printed as a single JSON line."""

    files_total: int = 0
    files_processed: int = 0  # at least one chunk processed in this run
    files_skipped_resume: int = 0
    files_skipped_other: int = 0  # empty, unreadable, bad encoding
    chunks_ok: int = 0
    chunks_failed: int = 0
    chunks_skipped_resume: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        """Return the stats as a JSON-serializable dict (elapsed_s rounded to 0.01)."""
        return {
            "files_total": self.files_total,
            "files_processed": self.files_processed,
            "files_skipped_resume": self.files_skipped_resume,
            "files_skipped_other": self.files_skipped_other,
            "chunks_ok": self.chunks_ok,
            "chunks_failed": self.chunks_failed,
            "chunks_skipped_resume": self.chunks_skipped_resume,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "elapsed_s": round(self.elapsed_s, 2),
        }
