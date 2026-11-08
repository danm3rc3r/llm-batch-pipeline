"""Minimal Ollama HTTP client (no SDK). One httpx.Client per run."""

import json
import logging
from dataclasses import dataclass

import httpx

from .config import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_S, PREFLIGHT_TIMEOUT_S

log = logging.getLogger("llm_pipeline")


class OllamaError(Exception):
    """Base error for Ollama client failures."""


class OllamaUnavailable(OllamaError):
    """Connection-level failure: the server is down or the request timed out."""


class OllamaBadOutput(OllamaError):
    """The model response is not a valid JSON object."""


@dataclass(frozen=True)
class GenerationResult:
    """Parsed model output plus the token counters from the same response."""

    output: dict
    tokens_in: int  # prompt_eval_count (0 when absent)
    tokens_out: int  # eval_count (0 when absent)


class OllamaClient:
    """Minimal Ollama HTTP client (no SDK). One httpx.Client per run."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)

    def is_available(self) -> bool:
        """Return True when GET /api/version answers 200 within the preflight timeout."""
        try:
            resp = self._http.get(f"{self._base}/api/version", timeout=PREFLIGHT_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    def list_models(self) -> list[str]:
        """Return the names of the models installed on the server."""
        resp = self._http.get(f"{self._base}/api/tags")
        resp.raise_for_status()
        body = resp.json()
        return [m["name"] for m in body.get("models", [])]

    def generate(self, model: str, prompt: str, *, response_json: bool = True) -> GenerationResult:
        """Call POST /api/generate once and return the parsed output with token counts."""
        payload: dict = {"model": model, "prompt": prompt, "stream": False}
        if response_json:
            payload["format"] = "json"
        try:
            resp = self._http.post(f"{self._base}/api/generate", json=payload)
        except httpx.TransportError as exc:
            raise OllamaUnavailable(f"Ollama request failed: {exc}") from exc
        if resp.status_code == 404:
            raise OllamaError(f"model '{model}' not found; run: ollama pull {model}")
        if resp.status_code >= 500:
            raise OllamaError(f"Ollama server error: HTTP {resp.status_code}")
        resp.raise_for_status()
        body = resp.json()
        text = body.get("response", "")
        if response_json:
            try:
                output = json.loads(text)
            except json.JSONDecodeError as exc:
                raise OllamaBadOutput(f"model returned invalid JSON: {text[:80]!r}") from exc
            if not isinstance(output, dict):
                raise OllamaBadOutput(f"model returned invalid JSON: {text[:80]!r}")
        else:
            output = {"raw": text}
        return GenerationResult(
            output=output,
            tokens_in=body.get("prompt_eval_count", 0),
            tokens_out=body.get("eval_count", 0),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
