# llm-batch-pipeline

Batch-process a folder of `.txt`/`.md` files with a local LLM (Ollama). No
cloud and no API keys: your text never leaves your machine.

## Features

- Overlapping chunking at word boundaries (`--chunk-size` / `--chunk-overlap`)
- YAML task configs: prompt template plus validated expected JSON output fields
- Forced JSON output via Ollama `format: json`, with 2 retries on invalid model output
- Content-hash resume: after an interrupt or re-run, only new or changed parts are processed
- Append-only `results.jsonl` (one 10-field record per file/task/chunk, with token counts)
- Per-run stats summary on stderr (files, chunks, tokens, elapsed)
- Local Ollama only: configurable model, chunking and timeout

## Quickstart (5 minutes)

```
# one-time: install Ollama (ollama.com/download) and pull a model
ollama pull llama3:8b

git clone https://github.com/danm3rc3r/llm-batch-pipeline.git
cd llm-batch-pipeline

python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python -m llm_pipeline run --input sample_data --tasks sample_tasks.yaml --out results/
```

The sample run processes 10 fake support tickets with 2 tasks. It writes
20 records to `results/results.jsonl`.

## Usage

```
python -m llm_pipeline run --input DIR --tasks FILE --out DIR [--model NAME]
                            [--chunk-size N] [--chunk-overlap N] [--base-url URL]
                            [--timeout SECS] [--verbose]
```

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Directory with `.txt`/`.md` files |
| `--tasks` | (required) | YAML file with the tasks to run |
| `--out` | `results` | Output directory (`results.jsonl`, `state.json`) |
| `--model` | `llama3:8b` | Ollama model name |
| `--chunk-size` | `500` | Max chunk size in characters (>= 100) |
| `--chunk-overlap` | `100` | Overlap between chunks in characters (< chunk-size) |
| `--base-url` | env `OLLAMA_BASE_URL`, then `http://localhost:11434` | Ollama server URL |
| `--timeout` | `300` | Timeout in seconds for one `generate()` call (>= 1) |
| `--verbose` | off | Debug logging |

```
# custom model
python -m llm_pipeline run --input notes --tasks tasks-extract.yaml --out results/ --model llama3:8b

# interrupt with Ctrl+C any time, then re-run the same command to resume
python -m llm_pipeline run --input docs --tasks sample_tasks.yaml --out results/

# bigger chunks for long documents
python -m llm_pipeline run --input docs --tasks sample_tasks.yaml --out results/ --chunk-size 800 --chunk-overlap 200
```

### Output format

Two JSONL lines from a run over `sample_data`:

```
{"file": "ticket-001.txt", "sha256": "9f2c…e81a", "task": "summarize", "chunk": 0, "chunk_count": 1, "output": {"summary": "Customer could not log in after password reset; …"}, "model": "llama3:8b", "elapsed_ms": 1234, "tokens_in": 210, "tokens_out": 96}
{"file": "ticket-001.txt", "sha256": "9f2c…e81a", "task": "classify", "chunk": 0, "chunk_count": 1, "output": {"category": "authentication", "priority": "high"}, "model": "llama3:8b", "elapsed_ms": 987, "tokens_in": 210, "tokens_out": 41}
```

| Field | Description |
|---|---|
| `file` | Path relative to `--input` (slash-separated) |
| `sha256` | SHA-256 of the file bytes |
| `task` | Task name from the YAML config |
| `chunk` / `chunk_count` | Chunk index / total chunks of the file |
| `output` | Parsed JSON object from the model; `{"error": "…"}` when the chunk failed |
| `model` | Model used |
| `elapsed_ms` | Milliseconds for the successful call (all attempts for errors) |
| `tokens_in` / `tokens_out` | `prompt_eval_count` / `eval_count` from Ollama (0 for errors) |

A record is written per (file, task, chunk): files longer than one chunk
produce one record per chunk.

## Configuration

- `OLLAMA_BASE_URL`: Ollama server base URL (default `http://localhost:11434`).
  `--base-url` overrides the env value.
- Tasks file format:

```yaml
- name: summarize
  prompt: "Summarize the following text in 3 bullet points. Return JSON: {\"summary\": string}\n{text}"
  output_fields: [summary]
```

  Each task needs `name` and `prompt`. The prompt must contain the `{text}`
  placeholder, which is substituted with the chunk text (JSON braces in prompts
  are fine). Optional `output_fields` lists the expected JSON keys: a response
  missing any of them is retried, then recorded as an error.
- `.env.example` is included for reference.

## Real run

10 documents, 2 tasks: 20 LLM calls in 6.7 minutes on an RTX 4090 Laptop
(16 GB VRAM) with a local build of `qwen3:4b` (~145 output tokens/sec,
measured on that machine). No API cost.

## Limitations

1. Per-chunk output for large files: no cross-chunk aggregation (v1).
2. Sequential single-model calls (no parallelism).
3. Resume state is bound to (chunking params, model, tasks). Changing any of
   them resets it, and a warning is logged.
4. `results.jsonl` is an append-only log: retries and reprocessed files leave
   older records. A consumer takes the last record per (file, task, chunk).
5. One `--out` per dataset.
6. There is no global run timeout. Ctrl+C works, and progress is saved.

## License

MIT. See [LICENSE](LICENSE).