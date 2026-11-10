"""YAML task configuration: load, validate and render task prompts."""

from pathlib import Path

import yaml

from .config import TaskSpec


class TaskConfigError(Exception):
    """The tasks file is missing, unreadable or contains invalid tasks."""


def load_tasks(path: Path) -> list[TaskSpec]:
    """Load and validate tasks from a YAML file.

    Expected format: a non-empty list of mappings with "name" (non-empty
    string), "prompt" (non-empty string containing the {text} placeholder)
    and optional "output_fields" (list of unique non-empty strings).
    Raises TaskConfigError on any violation.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TaskConfigError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise TaskConfigError("tasks file must be a non-empty list")
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise TaskConfigError("each task must be a mapping with name and prompt")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TaskConfigError("task 'name' must be a non-empty string")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise TaskConfigError(f"task '{name}': 'prompt' must be a non-empty string")
        if name in seen:
            raise TaskConfigError(f"duplicate task name: {name}")
        if "{text}" not in prompt:
            raise TaskConfigError(
                f"task '{name}': prompt must contain the {{text}} placeholder"
            )
        fields = item.get("output_fields")
        if fields is None:
            fields = []
        if not isinstance(fields, list) or not all(
            isinstance(field, str) and field for field in fields
        ):
            raise TaskConfigError(
                f"task '{name}': 'output_fields' must be a list of non-empty strings"
            )
        if len(set(fields)) != len(fields):
            raise TaskConfigError(f"task '{name}': duplicate entry in 'output_fields'")
        seen.add(name)
        tasks.append(TaskSpec(name=name, prompt=prompt, output_fields=tuple(fields)))
    return tasks


def render_prompt(task: TaskSpec, text: str) -> str:
    """Fill the {text} placeholder (str.replace, not str.format: prompts contain JSON braces)."""
    return task.prompt.replace("{text}", text)
