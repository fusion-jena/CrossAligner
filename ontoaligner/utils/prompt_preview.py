"""Validation helpers for configurable LLM prompt previews."""

from typing import Any


def validate_prompt_preview_count(value: Any) -> None:
    """Require a non-negative integer prompt-preview limit."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError("llm_prompt_preview_count must be a non-negative integer")


def resolve_prompt_preview_count(value: Any, dataset_length: int) -> int:
    """Resolve a validated setting to the number of prompts to print."""
    validate_prompt_preview_count(value)
    return min(value, dataset_length)