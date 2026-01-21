"""Groq LLM client wrapper (single integration point).

Design goals:
- Centralize provider specifics (model selection, retries, error handling).
- Keep calling code provider-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class GroqLLM:
    """Thin wrapper around Groq Chat Completions."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        if Groq is None:
            raise ImportError(
                "Groq SDK is not installed. Install dependencies with: pip install -r requirements.txt"
            )
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.model = model or DEFAULT_MODEL
        self._client = Groq(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """
        Perform a chat completion.

        messages format:
          [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        usage = getattr(completion, "usage", None)
        return LLMResponse(text=text, model=self.model, usage=usage.model_dump() if usage else None)


def safe_chat(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 800,
    top_p: float = 1.0,
    model: Optional[str] = None,
) -> LLMResponse:
    """
    Convenience: one-off call with clean fallback errors.
    """
    llm = GroqLLM(model=model)
    return llm.chat(messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

