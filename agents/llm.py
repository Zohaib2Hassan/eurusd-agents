"""One small client for several AI providers. Returns parsed JSON or None.

provider: anthropic | openai | ollama | lemonade | none
Keys come from .env: ANTHROPIC_API_KEY / OPENAI_API_KEY
If the provider is 'none' or a call fails, agents fall back to their rule-based logic.
"""
from __future__ import annotations

import json
import logging
import os
import re

import requests

log = logging.getLogger("llm")

DEFAULTS = {
    "anthropic": ("claude-sonnet-5-5", "https://api.anthropic.com/v1"),
    "openai": ("gpt-5-mini", "https://api.openai.com/v1"),
    "ollama": ("llama3.1", "http://localhost:11434/v1"),
    "lemonade": ("Qwen3-8B-GGUF", "http://localhost:8000/api/v1"),
}


class LLM:
    def __init__(self, cfg: dict):
        c = cfg.get("llm", {})
        self.provider = (c.get("provider") or "none").lower()
        dm, du = DEFAULTS.get(self.provider, ("", ""))
        self.model = c.get("model") or dm
        self.base = (c.get("base_url") or du).rstrip("/")
        self.temp = float(c.get("temperature", 0.2))
        self.enabled = self.provider != "none"

    def ask_json(self, system: str, user: str, max_tokens: int = 800) -> dict | None:
        if not self.enabled:
            return None
        system += "\nRespond with ONE valid JSON object only. No markdown, no extra text."
        try:
            text = self._anthropic(system, user, max_tokens) if self.provider == "anthropic" \
                else self._openai_compat(system, user, max_tokens)
            return _parse_json(text)
        except Exception as e:
            log.warning("LLM call failed (%s): %s -> using rule-based fallback", self.provider, e)
            return None

    def _anthropic(self, system, user, max_tokens):
        r = requests.post(f"{self.base}/messages", timeout=90, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"},
            json={"model": self.model, "max_tokens": max_tokens, "temperature": self.temp,
                  "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    def _openai_compat(self, system, user, max_tokens):
        key = os.environ.get("OPENAI_API_KEY", "local") if self.provider == "openai" else "local"
        r = requests.post(f"{self.base}/chat/completions", timeout=180,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": self.model, "temperature": self.temp, "max_tokens": max_tokens,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    return json.loads(m.group(0))
