"""Model clients.

Ollama is the default (local). The interface returns generated text and rough
token counts. Other backends can be plugged in by implementing `chat()`.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ChatResult:
    text: str
    tokens_in: int
    tokens_out: int


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int = 0,
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "seed": self.seed,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        text = body.get("message", {}).get("content", "")
        return ChatResult(
            text=text,
            tokens_in=body.get("prompt_eval_count", 0),
            tokens_out=body.get("eval_count", 0),
        )


def build_client(model_id: str, backend: str = "ollama", **kwargs) -> OllamaClient:
    if backend == "ollama":
        return OllamaClient(model=model_id, **kwargs)
    raise ValueError(f"unknown backend: {backend}")
