"""Small Ollama chat client for validated structured responses."""

from dataclasses import dataclass
import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ChatError(RuntimeError):
    pass


class ChatModel(Protocol):
    def complete(
        self, system: str, user: str, schema: dict, *, max_output_tokens: int = 1024
    ) -> dict: ...


@dataclass(frozen=True, slots=True)
class OllamaChatConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:latest"
    timeout_seconds: float = 120.0


class OllamaChatClient:
    def __init__(self, config: OllamaChatConfig | None = None):
        self.config = config or OllamaChatConfig()
        if not self.config.model.strip() or self.config.timeout_seconds <= 0:
            raise ValueError("A model and positive timeout are required")

    def complete(
        self, system: str, user: str, schema: dict, *, max_output_tokens: int = 1024
    ) -> dict:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        request = Request(
            f"{self.config.base_url.rstrip('/')}/api/chat",
            data=json.dumps({
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": schema,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": max_output_tokens},
            }, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ChatError(f"Ollama chat failed ({exc.code}): {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ChatError(f"Ollama chat failed: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("done") is not True:
            raise ChatError("Ollama returned an incomplete chat response")
        content = (payload.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ChatError("Ollama returned an empty chat response")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ChatError("Ollama returned invalid structured JSON") from exc
        if not isinstance(result, dict):
            raise ChatError("Ollama structured response must be an object")
        return result
