from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class OpenAIClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 60

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="replace")
            if len(err_text) > 600:
                err_text = f"{err_text[:600]}..."
            raise RuntimeError(
                f"OpenAI API error ({exc.code}) on {endpoint}: {err_text}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to connect to OpenAI API: {exc.reason}") from exc

    def embed_texts(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {
                "model": model,
                "input": batch,
            }
            data = self._post("embeddings", payload)
            rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
            embeddings.extend(row["embedding"] for row in rows)

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding count mismatch between input texts and API response."
            )

        return embeddings

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4.1-mini",
        temperature: float = 0.1,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        data = self._post("chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected OpenAI chat completion response format."
            ) from exc
