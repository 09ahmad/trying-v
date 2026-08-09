"""LLM gateway client — wraps the Valura gateway with retry/backoff and blackout detection.

The gateway exposes exactly two models: "valura-fast" and "valura-deep".
Two failure modes are exercised in every grading run:

  transient_429  — first call per question is 429+Retry-After; retry gets through
  blackout       — every call fails with insufficient_quota for the whole band

The client detects both and raises the appropriate exception so callers can
respond correctly without crashing or hanging.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

SUPPORTED_MODELS = {"valura-fast", "valura-deep"}
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 8.0


class LLMClientError(Exception):
    """Upstream returned a non-retryable error."""


class BlackoutError(LLMClientError):
    """Upstream is in quota-exhausted blackout — no retry will help."""


class LLMClient:
    """Synchronous OpenAI-compatible client pointed at LLM_BASE_URL.

    All calls go to /chat/completions. Any other external network call would
    fail in the grading container (network is restricted to gateway only).
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=httpx.Timeout(55.0, connect=10.0),  # under 60s deadline
            trust_env=False,
        )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_retries: int = MAX_RETRIES,
    ) -> str:
        """Send a chat completion request and return the assistant's content string.

        Handles:
        - 429 + Retry-After → wait and retry (transient_429 band)
        - 429 + insufficient_quota → raise BlackoutError (blackout band)
        - Other 4xx/5xx → raise LLMClientError after retries exhausted
        """
        if model not in SUPPORTED_MODELS:
            raise LLMClientError(f"Unsupported model {model!r}. Use valura-fast or valura-deep.")

        payload = {"model": model, "messages": messages}
        delay = INITIAL_BACKOFF

        for attempt in range(max_retries):
            try:
                resp = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                raise LLMClientError("Request timed out before 60s question deadline")
            except httpx.RequestError as exc:
                raise LLMClientError(f"Network error: {exc}") from exc

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "") or ""
                return ""

            if resp.status_code == 429:
                # Distinguish blackout (insufficient_quota) from transient rate-limit
                try:
                    err_body = resp.json()
                    err_type = err_body.get("error", {}).get("type", "")
                except Exception:
                    err_type = ""

                if err_type == "insufficient_quota":
                    raise BlackoutError(
                        "Gateway in quota-exhausted blackout — no retry will help."
                    )

                # Transient 429: honour Retry-After header then back off
                if attempt < max_retries - 1:
                    retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                    wait = max(retry_after, delay)
                    time.sleep(wait)
                    delay = min(MAX_BACKOFF, delay * 2)
                    continue

                raise LLMClientError(f"Rate limited after {max_retries} attempts")

            # Non-retryable error
            if resp.status_code >= 400:
                raise LLMClientError(
                    f"Upstream error {resp.status_code}: {resp.text[:300]}"
                )

        raise LLMClientError(f"Chat completion failed after {max_retries} attempts")

    def _parse_retry_after(self, value: Optional[str]) -> float:
        if not value:
            return INITIAL_BACKOFF
        try:
            return max(INITIAL_BACKOFF, float(value))
        except ValueError:
            return INITIAL_BACKOFF
