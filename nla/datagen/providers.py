"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
from abc import ABC, abstractmethod

import anthropic
import httpx


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        # refusal: source text tripped safety — no answer coming, drop this row.
        # content may be [] or the refusal message; either way, no explanation.
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out


class LocalChatProvider(CompletionProvider):
    """Provider for any OpenAI-compatible local server (SGLang, vLLM, llama.cpp).

    Launch your server first, e.g.:
        python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct \\
            --port 30000 --disable-radix-cache

    Then pass to stage 2:
        --provider-cls nla.datagen.providers.LocalChatProvider \\
        --provider-kwargs '{"base_url":"http://localhost:30000/v1","model":"Qwen/Qwen2.5-7B-Instruct"}'

    `base_url` must end with /v1. `model` must match what the server reports
    (check /v1/models if unsure — for SGLang it's the --model-path value).

    None is returned for any request that errors after `max_retries` attempts;
    stage 2 drops those rows rather than killing the whole chunk.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000/v1",
        model: str = "default",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout

    async def _one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        prompt: str,
    ) -> str | None:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        async with sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    return text if text else None
                except (httpx.HTTPStatusError, httpx.TransportError, KeyError) as exc:
                    if attempt == self.max_retries - 1:
                        print(f"  [LocalChatProvider] gave up after {self.max_retries} attempts: {exc}")
                        return None
        return None  # unreachable, satisfies type checker

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None]:
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient() as client:
                return list(
                    await asyncio.gather(*(self._one(client, sem, p) for p in prompts))
                )

        return asyncio.run(_run())
