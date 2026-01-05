"""
AI-based news summarizer using Groq.ai (HTTP) with a small pydantic-based request/response model.

This module provides a single function `summarize_news_df(news_df, max_tokens=150)` which:
- concatenates headlines and short snippets from the DataFrame
- sends a concise summarization prompt to Groq.ai (configurable via env vars)
- returns the text summary

Notes:
- Requires an environment variable `GROQ_API_KEY` with a valid API key.
- Optionally set `GROQ_MODEL` and `GROQ_API_URL` env vars. Defaults try sensible values.
- If the remote call fails, falls back to a safe local extractive summarizer.

This implementation intentionally keeps dependencies minimal (requests + pydantic)
so it works even if the full pydantic-ai client isn't installed.
"""
from __future__ import annotations

import os
import logging
import pandas as pd
from typing import Optional

# The module now requires the official groq package and uses its client directly.
try:
    import groq
except Exception as exc:
    raise ImportError("The 'groq' package is required. Install it with 'pip install groq'.") from exc

from .config import find_and_load_env

logger = logging.getLogger(__name__)

MAX_ITEMS_PER_CHUNK = 30  # legacy cap; adaptive chunking will be used instead
MAX_PROMPT_CHARS = 6000
FINAL_PROMPT_CHARS = 8000

# Load env before reading the debug flag so .env values apply
_CANDIDATES, _LOADED_ENV = find_and_load_env()
DEBUG_PROMPTS = os.getenv("BIG_MOVES_DEBUG_PROMPTS", "").lower() in ("1", "true", "yes")

# If prompt debugging is enabled, ensure we have a console handler and DEBUG level.
if DEBUG_PROMPTS:
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setLevel(logging.DEBUG)
        _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(_handler)
    if _LOADED_ENV:
        logger.debug("Loaded env from %s", _LOADED_ENV)
    else:
        logger.debug("No env file found in candidates: %s", _CANDIDATES)

# Model-aware budgets (char/4 heuristic). Base hints derived from Groq model list.
STATIC_MODEL_HINTS = {
    "qwen/qwen3-32b": {"context": 131072, "max_completion": 40960},
    "moonshotai/kimi-k2-instruct": {"context": 131072, "max_completion": 16384},
    "llama-3.3-70b-versatile": {"context": 131072, "max_completion": 32768},
    "canopylabs/orpheus-v1-english": {"context": 4000, "max_completion": 50000},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"context": 131072, "max_completion": 8192},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"context": 131072, "max_completion": 8192},
    "openai/gpt-oss-safeguard-20b": {"context": 131072, "max_completion": 65536},
    "meta-llama/llama-guard-4-12b": {"context": 131072, "max_completion": 1024},
    "whisper-large-v3": {"context": 448, "max_completion": 448},
    "openai/gpt-oss-20b": {"context": 131072, "max_completion": 65536},
    "canopylabs/orpheus-arabic-saudi": {"context": 4000, "max_completion": 50000},
    "meta-llama/llama-prompt-guard-2-22m": {"context": 512, "max_completion": 512},
    "moonshotai/kimi-k2-instruct-0905": {"context": 262144, "max_completion": 16384},
    "llama-3.1-8b-instant": {"context": 131072, "max_completion": 131072},
    "groq/compound-mini": {"context": 131072, "max_completion": 8192},
    "whisper-large-v3-turbo": {"context": 448, "max_completion": 448},
    "groq/compound": {"context": 131072, "max_completion": 8192},
    "openai/gpt-oss-120b": {"context": 131072, "max_completion": 65536},
    "allam-2-7b": {"context": 4096, "max_completion": 4096},
    "meta-llama/llama-prompt-guard-2-86m": {"context": 512, "max_completion": 512},
}
DEFAULT_CONTEXT = 8000
DEFAULT_COMPLETION_BUDGET = 400  # tokens reserved for model reply (small summaries)
HEADER_OVERHEAD = 80  # rough tokens for prompt scaffolding


def _fetch_groq_model_hints_from_api() -> dict:
    """Attempt to fetch model context info from Groq API. Falls back silently on error."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {}
    url = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/models")
    try:
        import requests
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        hints = {}
        for m in data.get("data", []):
            mid = m.get("id")
            ctx_win = m.get("context_window") or m.get("context_length") or m.get("max_context_length")
            max_comp = m.get("max_completion_tokens")
            if mid and ctx_win:
                hints[mid] = {"context": int(ctx_win), "max_completion": int(max_comp) if max_comp else None}
        if DEBUG_PROMPTS:
            logger.debug("Fetched %s model hints from Groq API", len(hints))
        return hints
    except Exception as exc:
        if DEBUG_PROMPTS:
            logger.debug("Groq model fetch failed: %s", exc)
        return {}


def _select_hints_for_model(model_name: str) -> dict:
    model_name = model_name or ""
    # Try dynamic fetch (cached per process)
    global _DYNAMIC_MODEL_HINTS
    try:
        _DYNAMIC_MODEL_HINTS
    except NameError:
        _DYNAMIC_MODEL_HINTS = _fetch_groq_model_hints_from_api()

    # exact match first
    if model_name in _DYNAMIC_MODEL_HINTS:
        return _DYNAMIC_MODEL_HINTS[model_name]
    if model_name in STATIC_MODEL_HINTS:
        return STATIC_MODEL_HINTS[model_name]

    # substring fallback
    lname = model_name.lower()
    for mid, hint in _DYNAMIC_MODEL_HINTS.items():
        if mid.lower() in lname or lname in mid.lower():
            return hint
    for mid, hint in STATIC_MODEL_HINTS.items():
        if mid.lower() in lname or lname in mid.lower():
            return hint

    return {"context": DEFAULT_CONTEXT, "max_completion": DEFAULT_COMPLETION_BUDGET}


def _estimate_tokens(text: str) -> int:
    # crude heuristic: approx 4 chars per token
    return max(1, (len(text) + 3) // 4)

def _build_prompt_from_df(
    news_df: pd.DataFrame,
    max_items: int | None = MAX_ITEMS_PER_CHUNK,
    max_chars: int | None = MAX_PROMPT_CHARS,
) -> str:
    # Build the chunk prompt with optional caps; when max_items/max_chars are None,
    # adaptive budgeting upstream controls truncation instead of legacy fixed limits.
    parts = []
    if news_df is None or news_df.empty:
        return ""

    # Use date + title + source to give context
    for _, row in news_df.iterrows():
        date = row.get('Date')
        title = row.get('Title') or ''
        src = row.get('Source') or ''
        snippet = row.get('Summary') or row.get('Content') or ''
        seg = f"{date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date} - {title}"
        if src:
            seg += f" ({src})"
        if snippet:
            seg += f" : {snippet}"
        parts.append(seg)

    if max_items is not None and len(parts) > max_items:
        keep_each_side = max_items // 2
        parts = parts[:keep_each_side] + ["... (truncated) ..."] + parts[-keep_each_side:]

    joined = '\n'.join(parts)

    if max_chars is not None and len(joined) > max_chars:
        joined = joined[:max_chars]
        if '\n' in joined:
            joined = joined.rsplit('\n', 1)[0]
        joined += "\n... (truncated to fit model context)"

    prompt = (
        "You are a concise financial news summarization assistant. "
        "Given a list of dated headlines (with optional short snippets), produce a brief, neutral summary "
        "that captures the main narrative and any major drivers or events. Aim for 2-3 sentences.\n\n"
        "Headlines and snippets:\n"
        f"{joined}\n\n"
        "Return only the summary, no commentary or attribution."
    )
    return prompt

def _call_groq(prompt: str, max_tokens: int = 1000) -> str:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError('GROQ_API_KEY not set')

    model = os.getenv('GROQ_MODEL', 'groq/compound-mini')

    # Create client and call completions API using the official SDK
    from groq import Groq

    client = Groq(api_key=api_key)

    logger.debug(
        "Groq request: model=%s, max_tokens=%s, prompt_chars=%s",
        model,
        max_tokens,
        len(prompt),
    )
    if DEBUG_PROMPTS:
        logger.debug("Groq prompt:\n%s", prompt)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
        max_tokens=max_tokens,
    )

    return chat_completion.choices[0].message.content

def _summarize_chunk(news_df: pd.DataFrame, max_tokens: int, *, max_items: int | None = None, max_chars: int | None = None) -> str:
    prompt = _build_prompt_from_df(news_df, max_items=max_items, max_chars=max_chars)
    if not prompt:
        return ""
    return _call_groq(prompt, max_tokens=max_tokens).strip()

def summarize_news_df(news_df: pd.DataFrame, max_tokens: int = 150) -> str:
    """Summarize a news DataFrame using Groq.ai.

    If the GROQ_API_KEY env var is not set or the request fails, a small local fallback summary is returned.
    """
    if news_df is None or news_df.empty:
        return ""

    try:
        model = os.getenv('GROQ_MODEL', 'groq/compound-mini')
        hints = _select_hints_for_model(model)
        context_window = hints.get("context", DEFAULT_CONTEXT)
        hinted_max_completion = hints.get("max_completion") or DEFAULT_COMPLETION_BUDGET
        # Reserve at least 10% of the context for completion, capped at 2048 tokens and the hinted max.
        completion_budget = min(
            hinted_max_completion,
            2048,
            max(int(context_window * 0.10), DEFAULT_COMPLETION_BUDGET),
        )
        prompt_budget_map = max(int((context_window - completion_budget) * 0.65), 800)
        prompt_budget_reduce = max(int((context_window - completion_budget) * 0.75), 1000)
        if DEBUG_PROMPTS:
            logger.debug(
                "Using model=%s context_window=%s, map_budget=%s, reduce_budget=%s, completion_budget=%s (hinted=%s)",
                model,
                context_window,
                prompt_budget_map,
                prompt_budget_reduce,
                completion_budget,
                hints,
            )

        # sort chronologically so chunk summaries preserve flow
        news_df = news_df.sort_values('Date', ascending=True)

        # Adaptive chunking based on token budget (char/4 heuristic)
        chunk_summaries = []
        start_idx = 0
        tokens_used = 0
        token_budget = prompt_budget_map - HEADER_OVERHEAD
        # derive per-chunk prompt char cap from map budget (approx 4 chars per token)
        chunk_max_chars = int(token_budget * 4)

        for idx, (_, row) in enumerate(news_df.iterrows()):
            date = row.get('Date')
            title = row.get('Title') or ''
            src = row.get('Source') or ''
            snippet = row.get('Summary') or row.get('Content') or ''
            seg = f"{date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date} - {title}"
            if src:
                seg += f" ({src})"
            if snippet:
                seg += f" : {snippet}"
            seg_tokens = _estimate_tokens(seg)

            # If adding this segment would exceed budget and we already have some rows, flush the chunk.
            if tokens_used + seg_tokens > token_budget and start_idx < idx:
                chunk = news_df.iloc[start_idx:idx]
                logger.debug(
                    "Summarizing chunk %s (rows %s-%s of %s, est_tokens=%s)",
                    len(chunk_summaries) + 1,
                    start_idx,
                    idx - 1,
                    len(news_df),
                    tokens_used,
                )
                chunk_summary = _summarize_chunk(
                    chunk,
                    max_tokens=max_tokens,
                    max_items=None,  # rely on budget, not fixed count
                    max_chars=chunk_max_chars,
                )
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
                start_idx = idx
                tokens_used = 0

            tokens_used += seg_tokens

        # Handle the tail chunk
        if start_idx < len(news_df):
            chunk = news_df.iloc[start_idx:]
            logger.debug(
                "Summarizing chunk %s (rows %s-%s of %s, est_tokens=%s)",
                len(chunk_summaries) + 1,
                start_idx,
                len(news_df) - 1,
                len(news_df),
                tokens_used,
            )
            chunk_summary = _summarize_chunk(
                chunk,
                max_tokens=max_tokens,
                max_items=None,
                max_chars=chunk_max_chars,
            )
            if chunk_summary:
                chunk_summaries.append(chunk_summary)

        if not chunk_summaries:
            return ""

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Combine chunk summaries into a final concise narrative
        bullets = "\n".join(f"- {s}" for s in chunk_summaries)
        logger.debug("Combining %s chunk summaries into final prompt", len(chunk_summaries))
        final_prompt = (
            "You are a concise financial news summarization assistant. "
            "Combine the following chunk summaries into a single 2-3 sentence narrative that captures the main drivers, "
            "themes, and timeline. Preserve chronological flow if relevant.\n\n"
            "Chunk summaries:\n"
            f"{bullets}\n\n"
            "Return only the combined summary, no commentary or attribution."
        )
        reduce_max_chars = int(prompt_budget_reduce * 4)
        if len(final_prompt) > reduce_max_chars:
            final_prompt = final_prompt[:reduce_max_chars]
            if '\n' in final_prompt:
                final_prompt = final_prompt.rsplit('\n', 1)[0]
            final_prompt += "\n... (truncated to fit model context)"
        logger.debug("Final prompt chars=%s (budget chars=%s)", len(final_prompt), reduce_max_chars)
        if DEBUG_PROMPTS:
            logger.debug("Final Groq prompt:\n%s", final_prompt)

        return _call_groq(final_prompt, max_tokens=max_tokens).strip()
    except Exception as e:
        logger.warning("Groq.ai summarization failed: %s.", e)
        raise
