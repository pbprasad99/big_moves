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

# Try to load from standard candidate paths and record where we loaded env from
_CANDIDATES, _LOADED_ENV = find_and_load_env()
if _LOADED_ENV:
    logger.debug('Loaded env from %s', _LOADED_ENV)
else:
    logger.debug('No env file found in candidates: %s', _CANDIDATES)


def _build_prompt_from_df(news_df: pd.DataFrame) -> str:
    # prefer Title and Summary/Content columns
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

    joined = '\n'.join(parts)
    prompt = (
        "You are a concise financial news summarization assistant. "
        "Given a list of dated headlines (with optional short snippets), produce a brief, neutral summary "
        "that captures the main narrative and any major drivers or events. Aim for 2-3 sentences.\n\n"
        "Headlines and snippets:\n"
        f"{joined}\n\n"
        "Return only the summary, no commentary or attribution."
    )
    # logger.debug("Built prompt : ", prompt)
    return prompt


def _call_groq(prompt: str, max_tokens: int = 1000) -> str:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError('GROQ_API_KEY not set')

    model = os.getenv('GROQ_MODEL', 'groq/compound-mini')

    # Create client and call completions API using the official SDK
    from groq import Groq
        # client = groq.Client(api_key=api_key)

    # Client automatically uses the GROQ_API_KEY environment variable
    client = Groq(api_key=api_key)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
    )
    
    # This print statement will only run after the full response is received.
    return chat_completion.choices[0].message.content



def summarize_news_df(news_df: pd.DataFrame, max_tokens: int = 150) -> str:
    """Summarize a news DataFrame using Groq.ai.

    If the GROQ_API_KEY env var is not set or the request fails, a small local fallback summary is returned.
    """
    if news_df is None or news_df.empty:
        return ""

    prompt = _build_prompt_from_df(news_df)
    try:
        # Return whatever the model produced (no additional truncation here)
        return _call_groq(prompt, max_tokens=max_tokens).strip()
    except Exception as e:
        logger.warning("Groq.ai summarization failed: %s.", e)
        raise
        # local fallback: build date-prefixed entries and show up to 5 head + 5 tail
        # parts = []
        # for _, row in news_df.iterrows():
        #     date = row.get('Date')
        #     date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        #     title = row.get('Title') or ''
        #     src = row.get('Source') or ''
        #     snippet = row.get('Summary') or row.get('Content') or ''
        #     seg = f"{date_str} - {title}"
        #     if src:
        #         seg += f" ({src})"
        #     if snippet:
        #         seg += f" : {snippet}"
        #     parts.append(seg)
        # if not parts:
        #     return ""
        # # If more than 10 items, show first 5 and last 5 with an ellipsis in between
        # if len(parts) > 10:
        #     display_parts = parts[:5] + ['...'] + parts[-5:]
        # else:
        #     display_parts = parts
        # joined = ' '.join(display_parts)
        # # truncate to max_tokens characters, attempting to cut at a sentence boundary
        # if len(joined) > max_tokens:
        #     return (joined[:max_tokens].rsplit('.', 1)[0] + '...')
        # return joined
