# Changelog

## [Unreleased]
- Cap Groq completion budget to max(10% of context, default) and 2048 tokens to avoid negative prompt budgets.
- Add adaptive chunking for Groq summarization using per-model context windows (dynamic fetch + static hints).
- Respect `.env` for `BIG_MOVES_DEBUG_PROMPTS` and gate prompt-level logging; added README/Agents notes for usage.
- Fetch Groq model hints via `requests` (per Groq docs) with graceful fallback to static table.
