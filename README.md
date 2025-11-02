# Big Moves - Stock Price Movement Analyzer

A Python tool that analyzes significant price movements in stocks and correlates them with news events.

## Features

- Identifies sustained upward price movements over a specified time window
- Fetches relevant news articles around the dates of significant moves
- Supports both console and JSON output formats
- Configurable threshold for what constitutes a "significant" move
- Customizable time window for analysis

## Quick Start

```bash
# Setup your environment (see SETUP.md for detailed instructions)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run the analyzer
./big_moves.py AAPL  # Analyze Apple stock
```

## Usage

```bash
./big_moves.py TICKER [--threshold PERCENT] [--window DAYS] [--output {console,json}]
```

Arguments:
- `TICKER`: Stock symbol (e.g., AAPL, MSFT, GOOGL)
- `--threshold`: Percentage threshold for significant moves (default: 30.0)
- `--window`: Time window in days to calculate moves (default: 30)
- `--output`: Output format, either 'console' or 'json' (default: console)

## Example

```bash
./big_moves.py TSLA --threshold 25 --window 14 --output json
```

## Setup

See [SETUP.md](SETUP.md) for detailed installation and setup instructions.
