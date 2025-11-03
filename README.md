# Big Moves - Stock Price Movement Analyzer

Status : Development

A CLI application to analyze big moves ( Linear price movements to the upside) in stocks and correlates them with news events.
What we want is to observe the relationship between financial media narrative and price movements.


## Features

- Identifies sustained upward price movements over a specified time window
- Fetches relevant news articles around the dates of significant moves
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
./big_moves.py TICKER [--threshold PERCENT] [--window DAYS] 
```

Arguments:
- `TICKER`: Stock symbol (e.g., AAPL, MSFT, GOOGL)
- `--threshold`: Percentage threshold for significant moves (default: 30.0)
- `--window`: Time window in days to calculate moves (default: 30)

## Example

```bash
./big_moves.py TSLA --threshold 25 --window 14 --output json
```

## Setup

See [SETUP.md](SETUP.md) for detailed installation and setup instructions.
