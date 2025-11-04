# Big Moves

![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen)


![Big Moves](images/bigmoves.png)

A CLI application to trace the relationship between financial media narrative and price movements of stocks. Uses linear regression to identify big moves.


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
uv pip install -r requirements_dev.txt

# Run the analyzer
./big_moves.py AAPL  # Analyze Apple stock
```

## Usage

```bash
big-moves TICKER 
```

```bash
big-moves ASTS --min_length 10 --max_length 252 --min_change 30.0 --r_squared 0.9 --min_slope 0.1 --period 1y --detailed_news 
```

