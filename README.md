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
uv pip install -e . 
```

## Usage

```bash
big-moves <TICKER> [OPTIONS]
```

Example :  

```bash
 big-moves DOCN --max_segments 6 --min_points 6 --detailed_news --big_move_threshold 30.0  
```

