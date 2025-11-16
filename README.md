# Big Moves

![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen)


![Big Moves](images/bigmoves.png)

A CLI application to trace the relationship between financial media narrative and price movements of stocks. Uses segmented linear regression to identify and highlight big moves and summarizes the news flow progressing through segments.


## Features

- Divides chart into segments using segmented linear regression and highlights big moves.
- Summarizes news narrative as we progress trough the segments.
- Configurable percentage threshold for what constitutes a big move
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

