# Setup Instructions for Big Moves

This document explains how to set up the development environment for the Big Moves stock analysis tool using `uv`.

## Prerequisites

1. Python 3.8 or higher
2. pip (for installing uv)

## Installation Steps

1. Install uv:
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
# Create a new virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# Or if you're on Windows:
# .venv\Scripts\activate
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

## Updating Dependencies

To update dependencies to their latest compatible versions:
```bash
uv pip install --upgrade -r requirements.txt
```

## Deactivating the Environment

When you're done, you can deactivate the virtual environment:
```bash
deactivate
```