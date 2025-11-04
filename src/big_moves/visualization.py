"""
Visualization utilities for displaying stock data and analysis results.
"""
import plotext as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

W, H = plt.terminal_size()

plt.plotsize(W*0.6, H*0.3)

def plot_candlestick_chart(data, ticker):
    """Plot candlestick chart using plotext."""
    plt.clear_data()
    
    # Get OHLC data
    dates = data.index.strftime('%Y-%m-%d').tolist()
    ohlc_data = {
        "Open": data['Open'].tolist(),
        "High": data['High'].tolist(),
        "Low": data['Low'].tolist(),
        "Close": data['Close'].tolist()
    }
    
    # Plot candles
    plt.date_form('Y-m-d')
    plt.candlestick(dates, ohlc_data)
    plt.title(f"{ticker} Price History")
    plt.show()

def plot_volume_chart(data, ticker):
    """Plot volume chart using plotext."""
    plt.clear_data()
    
    # Get volume data
    dates = data.index.strftime('%Y-%m-%d').tolist()
    volumes = data['Volume'].tolist()
    
    # Plot volume bars
    plt.date_form('Y-m-d')
    plt.bar(dates, volumes)
    plt.title(f"{ticker} Volume")
    plt.show()
