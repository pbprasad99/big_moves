"""
Visualization utilities for displaying stock data and analysis results.
"""
import plotext as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

W, H = plt.terminal_size()

# plt.plotsize(W*0.7, H*0.7)

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
    #plt.clf()
    plt.theme('dark')
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
    volumes = [ volumes/1000000 for volumes in volumes]
    
    # Plot volume bars
    plt.date_form('Y-m-d')
    plt.bar(dates, volumes)
    plt.ylabel("Volume (Millions)")
    plt.title(f"{ticker} Volume")
    plt.show()
