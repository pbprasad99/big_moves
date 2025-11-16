"""
Visualization utilities for displaying stock data and analysis results.
"""
import plotext as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

W, H = plt.terminal_size()

plt.plotsize(W*0.7, H*0.7)

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

def plot_candlestick_with_segments(data, ticker, models):
    """Plot candlestick chart and overlay fitted segment lines from segmented regression.

    Color logic:
    - Normalize each segment slope to percent-per-day using segment start price.
    - Use the maximum absolute percent-per-day across segments to normalize intensity.
    - If the percent-per-day is below a flat threshold, draw in white.
    - Positive -> green shades, negative -> red shades.

    Parameters:
    - data: DataFrame with OHLC and Volume indexed by date
    - ticker: str symbol
    - models: list of dicts with 'start_idx','end_idx','slope','intercept'
    """
    plt.clear_data()

    # Prepare OHLC
    dates = data.index.strftime('%Y-%m-%d').tolist()
    ohlc_data = {
        "Open": data['Open'].tolist(),
        "High": data['High'].tolist(),
        "Low": data['Low'].tolist(),
        "Close": data['Close'].tolist()
    }

    plt.theme('dark')
    plt.date_form('Y-m-d')
    plt.candlestick(dates, ohlc_data)

    # Compute percent-per-day for each model to determine color intensity
    pct_list = []
    seg_info = []
    for m in models:
        try:
            s = int(m['start_idx'])
            start_price = float(data['Close'].iloc[s])
            slope_raw = float(m.get('slope', 0.0))
            # percent per index (index ~ trading day)
            pct_per_day = (slope_raw / start_price) * 100 if start_price != 0 else 0.0
        except Exception:
            pct_per_day = 0.0
        pct_list.append(abs(pct_per_day))
        seg_info.append({'model': m, 'pct_per_day': pct_per_day})

    max_pct = max(pct_list) if pct_list else 1.0
    if max_pct == 0:
        max_pct = 1.0

    # palettes
    # up_palette = ['#0b6623', '#138a2a', '#39d353']      # dark -> bright green
    # down_palette = ['#5a0a0a', '#b30000', '#ff5c5c']    # dark -> bright red
    flat_color = 'white'
    flat_threshold = 0.05  # percent-per-day below which we consider segment flat

    for i, info in enumerate(seg_info):
        model = info['model']
        pct = info['pct_per_day']
        s = int(model['start_idx'])
        e = int(model['end_idx'])
        if s < 0 or e >= len(dates) or e < s:
            continue

        # Use raw slope for sign, pct for magnitude
        slope_raw = float(model.get('slope', 0.0))
        abs_pct = abs(pct)
        # mag = min(abs_pct / max_pct, 1.0)
        # # choose palette index with rounding for better distribution
        # idx = int(round(mag * (len(up_palette) - 1)))
        
        # print(idx, mag, abs_pct, slope_raw)
        if abs_pct <= flat_threshold and abs_pct >= -flat_threshold:
            col = flat_color
        elif slope_raw > 0:
            col = 'green'#up_palette[idx]
        else:
            col = 'red' #down_palette[idx]

        seg_x = list(range(s, e+1))
        # compute prediction using model slope/intercept (slope defined on absolute x)
        seg_y = [slope_raw * xi + float(model.get('intercept', 0.0)) for xi in seg_x]
        seg_dates = dates[s:e+1]

        plt.plot(seg_dates, seg_y, label=f"seg {i+1}", color=col)

    plt.title(f"{ticker} Price History with Segments")
    plt.show()
