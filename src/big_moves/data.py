"""
Stock data fetching and processing utilities.
"""
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(symbol, period="1y"):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            print(f"No data found for {symbol}")
            return [], [], None
        return df.index.strftime('%Y-%m-%d').tolist(), df['Close'].tolist(), df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return [], [], None
