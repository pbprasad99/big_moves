#!/usr/bin/env python3

import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from tabulate import tabulate
from finvizfinance.quote import finvizfinance

def fetch_stock_data(ticker, period="1y"):
    """Fetch historical stock data for a given ticker and period"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def identify_significant_moves(data, threshold=30.0, window=30):
    """Identify sustained upward moves of threshold percentage or more"""
    if data is None or len(data) < window:
        return []
    
    # Calculate rolling percentage changes over the specified window
    # data['pct_change'] = (data['Close'].rolling(window=window).apply(
    #     lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100, raw=True
    # ))
    data['pct_change'] = data['Close'].rolling(window=window).apply(
        lambda x: ((x[-1] - x[0]) / x[0]) * 100, raw=True
    )
    
    # Find dates where the percentage change exceeds the threshold
    significant_moves = data[data['pct_change'] > threshold].copy()
    
    # Filter to get only the first date of each sustained move
    if not significant_moves.empty:
        significant_moves = significant_moves.reset_index()
        moves = []
        last_move_date = None
        
        for idx, row in significant_moves.iterrows():
            current_date = row['Date']
            if last_move_date is None or (current_date - last_move_date).days > window:
                moves.append({
                    'date': current_date,
                    'start_price': data.loc[current_date - timedelta(days=window)]['Close'] 
                               if current_date - timedelta(days=window) in data.index else None,
                    'end_price': row['Close'],
                    'pct_change': row['pct_change'],
                    'volume': row['Volume']
                })
                last_move_date = current_date
        
        return moves
    
    return []

def fetch_finviz_news(ticker):
    """Fetch news for a ticker from Finviz"""
    try:
        stock = finvizfinance(ticker)
        news_df = stock.TickerNews()
        return news_df
    except Exception as e:
        print(f"Error fetching Finviz news for {ticker}: {e}")
        return pd.DataFrame()

def filter_news_by_date(news_df, target_date, days_before=7, days_after=7):
    """Filter news by date range around target_date"""
    if news_df.empty:
        return pd.DataFrame()
    
    # Convert date strings to datetime objects
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    # Define date range
    start_date = target_date - timedelta(days=days_before)
    end_date = target_date + timedelta(days=days_after)
    
    # Filter news within the date range
    filtered_news = news_df[(news_df['Date'] >= start_date) & 
                            (news_df['Date'] <= end_date)]
    
    return filtered_news

def format_news_output(news_df, ticker, move_date, pct_change):
    """Format news items for display"""
    if news_df.empty:
        return f"No news found for {ticker} around {move_date.strftime('%Y-%m-%d')} (Move: {pct_change:.2f}%)"
    
    print(f"\n{'='*80}")
    print(f"NEWS FOR {ticker} AROUND {move_date.strftime('%Y-%m-%d')} - PRICE MOVE: {pct_change:.2f}%")
    print(f"{'='*80}")
    
    table_data = []
    for _, row in news_df.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        title = row['Title']
        url = row['Link']
        table_data.append([date, title, url])
    
    return tabulate(table_data, headers=["Date", "Title", "URL"], tablefmt="grid")

def main():
    parser = argparse.ArgumentParser(description='Stock Price Movement and Finviz News Analyzer')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--threshold', type=float, default=30.0, 
                        help='Percentage threshold for significant moves (default: 30.0)')
    parser.add_argument('--window', type=int, default=30, 
                        help='Time window in days to calculate moves (default: 30)')
    parser.add_argument('--output', type=str, choices=['console', 'json'], default='console',
                        help='Output format (default: console)')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    print(f"Analyzing {ticker} for sustained upward moves of {args.threshold}% or more...")
    
    # Fetch stock data
    data = fetch_stock_data(ticker)
    if data is None:
        return
    
    # Identify significant moves
    moves = identify_significant_moves(data, args.threshold, args.window)
    
    if not moves:
        print(f"No significant upward moves of {args.threshold}% or more found for {ticker}")
        return
    
    # Fetch all news for the ticker (Finviz usually provides recent news)
    all_news = fetch_finviz_news(ticker)
    
    # Output results
    if args.output == 'json':
        import json
        results = []
        for move in moves:
            move_date = move['date']
            filtered_news = filter_news_by_date(all_news, move_date)
            news_list = filtered_news.to_dict('records') if not filtered_news.empty else []
            
            results.append({
                'ticker': ticker,
                'move_date': move_date.strftime('%Y-%m-%d'),
                'start_price': float(move['start_price']) if move['start_price'] is not None else None,
                'end_price': float(move['end_price']),
                'pct_change': float(move['pct_change']),
                'volume': int(move['volume']),
                'news': news_list
            })
        print(json.dumps(results, indent=2))
    else:
        print(f"\nFound {len(moves)} significant upward moves for {ticker}:")
        move_table = []
        for i, move in enumerate(moves, 1):
            move_date = move['date']
            move_table.append([
                i,
                move_date.strftime('%Y-%m-%d'),
                f"${move['start_price']:.2f}" if move['start_price'] is not None else "N/A",
                f"${move['end_price']:.2f}",
                f"{move['pct_change']:.2f}%",
                f"{move['volume']:,}"
            ])
        
        print(tabulate(move_table, 
                      headers=["#", "Date", "Start Price", "End Price", "% Change", "Volume"],
                      tablefmt="grid"))
        
        for move in moves:
            move_date = move['date']
            filtered_news = filter_news_by_date(all_news, move_date)
            print(format_news_output(filtered_news, ticker, move_date, move['pct_change']))

if __name__ == "__main__":
    main()