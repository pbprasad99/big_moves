#!/usr/bin/env python3
"""
Stock Linear Move Detector and News Analyzer

This tool detects linear upward price movements in stocks and fetches related news.
"""

import argparse
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
from finvizfinance.quote import finvizfinance
import textwrap
from news import fetch_yahoo_finance_news
from display import rich_display_dataframe


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from rich.console import Console
from rich.table import Table
from rich import box 
from rich.panel import Panel

console = Console()

def fetch_stock_data(ticker, period="1y"):
    """Fetch historical stock data for a given ticker and period"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            console.print(f"[red]No data found for ticker {ticker}[/red]")
            return None
        return data
    except Exception as e:
        console.print(f"[red]Error fetching data for {ticker}: {e}[/red]")
        return None

def identify_variable_length_linear_moves(data, min_length=10, min_r_squared=0.9, 
                                         min_slope=0.1, min_pct_change=30.0, max_length=252):
    """
    Identify linear upward moves of variable length
    
    Parameters:
    - min_length: minimum trading days to consider for a trend
    - max_length: maximum trading days to consider for a trend
    - min_r_squared: minimum R-squared value to consider linear
    - min_slope: minimum daily slope as percentage of price
    - min_pct_change: minimum percentage change required
    """
    if data is None or len(data) < min_length:
        return []
    
    data = data.reset_index()  # Convert index to column for easier date access
    moves = []
    i = 0
    
    # Sliding window approach with variable end point
    while i < len(data) - min_length:
        max_r_squared = 0
        best_length = 0
        best_model = None
        
        # Try different window sizes starting from min_length
        for window_size in range(min_length, min(len(data) - i, max_length + 1)):
            # Get the segment
            segment = data.iloc[i:i+window_size]
            
            # Linear regression
            x = np.arange(window_size).reshape(-1, 1)
            y = segment['Close'].values
            model = LinearRegression().fit(x, y)
            
            # Calculate R-squared
            r_squared = model.score(x, y)
            
            # If R-squared starts deteriorating significantly, we've reached the end of the linear move
            if r_squared > max_r_squared:
                max_r_squared = r_squared
                best_length = window_size
                best_model = model
            elif r_squared < max_r_squared - 0.05 and best_length >= min_length:
                # The trend is deteriorating - stop expanding
                break
        
        # If we found a good linear fit
        if best_length >= min_length and max_r_squared >= min_r_squared:
            segment = data.iloc[i:i+best_length]
            
            # Calculate overall percentage change
            start_price = segment['Close'].iloc[0]
            end_price = segment['Close'].iloc[-1]
            pct_change = ((end_price - start_price) / start_price) * 100
            
            # Calculate normalized slope (% of price per day)
            slope = best_model.coef_[0]
            norm_slope = (slope / start_price) * 100
            
            # Check if the move is significant and upward
            if pct_change >= min_pct_change and norm_slope >= min_slope:
                moves.append({
                    'start_date': segment['Date'].iloc[0],
                    'end_date': segment['Date'].iloc[-1],
                    'length_days': best_length,
                    'start_price': start_price,
                    'end_price': end_price,
                    'pct_change': pct_change,
                    'volume': segment['Volume'].mean(),
                    'r_squared': max_r_squared,
                    'slope': norm_slope
                })
                
                # Skip to the end of this trend
                i += best_length
            else:
                # Move forward by 1 day
                i += 1
        else:
            # Move forward by 1 day
            i += 1
    
    return moves



def filter_news_by_date(news_df, start_date, end_date, extra_days_before=10, extra_days_after=7):
    """Filter news by date range with extra days before and after"""
    if news_df.empty:
        return pd.DataFrame(columns=['Date', 'Title', 'Link', 'Source'])
    
    # Ensure start_date and end_date are pandas Timestamp objects
    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)
    
    # Define extended date range
    extended_start = start_date - pd.Timedelta(days=extra_days_before)
    extended_end = end_date + pd.Timedelta(days=extra_days_after)
    
    # Convert news dates to timezone-naive if they have timezone info
    news_df['Date'] = news_df['Date'].dt.tz_localize(None)
    
    # Filter news within the date range
    filtered_news = news_df[(news_df['Date'] >= extended_start) & 
                           (news_df['Date'] <= extended_end)]
    
    return filtered_news

def summarize_news(news_df, max_headlines=5, max_width=60):
    """Create a summary of news headlines for display in the main table"""
    if news_df.empty:
        return "No relevant news found"
    
    # Sort by date, newest first
    sorted_news = news_df.sort_values('Date', 
                                      ascending=True)
    
    # Get the top headlines
    headlines = []
    for i, (_, row) in enumerate(sorted_news.iterrows()):
        if i >= max_headlines:
            break
            
        # Format: [date] title
        date_str = row['Date'].strftime('%Y-%m-%d')
        headline = f"[{date_str}] {row['Title']}"
        
        # Wrap long headlines
        wrapped = textwrap.fill(headline, width=max_width)
        headlines.append(wrapped)
    
    # Add a count if there are more headlines
    if len(sorted_news) > max_headlines:
        headlines.append(f"... and {len(sorted_news) - max_headlines} more news items")
        
    return "\n\n".join(headlines)

def format_detailed_news(news_df, ticker, start_date, end_date, pct_change, days, r_squared):
    """Format detailed news items for display using Rich"""
    if news_df.empty:
        return Panel(
            f"No news found for {ticker} from 10 days before {start_date.strftime('%Y-%m-%d')} to 7 days after {end_date.strftime('%Y-%m-%d')} (Move: {pct_change:.2f}% over {days} days)",
            title="No News Found",
            style="yellow"
        )
    
    # Create header panel
    header = f"{ticker} News Summary\n"
    header += f"Period: 10 days before {start_date.strftime('%Y-%m-%d')} to 7 days after {end_date.strftime('%Y-%m-%d')}\n"
    header += f"Move: {pct_change:.2f}% over {days} days (R² = {r_squared:.3f})"
    
    header_panel = Panel(header, style="bold blue")
    
    # Create news table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, show_lines=True)
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Source", style="green")
    table.add_column("Title", style="white")
    table.add_column("URL", style="blue")
    
    # Sort news by date
    sorted_news = news_df.sort_values('Date', ascending=False)
    
    for _, row in sorted_news.iterrows():
        table.add_row(
            row['Date'].strftime('%Y-%m-%d'),
            row.get('Source', 'N/A'),
            row['Title'],
            row['Link']
        )
    
    console.print(header_panel)
    return table

def main():
    parser = argparse.ArgumentParser(description='Stock Linear Move Detector and News Analyzer')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--min_length', type=int, default=10, 
                        help='Minimum trend length in trading days (default: 10)')
    parser.add_argument('--max_length', type=int, default=252,
                        help='Maximum trend length in trading days (default: 252, ~1 year)')
    parser.add_argument('--min_change', type=float, default=30.0, 
                        help='Minimum percentage change required (default: 30.0)')
    parser.add_argument('--r_squared', type=float, default=0.9,
                        help='Minimum R-squared value for linear fit (default: 0.9)')
    parser.add_argument('--min_slope', type=float, default=0.1,
                        help='Minimum daily slope as percentage of price (default: 0.1)')
    parser.add_argument('--period', type=str, default="1y",
                        help='Lookback period for data (default: 1y)')
    parser.add_argument('--output', type=str, choices=['console'], default='console',
                        help='Output format (default: console)')
    parser.add_argument('--detailed_news', action='store_true',
                        help='Show detailed news for each move')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    # Analysis status panel
    status = f"🔍 Analyzing {ticker}\n"
    status += f"📈 Looking for upward moves of {args.min_change}% or more\n"
    status += f"📅 Over {args.min_length} to {args.max_length} trading days\n"
    status += f"📊 Minimum R² value: {args.r_squared}"
    
    console.print(Panel(
        status,
        title="Big Moves",
        border_style="cyan",
        padding=(1, 2)
    ))
    
    # Fetch stock data with longer period
    data = fetch_stock_data(ticker, period=args.period)



    if data is None:
        return

  
    
    # Identify linear moves of variable length
    moves = identify_variable_length_linear_moves(
        data, 
        min_length=args.min_length,
        max_length=args.max_length,
        min_r_squared=args.r_squared,
        min_slope=args.min_slope,
        min_pct_change=args.min_change
    )
    

    
    # plot candlestick and volume charts            
    from visualization import plot_candlestick_chart, plot_volume_chart
    plot_candlestick_chart(data, ticker)
    plot_volume_chart(data, ticker)              

    if not moves:
        console.print(Panel(
            f"No significant linear upward moves of {args.min_change}% or more found for {ticker}",
            style="yellow",
            title="No Moves Found"
        ))
        return
    
    # Fetch all news for the ticker
    console.print(Panel("🔎 Fetching relevant news articles...", 
                       style="blue", 
                       subtitle="Please wait"))
    all_news = fetch_yahoo_finance_news(ticker)
    
    # First get news for all moves
    for i, move in enumerate(moves):
        filtered_news = filter_news_by_date(
            all_news, 
            move['start_date'], 
            move['end_date'],
            extra_days_before=10,
            extra_days_after=7
        )
        # Add news to the move dictionary
        moves[i]['news'] = filtered_news
        # Add a news summary
        moves[i]['news_summary'] = summarize_news(filtered_news)
    
    # Output results
    if args.output == 'json':
        import json
        results = []
        for move in moves:
            news_list = move['news'].to_dict('records') if not move['news'].empty else []
            
            results.append({
                'ticker': ticker,
                'start_date': move['start_date'].strftime('%Y-%m-%d'),
                'end_date': move['end_date'].strftime('%Y-%m-%d'),
                'length_days': int(move['length_days']),
                'start_price': float(move['start_price']),
                'end_price': float(move['end_price']),
                'pct_change': float(move['pct_change']),
                'r_squared': float(move['r_squared']),
                'slope': float(move['slope']),
                'news': news_list
            })
        console.print_json(data=results)
    else:
        # print(f"\nFound {len(moves)} significant linear upward moves for {ticker}:")
        move_table = []
        for i, move in enumerate(moves, 1):
            move_table.append([
                i,
                move['start_date'].strftime('%Y-%m-%d'),
                move['end_date'].strftime('%Y-%m-%d'),
                move['length_days'],
                f"${move['start_price']:.2f}",
                f"${move['end_price']:.2f}",
                f"{move['pct_change']:.2f}%",
                f"{move['r_squared']:.3f}",
                f"{move['slope']:.2f}%/day",
                move['news_summary']
            ])
        
        # print(tabulate(move_table, 
        #               headers=["#", "Start Date", "End Date", "Days", "Start", "End", "% Change", "R²", "Slope", "Key News"],
        #               tablefmt="grid", maxcolwidths=[5, 12, 12, 8, 10, 10, 10, 8, 10, 60]))

        #rich_display_dataframe(move_table, title=f"Significant Linear Upward Moves for {ticker}")

        # 2. Initialize the Console and Table

        headers=["#", "Start Date", "End Date", "Days", "Start", "End", "% Change", "R²", "Slope", "Key News"]
        table = Table(title="Big Moves", show_header=True, header_style="bold magenta",box=box.SQUARE, show_lines=True)
       
        # 3. Add columns to the table
        for header in headers:
            table.add_column(header, justify="center", style="green")

        for row in move_table:
            # Use the unpack operator (*) to pass each element as a separate argument
            # and convert non-string values to strings if necessary (rich tables require strings)
            str_row = [str(item) for item in row]
            table.add_row(*str_row)

        console.print(table)
        # Display detailed news if requested
        if args.detailed_news:
            console.print("\n")  # Add some spacing
            for move in moves:
                news_table = format_detailed_news(
                    move['news'], 
                    ticker, 
                    move['start_date'], 
                    move['end_date'], 
                    move['pct_change'], 
                    move['length_days'],
                    move['r_squared']
                )
                console.print(news_table)
                console.print("\n")  # Add spacing between moves
        # 4. Add rows to the table


if __name__ == "__main__":
    main()