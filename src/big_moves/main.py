"""
Main entry point for the Big Moves application.
"""
import sys
import argparse
from . import data, analysis, visualization, cli
from rich.panel import Panel
from rich.table import Table
from rich import box

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Linear Move Detector and News Analyzer")
    parser.add_argument("symbol", type=str, help="Stock ticker symbol")
    parser.add_argument("--min_length", type=int, default=10,
                        help="Minimum trend length in trading days (default: 10)")
    parser.add_argument("--max_length", type=int, default=252,
                        help="Maximum trend length in trading days (default: 252, ~1 year)")
    parser.add_argument("--min_change", type=float, default=30.0,
                        help="Minimum percentage change required (default: 30.0)")
    parser.add_argument("--r_squared", type=float, default=0.9,
                        help="Minimum R-squared value for linear fit (default: 0.9)")
    parser.add_argument("--min_slope", type=float, default=0.1,
                        help="Minimum daily slope as percentage of price (default: 0.1)")
    parser.add_argument("--period", type=str, default="1y",
                        help="Lookback period for data (default: 1y)")
    parser.add_argument("--output", type=str, choices=["console"], default="console",
                        help="Output format (default: console)")
    parser.add_argument("--detailed_news", action="store_true",
                        help="Show detailed news for each move")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Display header
    # cli.display_header()
    
    # Analysis status panel
    status = f"🔍 Analyzing {args.symbol}\n"
    status += f"📈 Looking for upward moves of {args.min_change}% or more\n"
    status += f"📅 Over {args.min_length} to {args.max_length} trading days\n"
    status += f"📊 Minimum R² value: {args.r_squared}"
    
    cli.console.print(Panel(status, title="Big Moves", border_style="cyan", padding=(1, 2)))
    
    # Fetch data
    dates, prices, stock_data = data.fetch_stock_data(args.symbol, args.period)
    if not prices:
        return 1
        
    # Plot charts
    visualization.plot_candlestick_chart(stock_data, args.symbol)
    visualization.plot_volume_chart(stock_data, args.symbol)
    
    # Analyze movement
    moves = analysis.identify_linear_moves(
        stock_data,
        min_length=args.min_length,
        max_length=args.max_length,
        min_r_squared=args.r_squared,
        min_slope=args.min_slope,
        min_pct_change=args.min_change
    )
    
    if not moves:
        cli.console.print(Panel(
            f"No significant linear upward moves of {args.min_change}% or more found for {args.symbol}",
            style="yellow",
            title="No Moves Found"
        ))
        return 0
        
    # Fetch news for moves
    from . import news
    cli.console.print(Panel("🔎 Fetching relevant news articles...", style="blue", subtitle="Please wait"))
    all_news = news.fetch_yahoo_finance_news(args.symbol)
    
    # Process each move
    for i, move in enumerate(moves):
        filtered_news = news.filter_news_by_date(
            all_news,
            move['start_date'],
            move['end_date']
        )
        moves[i]['news'] = filtered_news
        moves[i]['news_summary'] = news.summarize_news(filtered_news)
    
    # Display results
    table = Table(show_header=True, header_style="bold magenta", box=box.SQUARE, show_lines=True)
    headers = ["#", "Start Date", "End Date", "Days", "Start", "End", "% Change", "R²", "Slope", "Key News"]
    for header in headers:
        table.add_column(header, justify="center", style="green")
    
    for i, move in enumerate(moves, 1):
        table.add_row(
            str(i),
            move['start_date'].strftime('%Y-%m-%d'),
            move['end_date'].strftime('%Y-%m-%d'),
            str(move['length_days']),
            f"${move['start_price']:.2f}",
            f"${move['end_price']:.2f}",
            f"{move['pct_change']:.2f}%",
            f"{move['r_squared']:.3f}",
            f"{move['slope']:.2f}%/day",
            move['news_summary']
        )
    
    cli.console.print(table)
    
    # Display detailed news if requested
    if args.detailed_news:
        cli.console.print("\n")
        for move in moves:
            cli.display_detailed_news(
                move['news'],
                args.symbol,
                move['start_date'],
                move['end_date'],
                move['pct_change'],
                move['length_days'],
                move['r_squared']
            )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
