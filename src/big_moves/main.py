"""
Main entry point for the Big Moves application.
"""
import sys
import argparse
from . import data, analysis, visualization, cli
from .segmented_regression import SegmentedRegression
from rich.panel import Panel
from rich.table import Table
from rich import box
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Move Detector and News Analyzer")
    parser.add_argument("symbol", type=str, help="Stock ticker symbol")
    parser.add_argument("--period", type=str, default="1y",
                        help="Lookback period for data (default: 1y)")
    parser.add_argument("--max_segments", type=int, default=6,
                        help="Maximum number of segments to search for (default: 6)")
    parser.add_argument("--min_points", type=int, default=3,
                        help="Minimum points per segment for segmented regression (default: 3)")
    # parser.add_argument("--output", type=str, choices=["console"], default="console",
    #                     help="Output format (default: console)")
    parser.add_argument("--detailed_news", action="store_true",
                        help="Show detailed news for each move")
    parser.add_argument("--big_move_threshold",type=float,default=20.0,
                        help="Moves exceeding this percent change will be highlighted")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    # Display header
    cli.display_header()
    
    # Fetch data
    dates, prices, stock_data = data.fetch_stock_data(args.symbol, args.period)
    if not prices:
        return 1
        
    # Use segmented regression to identify piecewise linear segments
    x = np.arange(len(stock_data))
    y = stock_data['Close'].values
    seg_model = SegmentedRegression(max_segments=args.max_segments, min_points_per_segment=args.min_points)
    seg_model.fit(x, y)

    # Overlay the fitted segments on the candlestick chart
    visualization.plot_candlestick_with_segments(stock_data, args.symbol, seg_model.models)

    # Also show volume chart below
    visualization.plot_volume_chart(stock_data, args.symbol)

    # Build segments list in the same format as previous 'moves' for downstream processing
    moves = []
    for model in seg_model.models:
        start = int(model['start_idx'])
        end = int(model['end_idx'])
        start_date = stock_data.index[start]
        end_date = stock_data.index[end]
        start_price = float(stock_data['Close'].iloc[start])
        end_price = float(stock_data['Close'].iloc[end])
        pct_change = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0.0
        length_days = end - start + 1
        pct_per_day = (pct_change / length_days) if length_days > 0 else 0.0
        volume = float(stock_data['Volume'].iloc[start:end+1].mean()) if 'Volume' in stock_data.columns else 0.0
        slope = float(model['slope'])
        # Compute R-squared for the segment
        y_seg = y[start:end+1]
        x_seg = x[start:end+1]
        y_pred = slope * x_seg + model['intercept']
        ss_res = float(((y_seg - y_pred) ** 2).sum())
        ss_tot = float(((y_seg - y_seg.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        moves.append({
            'start_date': start_date,
            'end_date': end_date,
            'length_days': length_days,
            'start_price': start_price,
            'end_price': end_price,
            'pct_change': pct_change,
            'pct_per_day': pct_per_day,
            'volume': volume,
            'r_squared': r_squared,
            'slope': slope,
            'rss': float(model.get('rss', ss_res))
        })

    if not moves:
        cli.console.print(Panel(
            f"No segments identified for {args.symbol}",
            style="yellow",
            title="No Segments Found"
        ))
        return 0
        
    # Fetch news for moves
    from . import news
    all_news = news.fetch_yahoo_finance_news(args.symbol)
    
    # Process each segment and attach news
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
    headers = ["#", "Start Date", "End Date", "Days", "Start", "End", "% Change", "R²", "Slope (raw)", "%/day", "Key News"]
    for header in headers:
        table.add_column(header, justify="center", style="green")

    # Highlight threshold (percent)
    # HIGHLIGHT_THRESHOLD = args.big_move_threshold

    for i, move in enumerate(moves, 1):
        abs_pct = abs(move['pct_change'])
        # If the move exceeds the highlight threshold, use a strong background style
        if abs_pct >= args.big_move_threshold:
            if move['pct_change'] > 0:
                row_style = "bold black on chartreuse3"
            elif move['pct_change'] < 0:
                row_style = "bold black on orange3"
            else:
                row_style = "bold black on white"
        else:
            # regular directional coloring
            if move['pct_change'] > 0:
                row_style = "bold black on honeydew2"
            elif move['pct_change'] < 0:
                row_style = "bold black on light_salmon1"
            else:
                row_style = "bold black on white"

        table.add_row(
            str(i),
            move['start_date'].strftime('%Y-%m-%d'),
            move['end_date'].strftime('%Y-%m-%d'),
            str(move['length_days']),
            f"${move['start_price']:.2f}",
            f"${move['end_price']:.2f}",
            f"{move['pct_change']:.2f}%",
            f"{move['r_squared']:.3f}",
            f"{move['slope']:.6f}",
            f"{move['pct_per_day']:.2f}%/day",
            move['news_summary'],
            style=row_style
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
                move['r_squared'],
                highlight_threshold=args.big_move_threshold
            )

    return 0

if __name__ == "__main__":
    sys.exit(main())
