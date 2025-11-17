"""
Main entry point for the Big Moves application.
"""
import sys
import argparse
from . import data, analysis, visualization, cli
from .segmented_regression import SegmentedRegression
from .ai_summarizer import summarize_news_df
from rich.panel import Panel
from rich.table import Table
from rich import box
import numpy as np
import os
from pathlib import Path
import getpass

# Helpers to read/write .env at project root
ENV_PATH = Path.cwd() / '.env'


def get_global_env_path(app_name: str = 'big-moves') -> Path:
    """Return a cross-platform per-user config .env path (creates parent dirs).
    Uses platformdirs.user_config_dir when available, otherwise falls back to XDG/AppData locations.
    """
    try:
        from platformdirs import user_config_dir
        cfg_dir = Path(user_config_dir(app_name))
    except Exception:
        if os.name == 'nt':
            appdata = os.getenv('LOCALAPPDATA') or os.getenv('APPDATA') or str(Path.home())
            cfg_dir = Path(appdata) / app_name
        else:
            xdg = os.getenv('XDG_CONFIG_HOME')
            if xdg:
                cfg_dir = Path(xdg) / app_name
            else:
                cfg_dir = Path.home() / '.config' / app_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / '.env'


def _read_env_file(path: Path):
    data = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def _write_env_file(path: Path, data: dict):
    lines = [f"{k}={v}" for k, v in data.items()]
    # ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _mask_value(val: str):
    if not val:
        return ''
    if len(val) <= 6:
        return '****'
    return val[:3] + '...' + val[-3:]


def parse_args():
    """Parse command line arguments, including a 'config' subcommand."""
    parser = argparse.ArgumentParser(description="Stock Move Detector and News Analyzer")

    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    # main run parser (default)
    run_parser = subparsers.add_parser('run', help='Run analysis (default)')
    run_parser.add_argument('symbol', type=str, help='Stock ticker symbol')
    run_parser.add_argument('--period', type=str, default='1y', help='Lookback period for data (default: 1y)')
    run_parser.add_argument('--max_segments', type=int, default=6, help='Maximum number of segments to search for (default: 6)')
    run_parser.add_argument('--min_points', type=int, default=3, help='Minimum points per segment for segmented regression (default: 3)')
    run_parser.add_argument('--detailed_news', action='store_true', help='Show detailed news for each move')
    run_parser.add_argument('--big_move_threshold', type=float, default=20.0, help='Moves exceeding this percent change will be highlighted')

    # allow running without the 'run' word by treating no subcommand as run
    # config subcommand
    cfg = subparsers.add_parser('config', help='Manage stored credentials (.env)')
    cfg.add_argument('action', choices=['show', 'set', 'unset'], help='Action to perform on config')
    cfg.add_argument('key', nargs='?', help='Environment variable key (e.g. GROQ_API_KEY). Use "all" to prompt for all required keys')
    cfg.add_argument('value', nargs='?', help='Value when setting a key')
    cfg.add_argument('-g', '--global', action='store_true', dest='global_flag', help='Store in per-user global config location')
    cfg.add_argument('--all', action='store_true', dest='all_keys', help='Prompt interactively for all required keys')

    # If no args provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    # normalize: if command is None, treat it as 'run' and allow compatibility with old usage
    if args.command is None:
        # map sys.argv into run args for backwards compatibility
        # Reparse using run_parser behavior
        # Build a small fallback parse
        simple_parser = argparse.ArgumentParser(add_help=False)
        simple_parser.add_argument('symbol', type=str)
        simple_parser.add_argument('--period', type=str, default='1y')
        simple_parser.add_argument('--max_segments', type=int, default=6)
        simple_parser.add_argument('--min_points', type=int, default=3)
        simple_parser.add_argument('--detailed_news', action='store_true')
        simple_parser.add_argument('--big_move_threshold', type=float, default=20.0)
        parsed, _ = simple_parser.parse_known_args()
        parsed.command = 'run'
        return parsed

    return args


# Simple config command handler to set/show/unset credentials
REQUIRED_CONFIG_KEYS = [
    ('GROQ_API_KEY', True),  # (key_name, is_sensitive)
    ('GROQ_MODEL', False),
]


def handle_config_command(action, key=None, value=None, global_flag=False, all_keys=False):
    """Handle config subcommand actions with interactive set/unset support.

    - `set` will prompt for a value if not provided on the command line. Sensitive keys
      (contains KEY/SECRET/TOKEN/PASS) will be prompted with hidden input.
    - `unset` will require confirmation interactively when running in a TTY.
    """
    target_env_path = get_global_env_path() if global_flag else ENV_PATH
    env_data = _read_env_file(target_env_path)

    if action == 'show':
        if not env_data:
            print(f"No env file found at {target_env_path}")
            return
        print(f"Configured environment variables in {target_env_path} (masked):")
        for k, v in env_data.items():
            print(f"{k}={_mask_value(v)}")
        return

    if action == 'set':
        # If user asked to set all keys, or specified key as 'all'
        if all_keys or (key and key.lower() == 'all'):
            for req_key, is_sensitive in REQUIRED_CONFIG_KEYS:
                # skip if already present
                if req_key in env_data:
                    print(f"{req_key} already set; skipping (use unset to change)")
                    continue
                prompt = f"Enter value for {req_key} ({'sensitive' if is_sensitive else 'optional'}): "
                try:
                    if is_sensitive:
                        val = getpass.getpass(prompt)
                    else:
                        val = input(prompt)
                except (KeyboardInterrupt, EOFError):
                    print('\nAborted.')
                    return
                if val:
                    env_data[req_key] = val
            _write_env_file(target_env_path, env_data)
            print(f"Updated {target_env_path}")
            return

        if not key:
            print("Usage: big-moves config set KEY [VALUE] [--global] or use --all to set required keys")
            return

        # If value not provided, prompt interactively
        if value is None:
            sensitive = any(tok in key.upper() for tok in ("KEY", "SECRET", "TOKEN", "PASS", "PWD"))
            prompt = f"Enter value for {key}: "
            try:
                if sensitive:
                    value = getpass.getpass(prompt)
                else:
                    value = input(prompt)
            except (KeyboardInterrupt, EOFError):
                print('\nAborted.')
                return
        if value is None or value == '':
            print("No value provided; aborting.")
            return
        env_data[key] = value
        _write_env_file(target_env_path, env_data)
        print(f"Set {key} in {target_env_path}")
        return

    if action == 'unset':
        if not key:
            print("Usage: big-moves config unset KEY [--global]")
            return
        if key not in env_data:
            print(f"{key} not found in {target_env_path}")
            return
        # Confirm with user if running interactively
        try:
            if sys.stdin.isatty():
                ans = input(f"Remove {key} from {target_env_path}? [y/N]: ").strip().lower()
                if ans not in ('y', 'yes'):
                    print("Aborted.")
                    return
        except Exception:
            # non-interactive, proceed
            pass
        env_data.pop(key, None)
        _write_env_file(target_env_path, env_data)
        print(f"Removed {key} from {target_env_path}")
        return

    print("Unknown config action. Use show|set|unset")
    return


def main():
    """Main entry point."""
    args = parse_args()

    # Handle config subcommand if invoked
    if args.command == 'config':
        handle_config_command(args.action, args.key, args.value, args.global_flag, args.all_keys)
        sys.exit(0)

    # Startup check for GROQ key
    if not os.getenv('GROQ_API_KEY'):
        cli.console.print(Panel(
            "GROQ_API_KEY is not set. AI summarization will fall back to a local summarizer.\n"
            "To enable AI summaries, create a .env file from sample.env or run:\n"
            "  big-moves config set GROQ_API_KEY <your_key>",
            title="Configuration Warning",
            style="yellow"
        ))

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
        # Generate AI summary using Groq (via ai_summarizer). Fall back to legacy summarizer on error.
        try:
            ai_summary = summarize_news_df(filtered_news, max_tokens=150)
            if not ai_summary:
                # fallback to existing summarizer if AI returns empty
                ai_summary = news.summarize_news(filtered_news)
        except Exception:
            ai_summary = news.summarize_news(filtered_news)
        moves[i]['news_summary'] = ai_summary
    
    # Display results
    table = Table(show_header=True, header_style="bold magenta", box=box.SQUARE, show_lines=True)
    # Add columns; the last column (Key News) should allow full multi-line content without ellipsis.
    table.add_column("#", justify="center", style="green")
    table.add_column("Start Date", justify="center", style="green")
    table.add_column("End Date", justify="center", style="green")
    table.add_column("Days", justify="center", style="green")
    table.add_column("Start", justify="center", style="green")
    table.add_column("End", justify="center", style="green")
    table.add_column("% Change", justify="center", style="green")
    table.add_column("R²", justify="center", style="green")
    table.add_column("Slope (raw)", justify="center", style="green")
    table.add_column("%/day", justify="center", style="green")
    # Key News may be long; allow folding/wrapping instead of ellipsizing so the full text is visible.
    table.add_column("Key News", justify="left", style="green", no_wrap=False, overflow="fold")

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
