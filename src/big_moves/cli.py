"""
CLI utilities and formatting functions.
"""
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

def display_header():
    """Display the application header."""
    title = Text("📈 Big Moves - Stock Linear Move Detector", style="bold cyan")
    subtitle = Text("\nDetecting significant linear price movements in stocks", style="italic")
    panel = Panel(title + subtitle, border_style="cyan")
    console.print(panel)

def display_results(symbol, slope, r_squared):
    """Display analysis results with formatting."""
    move_type = "UP 🚀" if slope > 0 else "DOWN 📉"
    strength = abs(slope) * r_squared
    
    result = Text()
    result.append(f"\nStock: ", style="bold")
    result.append(symbol, style="cyan")
    result.append(f"\nMove Direction: ", style="bold")
    result.append(f"{move_type}", style="green" if slope > 0 else "red")
    result.append(f"\nStrength: ", style="bold")
    result.append(f"{strength:.2f}", style="yellow")
    
    console.print(Panel(result, title="Analysis Results", border_style="cyan"))

def display_detailed_news(news_df, ticker, start_date, end_date, pct_change, days, r_squared):
    """Display detailed news with formatting."""
    if news_df.empty:
        return console.print(Panel(
            f"No news found for {ticker} from 10 days before {start_date.strftime('%Y-%m-%d')} "
            f"to 7 days after {end_date.strftime('%Y-%m-%d')} "
            f"(Move: {pct_change:.2f}% over {days} days)",
            title="No News Found",
            style="yellow"
        ))
    
    # Create header panel
    header = f"{ticker} News Summary\n"
    header += f"Period: 10 days before {start_date.strftime('%Y-%m-%d')} to 7 days after {end_date.strftime('%Y-%m-%d')}\n"
    header += f"Move: {pct_change:.2f}% over {days} days (R² = {r_squared:.3f})"
    
    console.print(Panel(header, style="bold blue"))
    
    # Create news table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, show_lines=True)
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Source", style="green")
    table.add_column("Title", style="white")
    table.add_column("URL", style="blue")
    
    sorted_news = news_df.sort_values('Date', ascending=False)
    for _, row in sorted_news.iterrows():
        table.add_row(
            row['Date'].strftime('%Y-%m-%d'),
            row.get('Source', 'N/A'),
            row['Title'],
            row['Link']
        )
    
    console.print(table)
    console.print("\n")
