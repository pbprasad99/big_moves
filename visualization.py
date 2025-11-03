import yfinance as yf
import plotext as plt

W, H = plt.terminal_size()
plt.plotsize(W*0.5, H*0.5)


def plot_candlestick_chart(data,ticker):
    """
    Plot a candlestick chart using plotext.
    Args:
        data (pd.DataFrame): DataFrame with stock data containing 'Open', 'High', 'Low', 'Close' columns.               
        ticker (str): Stock ticker symbol for title display.
    Returns:
        None

    """

    dates = plt.datetimes_to_string(data.index)
    plt.candlestick(dates, data,orientation='vertical') 
    plt.title(f"{ticker} Stock Price CandleSticks")
    plt.xlabel("Date")
    plt.ylabel("Stock Price $")
    plt.show()
    plt.clear_data()



def plot_volume_chart(data, ticker):
    """
    Plot a volume chart using plotext with green/red colors based on price movement.
    Args:
        data (pd.DataFrame): DataFrame with stock data containing 'Volume', 'Open', 'Close' columns.               
        ticker (str): Stock ticker symbol for title display.
    Returns:
        None
    """
    dates = plt.datetimes_to_string(data.index)
    volumes = data['Volume'].tolist()

    
    # Plot all volumes in a single pass with their respective colors
    plt.bar(
        dates,
        volumes,
        orientation='vertical',
        color='cyan',
        # width=0.8
    )
    
    plt.title(f"{ticker} Stock Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.show()
    plt.clear_data()


#Testing
# if __name__ == "__main__":
    # import pandas as pd
    # ticker = "AAPL"
    # stock = yf.Ticker(ticker)
    # data = stock.history(period="1y", interval="1d")
    # print(data.head())
    # plot_candlestick_chart(data,ticker)
    # plot_volume_chart(data,ticker)