import pandas as pd
import yfinance as yf

def fetch_yahoo_finance_news(ticker):
    """Fetch news for a ticker from Yahoo Finance and format like Finviz news"""
    try:
        import yfinance as yf
        
        # Get the ticker object
        stock = yf.Ticker(ticker)
        
        # Get the news data
        news_data = stock.get_news(count=1000)
        
        if not news_data:
            print(f"No news found for {ticker} on Yahoo Finance")
            return pd.DataFrame(columns=['Date', 'Title', 'Link', 'Source'])
            
        # Create a list to hold the formatted news items
        formatted_news = []
        
        for item in news_data:
            # Extract the publication date
            content = item.get('content', {}) if isinstance(item.get('content'), dict) else {}
            
            pub_date = content.get('pubDate')
            if not pub_date:
                # Try alternate location
                pub_date = item.get('pubDate')
                
            # Extract title
            title = content.get('title')
            if not title:
                # Try alternate location
                title = item.get('title')
                
            # Extract link - safely handle None values
            link = ""
            click_through = content.get('clickThroughUrl')
            if isinstance(click_through, dict):
                link = click_through.get('url', '')
            else:
                # Try alternate locations
                canonical = content.get('canonicalUrl')
                if isinstance(canonical, dict):
                    link = canonical.get('url', '')
                else:
                    link = item.get('link', '')
                
            # Extract source
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                provider = provider.get('displayName', 'Yahoo Finance')
            else:
                # Try alternate location
                provider = item.get('provider', 'Yahoo Finance')
                if isinstance(provider, dict):
                    provider = provider.get('displayName', 'Yahoo Finance')
                
            # Only add items with at least a date and title
            if pub_date and title:
                formatted_news.append({
                    'Date': pub_date,
                    'Title': title,
                    'Link': link,
                    'Source': provider
                })
                
        # Convert to DataFrame
        news_df = pd.DataFrame(formatted_news)
        
        # Convert dates to datetime objects
        if not news_df.empty:
            # Handle different date formats
            try:
                news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')
            except:
                # If standard parsing fails, try a specific format
                try:
                    news_df['Date'] = pd.to_datetime(news_df['Date'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
                except:
                    pass
                
            # Drop rows with invalid dates
            news_df = news_df.dropna(subset=['Date'])
            
        return news_df
    
    except Exception as e:
        import traceback
        print(f"Error fetching Yahoo Finance news for {ticker}: {e}")
        traceback.print_exc()  # Print the full stack trace for debugging
        return pd.DataFrame(columns=['Date', 'Title', 'Link', 'Source'])
    

    
# def fetch_yahoo_finance_news(ticker):
#     """Fetch news for a ticker from Yahoo Finance"""
#     try:
#         import yfinance as yf
#         stock = yf.ticker(ticker)
        
#         # Get news from Yahoo Finance
#         news_data = stock.news
        
#         if not news_data:
#             print(f"No news found for {ticker} on Yahoo Finance")
#             return pd.DataFrame()
            
#         # Convert to DataFrame
#         news_df = pd.DataFrame(news_data)
        
#         # Extract relevant columns and rename
#         if 'title' in news_df.columns and 'link' in news_df.columns and 'providerPublishTime' in news_df.columns:
#             news_df = news_df[['title', 'link', 'providerPublishTime']]
#             news_df.columns = ['Title', 'Link', 'Timestamp']
            
#             # Convert timestamp to datetime
#             news_df['Date'] = pd.to_datetime(news_df['Timestamp'], unit='s')
#             news_df = news_df.drop('Timestamp', axis=1)
            
#             return news_df
#         else:
#             print(f"Unexpected column format in Yahoo Finance news for {ticker}")
#             return pd.DataFrame()
            
#     except Exception as e:
#         print(f"Error fetching Yahoo Finance news for {ticker}: {e}")
#         return pd.DataFrame()



def fetch_finviz_news(ticker):
    """Fetch news for a ticker from Finviz using the current API"""
    try:
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(ticker)
        
        # Get the stock information dictionary
        stock_info = stock.ticker_fundament()
        
        # Access the news data - this is the updated method
        news_df = stock.ticker_news()
        
        if news_df is None or news_df.empty:
            print(f"No news found for {ticker}")
            return pd.DataFrame()
            
        return news_df
    except Exception as e:
        print(f"Error fetching Finviz news for {ticker}: {e}")
        # Return an empty DataFrame as fallback
        return pd.DataFrame(columns=['Date', 'Title', 'Link'])
    

if __name__ == "__main__":

    print("testing yahoo finance news fetch")
    # ticker = "AAPL"
    # news_df = fetch_finviz_news(ticker)
    # print(news_df)
    # print(news_df.dtypes)
    
    print("testing yahoo finance news fetch")
    ticker = "TSLA"
    news_df = fetch_yahoo_finance_news(ticker)
    print(news_df)