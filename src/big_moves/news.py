"""
News fetching and processing utilities.
"""
import pandas as pd
import yfinance as yf
from datetime import timedelta

def fetch_yahoo_finance_news(ticker):
    """Fetch news for a ticker from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.get_news(count=1000)
        
        if not news_data:
            print(f"No news found for {ticker} on Yahoo Finance")
            return pd.DataFrame(columns=['Date', 'Title', 'Link', 'Source'])
            
        formatted_news = []
        for item in news_data:
            content = item.get('content', {}) if isinstance(item.get('content'), dict) else {}
            
            pub_date = content.get('pubDate') or item.get('pubDate')
            title = content.get('title') or item.get('title')
            
            link = ""
            click_through = content.get('clickThroughUrl')
            if isinstance(click_through, dict):
                link = click_through.get('url', '')
            else:
                canonical = content.get('canonicalUrl')
                if isinstance(canonical, dict):
                    link = canonical.get('url', '')
                else:
                    link = item.get('link', '')
                
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                provider = provider.get('displayName', 'Yahoo Finance')
            else:
                provider = item.get('provider', 'Yahoo Finance')
                if isinstance(provider, dict):
                    provider = provider.get('displayName', 'Yahoo Finance')
                
            if pub_date and title:
                formatted_news.append({
                    'Date': pub_date,
                    'Title': title,
                    'Link': link,
                    'Source': provider
                })
                
        news_df = pd.DataFrame(formatted_news)
        
        if not news_df.empty:
            try:
                news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')
            except:
                try:
                    news_df['Date'] = pd.to_datetime(news_df['Date'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
                except:
                    pass
                
            news_df = news_df.dropna(subset=['Date'])
            
        return news_df
    
    except Exception as e:
        print(f"Error fetching Yahoo Finance news for {ticker}: {e}")
        return pd.DataFrame(columns=['Date', 'Title', 'Link', 'Source'])

def filter_news_by_date(news_df, start_date, end_date, extra_days_before=10, extra_days_after=7):
    """Filter news by date range with extra days before and after."""
    if news_df.empty:
        return pd.DataFrame(columns=['Date', 'Title', 'Link', 'Source'])
    
    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)
    
    extended_start = start_date - pd.Timedelta(days=extra_days_before)
    extended_end = end_date + pd.Timedelta(days=extra_days_after)
    
    news_df['Date'] = news_df['Date'].dt.tz_localize(None)
    
    filtered_news = news_df[
        (news_df['Date'] >= extended_start) & 
        (news_df['Date'] <= extended_end)
    ]
    
    return filtered_news

def summarize_news(news_df, max_headlines=5, max_width=60):
    """Create a summary of news headlines."""
    if news_df.empty:
        return "No relevant news found"
    
    sorted_news = news_df.sort_values('Date', ascending=True)
    headlines = []
    
    for i, (_, row) in enumerate(sorted_news.iterrows()):
        if i >= max_headlines:
            break
            
        date_str = row['Date'].strftime('%Y-%m-%d')
        headline = f"[{date_str}] {row['Title']}"
        wrapped = [headline[i:i+max_width] for i in range(0, len(headline), max_width)]
        headlines.extend(wrapped)
    
    if len(sorted_news) > max_headlines:
        headlines.append(f"... and {len(sorted_news) - max_headlines} more news items")
        
    return "\n".join(headlines)
