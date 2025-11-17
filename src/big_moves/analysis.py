"""
!Deprecated. Now, using src/big_moves/segmented_regression.py

Data analysis and filtering functions for detecting significant stock moves.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def identify_linear_moves(data, min_length=10, min_r_squared=0.9, 
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
            
            # If R-squared starts deteriorating significantly, we've reached the end
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
                i += 1
        else:
            i += 1
    
    return moves
