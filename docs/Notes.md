## WHat is R-squared?

R-squared (R²) is a statistical measure that shows how well a linear model fits the data. In the context of our stock trend detection algorithm, it measures how "straight" or linear the price movement actually is.

Here's what R-squared means in simpler terms:

- **Range**: R² ranges from 0 to 1
- **Perfect fit (R² = 1)**: All price points fall exactly on a straight line
- **No fit (R² = 0)**: There's no linear relationship at all (completely random)
- **Partial fit (e.g., R² = 0.85)**: 85% of the price movement follows a straight-line pattern

### Visual examples of different R-squared values:

1. **High R² (0.95+)**: 
   ```
   Price  │                                    /
          │                                  /
          │                               /
          │                            /
          │                         /
          │                      /
          └──────────────────────────── Time
   ```
   This is a very clean, straight-line uptrend.

2. **Medium R² (0.75-0.85)**:
   ```
   Price  │                               /
          │                            /  │
          │                         /     │
          │                   /  /        │
          │               /                
          │            /                   
          └──────────────────────────── Time
   ```
   General uptrend but with some deviation from a perfect line.

3. **Low R² (below 0.7)**:
   ```
   Price  │           /\          /
          │         /    \    /\/
          │       /        \/
          │     /
          │   /
          │ /
          └──────────────────────────── Time
   ```
   The price is generally increasing but with significant volatility.

In our algorithm, we use R-squared to:

1. Identify truly linear price movements (setting a high minimum R² like 0.9)
2. Determine when a trend ends (when R² starts to deteriorate)
3. Rank and compare different trends (higher R² means a more reliable linear movement)

By focusing on movements with high R-squared values, we're finding price trends that demonstrate consistent, predictable growth rather than erratic movements that happen to end higher than where they started.