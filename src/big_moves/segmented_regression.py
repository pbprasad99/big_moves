"""
Segmented Regression for Stock Price Data

This script performs segmented regression (piecewise linear regression) on stock price data
and automatically estimates optimal breakpoints using dynamic programming.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SegmentedRegression:
    """
    Performs segmented regression with automatic breakpoint estimation.
    """
    
    def __init__(self, max_segments=5):
        """
        Initialize the segmented regression model.
        
        Parameters:
        -----------
        max_segments : int
            Maximum number of segments to consider
        """
        self.max_segments = max_segments
        self.breakpoints = None
        self.models = None
        self.optimal_segments = None
        self.bic_scores = {}
        
    def fit_single_segment(self, x, y, start_idx, end_idx):
        """
        Fit a linear regression to a single segment.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        start_idx : int
            Start index of segment
        end_idx : int
            End index of segment (inclusive)
            
        Returns:
        --------
        dict : Contains slope, intercept, and residual sum of squares
        """
        x_seg = x[start_idx:end_idx+1]
        y_seg = y[start_idx:end_idx+1]
        
        # Fit linear regression: y = mx + b
        A = np.vstack([x_seg, np.ones(len(x_seg))]).T
        m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        
        # Calculate predictions and RSS
        y_pred = m * x_seg + b
        rss = np.sum((y_seg - y_pred) ** 2)
        
        return {
            'slope': m,
            'intercept': b,
            'rss': rss,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
    
    def calculate_bic(self, n, rss, k):
        """
        Calculate Bayesian Information Criterion.
        
        Parameters:
        -----------
        n : int
            Number of data points
        rss : float
            Residual sum of squares
        k : int
            Number of parameters (2 per segment: slope and intercept)
            
        Returns:
        --------
        float : BIC score
        """
        if rss <= 0:
            return np.inf
        return n * np.log(rss / n) + k * np.log(n)
    
    def find_optimal_breakpoints(self, x, y, n_segments):
        """
        Find optimal breakpoints for a given number of segments using dynamic programming.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        n_segments : int
            Number of segments
            
        Returns:
        --------
        tuple : (breakpoints, total_rss)
        """
        n = len(x)
        min_points_per_segment = 3  # Minimum points needed per segment
        
        if n_segments == 1:
            model = self.fit_single_segment(x, y, 0, n-1)
            return [], model['rss']
        
        # Dynamic programming approach
        # dp[i][j] = minimum RSS for fitting j segments to first i points
        dp = np.full((n, n_segments + 1), np.inf)
        splits = {}
        
        # Base case: 1 segment
        for i in range(min_points_per_segment, n):
            model = self.fit_single_segment(x, y, 0, i)
            dp[i][1] = model['rss']
        
        # Fill DP table
        for j in range(2, n_segments + 1):
            for i in range(j * min_points_per_segment, n):
                # Try all possible positions for the last breakpoint
                for k in range((j-1) * min_points_per_segment, i - min_points_per_segment + 1):
                    model = self.fit_single_segment(x, y, k, i)
                    cost = dp[k-1][j-1] + model['rss']
                    
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        splits[(i, j)] = k
        
        # Backtrack to find breakpoints
        breakpoints = []
        i = n - 1
        for j in range(n_segments, 1, -1):
            if (i, j) in splits:
                k = splits[(i, j)]
                breakpoints.append(k)
                i = k - 1
        
        breakpoints.reverse()
        return breakpoints, dp[n-1][n_segments]
    
    def fit(self, x, y):
        """
        Fit segmented regression with automatic segment selection.
        
        Parameters:
        -----------
        x : array-like
            Independent variable (e.g., time indices)
        y : array-like
            Dependent variable (e.g., stock prices)
        """
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        
        # Try different numbers of segments
        best_bic = np.inf
        best_n_segments = 1
        
        for n_segments in range(1, min(self.max_segments + 1, n // 6 + 1)):
            breakpoints, total_rss = self.find_optimal_breakpoints(x, y, n_segments)
            
            # Calculate BIC
            k = 2 * n_segments  # Each segment has slope and intercept
            bic = self.calculate_bic(n, total_rss, k)
            self.bic_scores[n_segments] = bic
            
            if bic < best_bic:
                best_bic = bic
                best_n_segments = n_segments
                self.breakpoints = breakpoints
        
        self.optimal_segments = best_n_segments
        
        # Fit models for each segment
        self.models = []
        segment_starts = [0] + self.breakpoints
        segment_ends = self.breakpoints + [n - 1]
        
        for start, end in zip(segment_starts, segment_ends):
            model = self.fit_single_segment(x, y, start, end)
            self.models.append(model)
        
        return self
    
    def predict(self, x):
        """
        Predict values for given x using the fitted segmented regression.
        
        Parameters:
        -----------
        x : array-like
            Independent variable values
            
        Returns:
        --------
        array : Predicted values
        """
        x = np.array(x)
        y_pred = np.zeros_like(x)
        
        for model in self.models:
            mask = (x >= x[model['start_idx']]) & (x <= x[model['end_idx']])
            y_pred[mask] = model['slope'] * x[mask] + model['intercept']
        
        return y_pred
    
    def plot_results(self, x, y, dates=None, title="Segmented Regression"):
        """
        Plot the original data and segmented regression results.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        dates : array-like, optional
            Date labels for x-axis
        title : str
            Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Data and segmented regression
        ax1.scatter(x, y, alpha=0.5, s=20, label='Actual Data', color='blue')
        
        # Plot each segment
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.models)))
        for i, model in enumerate(self.models):
            start, end = model['start_idx'], model['end_idx']
            x_seg = x[start:end+1]
            y_pred = model['slope'] * x_seg + model['intercept']
            
            ax1.plot(x_seg, y_pred, color=colors[i], linewidth=2.5, 
                    label=f'Segment {i+1}: slope={model["slope"]:.4f}')
            
            # Mark breakpoints
            if i < len(self.models) - 1:
                ax1.axvline(x[end], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel('Time Index' if dates is None else 'Date', fontsize=12)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title(f'{title}\n({self.optimal_segments} segments, BIC={self.bic_scores[self.optimal_segments]:.2f})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: BIC scores for different numbers of segments
        segments = sorted(self.bic_scores.keys())
        bic_values = [self.bic_scores[s] for s in segments]
        
        ax2.plot(segments, bic_values, marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax2.axvline(self.optimal_segments, color='red', linestyle='--', 
                   label=f'Optimal: {self.optimal_segments} segments', linewidth=2)
        ax2.set_xlabel('Number of Segments', fontsize=12)
        ax2.set_ylabel('BIC Score', fontsize=12)
        ax2.set_title('Model Selection: BIC vs Number of Segments', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(segments)
        
        plt.tight_layout()
        return fig
    
    def summary(self, x, dates=None):
        """
        Print a summary of the segmented regression results.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        dates : array-like, optional
            Date labels
        """
        print("=" * 70)
        print("SEGMENTED REGRESSION SUMMARY")
        print("=" * 70)
        print(f"\nOptimal number of segments: {self.optimal_segments}")
        print(f"BIC Score: {self.bic_scores[self.optimal_segments]:.2f}")
        print(f"\nNumber of breakpoints: {len(self.breakpoints)}")
        
        if dates is not None:
            print("\nBreakpoint dates:")
            for i, bp in enumerate(self.breakpoints):
                print(f"  Breakpoint {i+1}: {dates[bp]}")
        else:
            print(f"\nBreakpoint indices: {self.breakpoints}")
        
        print("\n" + "-" * 70)
        print("SEGMENT DETAILS:")
        print("-" * 70)
        
        for i, model in enumerate(self.models):
            start, end = model['start_idx'], model['end_idx']
            if dates is not None:
                print(f"\nSegment {i+1}: {dates[start]} to {dates[end]}")
            else:
                print(f"\nSegment {i+1}: Index {start} to {end}")
            
            print(f"  Slope: {model['slope']:.6f}")
            print(f"  Intercept: {model['intercept']:.4f}")
            print(f"  RSS: {model['rss']:.4f}")
            
            # Interpret slope
            if model['slope'] > 0.001:
                trend = "upward"
            elif model['slope'] < -0.001:
                trend = "downward"
            else:
                trend = "flat"
            print(f"  Trend: {trend}")
        
        print("=" * 70)


def generate_sample_stock_data(n_points=200):
    """
    Generate synthetic stock price data with multiple trend changes.
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
        
    Returns:
    --------
    tuple : (dates, prices)
    """
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    
    # Generate prices with different trends
    x = np.arange(n_points)
    prices = np.zeros(n_points)
    
    # Segment 1: Upward trend (0-60)
    prices[0:60] = 100 + 0.5 * x[0:60] + np.random.normal(0, 2, 60)
    
    # Segment 2: Flat/slight decline (60-120)
    prices[60:120] = prices[59] - 0.1 * (x[60:120] - 60) + np.random.normal(0, 2, 60)
    
    # Segment 3: Sharp upward (120-160)
    prices[120:160] = prices[119] + 0.8 * (x[120:160] - 120) + np.random.normal(0, 2.5, 40)
    
    # Segment 4: Decline (160-200)
    prices[160:200] = prices[159] - 0.4 * (x[160:200] - 160) + np.random.normal(0, 2, 40)
    
    return dates, prices


def main():
    """
    Main function demonstrating segmented regression on stock price data.
    """
    pass
    # # print("Generating sample stock price data...")
    # # dates, prices = generate_sample_stock_data(n_points=200)
    
    # # # Convert to time indices
    # # x = np.arange(len(prices))
    # import pandas as pd
    # import yfinance as yf
    
    # # Download stock data
    # ticker = yf.Ticker("ASST")
    # df = ticker.history(period="1y")
    
    # # Prepare data
    # dates = df.index
    # prices = df['Close'].values
    # # ma = df['Close'].rolling(window=5).mean().values
    # # prices = ma[~np.isnan(ma)]
    # # dates = dates[4:]
    # x = np.arange(len(prices))
    
    # # Create and fit the model
    # print("\nFitting segmented regression model...")
    # model = SegmentedRegression(max_segments=3)
    # model.fit(x, prices)
    
    # # Print summary
    # model.summary(x, dates=dates)
    
    # # Plot results
    # print("\nGenerating plots...")
    # fig = model.plot_results(x, prices, title="Stock Price Segmented Regression Analysis")
    # plt.savefig('segmented_regression_plot.png', dpi=300, bbox_inches='tight')
    # print("Plot saved to: segmented_regression_plot.png")
    
    # # Make predictions
    # y_pred = model.predict(x)
    
    # # Calculate overall R-squared
    # ss_res = np.sum((prices - y_pred) ** 2)
    # ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    # r_squared = 1 - (ss_res / ss_tot)
    # print(f"\nOverall R-squared: {r_squared:.4f}")
    
    # # Save results to CSV
    # results_df = pd.DataFrame({
    #     'Date': dates,
    #     'Actual_Price': prices,
    #     'Predicted_Price': y_pred,
    #     'Residual': prices - y_pred
    # })
    # results_df.to_csv('segmented_regression_results.csv', index=False)
    # print("Results saved to: segmented_regression_results.csv")
    
    # plt.show()

if __name__ == "__main__":
    main()
