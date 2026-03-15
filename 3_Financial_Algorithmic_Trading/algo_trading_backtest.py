import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 7)

def get_stock_data(ticker, start_date, end_date):
    print(f"Fetching historical data for {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        print("Data fetched successfully!")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def backtest_ma_crossover(df, short_window=50, long_window=200):
    print(f"Backtesting {short_window}-Day vs {long_window}-Day MA Crossover Strategy...")
    
    # Calculate Moving Averages
    df['SMA_Short'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate Trading Signals
    # 1 when Short MA > Long MA (Bullish), 0 otherwise
    df['Signal'] = 0
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate strategy returns (Shift signal by 1 day because you trade at the close of the signal day)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    
    # Calculate cumulative returns
    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    # Drop NAs
    df.dropna(inplace=True)
    
    return df

def generate_performance_report(df, ticker, output_dir):
    print("Generating Performance Analytics & Visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price Map with Moving Averages and Buy/Sell Signals
    plt.figure()
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    plt.plot(df.index, df['SMA_Short'], label='50-Day SMA', alpha=0.8)
    plt.plot(df.index, df['SMA_Long'], label='200-Day SMA', alpha=0.8)
    
    # Spot Buy/Sell signals to mark on the chart
    df['Trade_Type'] = df['Signal'].diff()
    buy_signals = df[df['Trade_Type'] == 1]
    sell_signals = df[df['Trade_Type'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)
    
    plt.title(f'{ticker} Simple Moving Average Crossover Backtest')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{ticker}_ma_crossover_plot.png"))
    plt.close()
    
    # 2. Cumulative Returns Comparison (Market vs Strategy)
    plt.figure()
    plt.plot(df.index, df['Cumulative_Market_Return'], label='Buy & Hold Return', color='blue')
    plt.plot(df.index, df['Cumulative_Strategy_Return'], label='SMA Strategy Return', color='orange')
    plt.title(f'Performance Comparison: {ticker} (Strategy vs. Buy & Hold)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{ticker}_cumulative_returns.png"))
    plt.close()
    
    # 3. Print Summary Stats
    total_market_return = (df['Cumulative_Market_Return'].iloc[-1] - 1) * 100
    total_strat_return = (df['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    
    print("\n" + "="*40)
    print(f"BACKTEST RESULTS FOR {ticker}")
    print("="*40)
    print(f"Total Market Return (Buy & Hold): {total_market_return:.2f}%")
    print(f"Total Strategy Return: {total_strat_return:.2f}%")
    
    if total_strat_return > total_market_return:
        print("\n🏆 The SMA Crossover Strategy OUTPERFORMED the market.")
    else:
        print("\n📉 The SMA Crossover Strategy UNDERPERFORMED the market.")
        
    print(f"Visualizations saved to: {output_dir}/")

if __name__ == "__main__":
    TICKER = 'AAPL' # Apple Inc. 
    START_DATE = '2015-01-01'
    END_DATE = '2023-12-31'
    
    base_dir = os.path.dirname(__file__)
    viz_dir = os.path.join(base_dir, "analytics_dashboard")
    
    # 1. Fetch Data
    df = get_stock_data(TICKER, START_DATE, END_DATE)
    
    if df is not None:
        # 2. Execute Strategy
        df_backtest = backtest_ma_crossover(df)
        
        # 3. Generate Analytics
        generate_performance_report(df_backtest, TICKER, viz_dir)
        
        # Save historical dataset
        df_backtest.to_csv(os.path.join(base_dir, f"{TICKER}_backtest_results.csv"))
        print("\nEnd of execution.")
