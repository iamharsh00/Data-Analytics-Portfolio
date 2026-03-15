# Financial Algorithmic Trading Analytics 📈💸

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![yfinance](https://img.shields.io/badge/yfinance-0.2.28-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.1.0-150458.svg)

## 📌 Project Overview
Quantitative Finance and Algorithmic Trading heavily rely on data manipulation to find edges in the market. This project simulates a quantitative research environment by pulling live/historical stock market data using the `yfinance` API and executing a backtest for the popular **SMA (Simple Moving Average) Crossover Strategy**.

### 🎯 Objective:
- Automatically fetch years of historical financial data for any given ticker (e.g., AAPL).
- Develop a Pythonic backtesting framework to calculate Daily and Cumulative Returns.
- Visually and statistically compare the mathematical trading strategy against a standard "Buy and Hold" benchmark.

---

## 🏗️ Project Structure
```text
📦 3_Financial_Algorithmic_Trading
 ┣ 📜 algo_trading_backtest.py      # Main ingestion, execution, and rendering logic
 ┣ 📜 AAPL_backtest_results.csv     # Exported historical dataframe with buy/sell signals
 ┣ 📜 requirements.txt              # Project dependencies
 ┗ 📂 analytics_dashboard           # Exported Matplotlib/Seaborn Strategy Charts
   ┣ 🖼️ AAPL_ma_crossover_plot.png
   ┗ 🖼️ AAPL_cumulative_returns.png
```

## 🛠️ Tech Stack & Skills
* **Libraries:** `pandas`, `numpy`, `yfinance`, `matplotlib`, `seaborn`
* **Finance Concepts:** Moving Averages (50-Day & 200-Day), Portfolio Returns, Algorithmic Buy/Sell Signals, Benchmarking.

## 🚀 How to Run the Strategy
1. **Ensure dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute the Backtest:**
   ```bash
   python algo_trading_backtest.py
   ```
   *The script will query Yahoo Finance, run the algorithmic calculations, log the total ROI compared to the market, and save the charts in the `/analytics_dashboard/` folder.*

---
*Developed with 💻 as part of a Data Analytics Portfolio.*
