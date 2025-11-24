"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    
    def calculate_weights(self):
        # 排除 SPY，取得可投資資產列表
        assets = self.price.columns[self.price.columns != self.exclude]
        
        # 定義策略參數 (可根據需要調整這些參數以優化結果)
        momentum_lookback = 120
        volatility_lookback = 50
        n_assets = 5 # 每次選擇前 5 個動量最強的資產

        # 初始化權重 DataFrame
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
# gemini        
        # 逐日迭代計算權重
        for i in range(momentum_lookback + 1, len(self.price)):
            current_date = self.price.index[i]
            
            # 1. 動量篩選 (Momentum Screening)
            
            # 獲取計算動量所需的數據 (t - momentum_lookback 到 t - 1)
            # 使用價格數據計算累積回報
            momentum_price_window = self.price[assets].iloc[i - momentum_lookback : i]
            
            # 計算回顧期內的累積回報
            period_returns = momentum_price_window.iloc[-1] / momentum_price_window.iloc[0] - 1.0
            
            # 選擇動量最強的 N 個資產 (N=5)
            # 使用 .nlargest() 獲取表現最好的資產名稱
            selected_assets = period_returns.nlargest(n_assets).index.tolist()
            
            # 2. 風險平價分配 (Risk Parity Allocation)
            
            # 獲取計算波動率所需的日回報數據 (t - volatility_lookback 到 t - 1)
            volatility_return_window = self.returns[selected_assets].iloc[i - volatility_lookback : i]
            
            # 計算波動率 (日回報標準差)
            sigma = volatility_return_window.std()

            # 處理潛在的零波動率
            sigma[sigma == 0] = 1e9 

            # 計算逆波動率 (1 / sigma_i)
            inverse_volatility = 1.0 / sigma

            # 計算總逆波動率 (分母)
            sum_inverse_volatility = inverse_volatility.sum()
            
            # 計算最終的風險平價權重
            if sum_inverse_volatility > 0:
                weights = inverse_volatility / sum_inverse_volatility
            else:
                weights = pd.Series(0.0, index=selected_assets)

            # 3. 賦值
            
            # 初始化當前日期的所有權重為 0
            current_weights = pd.Series(0.0, index=self.price.columns)
            
            # 將計算出的權重賦值給選定的資產
            current_weights[selected_assets] = weights
            
            # 將結果存儲到 portfolio_weights DataFrame 中
            self.portfolio_weights.loc[current_date] = current_weights

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
