"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
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

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust = False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 1 Below
        """
        # 1. 計算可投資資產的數量 m
        m = len(assets)

        # 2. 計算等權重
        if m > 0:
            equal_weight = 1.0 / m
        
        # 3. 將權重分配給所有可投資資產
            self.portfolio_weights.loc[:, assets] = equal_weight
        

        """
        TODO: Complete Task 1 Above
        """
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        m = len(assets) # 可投資資產的數量
        
        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 2 Below
        """

        # 1. 逐日計算權重，從 lookback 窗口結束後的下一天開始
        for i in range(self.lookback + 1, len(df)):
            # 獲取計算當前權重所需的前 lookback 天的回報數據 (t-lookback 到 t-1)
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]

            # 計算資產在該窗口內的波動率 (日回報標準差)
            # R_n.std() 是一個包含 m 個資產波動率的 Pandas Series
            sigma = R_n.std()

            # 處理潛在的零波動率 (避免除以零)
            # 將波動率為零的資產設為一個非常大的值，使其權重接近零
            sigma[sigma == 0] = 1e9 

            # 2. 計算逆波動率 (1 / sigma_i)
            inverse_volatility = 1.0 / sigma

            # 3. 計算總逆波動率 (分母)
            sum_inverse_volatility = inverse_volatility.sum()
            
            # 4. 計算最終的風險平價權重 (w_i)
            weights = inverse_volatility / sum_inverse_volatility

            # 5. 將計算出的權重賦值給當前日期
            # df.index[i] 是當前的再平衡日期 t
            # weights 已經是一個 Series，可以直接賦值給對應的資產列
            self.portfolio_weights.loc[df.index[i], assets] = weights.values


        """
        TODO: Complete Task 2 Above
        """

        # 確保任何遺漏的日期（例如前 lookback 天）或非投資資產（如 SPY）的權重為 0
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)



    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below
                """

                # 1. 初始化決策變數 (Portfolio Weights w)
                # w 是 n 個資產的權重向量。
                # Long-only 約束 (w_i >= 0) 可以通過設置 lower bound (lb=0) 實現。
                # 初始代碼中的 ub=1 也是正確的，因為權重上限是 1。
                w = model.addMVar(n, name="w", lb=0, ub=1)

                # 2. 定義目標函式 (Objective Function)
                # Max: w^T * mu - (gamma/2) * w^T * Sigma * w
                # 這是最大化 (期望回報 - 0.5 * gamma * 方差/風險)
                
                # 線性部分 (w^T * mu): 期望回報
                return_term = mu @ w
                
                # 二次部分 (w^T * Sigma * w): 投資組合方差/風險
                risk_term = w @ Sigma @ w
                
                # 設置目標函式：最大化回報並懲罰風險
                model.setObjective(return_term - (gamma / 2) * risk_term, gp.GRB.MAXIMIZE)


                # 3. 添加約束條件 (Constraints)
                # No-leverage 約束: Sum(w_i) = 1
                model.addConstr(w.sum() == 1, name="sum_to_one")

                """
                TODO: Complete Task 3 Above
                """
                model.optimize()

                # Check if the status is INF_OR_UNBD (code 4)
                if model.status == gp.GRB.INF_OR_UNBD:
                    print(
                        "Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0."
                    )
                elif model.status == gp.GRB.INFEASIBLE:
                    # Handle infeasible model
                    print("Model is infeasible.")
                elif model.status == gp.GRB.INF_OR_UNBD:
                    # Handle infeasible or unbounded model
                    print("Model is infeasible or unbounded.")

                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    # Extract the solution
                    solution = []
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        # print(f"w {i} = {var.X}")
                        solution.append(var.X)

        return solution

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
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

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)
