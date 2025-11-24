# """
# Package Import
# """
# import yfinance as yf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import quantstats as qs
# import gurobipy as gp
# import warnings
# import argparse
# import sys

# """
# Project Setup
# """
# warnings.simplefilter(action="ignore", category=FutureWarning)

# assets = [
#     "SPY",
#     "XLB",
#     "XLC",
#     "XLE",
#     "XLF",
#     "XLI",
#     "XLK",
#     "XLP",
#     "XLRE",
#     "XLU",
#     "XLV",
#     "XLY",
# ]

# # Initialize Bdf and df
# Bdf = pd.DataFrame()
# for asset in assets:
#     raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
#     Bdf[asset] = raw['Adj Close']

# df = Bdf.loc["2019-01-01":"2024-04-01"]

# """
# Strategy Creation

# Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
# """


# class MyPortfolio:
#     """
#     NOTE: You can modify the initialization function
#     """

#     def __init__(self, price, exclude, lookback=252, gamma=0):
#         self.price = price
#         self.returns = price.pct_change().fillna(0)
#         self.exclude = exclude
#         self.lookback = lookback
#         self.gamma = gamma

    
# #     def calculate_weights(self):
# #         # 排除 SPY，取得可投資資產列表
# #         assets = self.price.columns[self.price.columns != self.exclude]
        
# #         # 定義策略參數 (可根據需要調整這些參數以優化結果)
# #         momentum_lookback = 120   #120
# #         volatility_lookback = 70  #70
# #         n_assets = 5 # 每次選擇前 5 個動量最強的資產

# #         # 初始化權重 DataFrame
# #         self.portfolio_weights = pd.DataFrame(
# #             index=self.price.index, columns=self.price.columns
# #         )

# #         """
# #         TODO: Complete Task 4 Below
# #         """
# # # gemini        
# #         # 逐日迭代計算權重
# #         for i in range(momentum_lookback + 1, len(self.price)):
# #             current_date = self.price.index[i]
            
# #             # === 加這一行必殺技 ===
# #             # 每 22 天 (約一個月) 才做一次動作，其他時間沿用上一天的權重
# #             # 這算是一種「頻率參數」，能大幅降低波動
# #             # if i % 365 != 0: 
# #             #     # 如果不是換倉日，直接用上一天的權重 (前提是你上一輪迴圈有存)
# #             #     # 或是乾脆都不做，讓下面 ffill 去填補
# #             #     continue 
# #             # ====================

# #             # 1. 動量篩選 (Momentum Screening)
            
# #             # 獲取計算動量所需的數據 (t - momentum_lookback 到 t - 1)
# #             # 使用價格數據計算累積回報
# #             momentum_price_window = self.price[assets].iloc[i - momentum_lookback : i]
            
# #             # 計算回顧期內的累積回報
# #             period_returns = momentum_price_window.iloc[-1] / momentum_price_window.iloc[0] - 1.0
            
# #             # 選擇動量最強的 N 個資產 (N=5)
# #             # 使用 .nlargest() 獲取表現最好的資產名稱
# #             selected_assets = period_returns.nlargest(n_assets).index.tolist()
            
# #             # 2. 風險平價分配 (Risk Parity Allocation)
            
# #             # 獲取計算波動率所需的日回報數據 (t - volatility_lookback 到 t - 1)
# #             volatility_return_window = self.returns[selected_assets].iloc[i - volatility_lookback : i]
            
# #             # 計算波動率 (日回報標準差)
# #             sigma = volatility_return_window.std()

# #             # 處理潛在的零波動率
# #             sigma[sigma == 0] = 1e9 

# #             # 計算逆波動率 (1 / sigma_i)
# #             inverse_volatility = 1.0 / sigma

# #             # 計算總逆波動率 (分母)
# #             sum_inverse_volatility = inverse_volatility.sum()
            
# #             # 計算最終的風險平價權重
# #             if sum_inverse_volatility > 0:
# #                 weights = inverse_volatility / sum_inverse_volatility
# #             else:
# #                 weights = pd.Series(0.0, index=selected_assets)

# #             # 3. 賦值
            
# #             # 初始化當前日期的所有權重為 0
# #             current_weights = pd.Series(0.0, index=self.price.columns)
            
# #             # 將計算出的權重賦值給選定的資產
# #             current_weights[selected_assets] = weights
            
# #             # 將結果存儲到 portfolio_weights DataFrame 中
# #             self.portfolio_weights.loc[current_date] = current_weights

# #         """
# #         TODO: Complete Task 4 Above
# #         """

# #         self.portfolio_weights.ffill(inplace=True)
# #         self.portfolio_weights.fillna(0, inplace=True)
    
#     def calculate_weights(self):
#         # 1. 取得除了排除項 (SPY) 以外的所有資產
#         assets = self.price.columns[self.price.columns != self.exclude]
#         n_assets = len(assets)
        
#         # 定義修正後的參數
#         # 使用 4 個月 (約 84 天) 波動率
#         VOL_LOOKBACK = 84
#         # 使用 9 個月 (約 189 天) 動量
#         MOMENTUM_LOOKBACK = 189
#         # 固定選擇前 4 名高動量資產
#         N_SELECT = 4 

#         # 2. 初始化權重 DataFrame
#         self.portfolio_weights = pd.DataFrame(
#             index=self.price.index, columns=self.price.columns
#         ).fillna(0) # 預設為 0
        
#         # 計算資產的日回報
#         returns = self.price.pct_change().fillna(0)
        
#         # 確定滾動窗口啟動所需的最小天數
#         start_day = max(VOL_LOOKBACK, MOMENTUM_LOOKBACK)

#         """
#         TODO: Complete Task 4 Below
#         """

#         # 逐日迭代計算權重
#         for i in range(len(self.price)):
#             current_date = self.price.index[i]

#             if i < start_day:  # 暖機期 (數據不足)
#                 # 使用等權重作為初始策略
#                 equal_weight = 1.0 / n_assets if n_assets > 0 else 0
#                 self.portfolio_weights.loc[current_date, assets] = equal_weight
#                 self.portfolio_weights.loc[current_date, self.exclude] = 0.0
#                 continue


#             # *** 進入滾動窗口計算 ***
#             try:
#                 # 1. 計算資產波動度（4個月回報標準差，並年化）
#                 # 使用前 i-84 到 i-1 天的數據 (t-1 資訊)
#                 vol_window = returns[assets].iloc[i - VOL_LOOKBACK : i]
#                 volatility = vol_window.std() * np.sqrt(252)
                
#                 # 處理零波動率，使用一個極小的正數替換
#                 volatility[volatility == 0] = 1e-6 

#                 # 2. 計算動量（9個月累積回報）
#                 # 使用前 i-189 到 i-1 天的價格數據 (標準的期末到期末動量)
#                 price_window = self.price[assets].iloc[i - MOMENTUM_LOOKBACK : i]
#                 # 累積回報 = (t-1 價格 / t-189 價格) - 1
#                 momentum = price_window.iloc[-1] / price_window.iloc[0] - 1.0
                
#                 # 3. 選擇前 N_SELECT=4 動量的資產
#                 selected_assets = momentum.nlargest(N_SELECT).index.tolist()
                
                
#                 # 4. 基於逆波動度分配權重 (僅針對選定資產)
#                 weights = {}
#                 total_inv_vol = 0
                
#                 for asset in selected_assets:
#                     inv_vol = 1.0 / volatility[asset]
#                     weights[asset] = inv_vol
#                     total_inv_vol += inv_vol
                
                
#                 # 5. 歸一化權重 (確保總和為 1)
#                 final_weights = {}
#                 if total_inv_vol > 0:
#                     for asset in selected_assets:
#                         final_weights[asset] = weights[asset] / total_inv_vol
                
                
#                 # 6. 分配權重到 self.portfolio_weights
#                 # 初始化當前日期的所有權重
#                 current_day_weights = pd.Series(0.0, index=self.price.columns)

#                 for asset, weight_val in final_weights.items():
#                     current_day_weights[asset] = weight_val
                
#                 self.portfolio_weights.loc[current_date] = current_day_weights
                        

#             except Exception as e:
#                 # 錯誤處理：回退到等權重
#                 # print(f"Error on {current_date}: {e}") # 可以取消註釋來調試
#                 equal_weight = 1.0 / n_assets if n_assets > 0 else 0
#                 self.portfolio_weights.loc[current_date, assets] = equal_weight
            
#             # 確保SPY權重為0
#             self.portfolio_weights.loc[current_date, self.exclude] = 0.0


#         """
#         TODO: Complete Task 4 Above
#         """

#         # 填補初期的空值 (Forward Fill) 並將其餘 NaN 設為 0
#         self.portfolio_weights.ffill(inplace=True)
#         self.portfolio_weights.fillna(0, inplace=True)


#     def calculate_portfolio_returns(self):
#         # Ensure weights are calculated
#         if not hasattr(self, "portfolio_weights"):
#             self.calculate_weights()

#         # Calculate the portfolio returns
#         self.portfolio_returns = self.returns.copy()
#         assets = self.price.columns[self.price.columns != self.exclude]
#         self.portfolio_returns["Portfolio"] = (
#             self.portfolio_returns[assets]
#             .mul(self.portfolio_weights[assets])
#             .sum(axis=1)
#         )

#     def get_results(self):
#         # Ensure portfolio returns are calculated
#         if not hasattr(self, "portfolio_returns"):
#             self.calculate_portfolio_returns()

#         return self.portfolio_weights, self.portfolio_returns


# if __name__ == "__main__":
#     # Import grading system (protected file in GitHub Classroom)
#     from grader_2 import AssignmentJudge
    
#     parser = argparse.ArgumentParser(
#         description="Introduction to Fintech Assignment 3 Part 12"
#     )

#     parser.add_argument(
#         "--score",
#         action="append",
#         help="Score for assignment",
#     )

#     parser.add_argument(
#         "--allocation",
#         action="append",
#         help="Allocation for asset",
#     )

#     parser.add_argument(
#         "--performance",
#         action="append",
#         help="Performance for portfolio",
#     )

#     parser.add_argument(
#         "--report", action="append", help="Report for evaluation metric"
#     )

#     parser.add_argument(
#         "--cumulative", action="append", help="Cumulative product result"
#     )

#     args = parser.parse_args()

#     judge = AssignmentJudge()
    
#     # All grading logic is protected in grader_2.py
#     judge.run_grading(args)


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
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust=False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    Enhanced Momentum with Volatility Targeting Strategy
    Combines momentum ranking with volatility-adjusted weighting
    """
    def __init__(self, price, exclude, momentum_window=126, vol_window=63):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.momentum_window = momentum_window
        self.vol_window = vol_window

    def calculate_weights(self):
        # Get assets excluding SPY
        assets = self.price.columns[self.price.columns != self.exclude]

        # Initialize weights DataFrame
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )
        self.portfolio_weights.fillna(0, inplace=True)

        # Strategy implementation
        for i in range(len(self.price)):
            current_date = self.price.index[i]
            
            # Initial period: equal weighting for first 3 months
            if i < 63:
                equal_weight = 1.0 / len(assets) if len(assets) > 0 else 0
                for asset in assets:
                    self.portfolio_weights.loc[current_date, asset] = equal_weight
                self.portfolio_weights.loc[current_date, self.exclude] = 0.0
                continue
            
            try:
                # Calculate rolling volatility (3 months)
                vol_data = self.returns[assets].iloc[i-63:i]
                annualized_vol = vol_data.std() * np.sqrt(252)
                # Handle zero volatility cases
                annualized_vol = annualized_vol.replace(0, 0.001)
                
                # Calculate momentum (6-month total return)
                price_today = self.price[assets].iloc[i]
                price_6m_ago = self.price[assets].iloc[i-126]
                momentum_scores = (price_today / price_6m_ago - 1)
                
                # Select top half assets by momentum
                momentum_ranking = momentum_scores.rank(ascending=False)
                selected = momentum_ranking[momentum_ranking <= len(assets) * 0.5].index
                
                if len(selected) == 0:
                    selected = assets
                
                # Calculate inverse volatility weights for selected assets
                weight_dict = {}
                total_ivol = 0
                
                for asset in selected:
                    inverse_vol = 1.0 / annualized_vol[asset]
                    weight_dict[asset] = inverse_vol
                    total_ivol += inverse_vol
                
                # Normalize weights
                for asset in selected:
                    weight_dict[asset] = weight_dict[asset] / total_ivol
                
                # Assign weights to portfolio
                for asset in assets:
                    if asset in weight_dict:
                        self.portfolio_weights.loc[current_date, asset] = weight_dict[asset]
                    else:
                        self.portfolio_weights.loc[current_date, asset] = 0.0
                        
            except Exception as e:
                # Fallback to equal weight on error
                equal_weight = 1.0 / len(assets) if len(assets) > 0 else 0
                for asset in assets:
                    self.portfolio_weights.loc[current_date, asset] = equal_weight
            
            # Ensure SPY has zero weight
            self.portfolio_weights.loc[current_date, self.exclude] = 0.0

        # Handle missing values
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
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
    judge.run_grading(args)