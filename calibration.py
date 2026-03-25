#-----------------------------
# 0. GBM Tent Peg Calibrating
#-----------------------------

import yfinance as yf
import numpy as np

prices = yf.Ticker("SPY").history(period="2y")["Close"].values
daily_sigma = np.std(np.diff(np.log(prices)))
daily_drift = np.mean(np.diff(np.log(prices)))
print(f"SPY daily σ={daily_sigma:.4f}, μ={daily_drift:.4f}")

# Expected: σ≈0.0105, μ≈0.0005
# current GBM_volatility=0.20 and GBM_drift=0.05 are per-step values
# for 25 rounds, both are plausible and defensible

# After run_experiment() returns:

from main import run_experiment
result = run_experiment()

df = result["generation_counts_df"]

#-----------------------------------
# 1. Wealth premium — informed vs ZI
#-----------------------------------

df["wealth_premium_pct"] = (
    (df["mean_wealth_parameterised_informed"] - df["mean_wealth_zi"])
    / df["mean_wealth_zi"].abs() * 100
)
print(f"Final wealth premium: {df['wealth_premium_pct'].iloc[-1]:.1f}%")


#------------------------------------------
# 2. No-clear rate — market liquidity check
#------------------------------------------

print(f"Mean no-clear rate: {df['no_clear_rate'].mean()*100:.1f}%")

#------------------------------------------
# 3. Output kurtosis — endogenous fat tails
#------------------------------------------

from scipy.stats import kurtosis

# pool clearing price log-returns across all recent games
# (recent_games is returned from run_experiment)