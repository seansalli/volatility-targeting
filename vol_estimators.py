"""
Volatility Estimator Comparison
--------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model

# ----------------------------------------------------------------------
# Load & prep data
# ----------------------------------------------------------------------
DATA_PATH = "data/us_market_data.csv"
ANNUALIZATION = 252  # trading days per year

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df[["Date", "Adjusted Close"]].rename(columns={"Adjusted Close": "price"})
df = df.sort_values("Date").reset_index(drop=True)

# Restrict to a window that's rich in different vol regimes:
# dot-com bust, 2008 GFC, 2018 vol spike, COVID crash, 2022 rate-hike selloff
start_date = "2005-01-01"
df = df[df["Date"] >= start_date].reset_index(drop=True)

df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
df = df.dropna(subset=["log_ret"]).reset_index(drop=True)

print(f"Loaded {len(df):,} daily observations from {df['Date'].min().date()} to {df['Date'].max().date()}")

# ----------------------------------------------------------------------
# Estimator 1: Rolling realized volatility
# ----------------------------------------------------------------------
ROLL_WINDOW = 20  # ~1 trading month
df["vol_rolling"] = df["log_ret"].rolling(ROLL_WINDOW).std() * np.sqrt(ANNUALIZATION)

# ----------------------------------------------------------------------
# Estimator 2: EWMA volatility (RiskMetrics)
# ----------------------------------------------------------------------
LAMBDA = 0.94  # RiskMetrics standard daily decay factor
ewma_var = np.zeros(len(df))
ewma_var[0] = df["log_ret"].iloc[:ROLL_WINDOW].var()  # seed with a short sample var
for t in range(1, len(df)):
    ewma_var[t] = LAMBDA * ewma_var[t - 1] + (1 - LAMBDA) * df["log_ret"].iloc[t - 1] ** 2
df["vol_ewma"] = np.sqrt(ewma_var) * np.sqrt(ANNUALIZATION)

# ----------------------------------------------------------------------
# Estimator 3: GARCH(1,1)
# ----------------------------------------------------------------------
# arch_model expects returns in percentage points for numerical stability
returns_pct = df["log_ret"] * 100
garch = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal", mean="Constant")
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

# in-sample conditional volatility, annualized, converted back from % to decimal
df["vol_garch"] = garch_fit.conditional_volatility / 100 * np.sqrt(ANNUALIZATION)

# ----------------------------------------------------------------------
# Save processed dataset for reuse in step 2 (the vol-targeting overlay)
# ----------------------------------------------------------------------
out_cols = ["Date", "price", "log_ret", "vol_rolling", "vol_ewma", "vol_garch"]
df[out_cols].to_csv("data/vol_estimates.csv", index=False)
print("Saved data/vol_estimates.csv")

# ----------------------------------------------------------------------
# Plot comparison
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                          gridspec_kw={"height_ratios": [1, 2]})

axes[0].plot(df["Date"], df["price"], color="#1f2937", linewidth=1)
axes[0].set_yscale("log")
axes[0].set_title("S&P 500 / SPY Total Return Index (log scale)", fontsize=12, loc="left")
axes[0].grid(alpha=0.3)

axes[1].plot(df["Date"], df["vol_rolling"] * 100, label="Rolling 20d std", linewidth=1, alpha=0.8)
axes[1].plot(df["Date"], df["vol_ewma"] * 100, label="EWMA (λ=0.94)", linewidth=1, alpha=0.85)
axes[1].plot(df["Date"], df["vol_garch"] * 100, label="GARCH(1,1)", linewidth=1.3, color="#b91c1c")
axes[1].set_title("Annualized Volatility Estimates", fontsize=12, loc="left")
axes[1].set_ylabel("Annualized vol (%)")
axes[1].legend(loc="upper left", frameon=False)
axes[1].grid(alpha=0.3)
axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("vol_estimator_comparison.png", dpi=150)
print("Saved vol_estimator_comparison.png")

# ----------------------------------------------------------------------
# Quick numerical comparison: how differently do they react to shocks?
# ----------------------------------------------------------------------
summary = df[["vol_rolling", "vol_ewma", "vol_garch"]].describe().T
summary["reacts_fastest"] = None
print("\nSummary stats (annualized vol, decimal form):")
print(summary)

# Peak lag check: on the single worst return day (COVID crash), how many
# days did each estimator take to reach its post-shock peak?
crash_idx = df["log_ret"].idxmin()
crash_date = df.loc[crash_idx, "Date"]
window = df[(df["Date"] >= crash_date - pd.Timedelta(days=5)) &
            (df["Date"] <= crash_date + pd.Timedelta(days=30))]
print(f"\nWorst single-day return: {df.loc[crash_idx, 'log_ret']:.2%} on {crash_date.date()}")
for col in ["vol_rolling", "vol_ewma", "vol_garch"]:
    peak_row = window.loc[window[col].idxmax()]
    days_to_peak = (peak_row["Date"] - crash_date).days
    print(f"  {col}: peaked at {peak_row[col]:.1%} annualized, {days_to_peak} days after the shock")
