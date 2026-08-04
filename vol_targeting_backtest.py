"""
Vol-Targeting Overlay Backtest
--------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------------------------------------------------
# Load Step 1 output
# ----------------------------------------------------------------------
df = pd.read_csv("data/vol_estimates.csv", parse_dates=["Date"])
df = df.set_index("Date")

ANNUALIZATION = 252
TARGET_VOL = 0.10       # 10% annualized target — a common institutional choice
MAX_LEVERAGE = 1.5      # cap so we never over-lever in ultra-calm regimes
REBALANCE_LAG = 1       # use yesterday's vol estimate to size today's position

vol_cols = {
    "Rolling 20d": "vol_rolling",
    "EWMA": "vol_ewma",
    "GARCH(1,1)": "vol_garch",
}

# ----------------------------------------------------------------------
# Build each overlay
# ----------------------------------------------------------------------
results = {}
for label, col in vol_cols.items():
    vol_lagged = df[col].shift(REBALANCE_LAG)
    scalar = (TARGET_VOL / vol_lagged).clip(upper=MAX_LEVERAGE)
    strat_ret = scalar * df["log_ret"]
    results[label] = pd.DataFrame({
        "scalar": scalar,
        "strategy_log_ret": strat_ret,
    })

# Buy-and-hold benchmark
bh_log_ret = df["log_ret"]

# ----------------------------------------------------------------------
# Performance metrics
# ----------------------------------------------------------------------
def performance_stats(log_returns, scalars=None):
    log_returns = log_returns.dropna()
    total_days = len(log_returns)
    cagr = np.exp(log_returns.sum() * ANNUALIZATION / total_days) - 1
    ann_vol = log_returns.std() * np.sqrt(ANNUALIZATION)
    sharpe = (log_returns.mean() * ANNUALIZATION) / ann_vol if ann_vol > 0 else np.nan

    cum = log_returns.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max  # in log space, roughly = % drawdown for small moves
    max_dd = drawdown.min()
    max_dd_pct = np.exp(max_dd) - 1  # convert to a true percentage drawdown

    stats = {
        "CAGR": cagr,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd_pct,
    }
    if scalars is not None:
        turnover = scalars.dropna().diff().abs().sum() / (total_days / ANNUALIZATION)
        stats["Avg Position Scalar"] = scalars.mean()
        stats["Annual Turnover (abs scalar chg)"] = turnover
    return stats

summary_rows = {}
summary_rows["Buy & Hold"] = performance_stats(bh_log_ret)
for label, res in results.items():
    summary_rows[label] = performance_stats(res["strategy_log_ret"], res["scalar"])

summary = pd.DataFrame(summary_rows).T
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print("\n=== Performance Summary (2005-2025) ===")
print(summary)
summary.to_csv("vol_targeting_summary.csv")

# ----------------------------------------------------------------------
# Plot equity curves
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1, 1]})

# Equity curves (cumulative growth of $1, log scale)
axes[0].plot(df.index, np.exp(bh_log_ret.cumsum()), label="Buy & Hold", color="#1f2937", linewidth=1.3)
colors = {"Rolling 20d": "#3b82f6", "EWMA": "#f59e0b", "GARCH(1,1)": "#b91c1c"}
for label, res in results.items():
    axes[0].plot(df.index, np.exp(res["strategy_log_ret"].cumsum()),
                 label=f"{label} vol-target", color=colors[label], linewidth=1.1, alpha=0.9)
axes[0].set_yscale("log")
axes[0].set_title(f"Growth of $1 — Vol Targeting (target={TARGET_VOL:.0%}, max leverage={MAX_LEVERAGE}x) vs Buy & Hold",
                   fontsize=12, loc="left")
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
axes[0].grid(alpha=0.3)

# Position scalar over time (using GARCH as the representative example)
axes[1].plot(df.index, results["GARCH(1,1)"]["scalar"], color="#b91c1c", linewidth=0.9)
axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_title("Position Scalar Over Time (GARCH-based overlay)", fontsize=12, loc="left")
axes[1].set_ylabel("Leverage multiple")
axes[1].grid(alpha=0.3)

# Rolling drawdown comparison
bh_dd = np.exp(bh_log_ret.cumsum() - bh_log_ret.cumsum().cummax()) - 1
garch_dd = np.exp(results["GARCH(1,1)"]["strategy_log_ret"].cumsum() -
                   results["GARCH(1,1)"]["strategy_log_ret"].cumsum().cummax()) - 1
axes[2].fill_between(df.index, bh_dd * 100, 0, color="#1f2937", alpha=0.3, label="Buy & Hold")
axes[2].fill_between(df.index, garch_dd * 100, 0, color="#b91c1c", alpha=0.4, label="GARCH vol-target")
axes[2].set_title("Drawdown Comparison", fontsize=12, loc="left")
axes[2].set_ylabel("Drawdown (%)")
axes[2].legend(loc="lower left", frameon=False, fontsize=9)
axes[2].grid(alpha=0.3)
axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("vol_targeting_backtest.png", dpi=150)
print("\nSaved vol_targeting_backtest.png")
print("Saved vol_targeting_summary.csv")