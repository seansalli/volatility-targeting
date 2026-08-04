"""
Transaction Costs
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ANNUALIZATION = 252
TARGET_VOL = 0.10
MAX_LEVERAGE = 1.5
COST_LEVELS_BPS = [0, 2, 5, 10]  # basis points cost per unit of scalar change

df = pd.read_csv("data/vol_estimates.csv", parse_dates=["Date"]).set_index("Date")

rolling_vol_lagged = df["vol_rolling"].shift(1)
garch_vol_lagged = df["vol_garch"].shift(1)

rolling_scalar = (TARGET_VOL / rolling_vol_lagged).clip(upper=MAX_LEVERAGE)
garch_scalar = (TARGET_VOL / garch_vol_lagged).clip(upper=MAX_LEVERAGE)

# Hybrid scalar (same regime-switch logic as Step 3)
regime_threshold = rolling_vol_lagged.rolling(252, min_periods=60).quantile(0.75)
elevated = (rolling_vol_lagged > regime_threshold).fillna(False)
mode = np.empty(len(df), dtype=object)
cooldown = 0
for i in range(len(df)):
    if cooldown > 0:
        mode[i] = "garch"; cooldown -= 1
    elif elevated.iloc[i]:
        mode[i] = "garch"; cooldown = 10
    else:
        mode[i] = "rolling"
vol_used = pd.Series(np.where(mode == "garch", garch_vol_lagged, rolling_vol_lagged), index=df.index)
hybrid_scalar = (TARGET_VOL / vol_used).clip(upper=MAX_LEVERAGE)

scalars = {
    "Pure Rolling": rolling_scalar,
    "Pure GARCH": garch_scalar,
    "Hybrid": hybrid_scalar,
}

def performance_stats(log_returns):
    log_returns = log_returns.dropna()
    total_days = len(log_returns)
    cagr = np.exp(log_returns.sum() * ANNUALIZATION / total_days) - 1
    ann_vol = log_returns.std() * np.sqrt(ANNUALIZATION)
    sharpe = (log_returns.mean() * ANNUALIZATION) / ann_vol if ann_vol > 0 else np.nan
    cum = log_returns.cumsum()
    max_dd = np.exp((cum - cum.cummax()).min()) - 1
    return {"CAGR": cagr, "Ann. Vol": ann_vol, "Sharpe": sharpe, "Max Drawdown": max_dd}

# ----------------------------------------------------------------------
# Run every strategy at every cost level
# ----------------------------------------------------------------------
all_rows = {}
bh_stats = performance_stats(df["log_ret"])
all_rows["Buy & Hold"] = bh_stats  # no rebalancing, no cost, cost-invariant

for label, scalar in scalars.items():
    scalar_change = scalar.diff().abs()
    for cost_bps in COST_LEVELS_BPS:
        daily_cost = (cost_bps / 10_000) * scalar_change
        net_ret = scalar * df["log_ret"] - daily_cost
        row_label = f"{label} ({cost_bps}bps)"
        all_rows[row_label] = performance_stats(net_ret)

summary = pd.DataFrame(all_rows).T
print("\n=== Performance Across Transaction Cost Levels ===")
print(summary.round(3))
summary.to_csv("transaction_cost_summary.csv")

# ----------------------------------------------------------------------
# Sharpe ratio decay as costs rise, for each strategy
# ----------------------------------------------------------------------
sharpe_decay = pd.DataFrame(index=COST_LEVELS_BPS)
for label, scalar in scalars.items():
    scalar_change = scalar.diff().abs()
    sharpes = []
    for cost_bps in COST_LEVELS_BPS:
        daily_cost = (cost_bps / 10_000) * scalar_change
        net_ret = scalar * df["log_ret"] - daily_cost
        sharpes.append(performance_stats(net_ret)["Sharpe"])
    sharpe_decay[label] = sharpes

print("\n=== Sharpe Ratio vs. Cost Level ===")
print(sharpe_decay.round(3))
sharpe_decay.to_csv("sharpe_decay_by_cost.csv")

# Also compute annual turnover per strategy, for reference
print("\n=== Annual Turnover (reference) ===")
for label, scalar in scalars.items():
    turnover = scalar.diff().abs().sum() / (len(scalar.dropna()) / ANNUALIZATION)
    annual_cost_at_5bps = turnover * (5 / 10_000)
    print(f"{label:15s} turnover={turnover:.2f}x/yr  ->  ~{annual_cost_at_5bps:.2%} annual cost drag @ 5bps")

# ----------------------------------------------------------------------
# Sharpe ratio decay lines
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = {"Pure Rolling": "#3b82f6", "Pure GARCH": "#b91c1c", "Hybrid": "#059669"}
for label in scalars:
    axes[0].plot(sharpe_decay.index, sharpe_decay[label], marker="o", label=label, color=colors[label])
axes[0].axhline(bh_stats["Sharpe"], color="#1f2937", linestyle="--", linewidth=1, label="Buy & Hold (no cost)")
axes[0].set_xlabel("Transaction cost (basis points per unit of scalar change)")
axes[0].set_ylabel("Sharpe ratio")
axes[0].set_title("Sharpe Ratio Decay as Trading Costs Rise", fontsize=11, loc="left")
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(alpha=0.3)

# Equity curves at a realistic 5bps cost level
for label, scalar in scalars.items():
    scalar_change = scalar.diff().abs()
    daily_cost = (5 / 10_000) * scalar_change
    net_ret = scalar * df["log_ret"] - daily_cost
    axes[1].plot(df.index, np.exp(net_ret.cumsum()), label=f"{label} (5bps)", color=colors[label], linewidth=1.1)
axes[1].plot(df.index, np.exp(df["log_ret"].cumsum()), label="Buy & Hold", color="#1f2937", linewidth=1.2)
axes[1].set_yscale("log")
axes[1].set_title("Growth of $1 at a Realistic 5bps Cost", fontsize=11, loc="left")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
axes[1].grid(alpha=0.3)
axes[1].xaxis.set_major_locator(mdates.YearLocator(4))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("transaction_costs.png", dpi=150)
print("\nSaved transaction_costs.png")
print("Saved transaction_cost_summary.csv")
print("Saved sharpe_decay_by_cost.csv")