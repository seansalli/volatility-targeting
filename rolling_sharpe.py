"""
Rolling Sharpe Ratio
----------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ANNUALIZATION = 252
TARGET_VOL = 0.10
MAX_LEVERAGE = 1.5
ROLLING_WINDOW = 252  # 1 trading year

df = pd.read_csv("data/vol_estimates.csv", parse_dates=["Date"]).set_index("Date")

# Rebuild the four return streams (same logic as prior steps)
rolling_vol_lagged = df["vol_rolling"].shift(1)
garch_vol_lagged = df["vol_garch"].shift(1)

bh_ret = df["log_ret"]
rolling_ret = (TARGET_VOL / rolling_vol_lagged).clip(upper=MAX_LEVERAGE) * df["log_ret"]
garch_ret = (TARGET_VOL / garch_vol_lagged).clip(upper=MAX_LEVERAGE) * df["log_ret"]

# Hybrid (same regime-switch logic as step 3)
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
vol_used = np.where(mode == "garch", garch_vol_lagged, rolling_vol_lagged)
hybrid_ret = (TARGET_VOL / pd.Series(vol_used, index=df.index)).clip(upper=MAX_LEVERAGE) * df["log_ret"]

strategies = {
    "Buy & Hold": bh_ret,
    "Pure Rolling": rolling_ret,
    "Pure GARCH": garch_ret,
    "Hybrid": hybrid_ret,
}

# ----------------------------------------------------------------------
# Rolling Sharpe: annualized mean / annualized std, over a trailing
# 252-day window, recalculated fresh for every single day
# ----------------------------------------------------------------------
rolling_sharpe = pd.DataFrame(index=df.index)
for label, ret in strategies.items():
    roll_mean = ret.rolling(ROLLING_WINDOW).mean() * ANNUALIZATION
    roll_std = ret.rolling(ROLLING_WINDOW).std() * np.sqrt(ANNUALIZATION)
    rolling_sharpe[label] = roll_mean / roll_std

rolling_sharpe.to_csv("rolling_sharpe.csv")

# Quick stats: how often is each strategy's rolling Sharpe negative?
print("\n=== Rolling 1-Year Sharpe Ratio: Summary ===")
for label in strategies:
    s = rolling_sharpe[label].dropna()
    pct_negative = (s < 0).mean()
    print(f"{label:15s}  mean={s.mean():.2f}  min={s.min():.2f}  max={s.max():.2f}  "
          f"% of days negative={pct_negative:.1%}")

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
colors = {"Buy & Hold": "#1f2937", "Pure Rolling": "#3b82f6", "Pure GARCH": "#b91c1c", "Hybrid": "#059669"}
for label in strategies:
    ax.plot(rolling_sharpe.index, rolling_sharpe[label], label=label,
            color=colors[label], linewidth=1.1, alpha=0.9)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Rolling 1-Year Sharpe Ratio — Is the Edge Consistent Over Time?", fontsize=12, loc="left")
ax.set_ylabel("Annualized Sharpe ratio (trailing 252 trading days)")
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("rolling_sharpe.png", dpi=150)
print("\nSaved rolling_sharpe.png")
print("Saved rolling_sharpe.csv")