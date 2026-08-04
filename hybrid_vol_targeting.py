"""
Hybrid Regime-Switching Vol-Targeting
----------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------------------------------------------------
# Load Step 1 output
# ----------------------------------------------------------------------
df = pd.read_csv("data/vol_estimates.csv", parse_dates=["Date"]).set_index("Date")

ANNUALIZATION = 252
TARGET_VOL = 0.10
MAX_LEVERAGE = 1.5
COOLDOWN_DAYS = 10
REGIME_LOOKBACK = 252     # ~1 trading year of history to define "elevated"
REGIME_PERCENTILE = 0.75  # top quartile of trailing vol = "elevated regime"

# ----------------------------------------------------------------------
# Causal regime threshold: trailing 75th percentile of Rolling vol
# (shifted so day t only sees data through t-1 - no lookahead)
# ----------------------------------------------------------------------
rolling_vol_lagged = df["vol_rolling"].shift(1)
garch_vol_lagged = df["vol_garch"].shift(1)

regime_threshold = rolling_vol_lagged.rolling(REGIME_LOOKBACK, min_periods=60).quantile(REGIME_PERCENTILE)
elevated_trigger = rolling_vol_lagged > regime_threshold  # bool series, NaN-safe (False where NaN)
elevated_trigger = elevated_trigger.fillna(False)

# ----------------------------------------------------------------------
# State machine: pick which estimator each day actually uses
# ----------------------------------------------------------------------
n = len(df)
mode = np.empty(n, dtype=object)  # "rolling" or "garch"
cooldown = 0
for i in range(n):
    if cooldown > 0:
        mode[i] = "garch"
        cooldown -= 1
    elif elevated_trigger.iloc[i]:
        mode[i] = "garch"
        cooldown = COOLDOWN_DAYS
    else:
        mode[i] = "rolling"

df["hybrid_mode"] = mode
df["hybrid_vol_used"] = np.where(mode == "garch", garch_vol_lagged, rolling_vol_lagged)

hybrid_scalar = (TARGET_VOL / df["hybrid_vol_used"]).clip(upper=MAX_LEVERAGE)
hybrid_strat_ret = hybrid_scalar * df["log_ret"]

# ----------------------------------------------------------------------
# Compare against pure Rolling, pure GARCH, and buy & hold
# ----------------------------------------------------------------------
def performance_stats(log_returns, scalars=None):
    log_returns = log_returns.dropna()
    total_days = len(log_returns)
    cagr = np.exp(log_returns.sum() * ANNUALIZATION / total_days) - 1
    ann_vol = log_returns.std() * np.sqrt(ANNUALIZATION)
    sharpe = (log_returns.mean() * ANNUALIZATION) / ann_vol if ann_vol > 0 else np.nan
    cum = log_returns.cumsum()
    drawdown = cum - cum.cummax()
    max_dd_pct = np.exp(drawdown.min()) - 1
    stats = {"CAGR": cagr, "Ann. Vol": ann_vol, "Sharpe": sharpe, "Max Drawdown": max_dd_pct}
    if scalars is not None:
        aligned = scalars.reindex(log_returns.index).dropna()
        turnover = aligned.diff().abs().sum() / (total_days / ANNUALIZATION)
        stats["Avg Position Scalar"] = aligned.mean()
        stats["Annual Turnover"] = turnover
    return stats

pure_rolling_scalar = (TARGET_VOL / rolling_vol_lagged).clip(upper=MAX_LEVERAGE)
pure_rolling_ret = pure_rolling_scalar * df["log_ret"]

pure_garch_scalar = (TARGET_VOL / garch_vol_lagged).clip(upper=MAX_LEVERAGE)
pure_garch_ret = pure_garch_scalar * df["log_ret"]

summary = pd.DataFrame({
    "Buy & Hold": performance_stats(df["log_ret"]),
    "Pure Rolling": performance_stats(pure_rolling_ret, pure_rolling_scalar),
    "Pure GARCH": performance_stats(pure_garch_ret, pure_garch_scalar),
    "Hybrid (Rolling + GARCH-on-shock)": performance_stats(hybrid_strat_ret, hybrid_scalar),
}).T

print("\n=== Hybrid vs Pure Estimators (2005-2025) ===")
print(summary.round(3))
summary.to_csv("hybrid_vol_targeting_summary.csv")

pct_days_in_garch_mode = (df["hybrid_mode"] == "garch").mean()
print(f"\nDays spent in GARCH mode: {pct_days_in_garch_mode:.1%} of the backtest")

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1, 1]})

axes[0].plot(df.index, np.exp(df["log_ret"].cumsum()), label="Buy & Hold", color="#1f2937", linewidth=1.2)
axes[0].plot(df.index, np.exp(pure_rolling_ret.cumsum()), label="Pure Rolling", color="#3b82f6", linewidth=1.0, alpha=0.85)
axes[0].plot(df.index, np.exp(pure_garch_ret.cumsum()), label="Pure GARCH", color="#b91c1c", linewidth=1.0, alpha=0.7)
axes[0].plot(df.index, np.exp(hybrid_strat_ret.cumsum()), label="Hybrid", color="#059669", linewidth=1.4)
axes[0].set_yscale("log")
axes[0].set_title("Growth of $1 — Hybrid Regime-Switching vs Pure Estimators", fontsize=12, loc="left")
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
axes[0].grid(alpha=0.3)

# Shade periods where hybrid is in GARCH mode
in_garch = df["hybrid_mode"] == "garch"
axes[1].fill_between(df.index, 0, 1, where=in_garch, color="#b91c1c", alpha=0.3,
                      transform=axes[1].get_xaxis_transform(), label="GARCH mode active")
axes[1].set_yticks([])
axes[1].set_title(f"When the Hybrid Strategy Switches to GARCH ({pct_days_in_garch_mode:.0%} of days)",
                   fontsize=12, loc="left")
axes[1].legend(loc="upper left", frameon=False, fontsize=9)

axes[2].plot(df.index, hybrid_scalar, color="#059669", linewidth=0.9)
axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
axes[2].set_title("Hybrid Position Scalar Over Time", fontsize=12, loc="left")
axes[2].set_ylabel("Leverage multiple")
axes[2].grid(alpha=0.3)
axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("hybrid_vol_targeting.png", dpi=150)
print("Saved hybrid_vol_targeting.png")
print("Saved hybrid_vol_targeting_summary.csv")