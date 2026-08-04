"""
Out-of-Sample Testing + Statistical Significance
------------------------------------------------------
"""

import pandas as pd
import numpy as np

ANNUALIZATION = 252
TARGET_VOL = 0.10
MAX_LEVERAGE = 1.5
COST_BPS = 5
BLOCK_SIZE = 21          # ~1 trading month, to preserve vol clustering
N_BOOTSTRAP = 5000
TRAIN_END = "2018-12-31"
RNG_SEED = 42

df = pd.read_csv("data/vol_estimates.csv", parse_dates=["Date"]).set_index("Date")

rolling_vol_lagged = df["vol_rolling"].shift(1)
garch_vol_lagged = df["vol_garch"].shift(1)

rolling_scalar = (TARGET_VOL / rolling_vol_lagged).clip(upper=MAX_LEVERAGE)
garch_scalar = (TARGET_VOL / garch_vol_lagged).clip(upper=MAX_LEVERAGE)

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

def net_returns(scalar, cost_bps=COST_BPS):
    scalar_change = scalar.diff().abs()
    cost = (cost_bps / 10_000) * scalar_change
    return (scalar * df["log_ret"] - cost).dropna()

strategies_net = {
    "Buy & Hold": df["log_ret"].dropna(),
    "Pure Rolling": net_returns(rolling_scalar),
    "Pure GARCH": net_returns(garch_scalar),
    "Hybrid": net_returns(hybrid_scalar),
}

def sharpe(returns):
    return (returns.mean() * ANNUALIZATION) / (returns.std() * np.sqrt(ANNUALIZATION))

def cagr(returns):
    return np.exp(returns.sum() * ANNUALIZATION / len(returns)) - 1

# ----------------------------------------------------------------------
# (A) OUT-OF-SAMPLE SPLIT
# ----------------------------------------------------------------------
print("=" * 70)
print("(A) OUT-OF-SAMPLE: TRAIN (2005-2018) vs TEST (2019-2025)")
print("=" * 70)

split_rows = []
for label, ret in strategies_net.items():
    train = ret[ret.index <= TRAIN_END]
    test = ret[ret.index > TRAIN_END]
    split_rows.append({
        "Strategy": label,
        "Train Sharpe": sharpe(train), "Train CAGR": cagr(train),
        "Test Sharpe": sharpe(test), "Test CAGR": cagr(test),
    })
split_df = pd.DataFrame(split_rows).set_index("Strategy")
print(split_df.round(3))
split_df.to_csv("out_of_sample_split.csv")

# ----------------------------------------------------------------------
# (B) BLOCK BOOTSTRAP SIGNIFICANCE TEST
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"(B) BLOCK BOOTSTRAP ({N_BOOTSTRAP} resamples, block size={BLOCK_SIZE} days)")
print("=" * 70)

rng = np.random.default_rng(RNG_SEED)

def block_bootstrap_indices(n_obs, block_size, rng):
    """Circular block bootstrap"""
    n_blocks_needed = int(np.ceil(n_obs / block_size))
    starts = rng.integers(0, n_obs, size=n_blocks_needed)
    idx = np.concatenate([np.arange(s, s + block_size) % n_obs for s in starts])
    return idx[:n_obs]

def bootstrap_sharpe_diff(ret_a, ret_b, n_boot, block_size, rng):
    """Paired block bootstrap on the Sharpe ratio difference (a - b)."""
    common_idx = ret_a.index.intersection(ret_b.index)
    a = ret_a.loc[common_idx].to_numpy()
    b = ret_b.loc[common_idx].to_numpy()
    n = len(common_idx)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = block_bootstrap_indices(n, block_size, rng)
        ra, rb = a[idx], b[idx]
        sa = (ra.mean() * ANNUALIZATION) / (ra.std() * np.sqrt(ANNUALIZATION))
        sb = (rb.mean() * ANNUALIZATION) / (rb.std() * np.sqrt(ANNUALIZATION))
        diffs[k] = sa - sb
    return diffs

comparisons = [
    ("Hybrid", "Pure Rolling"),
    ("Pure Rolling", "Buy & Hold"),
    ("Hybrid", "Buy & Hold"),
    ("Pure GARCH", "Pure Rolling"),
]

sig_rows = []
for a_label, b_label in comparisons:
    diffs = bootstrap_sharpe_diff(strategies_net[a_label], strategies_net[b_label],
                                   N_BOOTSTRAP, BLOCK_SIZE, rng)
    point_estimate = sharpe(strategies_net[a_label]) - sharpe(strategies_net[b_label])
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: how often does the bootstrap distribution cross zero,
    # doubled to reflect a two-sided test
    p_value = 2 * min((diffs <= 0).mean(), (diffs > 0).mean())
    sig_rows.append({
        "Comparison": f"{a_label} minus {b_label}",
        "Sharpe Diff (point est.)": point_estimate,
        "95% CI Low": ci_low,
        "95% CI High": ci_high,
        "p-value (approx.)": p_value,
        "Significant at 5%?": "Yes" if p_value < 0.05 else "No",
    })

sig_df = pd.DataFrame(sig_rows).set_index("Comparison")
print(sig_df.round(4).to_string())
sig_df.to_csv("bootstrap_significance.csv")

print("""
How to read this:
  - The 95% CI is the bootstrap's estimate of the range this difference could
    plausibly fall in, given sampling uncertainty in 20 years of market history.
  - If the CI does NOT include 0, the difference is unlikely to be pure noise
    at the 5% significance level (p < 0.05).
  - If the CI DOES include 0, we cannot rule out that the observed difference
    is just noise from this particular historical path.
""")