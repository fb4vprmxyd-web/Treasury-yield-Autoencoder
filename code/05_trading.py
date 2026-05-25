"""
Long-short trading strategy on yield-curve mispricing.

For each bond (maturity tau) and each date t:
  - Reconstruct yield y_hat_t from the factor model (AE, PCA, NS).
  - Residual e_t = y_t - y_hat_t.
    e_t > 0: yield is above the model -> bond is "cheap" -> expect mean reversion
             (yield falls, price rises). Take LONG position.
    e_t < 0: yield is below the model -> bond is "rich" -> expect mean reversion
             (yield rises, price falls). Take SHORT position.
  - PnL from t to t+1 (one week) approximated by   -tau * (y_{t+1} - y_t) * position_t
    where -tau approximates negative modified duration for a small Delta y.

Strategies compared:
  - "AE-3"        : positions from AE-3 residuals
  - "PCA-3"       : positions from PCA-3 residuals
  - "NS"          : positions from Nelson-Siegel residuals
  - "TrendFollow" : position = sign of 4-week trailing yield change (long when
                    yields have fallen recently)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import os
_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("COURSEWORK_DIR", _HERE.parent))
RES = OUT / "results"
FIG = OUT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y"]
TAU = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0])

OOS_START = pd.Timestamp("2020-01-01")


def load_all():
    """Load yields + each model's reconstruction."""
    full = pd.read_csv(RES / "weekly_yields.csv", index_col="Date", parse_dates=True)[TENORS]
    ae   = pd.read_csv(RES / "ae3_reconstruction.csv", index_col="Date", parse_dates=True)[TENORS]
    pca  = pd.read_csv(RES / "pca_reconstruction.csv", index_col="Date", parse_dates=True)[TENORS]
    ns   = pd.read_csv(RES / "ns_reconstruction.csv",  index_col="Date", parse_dates=True)[TENORS]
    # align
    common = full.index.intersection(ae.index).intersection(pca.index).intersection(ns.index)
    return full.loc[common], ae.loc[common], pca.loc[common], ns.loc[common]


def positions_from_residual(yields, recon):
    e = yields - recon
    return np.sign(e)            # +1 long, -1 short, 0 if equal


def positions_trend_follow(yields, lookback=4):
    # +1 long bond when yields fell over the lookback (price rose);
    # -1 short when yields rose.
    return -np.sign(yields - yields.shift(lookback))


def backtest(yields, positions, label):
    """One-week holding, returns ~ -tau * Delta y * position_(t-1).
    The position used to realise the t->t+1 return is set at time t (lagged once).
    """
    dy = yields.diff()                                  # yield change, %
    pos_lag = positions.shift(1)                        # use last week's position
    # PnL by bond per week, in % terms (treat 1% yield = 1 unit)
    pnl_each = -TAU * dy * pos_lag                      # broadcasting on columns
    pnl_each = pnl_each.dropna(how="any")
    cumret_each = pnl_each.cumsum()
    pnl_port = pnl_each.mean(axis=1)                    # equal-weighted across bonds
    cumret_port = pnl_port.cumsum()
    return pnl_each, cumret_each, pnl_port, cumret_port


def stats(pnl, ann_factor=52):
    mu  = pnl.mean()  * ann_factor
    sd  = pnl.std()   * np.sqrt(ann_factor)
    sr  = mu / sd if sd > 0 else np.nan
    hit = (pnl > 0).mean()
    dd  = (pnl.cumsum() - pnl.cumsum().cummax()).min()
    return pd.Series({"AnnRet (%-yr)": mu, "AnnVol": sd, "Sharpe": sr,
                      "Hit": hit, "MaxDD": dd})


def main():
    full, ae, pca, ns = load_all()

    strategies = {
        "AE-3"  : positions_from_residual(full, ae),
        "PCA-3" : positions_from_residual(full, pca),
        "NS"    : positions_from_residual(full, ns),
        "Trend" : positions_trend_follow(full, lookback=4),
    }

    results = {}
    for name, pos in strategies.items():
        pnl_each, cum_each, pnl_port, cum_port = backtest(full, pos, name)
        results[name] = {"pnl_each": pnl_each, "cum_each": cum_each,
                         "pnl_port": pnl_port, "cum_port": cum_port}

    # ---- Summary table: portfolio level, full sample / pre-2020 / 2020+ -------
    rows = []
    for name, r in results.items():
        for label, mask in [("Full", r["pnl_port"].index),
                            ("Pre-2020", r["pnl_port"].index < OOS_START),
                            ("OOS 2020+", r["pnl_port"].index >= OOS_START)]:
            s = stats(r["pnl_port"].loc[mask])
            s.name = (name, label)
            rows.append(s)
    summary_port = pd.concat(rows, axis=1).T
    summary_port.index = pd.MultiIndex.from_tuples(summary_port.index, names=["Strategy", "Sample"])
    summary_port.to_csv(RES / "trading_summary_portfolio.csv")
    print("Portfolio-level strategy stats (equal-weight across 6 maturities):")
    print(summary_port.round(3).to_string())

    # ---- Summary table: by maturity, full sample only ------------------------
    rows = []
    for name, r in results.items():
        for tenor in TENORS:
            s = stats(r["pnl_each"][tenor])
            s.name = (name, tenor)
            rows.append(s)
    summary_bond = pd.concat(rows, axis=1).T
    summary_bond.index = pd.MultiIndex.from_tuples(summary_bond.index, names=["Strategy","Bond"])
    summary_bond.to_csv(RES / "trading_summary_bymaturity.csv")
    print("\nBy-maturity Sharpe ratios (full sample):")
    print(summary_bond["Sharpe"].unstack("Bond").round(2).to_string())

    # ---- Figure 4: Cumulative PnL ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for name, r in results.items():
        ax.plot(r["cum_port"].index, r["cum_port"].values, lw=1, label=name)
    ax.axvspan(OOS_START, full.index.max(), color="grey", alpha=0.15)
    ax.set_title("(a) Cumulative PnL, equal-weight across 6 bonds")
    ax.set_ylabel("Cumulative return (duration-weighted, %)")
    ax.legend(); ax.grid(alpha=0.3)

    # Per-bond cumulative for the AE strategy (most directly mirrors the paper)
    ax = axes[1]
    for tenor, c in zip(TENORS, plt.cm.viridis(np.linspace(0,0.9,len(TENORS)))):
        ax.plot(results["AE-3"]["cum_each"].index,
                results["AE-3"]["cum_each"][tenor].values,
                lw=0.9, color=c, label=tenor)
    ax.axvspan(OOS_START, full.index.max(), color="grey", alpha=0.15)
    ax.set_title("(b) AE-3 strategy by bond maturity")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(ncol=3, fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "fig4_trading.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 4 -> {FIG / 'fig4_trading.png'}")


if __name__ == "__main__":
    main()
