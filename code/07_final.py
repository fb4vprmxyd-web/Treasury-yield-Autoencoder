"""
Final consolidation: master metrics table + 3 polished figures for the report.

Figures:
  fig_report_1: 4-panel — yields, PCA loadings, AE loadings, NS loadings (comparison)
  fig_report_2: 2-panel — subspace correlations + principal angles (rotation finding)
  fig_report_3: 2-panel — trading cum PnL aggregate + by-maturity (trading finding)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import subspace_angles

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

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def consolidate_reconstruction_table():
    pca = pd.read_csv(RES / "pca_metrics.csv", index_col=0).assign(model="PCA-3")
    ae  = pd.read_csv(RES / "ae_metrics.csv",  index_col=0)
    ns  = pd.read_csv(RES / "ns_metrics.csv",  index_col=0).assign(model="NS")
    full = pd.concat([pca, ae, ns])

    full = full.reset_index().rename(columns={"index": "tenor"})
    # overall RMSE only
    overall = full[full["tenor"] == "ALL"][["model", "sample", "RMSE", "R2"]]
    # express RMSE in basis points for clarity
    overall["RMSE_bps"] = (overall["RMSE"] * 100).round(2)
    pivot_rmse = overall.pivot(index="model", columns="sample", values="RMSE_bps")
    pivot_rmse = pivot_rmse.reindex(["PCA-3", "AE-2", "AE-3", "AE-4", "NS"])
    pivot_rmse = pivot_rmse[["train", "oos"]]
    pivot_rmse.columns = ["Train RMSE (bp)", "OOS RMSE (bp)"]
    pivot_rmse.to_csv(RES / "table1_reconstruction.csv")
    print("Table 1: Reconstruction RMSE (basis points)")
    print(pivot_rmse.round(2))
    return pivot_rmse


def consolidate_trading_table():
    """Headline = rolling (honest OOS) results. Static kept only as a labelled
    in-sample contrast, if present."""
    # --- headline: rolling OOS ---
    roll = pd.read_csv(RES / "rolling_trading_portfolio.csv")
    samp_col = "Sample"
    pivot = roll.pivot(index="Strategy", columns=samp_col, values="Sharpe").reindex(
        ["AE-3", "PCA-3", "NS", "Trend"])
    # order columns: pre, oos, full (column labels come from 05b)
    col_order = [c for c in ["1998-2019", "2020+", "Full (98-23 OOS)"] if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.to_csv(RES / "table2_trading.csv")
    print("\nTable 2 (HEADLINE, rolling OOS): Portfolio Sharpe by strategy and sample")
    print(pivot.round(3))

    bymat = pd.read_csv(RES / "rolling_trading_bymaturity.csv", index_col=0)
    ae_only = bymat["Sharpe"].reindex(TENORS)
    ae_only.to_csv(RES / "table3_ae_bymaturity.csv")
    print("\nTable 3 (rolling OOS): AE-3 strategy Sharpe by maturity")
    print(ae_only.round(3))

    # --- contrast: static in-sample (only if the static pipeline was run) ---
    static_path = RES / "trading_summary_portfolio.csv"
    if static_path.exists():
        s = pd.read_csv(static_path)
        sp = s.pivot(index="Strategy", columns="Sample", values="Sharpe").reindex(
            ["AE-3", "PCA-3", "NS", "Trend"])
        cols = [c for c in ["Pre-2020", "OOS 2020+", "Full"] if c in sp.columns]
        sp = sp[cols]
        sp.to_csv(RES / "table2_trading_static_insample.csv")
        print("\n(Contrast) static IN-SAMPLE portfolio Sharpe — do NOT headline:")
        print(sp.round(3))
    else:
        print("\n(Static trading summary not found — skipping in-sample contrast table.)")


def make_figure_1():
    """4-panel: yields + 3 loadings."""
    weekly = pd.read_csv(RES / "weekly_yields.csv", index_col="Date", parse_dates=True)[TENORS]
    pca = pd.read_csv(RES / "pca_loadings.csv", index_col=0)
    ae  = pd.read_csv(RES / "ae3_loadings.csv", index_col=0)
    ns  = pd.read_csv(RES / "ns_loadings.csv", index_col=0)
    ns.columns = ["Level", "Slope", "Curvature"]   # rename for consistency

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # Yield curve time series spanning both columns of top row
    ax0 = fig.add_subplot(gs[0, :])
    for col, c in zip(TENORS, plt.cm.viridis(np.linspace(0, 0.9, len(TENORS)))):
        ax0.plot(weekly.index, weekly[col], lw=0.7, color=c, label=col)
    ax0.axvspan(OOS_START, weekly.index.max(), color="grey", alpha=0.15)
    ax0.set_ylabel("Yield (%)")
    ax0.set_title("(a) US Treasury yields, weekly Friday close (grey shading = OOS 2020-2023)")
    ax0.legend(ncol=6, fontsize=9, loc="upper right")

    # Three loadings panels
    for ax, df, title in [
        (fig.add_subplot(gs[1, 0]), pca, "(b) PCA-3 loadings"),
        (fig.add_subplot(gs[1, 1]), ae,  "(c) AE-3 decoder weights"),
        (fig.add_subplot(gs[1, 2]), ns,  "(d) Nelson-Siegel loadings"),
    ]:
        for col, m in zip(["Level", "Slope", "Curvature"], ["o", "s", "D"]):
            # normalise each method's loadings to unit max-abs for shape comparison
            v = df[col].values; v = v / max(abs(v).max(), 1e-9)
            ax.plot(TAU, v, marker=m, label=col)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Normalised loading")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "fig_report_1.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig_report_1.png'}")


def make_figure_2():
    """2-panel rotation analysis."""
    pca_load = pd.read_csv(RES / "pca_loadings.csv", index_col=0)
    ae_load  = pd.read_csv(RES / "ae3_loadings.csv", index_col=0)
    pca_Z    = pd.read_csv(RES / "pca_components.csv", index_col="Date", parse_dates=True)
    ae_Z     = pd.read_csv(RES / "ae3_components.csv",  index_col="Date", parse_dates=True)
    common = pca_Z.index.intersection(ae_Z.index)
    pca_Z = pca_Z.loc[common]; ae_Z = ae_Z.loc[common]

    corr = pd.DataFrame(index=ae_Z.columns, columns=pca_Z.columns)
    for ac in ae_Z.columns:
        for pc in pca_Z.columns:
            corr.loc[ac, pc] = ae_Z[ac].corr(pca_Z[pc])
    corr = corr.astype(float)

    angles_deg = np.degrees(subspace_angles(pca_load.values, ae_load.values))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["PCA Level", "PCA Slope", "PCA Curv."])
    ax.set_yticks(range(3)); ax.set_yticklabels(["AE \"Level\"", "AE \"Slope\"", "AE \"Curv.\""])
    for i in range(3):
        for j in range(3):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=11)
    ax.set_title("(a) Correlation of AE-3 factors with PCA-3 factors")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ax = axes[1]
    bars = ax.bar(range(len(angles_deg)), angles_deg, color=["#377eb8","#4daf4a","#984ea3"])
    ax.set_xticks(range(len(angles_deg)))
    ax.set_xticklabels([f"$\\theta_{{{i+1}}}$" for i in range(len(angles_deg))])
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title("(b) Principal angles between AE-3 and PCA-3 subspaces")
    ax.set_ylim(0, max(3, angles_deg.max() * 1.4))
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(angles_deg):
        ax.text(i, v + 0.06, f"{v:.1f}°", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(FIG / "fig_report_2.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'fig_report_2.png'}")


def make_figure_3():
    """Trading figure is produced by 05b_rolling_trading.py as
    fig_report_3_rolling.png (rolling OOS residuals). 07 no longer regenerates a
    static version; this avoids importing the digit-prefixed 05_trading module and
    prevents a stale in-sample figure from entering the report."""
    target = FIG / "fig_report_3_rolling.png"
    if target.exists():
        print(f"Trading figure already present (rolling): {target}")
    else:
        print("WARNING: fig_report_3_rolling.png not found — run 05b_rolling_trading.py.")


if __name__ == "__main__":
    consolidate_reconstruction_table()
    consolidate_trading_table()
    make_figure_1()
    make_figure_2()
    make_figure_3()
