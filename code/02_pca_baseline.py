"""
PCA baseline (3 components) on the standardised yield curve.

Mirrors the StandardScaler + PCA(n_components=3) pipeline from the course's
PCA_YC.ipynb so the AE-vs-PCA comparison later is apples-to-apples.

Outputs:
  results/pca_components.csv     (PC1..PC3 time series, full sample)
  results/pca_loadings.csv       (component loadings per tenor)
  results/pca_reconstruction.csv (in-sample and OOS reconstruction in yield space)
  results/pca_metrics.csv        (RMSE / R^2 by tenor and by sample)
  figures/fig2_pca_factors.png   (loadings + factor time series)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("COURSEWORK_DIR", _HERE.parent))
RES = OUT / "results"
FIG = OUT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y"]
TENOR_YEARS = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0])


def _metrics(y, yhat, label):
    """RMSE and R^2 per tenor and overall, in yield-percent space."""
    rmse = np.sqrt(((y - yhat) ** 2).mean(axis=0))
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot
    out = pd.DataFrame({"RMSE": rmse, "R2": r2}, index=TENORS)
    out["sample"] = label
    # overall row
    rmse_all = np.sqrt(((y - yhat) ** 2).mean())
    ss_res_all = ((y - yhat) ** 2).sum()
    ss_tot_all = ((y - y.mean()) ** 2).sum()
    r2_all = 1 - ss_res_all / ss_tot_all
    out.loc["ALL", :] = [rmse_all, r2_all, label]
    return out


def main():
    train = pd.read_csv(RES / "train_yields.csv", index_col="Date", parse_dates=True)[TENORS]
    oos   = pd.read_csv(RES / "oos_yields.csv",   index_col="Date", parse_dates=True)[TENORS]

    scaler = StandardScaler().fit(train.values)
    Xtr = scaler.transform(train.values)
    Xoos = scaler.transform(oos.values)

    pca = PCA(n_components=3).fit(Xtr)
    Ztr  = pca.transform(Xtr)
    Zoos = pca.transform(Xoos)

    # Sign convention for interpretability (paper labels them "level/slope/curvature"):
    #   PC1: positive loadings on all tenors  -> level    (flip if average loading < 0)
    #   PC2: monotone increasing in maturity  -> slope    (flip if loading at 20Y < at 2Y)
    #   PC3: hump in the middle               -> curvature (flip if belly loading < end loadings)
    L = pca.components_.copy()        # shape (3, n_tenors)
    if L[0].mean() < 0:                   L[0] *= -1; Ztr[:,0] *= -1; Zoos[:,0] *= -1
    if L[1, -1] < L[1, 0]:                L[1] *= -1; Ztr[:,1] *= -1; Zoos[:,1] *= -1
    if L[2, len(TENORS)//2] < L[2, 0]:    L[2] *= -1; Ztr[:,2] *= -1; Zoos[:,2] *= -1

    loadings = pd.DataFrame(L.T, index=TENORS, columns=["Level", "Slope", "Curvature"])
    loadings.to_csv(RES / "pca_loadings.csv")
    print("PCA loadings (after sign flip):")
    print(loadings.round(3))
    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_.round(4).tolist()}")
    print(f"Cumulative: {pca.explained_variance_ratio_.cumsum().round(4).tolist()}")

    # Reconstruction in yield-percent space
    Xtr_hat  = scaler.inverse_transform(pca.inverse_transform(pca.transform(Xtr)))
    Xoos_hat = scaler.inverse_transform(pca.inverse_transform(pca.transform(Xoos)))

    recon_tr  = pd.DataFrame(Xtr_hat,  index=train.index, columns=TENORS)
    recon_oos = pd.DataFrame(Xoos_hat, index=oos.index,   columns=TENORS)
    recon = pd.concat([recon_tr.assign(sample="train"), recon_oos.assign(sample="oos")])
    recon.to_csv(RES / "pca_reconstruction.csv")

    # Metrics
    m_tr  = _metrics(train.values,  Xtr_hat,  "train")
    m_oos = _metrics(oos.values,    Xoos_hat, "oos")
    metrics = pd.concat([m_tr, m_oos])
    metrics.to_csv(RES / "pca_metrics.csv")
    print("\nPCA reconstruction metrics (yield-% space):")
    print(metrics.round(4))

    # Save components time series (training + OOS), aligned
    Z = np.vstack([Ztr, Zoos])
    idx = train.index.append(oos.index)
    Zdf = pd.DataFrame(Z, index=idx, columns=["PC1_level", "PC2_slope", "PC3_curv"])
    Zdf.to_csv(RES / "pca_components.csv")

    # ---- Figure 2: PCA loadings + factor time series --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for col, marker in zip(loadings.columns, ["o", "s", "D"]):
        ax.plot(TENOR_YEARS, loadings[col].values, marker=marker, label=col)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Loading")
    ax.set_title("(a) PCA factor loadings")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    Zdf.plot(ax=ax, lw=0.7)
    ax.set_title("(b) PCA factor time series (full sample)")
    ax.axvspan(oos.index.min(), oos.index.max(), color="grey", alpha=0.15)
    ax.set_xlabel("Date"); ax.set_ylabel("Standardised factor value")
    ax.legend(ncol=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "fig2_pca_factors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 2 -> {FIG / 'fig2_pca_factors.png'}")


if __name__ == "__main__":
    main()
