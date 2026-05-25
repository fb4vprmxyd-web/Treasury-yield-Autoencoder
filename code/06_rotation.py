"""
Subspace/rotation analysis: how does AE-3 relate to PCA-3?

The reconstruction RMSEs are nearly identical, so AE-3 and PCA-3 ought to span
essentially the same 3D subspace of yield-curve dynamics. We test this in two
ways:

(1) Principal angles between the 3D subspaces spanned by the two decoder weight
    matrices (smaller -> more overlap). Three principal angles, all close to 0
    means the subspaces are nearly identical.

(2) Regression of each AE factor on the three PCA factors. R^2 close to 1 means
    each AE factor is a near-perfect linear combination of PCA factors. The
    coefficient matrix tells us the rotation.

Outputs:
  results/subspace_angles.csv
  results/ae_in_pca_basis.csv
  figures/fig5_rotation.png     (regression coefficients heatmap + correlation panel)
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


def main():
    pca_load = pd.read_csv(RES / "pca_loadings.csv", index_col=0)        # 6 x 3
    ae_load  = pd.read_csv(RES / "ae3_loadings.csv", index_col=0)        # 6 x 3
    pca_Z    = pd.read_csv(RES / "pca_components.csv", index_col="Date", parse_dates=True)
    ae_Z     = pd.read_csv(RES / "ae3_components.csv",  index_col="Date", parse_dates=True)

    # --- (1) Principal angles between subspaces ----
    A = pca_load.values   # (6, 3)
    B = ae_load.values    # (6, 3)
    angles_rad = subspace_angles(A, B)
    angles_deg = np.degrees(angles_rad)
    pd.DataFrame({"angle_rad": angles_rad, "angle_deg": angles_deg},
                 index=[f"angle_{i+1}" for i in range(len(angles_rad))]).to_csv(RES / "subspace_angles.csv")
    print(f"Principal angles between PCA-3 and AE-3 subspaces (degrees): "
          f"{[f'{a:.2f}' for a in angles_deg]}")

    # --- (2) Express each AE factor as a linear combination of PCA factors ----
    # use the training window (both PCA and AE factor time series are aligned)
    common = pca_Z.index.intersection(ae_Z.index)
    pca_Z = pca_Z.loc[common]
    ae_Z  = ae_Z.loc[common]

    # standardise each set of factors so the regression coefficients are
    # comparable in magnitude (correlation interpretation)
    pca_Zs = (pca_Z - pca_Z.mean()) / pca_Z.std()
    ae_Zs  = (ae_Z  - ae_Z.mean())  / ae_Z.std()

    coef_rows = []
    for col in ae_Zs.columns:
        y = ae_Zs[col].values
        X = np.column_stack([np.ones_like(y), pca_Zs.values])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]            # intercept + 3 betas
        yhat = X @ beta
        r2 = 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()
        coef_rows.append({"AE_factor": col, "intercept": beta[0],
                          "PC1_level": beta[1], "PC2_slope": beta[2],
                          "PC3_curv": beta[3], "R2": r2})
    coef_df = pd.DataFrame(coef_rows).set_index("AE_factor")
    coef_df.to_csv(RES / "ae_in_pca_basis.csv")
    print("\nRegression of AE factors (standardised) on PCA factors (standardised):")
    print(coef_df.round(3))

    # Correlation matrix between AE and PCA factors (for the figure)
    corr = pd.DataFrame(index=ae_Z.columns, columns=pca_Z.columns)
    for ac in ae_Z.columns:
        for pc in pca_Z.columns:
            corr.loc[ac, pc] = ae_Z[ac].corr(pca_Z[pc])
    corr = corr.astype(float)
    print("\nCorrelation matrix AE vs PCA factors:")
    print(corr.round(3))

    # ---- Figure 5 -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=10)
    ax.set_title("(a) Correlation: AE-3 factors vs PCA-3 factors")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # principal-angles bar
    ax = axes[1]
    ax.bar(range(len(angles_deg)), angles_deg, color=["#2c7bb6","#abd9e9","#fdae61"])
    ax.set_xticks(range(len(angles_deg)))
    ax.set_xticklabels([f"angle {i+1}" for i in range(len(angles_deg))])
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title("(b) Principal angles between AE-3 and PCA-3 subspaces")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(angles_deg):
        ax.text(i, v + max(angles_deg)*0.02, f"{v:.1f}°", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG / "fig5_rotation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 5 -> {FIG / 'fig5_rotation.png'}")


if __name__ == "__main__":
    main()
