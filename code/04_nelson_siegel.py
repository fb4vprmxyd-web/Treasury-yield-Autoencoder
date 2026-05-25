"""
Nelson-Siegel (1987) three-factor parametric yield curve.

For each date t, the cross-sectional yield curve is modelled as

    y(tau) = beta0 + beta1 * (1 - exp(-lam*tau)) / (lam*tau)
                   + beta2 * [(1 - exp(-lam*tau)) / (lam*tau) - exp(-lam*tau)]

where tau is maturity in years and lam is a fixed decay RATE (units 1/year)
that controls where the curvature factor peaks. Diebold and Li (2006) use
lam = 0.0609 with tau measured in MONTHS, which places the curvature-loading
peak at tau ~ 30 months. With tau in years we rescale lam by 12:
    lam = 12 * 0.0609 = 0.7308 per year   (curvature peak still at ~2.5 years).
We estimate the three betas by OLS at each date (the standard "two-step"
Diebold-Li implementation: fix lam, then OLS the betas date by date).

beta0 = long-run level, beta1 = -(slope), beta2 = curvature.
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
LAM = 0.7308          # = 12 * 0.0609 (Diebold-Li monthly lambda rescaled to 1/year);
                       # We use LAM in (1/yr) units consistent with TAU in years.


def ns_factor_matrix(tau, lam):
    """Build (n_tenors, 3) factor-loading matrix for Nelson-Siegel."""
    f1 = np.ones_like(tau)
    f2 = (1 - np.exp(-lam * tau)) / (lam * tau)
    f3 = f2 - np.exp(-lam * tau)
    return np.column_stack([f1, f2, f3])


def fit_ns(yields, lam=LAM):
    """OLS fit per row: yields is (T, n_tenors). Returns betas (T, 3) and fit (T, n_tenors)."""
    F = ns_factor_matrix(TAU, lam)                  # (n_tenors, 3)
    # solve y_t = F beta_t  =>  beta_t = (F'F)^-1 F' y_t.  Vectorise across t.
    pinv = np.linalg.pinv(F)                        # (3, n_tenors)
    betas = yields.values @ pinv.T                  # (T, 3)
    fit   = betas @ F.T                             # (T, n_tenors)
    return betas, fit


def metrics(y, yhat, label, model):
    rmse = np.sqrt(((y - yhat) ** 2).mean(axis=0))
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot
    out = pd.DataFrame({"RMSE": rmse, "R2": r2}, index=TENORS)
    out["sample"] = label; out["model"] = model
    rmse_all = np.sqrt(((y - yhat) ** 2).mean())
    ss_res_all = ((y - yhat) ** 2).sum(); ss_tot_all = ((y - y.mean()) ** 2).sum()
    out.loc["ALL", :] = [rmse_all, 1 - ss_res_all / ss_tot_all, label, model]
    return out


def main():
    train = pd.read_csv(RES / "train_yields.csv", index_col="Date", parse_dates=True)[TENORS]
    oos   = pd.read_csv(RES / "oos_yields.csv",   index_col="Date", parse_dates=True)[TENORS]

    F = ns_factor_matrix(TAU, LAM)
    loadings = pd.DataFrame(F, index=TENORS,
                            columns=["Level (beta0)", "Slope (beta1)", "Curvature (beta2)"])
    loadings.to_csv(RES / "ns_loadings.csv")
    print(f"Nelson-Siegel loadings (lambda = {LAM}):")
    print(loadings.round(3))

    betas_tr,  fit_tr  = fit_ns(train)
    betas_oos, fit_oos = fit_ns(oos)
    fit_tr_df  = pd.DataFrame(fit_tr,  index=train.index, columns=TENORS)
    fit_oos_df = pd.DataFrame(fit_oos, index=oos.index,   columns=TENORS)
    pd.concat([fit_tr_df.assign(sample="train"), fit_oos_df.assign(sample="oos")]) \
      .to_csv(RES / "ns_reconstruction.csv")

    m_tr  = metrics(train.values, fit_tr,  "train", "NS")
    m_oos = metrics(oos.values,   fit_oos, "oos",   "NS")
    metrics_df = pd.concat([m_tr, m_oos])
    metrics_df.to_csv(RES / "ns_metrics.csv")
    print("\nNelson-Siegel reconstruction metrics (yield-% space):")
    print(metrics_df.round(4))

    # Save betas as time series for comparison with PCA / AE factors
    betas_full = pd.DataFrame(np.vstack([betas_tr, betas_oos]),
                              index=train.index.append(oos.index),
                              columns=["NS_level", "NS_slope", "NS_curv"])
    betas_full.to_csv(RES / "ns_betas.csv")


if __name__ == "__main__":
    main()
