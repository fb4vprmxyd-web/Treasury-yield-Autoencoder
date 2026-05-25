"""
Rolling-window factor models for an honest out-of-sample trading test.

Addresses the central methodological gap in the static version: previously the
AE was fit once on 1993-2019 and its residuals over that same period were used
to trade, so all Pre-2020 PnL was in-sample. Here we follow the paper's design:

  For each calendar year Y (from the first year with >=5 years of history):
    - Fit the factor model on the trailing WINDOW_YEARS (default 5) of weekly data,
      ending 31 Dec of year Y-1.
    - Generate reconstructions / residuals ONLY for year Y.
  Concatenate the per-year residuals into a single genuinely-OOS residual series.

We do this for both the AE-3 and PCA-3 so the comparison stays like-for-like.
Nelson-Siegel is already estimated independently each date (cross-sectional OLS),
so it is effectively rolling already and is reused from 04_nelson_siegel outputs.

Outputs:
  results/rolling_ae3_residuals.csv
  results/rolling_pca3_residuals.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize

# --- Paths -------------------------------------------------------------------
# Resolve everything relative to this script so it runs on any machine.
# Layout assumed:  <project>/code/03b_rolling.py  and  <project>/results/*.csv
# Override OUT with the COURSEWORK_DIR environment variable if your layout differs.
import os
_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("COURSEWORK_DIR", _HERE.parent))
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)

TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y"]
WINDOW_YEARS = 5
N_RESTARTS = 10          # per refit; fewer than the static 30 because we refit ~25x


# ---- AE primitives (same math as 03_autoencoder.py) ----
def unpack(theta, n_in, h):
    i = 0
    W1 = theta[i:i+n_in*h].reshape(n_in, h); i += n_in*h
    b1 = theta[i:i+h];                       i += h
    W2 = theta[i:i+h*n_in].reshape(h, n_in); i += h*n_in
    b2 = theta[i:i+n_in]
    return W1, b1, W2, b2

def pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

def forward(X, theta, n_in, h):
    W1, b1, W2, b2 = unpack(theta, n_in, h)
    Z = np.tanh(X @ W1 + b1)
    return Z, Z @ W2 + b2

def loss_and_grad(theta, X, n_in, h):
    N = X.shape[0]
    W1, b1, W2, b2 = unpack(theta, n_in, h)
    Z = np.tanh(X @ W1 + b1)
    Xhat = Z @ W2 + b2
    diff = Xhat - X
    loss = (diff**2).mean()
    dXhat = (2.0 / (N * n_in)) * diff
    dW2 = Z.T @ dXhat
    db2 = dXhat.sum(axis=0)
    dpre = (dXhat @ W2.T) * (1.0 - Z**2)
    dW1 = X.T @ dpre
    db1 = dpre.sum(axis=0)
    return loss, pack(dW1, db1, dW2, db2)

def train_ae(X, h, seed):
    rng = np.random.default_rng(seed)
    n_in = X.shape[1]
    scale = np.sqrt(1.0 / n_in)
    theta0 = pack(rng.normal(0, scale, (n_in, h)), np.zeros(h),
                  rng.normal(0, scale, (h, n_in)), np.zeros(n_in))
    return minimize(loss_and_grad, theta0, args=(X, n_in, h), jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})


def rolling_residuals(weekly, model="ae", h=3):
    resid = pd.DataFrame(index=weekly.index, columns=TENORS, dtype=float)
    first_year = weekly.index.year.min()
    last_year  = weekly.index.year.max()

    for Y in range(first_year + WINDOW_YEARS, last_year + 1):
        tr_start = pd.Timestamp(f"{Y - WINDOW_YEARS}-01-01")
        tr_end   = pd.Timestamp(f"{Y - 1}-12-31")
        ev_start = pd.Timestamp(f"{Y}-01-01")
        ev_end   = pd.Timestamp(f"{Y}-12-31")

        Xtr_df = weekly.loc[tr_start:tr_end]
        Xev_df = weekly.loc[ev_start:ev_end]
        if len(Xtr_df) < 100 or len(Xev_df) == 0:
            continue

        # No-leakage guarantee: the training window must end strictly before the
        # evaluation window begins. If this ever fails, the "OOS" residuals are
        # contaminated and every downstream Sharpe is meaningless.
        assert Xtr_df.index.max() < Xev_df.index.min(), (
            f"LEAK in year {Y}: train ends {Xtr_df.index.max().date()} "
            f"but eval starts {Xev_df.index.min().date()}"
        )

        scaler = StandardScaler().fit(Xtr_df.values)
        Xtr = scaler.transform(Xtr_df.values)
        Xev = scaler.transform(Xev_df.values)

        if model == "ae":
            best_theta, best_mse = None, np.inf
            for s in range(N_RESTARTS):
                res = train_ae(Xtr, h, seed=s)
                if res.fun < best_mse:
                    best_mse, best_theta = res.fun, res.x
            _, Xev_hat = forward(Xev, best_theta, Xtr.shape[1], h)
        elif model == "pca":
            pca = PCA(n_components=h).fit(Xtr)
            Xev_hat = pca.inverse_transform(pca.transform(Xev))

        recon = scaler.inverse_transform(Xev_hat)
        resid.loc[Xev_df.index] = Xev_df.values - recon

    return resid.dropna(how="all")


def main():
    weekly = pd.read_csv(RES / "weekly_yields.csv", index_col="Date", parse_dates=True)[TENORS]
    print(f"Weekly obs: {len(weekly)}  {weekly.index.min().date()} -> {weekly.index.max().date()}")
    print(f"Rolling {WINDOW_YEARS}y window, refit annually, {N_RESTARTS} restarts per AE fit.\n")

    print("Rolling AE-3 ...")
    ae_resid = rolling_residuals(weekly, model="ae", h=3)
    ae_resid.to_csv(RES / "rolling_ae3_residuals.csv")
    print(f"  OOS residual rows: {len(ae_resid)}  {ae_resid.index.min().date()} -> {ae_resid.index.max().date()}")

    print("Rolling PCA-3 ...")
    pca_resid = rolling_residuals(weekly, model="pca", h=3)
    pca_resid.to_csv(RES / "rolling_pca3_residuals.csv")
    print(f"  OOS residual rows: {len(pca_resid)}")

    # quick reconstruction-quality sanity check on the OOS residuals
    for name, r in [("AE-3", ae_resid), ("PCA-3", pca_resid)]:
        rmse_bp = np.sqrt((r.values.astype(float)**2).mean()) * 100
        print(f"  {name} rolling-OOS reconstruction RMSE: {rmse_bp:.2f} bp")


if __name__ == "__main__":
    main()
