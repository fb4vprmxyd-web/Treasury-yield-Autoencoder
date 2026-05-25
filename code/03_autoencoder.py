"""
Autoencoder replication of Suimon et al. (2020), applied to US Treasury yields.

Model (paper equation 2):
    Z    = tanh(X W1 + b1)       # encoder, single hidden layer, tanh
    Xhat = Z W2 + b2              # decoder, linear (no activation)
    Loss = mean( (Xhat - X)^2 )

Parameters for h=3: 6*3 + 3 + 3*6 + 6 = 45 floats. Trained with L-BFGS-B
using analytical gradients. Multiple random restarts to avoid local minima
(the paper notes weights are re-estimated from many random starts).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

import os
_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("COURSEWORK_DIR", _HERE.parent))
RES = OUT / "results"
FIG = OUT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y"]
TENOR_YEARS = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0])
N_RESTARTS = 30


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
    Z    = np.tanh(X @ W1 + b1)
    Xhat = Z @ W2 + b2
    return Z, Xhat

def loss_and_grad(theta, X, n_in, h):
    N = X.shape[0]
    W1, b1, W2, b2 = unpack(theta, n_in, h)
    Z    = np.tanh(X @ W1 + b1)
    Xhat = Z @ W2 + b2
    diff = Xhat - X
    loss = (diff**2).mean()

    dXhat = (2.0 / (N * n_in)) * diff
    dW2   = Z.T @ dXhat
    db2   = dXhat.sum(axis=0)
    dZ    = dXhat @ W2.T
    dpre  = dZ * (1.0 - Z**2)
    dW1   = X.T @ dpre
    db1   = dpre.sum(axis=0)
    return loss, pack(dW1, db1, dW2, db2)


def train_ae(X, h, seed):
    rng = np.random.default_rng(seed)
    n_in = X.shape[1]
    scale = np.sqrt(1.0 / n_in)
    theta0 = pack(rng.normal(0, scale, (n_in, h)),
                  np.zeros(h),
                  rng.normal(0, scale, (h, n_in)),
                  np.zeros(n_in))
    res = minimize(loss_and_grad, theta0, args=(X, n_in, h),
                   jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
    return res


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

    # last 15% of training window held out for validation (restart selection)
    n_val = int(0.15 * len(train))
    train_in  = train.iloc[:-n_val]
    train_val = train.iloc[-n_val:]

    scaler = StandardScaler().fit(train_in.values)
    Xtr_in   = scaler.transform(train_in.values)
    Xtr_val  = scaler.transform(train_val.values)
    Xtr_full = scaler.transform(train.values)
    Xoos     = scaler.transform(oos.values)

    all_metrics, best = [], {}

    for h in [2, 3, 4]:
        print(f"\n=== AE  h={h}  ({N_RESTARTS} restarts) ===")
        best_fit = {"val_mse": np.inf}
        for s in range(N_RESTARTS):
            res = train_ae(Xtr_in, h, seed=s)
            _, Xhat_val = forward(Xtr_val, res.x, Xtr_val.shape[1], h)
            val_mse = ((Xhat_val - Xtr_val)**2).mean()
            if val_mse < best_fit["val_mse"]:
                best_fit = {"theta": res.x.copy(), "tr_mse": res.fun,
                            "val_mse": val_mse, "seed": s}
        print(f"  best seed={best_fit['seed']}  train MSE={best_fit['tr_mse']:.5f}  val MSE={best_fit['val_mse']:.5f}")

        # refit on full training (in + val) starting from the best init
        res_full = minimize(loss_and_grad, best_fit["theta"],
                            args=(Xtr_full, Xtr_full.shape[1], h),
                            jac=True, method="L-BFGS-B",
                            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        best_fit["theta_full"] = res_full.x
        best[h] = best_fit

        _, Xhat_tr  = forward(Xtr_full, res_full.x, Xtr_full.shape[1], h)
        _, Xhat_oos = forward(Xoos,     res_full.x, Xtr_full.shape[1], h)
        all_metrics.append(metrics(train.values, scaler.inverse_transform(Xhat_tr),  "train", f"AE-{h}"))
        all_metrics.append(metrics(oos.values,   scaler.inverse_transform(Xhat_oos), "oos",   f"AE-{h}"))

    metrics_df = pd.concat(all_metrics)
    metrics_df.to_csv(RES / "ae_metrics.csv")
    print("\nAE reconstruction metrics (yield-% space):")
    print(metrics_df.round(4))

    # ---- 3-node headline model ----
    h = 3
    theta = best[h]["theta_full"]
    W1, b1, W2, b2 = unpack(theta, 6, h)
    loadings = pd.DataFrame(W2.T, index=TENORS, columns=["F1", "F2", "F3"])

    def classify(col):
        v = col.values
        flat = 1 - (v.std() / (abs(v.mean()) + 1e-9))
        mono = abs(np.corrcoef(v, TENOR_YEARS)[0, 1])
        interior = v[1:-1]; ends = (v[0] + v[-1]) / 2
        hump = (interior.max() - ends) / (abs(v).max() + 1e-9)
        return flat, mono, hump

    sc = {c: classify(loadings[c]) for c in loadings.columns}
    flat_c  = max(sc, key=lambda c: sc[c][0])
    rest    = [c for c in loadings.columns if c != flat_c]
    mono_c  = max(rest, key=lambda c: sc[c][1])
    hump_c  = [c for c in rest if c != mono_c][0]
    rename = {flat_c: "Level", mono_c: "Slope", hump_c: "Curvature"}
    loadings = loadings.rename(columns=rename)[["Level", "Slope", "Curvature"]]

    if loadings["Level"].mean() < 0: loadings["Level"] *= -1
    if loadings["Slope"].iloc[-1] < loadings["Slope"].iloc[0]: loadings["Slope"] *= -1
    if loadings["Curvature"].iloc[len(TENORS)//2] < loadings["Curvature"].iloc[0]:
        loadings["Curvature"] *= -1
    loadings.to_csv(RES / "ae3_loadings.csv")
    print("\nAE-3 decoder loadings:")
    print(loadings.round(3))

    Z_tr,  _ = forward(Xtr_full, theta, 6, h)
    Z_oos, _ = forward(Xoos,     theta, 6, h)
    Zdf = pd.DataFrame(np.vstack([Z_tr, Z_oos]),
                       index=train.index.append(oos.index),
                       columns=["F1", "F2", "F3"]).rename(columns=rename)
    Zdf = Zdf[["Level", "Slope", "Curvature"]]

    # match signs of latent factors to the loadings interpretation
    pca_load = pd.read_csv(RES / "pca_loadings.csv", index_col=0)
    for col in ["Level", "Slope", "Curvature"]:
        if (loadings[col].values @ pca_load[col].values) < 0:
            Zdf[col]      *= -1
            loadings[col] *= -1
    Zdf.to_csv(RES / "ae3_components.csv")

    _, Xhat_tr  = forward(Xtr_full, theta, 6, h)
    _, Xhat_oos = forward(Xoos,     theta, 6, h)
    recon_tr  = pd.DataFrame(scaler.inverse_transform(Xhat_tr),  index=train.index, columns=TENORS)
    recon_oos = pd.DataFrame(scaler.inverse_transform(Xhat_oos), index=oos.index,   columns=TENORS)
    pd.concat([recon_tr.assign(sample="train"), recon_oos.assign(sample="oos")]) \
      .to_csv(RES / "ae3_reconstruction.csv")

    # ---- Figure 3 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for col, marker in zip(loadings.columns, ["o", "s", "D"]):
        ax.plot(TENOR_YEARS, loadings[col].values, marker=marker, label=col)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Decoder weight")
    ax.set_title("(a) AE-3 decoder weights")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    Zdf.plot(ax=ax, lw=0.7)
    ax.axvspan(oos.index.min(), oos.index.max(), color="grey", alpha=0.15)
    ax.set_title("(b) AE-3 latent factors")
    ax.set_xlabel("Date"); ax.set_ylabel("Hidden-unit activation")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ae_factors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 3 -> {FIG / 'fig3_ae_factors.png'}")


if __name__ == "__main__":
    main()
