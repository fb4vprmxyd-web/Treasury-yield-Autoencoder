# Latent Factors in the US Treasury Yield Curve via an Autoencoder

Replication and critical extension of Suimon, Watanabe & Sakaji (2020), *Discovering
Latent Factors in the Japanese Government Bond Market Using an Autoencoder*, applied
to US Treasury data.

A single-hidden-layer autoencoder (tanh encoder, linear decoder) is compared against
PCA and the Nelson–Siegel model on three tasks:

1. **Reconstruction** — how accurately each model compresses and rebuilds the
   six-tenor yield curve.
2. **Interpretation** — whether the autoencoder's latent factors correspond to the
   classic Level / Slope / Curvature, assessed quantitatively via principal-angle
   subspace analysis against PCA.
3. **Trading** — a long–short relative-value strategy driven by reconstruction
   residuals, evaluated with an annually-refit **rolling window** so the backtest is
   genuinely out-of-sample (the key methodological point versus a single static fit).

## Repository structure

```
.
├── code/
│   ├── 01_data_prep.py          # load, weekly-resample, train/OOS split
│   ├── 02_pca_baseline.py       # PCA(3) baseline
│   ├── 03_autoencoder.py        # autoencoder, h in {2,3,4}, NumPy + L-BFGS-B
│   ├── 03b_rolling.py           # rolling-window OOS residuals (AE-3 and PCA-3)
│   ├── 04_nelson_siegel.py      # Nelson–Siegel benchmark (Diebold–Li two-step)
│   ├── 05_trading.py            # static long–short backtest (in-sample contrast)
│   ├── 05b_rolling_trading.py   # rolling-OOS long–short backtest (headline)
│   ├── 06_rotation.py           # principal-angle subspace analysis (AE vs PCA)
│   └── 07_final.py              # consolidates headline tables + report figures
├── data/
│   └── USdataYC.csv             # daily US Treasury CMT yields, 1990–2023
├── requirements.txt
└── README.md
```

`results/` and `figures/` are created on the fly when you run the pipeline and are
git-ignored (they are reproducible outputs, not source).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Running

Run from the repository root, in order:

```bash
python code/01_data_prep.py
python code/02_pca_baseline.py
python code/03_autoencoder.py
python code/04_nelson_siegel.py
python code/05_trading.py          # optional; only feeds the in-sample contrast table
python code/03b_rolling.py         # ~45 s (annual refits)
python code/05b_rolling_trading.py
python code/06_rotation.py
python code/07_final.py
```

Each script resolves paths relative to itself, so no configuration is needed when run
from a standard checkout. If your layout differs, set `COURSEWORK_DIR` to the project
root, or `DATA_PATH` to the CSV.

Outputs are written to `results/` (CSV tables) and `figures/` (PNG). The headline
report artefacts are `figures/fig_report_1.png`, `fig_report_2.png`,
`fig_report_3_rolling.png`, and the tables `results/table1_reconstruction.csv`,
`table2_trading.csv`, and `table3_ae_bymaturity.csv`.

## Method notes

- **Tenors.** The paper uses JGB 2/5/7/10/15/20Y. The closest continuously-available
  US analogue is **2/3/5/7/10/20Y**: the US Treasury publishes no 15Y CMT (so 3Y
  substitutes), and the 30Y was discontinued Feb 2002 – Feb 2006 (so it is dropped to
  avoid a four-year hole). Continuous data on all six begins October 1993.
- **Autoencoder.** Single hidden layer, `tanh` encoder, linear decoder, MSE loss.
  Implemented in NumPy with analytical gradients and fitted with L-BFGS-B; 30 random
  restarts (static) / 10 per refit (rolling) handle local minima. Inputs are
  standardised, matching the PCA pipeline so the comparison is on identical scaling.
- **Rolling evaluation.** For each year *Y*, models are fitted on the trailing five
  years (ending 31 Dec of *Y*−1) and used to score only year *Y*. A runtime assertion
  guarantees the training window ends strictly before the evaluation window begins, so
  there is no look-ahead leakage.
- **Trading.** Position = sign of the reconstruction residual (long when yield is above
  the model); weekly P&L ≈ −τ · Δy · position. A 1-week-lookback trend-follow rule is
  included as a momentum benchmark.

## Data

`data/USdataYC.csv` contains daily US Treasury constant-maturity yields (public data,
US Department of the Treasury / FRED), January 1990 – December 2023.

## References

- Suimon, Y., Watanabe, H., & Sakaji, H. (2020). Discovering latent factors in the
  Japanese government bond market using an autoencoder. *Journal of Mathematical
  Finance*, 10(3).
- Diebold, F. X., & Li, C. (2006). Forecasting the term structure of government bond
  yields. *Journal of Econometrics*, 130(2), 337–364.
- Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of yield curves.
  *Journal of Business*, 60(4), 473–489.
