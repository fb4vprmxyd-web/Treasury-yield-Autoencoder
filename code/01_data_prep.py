"""
Data preparation for the autoencoder replication.

Reproduces the paper's setup as closely as possible on US Treasury data:
  - Paper uses JGB tenors: 2Y, 5Y, 7Y, 10Y, 15Y, 20Y.
  - Closest US analogue (no 15Y on UST): 2Y, 5Y, 7Y, 10Y, 20Y, 30Y.
  - Paper uses weekly data, Jul 1992 - Jul 2019. We use weekly (Friday close)
    starting Jul 1992. Replication window: Jul 1992 - Dec 2019.
    Out-of-sample stress test: Jan 2020 - Dec 2023.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import os

DATA_ENV = os.environ.get("DATA_PATH")
_HERE   = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("COURSEWORK_DIR", _HERE.parent))
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

def _find_data():
    """Locate USdataYC.csv. Checks $DATA_PATH, then a few sensible locations."""
    candidates = [
        Path(DATA_ENV) if DATA_ENV else None,
        OUT_DIR / "data" / "USdataYC.csv",
        OUT_DIR / "USdataYC.csv",
        _HERE / "USdataYC.csv",
    ]
    for c in candidates:
        if c and c.exists():
            return c
    raise FileNotFoundError(
        "USdataYC.csv not found. Put it in <project>/data/ or <project>/, "
        "or set the DATA_PATH environment variable to its full path."
    )

DATA_PATH = _find_data()

# Tenors used.
# Paper (JGB):                       2Y, 5Y, 7Y, 10Y, 15Y, 20Y.
# Closest continuous UST analogue:   2Y, 3Y, 5Y, 7Y, 10Y, 20Y.
#   - 15Y CMT yield is not published by US Treasury -> substitute 3Y.
#   - 30Y is discontinued Feb 2002 - Feb 2006 (4-year structural gap) -> dropped.
TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y"]
TENOR_YEARS = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0])

# Date conventions
# 20Y CMT not continuously available until Oct 1993, so the effective sample
# starts there rather than at the paper's Jul 1992.
TRAIN_START = "1993-10-01"
TRAIN_END   = "2019-12-31"
OOS_END     = "2023-12-29"


def load_raw():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    # File is reverse-chronological; sort ascending
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_weekly(df):
    """Select tenors, drop rows with any NA in selected columns, resample to weekly (Fri)."""
    sub = df[["Date"] + TENORS].copy()
    sub = sub.dropna(subset=TENORS).reset_index(drop=True)
    sub = sub.set_index("Date")
    # Weekly: take Friday close. If Friday missing (holiday) take last available within the week.
    weekly = sub.resample("W-FRI").last().dropna()
    return weekly


def split(weekly):
    train = weekly.loc[TRAIN_START:TRAIN_END]
    oos   = weekly.loc["2020-01-01":OOS_END]
    return train, oos


def main():
    raw = load_raw()
    print(f"Raw rows: {len(raw):,}  date range: {raw.Date.min().date()} -> {raw.Date.max().date()}")

    weekly = build_weekly(raw)
    print(f"Weekly rows after dropna on selected tenors: {len(weekly):,}")
    print(f"Weekly range: {weekly.index.min().date()} -> {weekly.index.max().date()}")

    train, oos = split(weekly)
    print(f"Train (1992-2019): {len(train):,} weeks")
    print(f"OOS   (2020-2023): {len(oos):,} weeks")

    # Save processed data
    weekly.to_csv(RES_DIR / "weekly_yields.csv")
    train.to_csv(RES_DIR / "train_yields.csv")
    oos.to_csv(RES_DIR / "oos_yields.csv")

    # Summary statistics
    print("\nWeekly yield summary (full sample, %):")
    summary = weekly.describe().T[["mean", "std", "min", "max"]].round(2)
    print(summary)
    summary.to_csv(RES_DIR / "summary_stats.csv")

    # ---- Figure 1: Yield curve through time -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) Time series of selected tenors
    ax = axes[0]
    for col, c in zip(TENORS, plt.cm.viridis(np.linspace(0, 0.9, len(TENORS)))):
        ax.plot(weekly.index, weekly[col], lw=0.7, color=c, label=col)
    ax.set_ylabel("Yield (%)"); ax.set_xlabel("Date")
    ax.set_title("(a) US Treasury yields, weekly (Friday close)")
    ax.axvspan(pd.Timestamp("2020-01-01"), weekly.index.max(),
               color="grey", alpha=0.15, label="OOS")
    ax.legend(ncol=4, fontsize=8, loc="upper right")

    # (b) Snapshot curves across different regimes
    ax = axes[1]
    snapshots = {
        "1994-12-30": "Mid-90s (positive carry)",
        "2000-12-29": "Dot-com peak (inverted)",
        "2008-12-26": "GFC (ZIRP onset)",
        "2015-12-31": "QE3 era",
        "2020-03-27": "COVID shock",
        "2023-12-29": "Post-hiking, inverted",
    }
    for date_str, label in snapshots.items():
        if pd.Timestamp(date_str) in weekly.index:
            ax.plot(TENOR_YEARS, weekly.loc[date_str].values, marker="o", label=label)
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)")
    ax.set_title("(b) Yield curve snapshots across regimes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_yields.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 1 -> {FIG_DIR / 'fig1_yields.png'}")


if __name__ == "__main__":
    main()
