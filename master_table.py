"""Pools the fold results of every method into one comparison table.

Results accumulate across runs and fold ranges that can overlap, so rows are deduplicated
by (fold, method, evaluation mode, target) before anything is pooled: the same fold
counted twice would tighten the spread without adding evidence.
"""

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DIRS = [
    "results_paper_baseline_legacy/save/downstream_results",
    "results_neighbor_legacy",
    "results_ablation_legacy",
    "results_vae_legacy",
    "results_expclr",
]
frames = []
for d in DIRS:
    files = sorted(glob.glob(f"{d}/downstream_raw_results_kfold_folds*.csv"))
    files += sorted(glob.glob(f"{d}/downstream_raw_results_kfold.csv"))
    if not files:
        print(f"[WARN] no result CSVs under {d}")
    frames.extend(pd.read_csv(f) for f in files)
df = pd.concat(frames, ignore_index=True)

# Re-runs and overlapping fold ranges must not double-count subject averages.
n_before = len(df)
df = df.drop_duplicates(subset=["fold", "method", "eval_mode", "target"], keep="first")
if len(df) != n_before:
    print(f"[WARN] dropped {n_before - len(df)} duplicated (fold, method, "
          f"eval_mode, target) rows before pooling")


def parse(s):
    out = []
    if isinstance(s, str) and s:
        for p in s.split(";"):
            if p.strip():
                a, b = p.split(",")
                out.append((float(a), float(b)))
    return out


CATS = [
    ("Baseline / SSL puro", ["PCA", "AE", "MAE", "SimCLR", "VAE", "InterFusion"]),
    ("Informado por etiqueta", ["TripletLoss", "supervised"]),
    ("Guiado por descriptor experto continuo (ExpCLR)", ["ExpCLR"]),
    ("Neighbor - same session", ["SimCLR-nbr-cosine", "SimCLR-nbr-wasser", "SimCLR-nbr-riemann"]),
    ("Neighbor - cross-subject same-age", ["SimCLR-xsubj-cosine", "SimCLR-xsubj-wasser", "SimCLR-xsubj-riemann"]),
    ("Neighbor - same-subject diff-age", ["SimCLR-diffage-cosine", "SimCLR-diffage-wasser", "SimCLR-diffage-riemann"]),
]


def row(m, mode):
    r = df[(df["method"] == m) & (df["eval_mode"] == mode)]
    if len(r) == 0:
        return None
    avgs = []
    for s in r["subject_avgs"]:
        avgs += parse(s)
    yt = np.array([a[0] for a in avgs])
    yp = np.array([a[1] for a in avgs])
    rmg = np.sqrt(np.mean((yt - yp) ** 2))
    std_yt = np.std(yt)
    nrmse = np.nan if std_yt < 1e-8 else rmg / std_yt
    return (nrmse, r["RMSE"].mean(), r["RMSE"].std(), r2_score(yt, yp), len(r))


for mode in ["linear_probe", "fine_tuning"]:
    print(f"\n################  {mode.upper()}  ################")
    print(f"{'method':<26}{'nRMSE':>7}{'RMSE':>7}{'±std':>7}{'R2':>7}{'folds':>6}")
    for cat, methods in CATS:
        printed = False
        for m in methods:
            res = row(m, mode)
            if res is None:
                continue
            if not printed:
                print(f"-- {cat}")
                printed = True
            nr, rm, sd, r2, nf = res
            print(f"   {m:<23}{nr:>7.2f}{rm:>7.2f}{sd:>7.2f}{r2:>7.2f}{nf:>6}")
