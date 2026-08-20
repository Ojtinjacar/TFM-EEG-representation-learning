from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval_expclr import (
    aggregate_to_sessions, extract_embeddings, fit_probe, session_metrics,
)
from loss import ExpCLRLoss
from models import EnhancedAttentionLSTM
from train_expclr import (
    checkpoint_is_reusable, effective_dimensionality, max_pairwise_distance, prepare_descriptor,
)

DATA = Path("data/processed/all_all")
OUT = Path("save/e3_tuning")

DEFAULT_DESCRIPTOR = "P_diverso"


def features_path(descriptor):
    return Path("data/processed/expert_features") / f"expert_features_{descriptor}.npy"


FEATURES = features_path(DEFAULT_DESCRIPTOR)

DELTA_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]

LR_GRID = [5e-5, 1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]

PAPER_LR = 5e-3

TAU_GRID = [0.5, 0.7, 1.0, 2.0]

VALIDATION_FRACTION = 0.2


def suggest_delta(features: np.ndarray, subjects: np.ndarray, holdout: set[str],
                  sim_max: str = "train") -> float:
    train_mask = ~np.isin(subjects, list(holdout))
    prepared, _ = prepare_descriptor(features[train_mask])
    max_dist = max_pairwise_distance(prepared) if sim_max == "train" else None
    criterion = ExpCLRLoss(torch.device("cpu"), feat_max_dist=max_dist, temperature=None)

    rng = np.random.default_rng(0)
    means = []
    for _ in range(50):
        idx = rng.choice(len(prepared), size=min(64, len(prepared)), replace=False)
        batch = torch.tensor(prepared[idx], dtype=torch.float32)
        means.append(float((1.0 - criterion.expert_similarity(batch)).mean()))
    return 1.0 / float(np.mean(means))


def split_subjects(subjects: np.ndarray, seed: int = 42) -> tuple[list[str], list[str]]:
    unique = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_val = max(2, int(round(len(unique) * VALIDATION_FRACTION)))
    return sorted(shuffled[n_val:].tolist()), sorted(shuffled[:n_val].tolist())


def train_one(delta: float, lr: float, sim_max: str, epochs: int, holdout: list[str],
              seed: int, tau: float = 1.0, descriptor: str = DEFAULT_DESCRIPTOR) -> Path | None:
    fold_id = f"tune_d{delta}_lr{lr}_t{tau}_{sim_max}_e{epochs}_s{seed}"
    ckpt = Path("save/models") / expclr_checkpoint_name(
        "all", "all", fold_id, descriptor,
        batch_size=64, lr=lr, temperature=tau, delta=delta)
    if checkpoint_is_reusable(ckpt, {"delta": delta, "lr": lr, "temperature": tau,
                                     "sim_max": sim_max, "loss_on": "projection",
                                     "dropout": 0.0, "num_epochs": epochs, "seed": seed}):
        return ckpt
    cmd = [sys.executable, "src/train_expclr.py",
           "--data_path", str(DATA), "--expert_features", str(features_path(descriptor)),
           "--descriptor", descriptor, "--zone", "all", "--frequency", "all",
           "--fold_id", fold_id, "--num_epochs", str(epochs), "--seed", str(seed),
           "--delta", str(delta), "--lr", str(lr), "--temperature", str(tau),
           "--loss_on", "projection", "--dropout", "0.0",
           "--sim_max", sim_max, "--diagnose_every", str(max(1, epochs // 4)),
           "--exclude_subjects", *holdout]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not ckpt.exists():
        print(f"  fallo delta={delta} lr={lr} tau={tau} sim_max={sim_max}: "
              f"{res.stderr.strip()[-300:]}")
        return None
    return ckpt


def score(ckpt: Path | None, X, meta, y, groups, val_mask, device, random_seed: int = 42) -> dict:
    if ckpt is None:
        torch.manual_seed(random_seed)
    model = EnhancedAttentionLSTM(input_size=X.shape[2], hidden_size=128, n_channels=X.shape[1],
                                  sfreq=250, lstm_hidden_size=64, dropout=0.0).to(device)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    emb = extract_embeddings(model, X, device)

    train_mask = ~val_mask
    scaler, probe = fit_probe(emb[train_mask], y[train_mask], groups[train_mask])
    pred = probe.predict(scaler.transform(emb[val_mask]))
    sessions = aggregate_to_sessions(meta[val_mask], y[val_mask], pred)
    return {**session_metrics(sessions), "effective_dim": effective_dimensionality(emb)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tuning of Delta and lr for ExpCLR following the paper's procedure.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per configuration. Enough to rank them, not for the final "
                             "result.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Delta axis only, at the paper's lr for SleepEDF/Waveform (5e-3).")
    parser.add_argument("--descriptor", default=DEFAULT_DESCRIPTOR,
                        help="Expert descriptor the search runs on. Each one needs its own search, "
                             "since Delta depends on the scale of its similarities.")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    features = features_path(args.descriptor)
    if not features.exists():
        raise SystemExit(f"Descriptor {args.descriptor} does not exist at {features}. "
                         "Build it first with code/src/build_expert_features.py.")

    X = np.asarray(np.load(DATA / "processed_windows.npy", mmap_mode="r"))
    meta = pd.read_csv(DATA / "processed_metadata.csv")
    F_raw = np.load(features)
    print(f"descriptor: {args.descriptor} ({F_raw.shape[1]} features)")
    y = meta.age.values.astype(float)
    groups = meta.subject.values

    train_subjects, val_subjects = split_subjects(groups, args.seed)
    val_mask = np.isin(groups, val_subjects)
    print(f"device: {device} | {len(train_subjects)} training subjects, "
          f"{len(val_subjects)} for validation: {val_subjects}")

    suggested = {sm: suggest_delta(F_raw, groups, set(val_subjects), sm)
                 for sm in ("train", "batch")}
    for sm, value in suggested.items():
        print(f"Suggested delta with --sim_max {sm}: {value:.2f} "
              f"(= 1 / E[1 - s_ij]; the bias with Delta=1 would be {abs(1 - 1 / value):.3f})")

    baseline = score(None, X, meta, y, groups, val_mask, device, args.seed)
    print(f"referencia, encoder aleatorio: MAE={baseline['mae']:.2f} R2={baseline['r2']:.3f} "
          f"dim_efectiva={baseline['effective_dim']:.2f}\n")

    rows = []

    def evaluate(delta, lr, sim_max, stage, index, total, tau=1.0):
        t0 = time.time()
        ckpt = train_one(delta, lr, sim_max, args.epochs, val_subjects, args.seed, tau,
                         descriptor=args.descriptor)
        if ckpt is None:
            return None
        result = score(ckpt, X, meta, y, groups, val_mask, device, args.seed)
        label = f"delta={delta:<5} lr={lr:<7} tau={tau:<4} {sim_max:<5}"
        if not np.isfinite(result["mae"]):
            print(f"  [{stage} {index:2d}/{total}] {label} "
                  f"DESCARTADA (metrica no finita, probable divergencia)", flush=True)
            return None
        rows.append({"stage": stage, "delta": delta, "lr": lr, "tau": tau, "sim_max": sim_max,
                     "is_suggested": abs(delta - suggested[sim_max]) < 0.01, **result})
        print(f"  [{stage} {index:2d}/{total}] {label} "
              f"MAE={result['mae']:5.2f} R2={result['r2']:+.3f} "
              f"dim={result['effective_dim']:5.2f} ({time.time() - t0:.0f}s)", flush=True)
        return result

    stage1 = [(d, sm) for sm in ("train", "batch")
              for d in sorted(set(DELTA_GRID + [round(suggested[sm], 2)]))]
    print(f"Etapa 1: {len(stage1)} configuraciones de (Delta, sim_max) a lr={PAPER_LR}")
    for i, (delta, sim_max) in enumerate(stage1, 1):
        evaluate(delta, PAPER_LR, sim_max, "1", i, len(stage1))

    if not rows:
        raise RuntimeError("no configuration completed training")

    best1 = min(rows, key=lambda r: r["mae"])
    print(f"\nBest of stage 1: delta={best1['delta']}, sim_max={best1['sim_max']}, "
          f"MAE={best1['mae']:.2f}")

    if not args.quick:
        lrs = [lr for lr in LR_GRID if lr != PAPER_LR]
        print(f"\nStage 2: {len(lrs)} learning rates with delta={best1['delta']}, "
              f"sim_max={best1['sim_max']}")
        for i, lr in enumerate(lrs, 1):
            evaluate(best1["delta"], lr, best1["sim_max"], "2", i, len(lrs))

        best2 = min(rows, key=lambda r: r["mae"])
        taus = [t for t in TAU_GRID if t != 1.0]
        print(f"\nStage 3: {len(taus)} temperatures with delta={best2['delta']}, "
              f"lr={best2['lr']}, sim_max={best2['sim_max']}")
        for i, tau in enumerate(taus, 1):
            evaluate(best2["delta"], best2["lr"], best2["sim_max"], "3", i, len(taus), tau)

    table = pd.DataFrame(rows).sort_values("mae")
    table.to_csv(OUT / f"tuning_results_{args.descriptor}.csv", index=False)
    best = table.iloc[0]

    print("\n=== best configurations (sorted by MAE) ===")
    print(table.head(10).round(3).to_string(index=False))
    print(f"\nreferencia aleatoria: MAE={baseline['mae']:.2f}, R2={baseline['r2']:.3f}, "
          f"dim_efectiva={baseline['effective_dim']:.2f}")
    print(f"ganadora: delta={best.delta}, lr={best.lr}, sim_max={best.sim_max}, "
          f"MAE={best.mae:.2f}, R2={best.r2:.3f}, dim_efectiva={best.effective_dim:.2f}")

    if best.mae >= baseline["mae"]:
        print("\nNO configuration beats the random encoder. The result to report is that "
              "ExpCLR does not transfer in this domain, not a tuning failure.")
    print(f"\nWarning: the winner is picked among {len(rows)} configurations over "
          f"{int(baseline['n_sessions'])} sessions from {len(val_subjects)} subjects, so its "
          f"validation metric is optimistically biased by selection. The reportable figure is "
          f"the later LOSO over the remaining subjects, not this one.")

    config_path = OUT / f"best_config_{args.descriptor}.json"
    json.dump({"descriptor": args.descriptor,
               "delta": float(best.delta), "lr": float(best.lr), "tau": float(best.tau),
               "sim_max": str(best.sim_max),
               "delta_suggested": {k: float(v) for k, v in suggested.items()},
               "mae": float(best.mae), "r2": float(best.r2),
               "random_baseline_mae": float(baseline["mae"]),
               "random_baseline_r2": float(baseline["r2"]),
               "beats_random_baseline": bool(best.mae < baseline["mae"]),
               "n_configs_compared": len(rows),
               "validation_subjects": val_subjects, "epochs": args.epochs, "seed": args.seed},
              open(config_path, "w"), indent=2)
    print(f"\nGuardado en {config_path}. Lanzar despues:")
    print(f"  python -u run_e3_loso.py --epochs 50 --config {config_path}")
    print(f"  or, for the comparable table: run_downstream.py --expclr_config {OUT}")


if __name__ == "__main__":
    main()
