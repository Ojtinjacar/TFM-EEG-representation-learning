"""Grouped-by-subject cross-validation of ExpCLR against its baselines.

The folds are the ones every other experiment here is evaluated under: KFold over the
sorted subjects, with the number of splits and the seed that run_downstream.py uses, so a
figure from this script and one from that one partition the cohort identically. One encoder
is pre-trained per fold, on a training set that excludes both the subjects held out by the
fold and any subject used to choose hyperparameters. Every fold contributes out-of-fold
predictions to a single pool, which is what the metrics are computed on.

The baselines are what makes the result falsifiable. B0 predicts the training mean, B1
regresses the descriptor directly without any encoder, B2 uses an untrained encoder, B3
repeats ExpCLR against a descriptor whose rows have been shuffled, and B7 concatenates the
embedding with the descriptor. An encoder that cannot beat B2 has learned nothing, and one
that matches B3 is not using the descriptor it was given.
"""

from __future__ import annotations

import argparse
import hashlib
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
    aggregate_to_sessions, bootstrap_ci, extract_embeddings, fit_knn_probe, fit_probe,
    metrics_by_visit, paired_bootstrap_difference, session_metrics, subject_metrics,
)
from models import EnhancedAttentionLSTM
from build_expert_features import DESCRIPTORS, descriptor_path
from checkpoint_naming import checkpoint_is_reusable, expclr_checkpoint_name
from folds import canonical_subject_folds
from window_loading import NORM_PROVENANCE, apply_fold_normalisation

DATA = Path("data/processed/all_all")
DESCRIPTOR = "P_madurativo"
FEATURES = Path(descriptor_path(DESCRIPTOR))
OUT = Path("save/expclr_folds")


def load_data() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    X = np.load(DATA / "processed_windows.npy", mmap_mode="r")
    meta = pd.read_csv(DATA / "processed_metadata.csv")
    F = np.load(FEATURES)
    if not (len(X) == len(meta) == len(F)):
        raise ValueError(f"desalineados: X={len(X)}, meta={len(meta)}, F={len(F)}")
    return X, meta, F


def impute_with_train_medians(features: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Fills the gaps of the descriptor with the column medians of the training split.

    The probe cannot take a missing value, and the encoder path never sees one because
    training imputes before it starts. The medians come from the training side alone: taken
    from the whole set they would let the held-out subject inform the values the probe is
    then scored on.

    Args:
        features (np.ndarray): Descriptor of every window, shape (n_windows, n_features).
        train (np.ndarray): Boolean mask selecting the training windows.

    Returns:
        np.ndarray: The descriptor with no missing values left.

    Raises:
        ValueError: If a column is missing throughout the training split, which leaves
            nothing to fill it with.
    """
    empty = np.isnan(features[train]).all(axis=0)
    if empty.any():
        raise ValueError(
            f"{int(empty.sum())} descriptor columns are missing across the whole training "
            "split, so there is no median to fill them with."
        )
    medians = np.nanmedian(features[train], axis=0)
    return np.where(np.isnan(features), medians, features)


def pretrain_fold(held_out: list[str], fold_idx: int, n_folds: int, epochs: int,
                  features: Path, tag: str, seed: int = 42,
                  delta: float = 1.0, lr: float = 5e-3, sim_max: str = "train",
                  tau: float = 1.0, tuning_excluded: list[str] | None = None) -> Path:
    # The fold encoder must not see the hyperparameter-tuning subjects either;
    # a short hash of the full exclusion list keeps configurations with
    # different tuning sets from sharing (and overwriting) checkpoint files. The
    # index alone would not: two folds of different partitions share it.
    excl = sorted({*held_out, *(tuning_excluded or [])})
    excl_tag = "_x" + hashlib.md5("_".join(excl).encode()).hexdigest()[:6]
    fold_id = f"kf{n_folds}_{tag}_e{epochs}_s{seed}_{sim_max}{excl_tag}_f{fold_idx}"
    ckpt = Path("save/models") / expclr_checkpoint_name(
        "all", "all", fold_id, DESCRIPTOR,
        batch_size=64, lr=lr, temperature=tau, delta=delta)
    expected = {"delta": delta, "lr": lr, "lr_gamma": 0.99, "temperature": tau,
                "loss_on": "projection", "sim_max": sim_max, "dropout": 0.0,
                "num_epochs": epochs, "seed": seed, "descriptor": DESCRIPTOR,
                # Neither the width of the model nor that of the descriptor reaches the
                # filename, and both change what the weights mean.
                "embedding_size": 128, "descriptor_dim": len(DESCRIPTORS[DESCRIPTOR]),
                "exclude_subjects": excl,
                # Refusing checkpoints trained under cohort-wide amplitude statistics.
                "norm_stats": NORM_PROVENANCE}
    if checkpoint_is_reusable(ckpt, expected):
        return ckpt
    cmd = [sys.executable, "src/train_expclr.py",
           "--data_path", str(DATA), "--expert_features", str(features),
           "--descriptor", DESCRIPTOR, "--zone", "all", "--frequency", "all",
           "--fold_id", fold_id, "--num_epochs", str(epochs), "--seed", str(seed),
           "--delta", str(delta), "--lr", str(lr), "--temperature", str(tau),
           "--loss_on", "projection", "--dropout", "0.0", "--sim_max", sim_max,
           "--exclude_subjects", *excl]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not ckpt.exists():
        raise RuntimeError(
            f"pre-training failed for fold {fold_idx}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return ckpt


def build_encoder(X: np.ndarray, device, checkpoint: Path | None, seed: int = 0):
    if checkpoint is None:
        torch.manual_seed(seed)
    model = EnhancedAttentionLSTM(input_size=X.shape[2], hidden_size=128, n_channels=X.shape[1],
                                  sfreq=250, lstm_hidden_size=64, dropout=0.0).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


class EmbeddingCache:
    def __init__(self, X: np.ndarray, device, random_seed: int = 42) -> None:
        self._X = np.asarray(X)
        self._device = device
        self._random_seed = random_seed
        self._embeddings: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def get(self, checkpoint: Path | None) -> np.ndarray:
        if checkpoint is not None:
            # The mtime keys out stale embeddings if a checkpoint file is
            # retrained in place during the lifetime of this cache.
            key = f"{checkpoint}:{Path(checkpoint).stat().st_mtime_ns}"
        else:
            key = f"__random_s{self._random_seed}__"
        cached = self._embeddings.get(key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        model = build_encoder(self._X, self._device, checkpoint, seed=self._random_seed)
        self._embeddings[key] = extract_embeddings(model, self._X, self._device)
        return self._embeddings[key]

    def set_windows(self, X: np.ndarray) -> None:
        """Rebinds the cache to the windows of a new fold.

        The amplitude normalisation is refitted per fold, so an embedding
        extracted under the previous fold's transform no longer describes these
        windows. Dropping what is cached is what keeps a checkpoint-keyed entry
        from outliving the data it was computed on: B2 reuses the same untrained
        encoder across folds and would otherwise hit a stale entry.

        Args:
            X (np.ndarray): Window tensor for the current fold.
        """
        self._X = np.asarray(X)
        self._embeddings.clear()

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}


def subject_folds(subjects: list[str], n_folds: int, base_seed: int) -> list[list[str]]:
    """Splits the subjects into folds, exactly as the rest of the pipeline does.

    Mirrors ``run_downstream.py``: ``KFold`` over the sorted subjects with ``shuffle`` and
    the same seed. The partition is built over every subject, tuning ones included, so that
    which fold a subject belongs to does not depend on how many are later withheld from
    scoring. That is what makes a figure here comparable, fold by fold, with one from there.

    The rule itself lives in ``src/folds.py``, so that the spectral baselines split the same
    way without a third copy of it drifting from these two.

    Args:
        subjects (list[str]): Every subject in the cohort, sorted.
        n_folds (int): Number of splits.
        base_seed (int): Seed of the shuffle.

    Returns:
        list[list[str]]: Held-out subjects of each fold, in fold order.
    """
    return canonical_subject_folds(subjects, n_folds, base_seed)


def run_folds(methods: list[str], epochs: int, device, seed: int = 42, delta: float = 1.0,
              lr: float = 5e-3, sim_max: str = "train",
              exclude_subjects: list[str] | None = None,
              tau: float = 1.0, n_folds: int = 10,
              base_seed: int = 1234) -> dict[str, pd.DataFrame]:
    X, meta, F = load_data()
    # The partition comes first and the tuning subjects are withdrawn afterwards, so their
    # removal shifts nobody else between folds.
    all_subjects = sorted(meta.subject.unique())
    folds = subject_folds(all_subjects, n_folds, base_seed)
    tuning = set(exclude_subjects or [])
    if tuning:
        keep = ~meta.subject.isin(tuning).values
        meta = meta[keep].reset_index(drop=True)
        X, F = X[keep], F[keep]
        print(f"Excluded {len(tuning)} subjects used to tune hyperparameters: "
              f"{sorted(tuning)}")
    subjects = sorted(meta.subject.unique())
    y = meta.age.values.astype(float)
    groups = meta.subject.values
    print(f"{n_folds}-fold by subject (seed {base_seed}) over {len(subjects)} subjects | "
          f"{len(meta)} windows | descriptor {F.shape}")

    # B3 permutes the FULL on-disk descriptor so the subprocess (which reloads
    # the complete dataset) stays row-aligned; the artifact is derived data and
    # lives under save/, not next to the real descriptors.
    shuffled_path = OUT / f"{DESCRIPTOR}_shuffled_s{seed}.npy"
    if "B3" in methods:
        rng = np.random.default_rng(seed)
        F_full = np.load(FEATURES)
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(shuffled_path, F_full[rng.permutation(len(F_full))])

    cache = EmbeddingCache(X, device, random_seed=seed)
    preds: dict[str, list[pd.DataFrame]] = {m: [] for m in methods}
    for fold_idx, fold_subjects in enumerate(folds):
        t0 = time.time()
        held_out = [s for s in fold_subjects if s not in tuning]
        if not held_out:
            print(f"  [{fold_idx + 1:2d}/{len(folds)}] skipped: every subject of this fold "
                  "was used for tuning", flush=True)
            continue
        test = np.isin(meta.subject.values, held_out)
        train = ~test
        if len(np.unique(y[train])) < 2:
            print(f"  [{fold_idx + 1:2d}/{len(folds)}] skipped: the training split has a "
                  "single age", flush=True)
            continue

        # The amplitude statistics are refitted without the held-out subjects and
        # applied to every window, theirs included. Without this the transform
        # that the probe is scored under would have been informed by the very
        # subjects it is meant not to have seen.
        cache.set_windows(
            apply_fold_normalisation(X, meta.subject.values, held_out, verbose=False)
        )

        for method in methods:
            if method == "B0":
                pred_test = np.full(test.sum(), y[train].mean())
            elif method in ("B1", "B7"):
                F_filled = impute_with_train_medians(F, train)
                if method == "B1":
                    Xtr, Xte = F_filled[train], F_filled[test]
                else:
                    emb = cache.get(pretrain_fold(held_out, fold_idx, n_folds, epochs,
                                                  FEATURES, "expclr", seed, delta, lr,
                                                  sim_max, tau,
                                                  tuning_excluded=exclude_subjects))
                    Xtr = np.hstack([emb[train], F_filled[train]])
                    Xte = np.hstack([emb[test], F_filled[test]])
                scaler, probe = fit_probe(Xtr, y[train], groups[train])
                pred_test = probe.predict(scaler.transform(Xte))
            else:
                if method == "ExpCLR":
                    ckpt = pretrain_fold(held_out, fold_idx, n_folds, epochs, FEATURES,
                                         "expclr", seed, delta, lr, sim_max, tau,
                                         tuning_excluded=exclude_subjects)
                elif method == "B3":
                    ckpt = pretrain_fold(held_out, fold_idx, n_folds, epochs, shuffled_path,
                                         "shuffled", seed, delta, lr, sim_max, tau,
                                         tuning_excluded=exclude_subjects)
                elif method == "B2":
                    ckpt = None
                else:
                    raise ValueError(f"unknown method: {method}")
                emb = cache.get(ckpt)
                scaler, probe = fit_probe(emb[train], y[train], groups[train])
                pred_test = probe.predict(scaler.transform(emb[test]))
                knn_scaler, knn = fit_knn_probe(emb[train], y[train])
                knn_pred = knn.predict(knn_scaler.transform(emb[test]))
                preds.setdefault(f"{method}_KNN", []).append(
                    aggregate_to_sessions(meta[test], y[test], knn_pred))

            preds[method].append(aggregate_to_sessions(meta[test], y[test], pred_test))

        print(f"  [{fold_idx + 1:2d}/{len(folds)}] {len(held_out)} subjects  "
              f"{time.time()-t0:5.1f}s", flush=True)

    stats = cache.stats
    print(f"embeddings: {stats['misses']} extracciones, {stats['hits']} reutilizadas", flush=True)
    return {m: pd.concat(v, ignore_index=True) for m, v in preds.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExpCLR under the grouped-by-subject cross-validation of the rest of "
                    "the pipeline, against its falsifiability baselines.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Pretraining epochs; the loss converges by epoch 50.")
    parser.add_argument("--methods", nargs="+",
                        default=["ExpCLR", "B0", "B1", "B2", "B3", "B7"],
                        help="ExpCLR, B0 mean, B1 Ridge on the descriptor, B2 random encoder, "
                             "B3 shuffled descriptor, B7 embedding+descriptor.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed shared by every fold: what varies across folds is which "
                             "subjects are held out, not the initialisation.")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Folds of the subject-wise cross-validation. The default is the "
                             "one run_downstream.py uses, and changing it breaks the "
                             "correspondence with the other experiments.")
    parser.add_argument("--base_seed", type=int, default=1234,
                        help="Seed of the fold shuffle. Together with --n_folds it is what "
                             "makes this partition identical to the one of run_downstream.py; "
                             "it is not the pre-training seed, which is --seed.")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Target scale Delta. A value of 1.0 is only self-consistent when the "
                             "mean of (1 - s_ij) is itself 1, which depends on how the descriptor "
                             "distributes its distances; take the value from tune_expclr.py.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Adam learning rate. The paper tunes it per dataset (Tab. 9), so take "
                             "it from tune_expclr.py as well.")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Hard-negative mining temperature. The paper states its own 1.0 is not "
                             "optimal (App. B.4: 'other tau < 1.0 could improve the performance "
                             "slightly'), so take it from tune_expclr.py.")
    parser.add_argument("--sim_max", choices=["train", "batch"], default="train",
                        help="Where max_kl ||f_k - f_l|| is taken. The paper allows both (Sec. 3.4); "
                             "it changes the scale of s_ij and therefore interacts with Delta. "
                             "Take it from tune_expclr.py.")
    parser.add_argument("--exclude_subjects", nargs="*", default=[],
                        help="Subjects excluded from the evaluation because they were used to tune "
                             "hyperparameters. See validation_subjects in best_config.json.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Read delta, lr, sim_max and the validation subjects from the "
                             "best_config.json written by tune_expclr.py, instead of by hand.")
    args = parser.parse_args()

    if args.config:
        cfg = json.loads(args.config.read_text())
        args.delta, args.lr = cfg["delta"], cfg["lr"]
        args.sim_max = cfg.get("sim_max", args.sim_max)
        args.tau = cfg.get("tau", args.tau)
        args.exclude_subjects = cfg["validation_subjects"]
        print(f"config from {args.config}: delta={args.delta} lr={args.lr} tau={args.tau} "
              f"sim_max={args.sim_max} | {len(args.exclude_subjects)} excluded subjects")

    if not args.exclude_subjects:
        print("[WARNING] Running without --config or --exclude_subjects: the subjects used "
              "to tune hyperparameters will enter the evaluation (optimistic bias).")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | epochs: {args.epochs} | methods: {args.methods}")

    results = run_folds(args.methods, args.epochs, device, args.seed, args.delta, args.lr,
                        args.sim_max, args.exclude_subjects, args.tau, args.n_folds,
                        args.base_seed)

    rows = []
    for method, sessions in results.items():
        sessions.to_csv(OUT / f"predicciones_{method}.csv", index=False)
        m = session_metrics(sessions)
        low, high = bootstrap_ci(sessions, "mae")
        rows.append({"method": method, **m, "mae_ci_low": low, "mae_ci_high": high,
                     **subject_metrics(sessions)})
    summary = pd.DataFrame(rows).sort_values("mae")
    summary.to_csv(OUT / "method_summary.csv", index=False)

    session_cols = ["method", "n_sessions", "mae", "mae_ci_low", "mae_ci_high", "r2", "rho"]
    print("\n=== nivel sesion, out-of-fold (cifra principal) ===")
    print(summary[session_cols].round(3).to_string(index=False))
    print("\n=== subject level, master_table.py formula (comparable with E0-E2) ===")
    print(summary[["method", "nrmse_subject", "rmse_subject", "r2_subject"]]
          .round(3).to_string(index=False))
    print("  (averages the visits of each child, hence not the headline figure)")

    gaps = [(m, summary.loc[summary.method == m, "mae"].iloc[0],
             summary.loc[summary.method == f"{m}_KNN", "mae"].iloc[0])
            for m in results if f"{m}_KNN" in results.keys()]
    if gaps:
        print("\n=== linear vs KNN(k=1) difference, MAE in months ===")
        for method, linear, knn in sorted(gaps, key=lambda g: g[1]):
            print(f"  {method:10s} linear {linear:5.2f} | KNN {knn:5.2f} | difference {knn-linear:+5.2f}")

    if "ExpCLR" in results and "B1" in results:
        diff = paired_bootstrap_difference(results["ExpCLR"], results["B1"], "mae")
        json.dump(diff, open(OUT / "contraste_primario.json", "w"), indent=2)
        print("\nprimary contrast ExpCLR - B1 (Ridge on the descriptor), MAE in months:")
        print(f"  diferencia {diff['diff']:+.3f}  IC95 [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]"
              f"  over {diff['n_paired_sessions']} sessions")
        print("  (negative = ExpCLR better; if the CI crosses 0, there is no evidence of a "
              "difference)")

    if "ExpCLR" in results:
        visits = metrics_by_visit(results["ExpCLR"])
        visits.to_csv(OUT / "expclr_por_visita.csv", index=False)
        print("\nExpCLR, error per visit (the target has only 4 levels):")
        print(visits.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
