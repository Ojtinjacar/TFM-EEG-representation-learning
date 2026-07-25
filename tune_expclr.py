"""Selects Delta and the learning rate for ExpCLR the way the paper does, on our data.

Nonnenmacher et al. (ICML 2022) do not hand down Delta = 1 as a constant of the method: they search
it (App. A.2, Fig. 4) and report it as "a good compromise over all datasets and algorithms" (p. 6),
and they optimise the learning rate per dataset over an eight-value grid (App. B.3, Tab. 9). Copying
their numbers without repeating their search is what would be unfaithful, because Delta only works
when it matches the scale of the descriptor similarities.

Why that matters here concretely: the mean-normalisation adopted from Kim et al. (2021) forces every
row mean of D_ij to equal 1 by construction, while the target (1 - s_ij) * Delta has row mean
E[1 - s_ij] * Delta. With P_diverso, E[1 - s_ij] is about 0.54, so Delta = 1 leaves a constant
mismatch that no encoder can remove and the objective converges to the equidistant geometry, the
least informative one available.

Following the paper (p. 6), hyperparameters are chosen on a held-out split of subjects and never
inside the LOSO loop used for the final comparison.

Usage:
    python tune_expclr.py --epochs 30
    python tune_expclr.py --epochs 30 --quick   # only the Delta axis, at the paper's lr
"""
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

from eval_expclr import (  # noqa: E402
    aggregate_to_sessions, extract_embeddings, fit_probe, session_metrics,
)
from loss import ExpCLRLoss  # noqa: E402
from models import EnhancedAttentionLSTM  # noqa: E402
from train_expclr import (  # noqa: E402
    effective_dimensionality, max_pairwise_distance, prepare_descriptor,
)

DATA = Path("data/processed/all_all")
FEATURES = Path("data/processed/expert_features/expert_features_P_diverso.npy")
OUT = Path("save/e3_tuning")

#: Delta grid of the paper (App. A.2), plus the value predicted by the bias/variance decomposition.
#: The predicted one is our addition and is reported as such.
DELTA_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]

#: Learning-rate grid of the paper (App. B.3): "lr in {5e-5, 1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2}".
LR_GRID = [5e-5, 1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]

#: The rate the paper selects for ExpCLR on the two EEG-like datasets, SleepEDF and Waveform
#: (Tab. 9). Used as the fixed rate while the Delta axis is explored.
PAPER_LR = 5e-3

#: Fraction of subjects held out to score configurations, matching the paper's 80/20 split (p. 6).
VALIDATION_FRACTION = 0.2


def suggest_delta(features: np.ndarray, subjects: np.ndarray, holdout: set[str],
                  sim_max: str = "train") -> float:
    """Derives the Delta that removes the irreducible bias of the objective.

    The normalisation pins each row mean of ``D_ij`` at 1, so the target's row mean must also be 1
    for the loss to be about the shape of the geometry rather than a constant offset. That gives
    ``Delta* = 1 / E[1 - s_ij]``, computed on the training subjects only.

    The estimate depends on ``sim_max`` because a dataset-wide maximum makes ``s_ij`` much larger
    than a per-batch one, and therefore the target much smaller.

    Args:
        features: Raw descriptor matrix, one row per window.
        subjects: Subject identifier per window.
        holdout: Subjects excluded from the estimate.
        sim_max: Either "train" (dataset-wide maximum) or "batch".

    Returns:
        The suggested Delta.
    """
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
    """Splits subjects 80/20, as the paper does for hyperparameter selection.

    Args:
        subjects: Subject identifier per window.
        seed: Seed for the split.

    Returns:
        The training and validation subject lists.
    """
    unique = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_val = max(2, int(round(len(unique) * VALIDATION_FRACTION)))
    return sorted(shuffled[n_val:].tolist()), sorted(shuffled[:n_val].tolist())


def train_one(delta: float, lr: float, sim_max: str, epochs: int, holdout: list[str],
              seed: int) -> Path | None:
    """Pre-trains one configuration, excluding the validation subjects.

    Args:
        delta: Margin under test.
        lr: Learning rate under test.
        sim_max: Whether ``max_kl`` is taken over the training split or per batch.
        epochs: Training epochs.
        holdout: Validation subjects, excluded from pre-training.
        seed: Seed for reproducibility.

    Returns:
        Checkpoint path, or None if training failed.
    """
    fold_id = f"tune_d{delta}_lr{lr}_{sim_max}_e{epochs}_s{seed}"
    ckpt = Path("save/models") / (
        f"ExpCLR_all_all_{fold_id}_P_diverso_batch_64_lr_{lr}_tau_1.0_delta_{delta}.pth")
    if ckpt.exists():
        return ckpt
    cmd = [sys.executable, "src/train_expclr.py",
           "--data_path", str(DATA), "--expert_features", str(FEATURES),
           "--descriptor", "P_diverso", "--zone", "all", "--frequency", "all",
           "--fold_id", fold_id, "--num_epochs", str(epochs), "--seed", str(seed),
           "--delta", str(delta), "--lr", str(lr), "--loss_on", "embedding",
           "--sim_max", sim_max, "--diagnose_every", str(max(1, epochs // 4)),
           "--exclude_subjects", *holdout]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not ckpt.exists():
        print(f"  fallo delta={delta} lr={lr} sim_max={sim_max}: {res.stderr.strip()[-300:]}")
        return None
    return ckpt


def score(ckpt: Path | None, X, meta, y, groups, val_mask, device, random_seed: int = 42) -> dict:
    """Scores a checkpoint on the held-out subjects with the same probe used in the final run.

    Args:
        ckpt: Checkpoint to score, or None for a randomly initialised encoder.
        X: Windows.
        meta: Window metadata.
        y: Target per window.
        groups: Subject per window.
        val_mask: Boolean mask selecting the validation windows.
        device: Torch device.

    Returns:
        Dict with session-level metrics and the effective dimensionality.
    """
    if ckpt is None:
        # Same seed as B2 in run_e3_loso.py: the reference against which the method is declared to
        # fail must be the very encoder that appears in the final table. Across seeds the random
        # baseline moves by sd = 0.024 in R2, comparable to the effects being chased.
        torch.manual_seed(random_seed)
    model = EnhancedAttentionLSTM(input_size=X.shape[2], hidden_size=128, n_channels=X.shape[1],
                                  sfreq=250, lstm_hidden_size=64).to(device)
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
        description="Ajuste de Delta y lr para ExpCLR con el procedimiento del paper.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epocas por configuracion. Suficientes para ordenarlas, no para el "
                             "resultado final.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Solo el eje Delta, al lr del paper para SleepEDF/Waveform (5e-3).")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    X = np.asarray(np.load(DATA / "processed_windows.npy", mmap_mode="r"))
    meta = pd.read_csv(DATA / "processed_metadata.csv")
    F_raw = np.load(FEATURES)
    y = meta.age.values.astype(float)
    groups = meta.subject.values

    train_subjects, val_subjects = split_subjects(groups, args.seed)
    val_mask = np.isin(groups, val_subjects)
    print(f"device: {device} | {len(train_subjects)} sujetos de entrenamiento, "
          f"{len(val_subjects)} de validacion: {val_subjects}")

    suggested = {sm: suggest_delta(F_raw, groups, set(val_subjects), sm)
                 for sm in ("train", "batch")}
    for sm, value in suggested.items():
        print(f"Delta sugerido con --sim_max {sm}: {value:.2f} "
              f"(= 1 / E[1 - s_ij]; el sesgo con Delta=1 seria {abs(1 - 1 / value):.3f})")

    baseline = score(None, X, meta, y, groups, val_mask, device, args.seed)
    print(f"referencia, encoder aleatorio: MAE={baseline['mae']:.2f} R2={baseline['r2']:.3f} "
          f"dim_efectiva={baseline['effective_dim']:.2f}\n")

    rows = []

    def evaluate(delta, lr, sim_max, stage, index, total):
        """Trains and scores one configuration, recording it in `rows`.

        Configurations whose metric is not finite are dropped rather than recorded: a diverged run
        would otherwise be picked as the winner by a comparison against NaN.
        """
        t0 = time.time()
        ckpt = train_one(delta, lr, sim_max, args.epochs, val_subjects, args.seed)
        if ckpt is None:
            return None
        result = score(ckpt, X, meta, y, groups, val_mask, device, args.seed)
        if not np.isfinite(result["mae"]):
            print(f"  [{stage} {index:2d}/{total}] delta={delta:<5} lr={lr:<7} {sim_max:<5} "
                  f"DESCARTADA (metrica no finita, probable divergencia)", flush=True)
            return None
        rows.append({"stage": stage, "delta": delta, "lr": lr, "sim_max": sim_max,
                     "is_suggested": abs(delta - suggested[sim_max]) < 0.01, **result})
        print(f"  [{stage} {index:2d}/{total}] delta={delta:<5} lr={lr:<7} {sim_max:<5} "
              f"MAE={result['mae']:5.2f} R2={result['r2']:+.3f} "
              f"dim={result['effective_dim']:5.2f} ({time.time() - t0:.0f}s)", flush=True)
        return result

    # Etapa 1: Delta y sim_max al learning rate del paper. Delta es el eje que domina el problema,
    # y sim_max cambia la escala de las similitudes, así que se exploran juntos.
    stage1 = [(d, sm) for sm in ("train", "batch")
              for d in sorted(set(DELTA_GRID + [round(suggested[sm], 2)]))]
    print(f"Etapa 1: {len(stage1)} configuraciones de (Delta, sim_max) a lr={PAPER_LR}")
    for i, (delta, sim_max) in enumerate(stage1, 1):
        evaluate(delta, PAPER_LR, sim_max, "1", i, len(stage1))

    if not rows:
        raise RuntimeError("ninguna configuracion completo el entrenamiento")

    # Selección por MAE, la métrica primaria declarada en eval_expclr.py y la del contraste
    # pre-registrado. Con un target de cuatro niveles el R2 lo domina la visita de 36 meses.
    best1 = min(rows, key=lambda r: r["mae"])
    print(f"\nMejor de la etapa 1: delta={best1['delta']}, sim_max={best1['sim_max']}, "
          f"MAE={best1['mae']:.2f}")

    # Etapa 2: learning rate sobre la rejilla del paper, con el (Delta, sim_max) ganador.
    if not args.quick:
        lrs = [lr for lr in LR_GRID if lr != PAPER_LR]
        print(f"\nEtapa 2: {len(lrs)} learning rates con delta={best1['delta']}, "
              f"sim_max={best1['sim_max']}")
        for i, lr in enumerate(lrs, 1):
            evaluate(best1["delta"], lr, best1["sim_max"], "2", i, len(lrs))

    table = pd.DataFrame(rows).sort_values("mae")
    table.to_csv(OUT / "tuning_results.csv", index=False)
    best = table.iloc[0]

    print("\n=== mejores configuraciones (ordenadas por MAE) ===")
    print(table.head(10).round(3).to_string(index=False))
    print(f"\nreferencia aleatoria: MAE={baseline['mae']:.2f}, R2={baseline['r2']:.3f}, "
          f"dim_efectiva={baseline['effective_dim']:.2f}")
    print(f"ganadora: delta={best.delta}, lr={best.lr}, sim_max={best.sim_max}, "
          f"MAE={best.mae:.2f}, R2={best.r2:.3f}, dim_efectiva={best.effective_dim:.2f}")

    if best.mae >= baseline["mae"]:
        print("\nNINGUNA configuracion supera al encoder aleatorio. El resultado a reportar es "
              "que ExpCLR no transfiere en este dominio, no un fallo de ajuste.")
    print(f"\nAviso: la ganadora se elige entre {len(rows)} configuraciones sobre "
          f"{int(baseline['n_sessions'])} sesiones de {len(val_subjects)} sujetos, asi que su "
          f"metrica de validacion esta sesgada al alza por seleccion. La cifra reportable es la "
          f"del LOSO posterior sobre los sujetos restantes, no esta.")

    json.dump({"delta": float(best.delta), "lr": float(best.lr), "sim_max": str(best.sim_max),
               "delta_suggested": {k: float(v) for k, v in suggested.items()},
               "mae": float(best.mae), "r2": float(best.r2),
               "random_baseline_mae": float(baseline["mae"]),
               "random_baseline_r2": float(baseline["r2"]),
               "n_configs_compared": len(rows),
               "validation_subjects": val_subjects, "epochs": args.epochs, "seed": args.seed},
              open(OUT / "best_config.json", "w"), indent=2)
    print(f"\nGuardado en {OUT}/best_config.json. Lanzar despues:")
    print(f"  python -u run_e3_loso.py --epochs 50 --config {OUT}/best_config.json")


if __name__ == "__main__":
    main()
