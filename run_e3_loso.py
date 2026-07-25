"""Runs experiment E3 (ExpCLR with P_diverso) end to end under leave-one-subject-out.

Uses its own evaluation path (``src/eval_expclr.py``) rather than ``downstream.py``, because the
shared one does not freeze the encoder, uses a non-standard probe, aggregates predictions to the
subject (averaging targets across visits months apart) and averages per-fold metrics. See that
module's docstring for the details.

Every method here is evaluated identically: same folds, same probe, same aggregation, same metric.
That is what makes the comparison against the baselines informative rather than decorative.

Usage:
    python run_e3_loso.py --epochs 50 --methods ExpCLR B0 B1 B2 B7
"""
from __future__ import annotations

import argparse
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
    aggregate_to_sessions, bootstrap_ci, extract_embeddings, fit_probe,
    metrics_by_visit, paired_bootstrap_difference, session_metrics,
)
from models import EnhancedAttentionLSTM  # noqa: E402

DATA = Path("data/processed/all_all")
FEATURES = Path("data/processed/expert_features/expert_features_P_diverso.npy")
OUT = Path("save/e3_results")


def load_data() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Loads windows, metadata and the expert descriptor, checking they are aligned."""
    X = np.load(DATA / "processed_windows.npy", mmap_mode="r")
    meta = pd.read_csv(DATA / "processed_metadata.csv")
    F = np.load(FEATURES)
    if not (len(X) == len(meta) == len(F)):
        raise ValueError(f"desalineados: X={len(X)}, meta={len(meta)}, F={len(F)}")
    return X, meta, F


def pretrain_fold(subject: str, epochs: int, features: Path, tag: str) -> Path:
    """Pre-trains one ExpCLR encoder excluding a subject, and returns its checkpoint path.

    The fold id carries the held-out subject and a tag, so checkpoints of different variants or
    targets can never collide, which is a defect the shared orchestrator has.

    Args:
        subject: Subject held out for testing.
        epochs: Training epochs.
        features: Descriptor matrix to guide the loss.
        tag: Short label distinguishing the variant.

    Returns:
        Path of the written checkpoint.

    Raises:
        RuntimeError: If the training subprocess fails or the checkpoint does not appear.
    """
    fold_id = f"loso_{tag}_{subject}"
    ckpt = Path("save/models") / (
        f"ExpCLR_all_all_{fold_id}_P_diverso_batch_64_lr_0.005_tau_1.0_delta_1.0.pth")
    if ckpt.exists():
        return ckpt
    cmd = [sys.executable, "src/train_expclr.py",
           "--data_path", str(DATA), "--expert_features", str(features),
           "--descriptor", "P_diverso", "--zone", "all", "--frequency", "all",
           "--fold_id", fold_id, "--num_epochs", str(epochs),
           "--exclude_subjects", subject]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not ckpt.exists():
        raise RuntimeError(f"fallo el preentrenamiento de {subject}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return ckpt


def build_encoder(X: np.ndarray, device, checkpoint: Path | None, seed: int = 0):
    """Instantiates the encoder, optionally loading pre-trained weights.

    With ``checkpoint=None`` this is baseline B2: the same architecture with random weights, which
    tests whether the merit lies in the pre-training or merely in a random non-linear projection.

    Args:
        X: Window tensor, used only for its shape.
        device: Torch device.
        checkpoint: Weights to load, or None for a randomly initialised encoder.
        seed: Seed used when the encoder is random, so B2 is reproducible.

    Returns:
        The encoder, on the requested device.
    """
    if checkpoint is None:
        torch.manual_seed(seed)
    model = EnhancedAttentionLSTM(input_size=X.shape[2], hidden_size=128, n_channels=X.shape[1],
                                  sfreq=250, lstm_hidden_size=64).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def run_loso(methods: list[str], epochs: int, device) -> dict[str, pd.DataFrame]:
    """Runs leave-one-subject-out for every requested method, sharing folds and probe.

    Args:
        methods: Method identifiers to evaluate.
        epochs: Pre-training epochs for the methods that need an encoder.
        device: Torch device.

    Returns:
        Mapping from method name to its out-of-fold session-level predictions.
    """
    X, meta, F = load_data()
    subjects = sorted(meta.subject.unique())
    y = meta.age.values.astype(float)
    groups = meta.subject.values
    print(f"LOSO sobre {len(subjects)} sujetos | {len(meta)} ventanas | descriptor {F.shape}")

    shuffled_path = FEATURES.with_name("expert_features_P_diverso_shuffled.npy")
    if "B3" in methods and not shuffled_path.exists():
        # B3: same descriptor, permuted across windows. Isolates the geometry from the loss shape.
        rng = np.random.default_rng(0)
        np.save(shuffled_path, F[rng.permutation(len(F))])

    preds: dict[str, list[pd.DataFrame]] = {m: [] for m in methods}
    for i, subject in enumerate(subjects, 1):
        t0 = time.time()
        test = meta.subject.values == subject
        train = ~test
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue

        for method in methods:
            if method == "B0":                      # media del train
                pred_test = np.full(test.sum(), y[train].mean())
            elif method in ("B1", "B7"):            # Ridge sobre descriptor (+ embedding en B7)
                if method == "B1":
                    Xtr, Xte = F[train], F[test]
                else:
                    ckpt = pretrain_fold(subject, epochs, FEATURES, "expclr")
                    emb = extract_embeddings(build_encoder(X, device, ckpt), np.asarray(X), device)
                    Xtr = np.hstack([emb[train], F[train]])
                    Xte = np.hstack([emb[test], F[test]])
                scaler, probe = fit_probe(Xtr, y[train], groups[train])
                pred_test = probe.predict(scaler.transform(Xte))
            else:                                   # métodos con encoder
                if method == "ExpCLR":
                    ckpt = pretrain_fold(subject, epochs, FEATURES, "expclr")
                elif method == "B3":
                    ckpt = pretrain_fold(subject, epochs, shuffled_path, "shuffled")
                elif method == "B2":
                    ckpt = None                     # encoder aleatorio
                else:
                    raise ValueError(f"metodo desconocido: {method}")
                emb = extract_embeddings(build_encoder(X, device, ckpt), np.asarray(X), device)
                scaler, probe = fit_probe(emb[train], y[train], groups[train])
                pred_test = probe.predict(scaler.transform(emb[test]))

            preds[method].append(aggregate_to_sessions(meta[test], y[test], pred_test))

        print(f"  [{i:2d}/{len(subjects)}] {subject}  {time.time()-t0:5.1f}s", flush=True)

    return {m: pd.concat(v, ignore_index=True) for m, v in preds.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: ExpCLR con P_diverso bajo LOSO.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epocas de preentrenamiento; la perdida converge hacia la 50.")
    parser.add_argument("--methods", nargs="+",
                        default=["ExpCLR", "B0", "B1", "B2", "B7"],
                        help="ExpCLR, B0 media, B1 Ridge sobre descriptor, B2 encoder aleatorio, "
                             "B3 descriptor permutado, B7 embedding+descriptor.")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | epocas: {args.epochs} | metodos: {args.methods}")

    results = run_loso(args.methods, args.epochs, device)

    rows = []
    for method, sessions in results.items():
        sessions.to_csv(OUT / f"predicciones_{method}.csv", index=False)
        m = session_metrics(sessions)
        low, high = bootstrap_ci(sessions, "mae")
        rows.append({"metodo": method, **m, "mae_ci_low": low, "mae_ci_high": high})
    summary = pd.DataFrame(rows).sort_values("mae")
    summary.to_csv(OUT / "resumen_metodos.csv", index=False)

    print("\n=== resultados a nivel sesion (out-of-fold) ===")
    print(summary.round(3).to_string(index=False))

    if "ExpCLR" in results and "B1" in results:
        # Contraste primario, pre-registrado: el test de falsabilidad del proyecto.
        diff = paired_bootstrap_difference(results["ExpCLR"], results["B1"], "mae")
        json.dump(diff, open(OUT / "contraste_primario.json", "w"), indent=2)
        print("\ncontraste primario ExpCLR - B1 (Ridge sobre descriptor), MAE en meses:")
        print(f"  diferencia {diff['diff']:+.3f}  IC95 [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]"
              f"  sobre {diff['n_paired_sessions']} sesiones")
        print("  (negativo = ExpCLR mejor; si el IC cruza 0, no hay evidencia de diferencia)")

    if "ExpCLR" in results:
        visits = metrics_by_visit(results["ExpCLR"])
        visits.to_csv(OUT / "expclr_por_visita.csv", index=False)
        print("\nExpCLR, error por visita (el target solo tiene 4 niveles):")
        print(visits.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
