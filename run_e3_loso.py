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
from train_expclr import checkpoint_is_reusable  # noqa: E402

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


def pretrain_fold(subject: str, epochs: int, features: Path, tag: str, seed: int = 42,
                  delta: float = 1.0, lr: float = 5e-3, sim_max: str = "train") -> Path:
    """Pre-trains one ExpCLR encoder excluding a subject, and returns its checkpoint path.

    The fold id carries the held-out subject, a tag, the epoch count and the seed, and the file
    name additionally carries the learning rate, tau and Delta, so checkpoints of different
    variants, budgets, seeds or hyperparameters can never collide. Checkpoint reuse is keyed on
    that path, so leaving any of them out would silently resume a sweep with encoders trained
    under different settings.

    The loss is applied to the embedding, not to the projection: ExpCLR has no projection head and
    optimises the same representation it evaluates.

    Args:
        subject: Subject held out for testing.
        epochs: Training epochs.
        features: Descriptor matrix to guide the loss.
        tag: Short label distinguishing the variant.
        seed: Seed for the encoder initialisation and the batch order.
        delta: Margin ``Delta`` of Eq. 3/4. Must match the scale of the descriptor similarities,
            see :func:`suggest_delta` in ``tune_expclr.py``.
        lr: Learning rate for Adam.
        sim_max: Whether ``max_kl ||f_k - f_l||`` is taken over the training split or per batch.
            It changes the scale of the similarities and therefore interacts with ``delta``.

    Returns:
        Path of the written checkpoint.

    Raises:
        RuntimeError: If the training subprocess fails or the checkpoint does not appear.
    """
    fold_id = f"loso_{tag}_e{epochs}_s{seed}_{sim_max}_{subject}"
    ckpt = Path("save/models") / (
        f"ExpCLR_all_all_{fold_id}_P_diverso_batch_64_lr_{lr}_tau_1.0_delta_{delta}.pth")
    # Reuse is decided against the recorded configuration, never against the path alone: settings
    # absent from the file name would otherwise be inherited silently from an earlier run.
    expected = {"delta": delta, "lr": lr, "loss_on": "embedding", "sim_max": sim_max,
                "num_epochs": epochs, "seed": seed, "exclude_subjects": [subject]}
    if checkpoint_is_reusable(ckpt, expected):
        return ckpt
    cmd = [sys.executable, "src/train_expclr.py",
           "--data_path", str(DATA), "--expert_features", str(features),
           "--descriptor", "P_diverso", "--zone", "all", "--frequency", "all",
           "--fold_id", fold_id, "--num_epochs", str(epochs), "--seed", str(seed),
           "--delta", str(delta), "--lr", str(lr), "--loss_on", "embedding",
           "--sim_max", sim_max, "--exclude_subjects", subject]
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


class EmbeddingCache:
    """Extracts frozen embeddings once per encoder and reuses them across methods and folds.

    Two savings, both exact rather than approximate:

    1. Within a fold, ExpCLR and B7 read the same pre-trained encoder, so one forward pass serves
       both instead of two identical ones.
    2. Across folds, B2's encoder is randomly initialised from a fixed seed and never trained, so
       its embeddings are the same in every fold and are computed once for the whole sweep.

    Caching is keyed on the checkpoint path, which encodes the variant, the held-out subject and
    the epoch budget, so two different encoders can never share an entry. Each matrix is a few
    megabytes, so nothing is evicted.
    """

    def __init__(self, X: np.ndarray, device, random_seed: int = 42) -> None:
        """Initialises an empty cache over a fixed set of windows.

        Args:
            X: Window tensor. Wrapped with ``np.asarray`` once; on a memmap this is a view, so it
                costs no memory and avoids re-wrapping it on every extraction.
            device: Torch device.
            random_seed: Seed of B2's untrained encoder. It is part of the cache key, so encoders
                drawn from different seeds cannot share an entry.
        """
        self._X = np.asarray(X)
        self._device = device
        self._random_seed = random_seed
        self._embeddings: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def get(self, checkpoint: Path | None) -> np.ndarray:
        """Returns the embeddings of one encoder, computing them only the first time.

        Args:
            checkpoint: Weights to load, or None for the fixed-seed random encoder of B2.

        Returns:
            Embedding matrix of shape (n_windows, embedding_dim).
        """
        key = str(checkpoint) if checkpoint is not None else f"__random_s{self._random_seed}__"
        cached = self._embeddings.get(key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        model = build_encoder(self._X, self._device, checkpoint, seed=self._random_seed)
        self._embeddings[key] = extract_embeddings(model, self._X, self._device)
        return self._embeddings[key]

    @property
    def stats(self) -> dict[str, int]:
        """Returns how many extractions were reused versus actually computed."""
        return {"hits": self._hits, "misses": self._misses}


def run_loso(methods: list[str], epochs: int, device, seed: int = 42, delta: float = 1.0,
             lr: float = 5e-3, sim_max: str = "train",
             exclude_subjects: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Runs leave-one-subject-out for every requested method, sharing folds and probe.

    Args:
        methods: Method identifiers to evaluate.
        epochs: Pre-training epochs for the methods that need an encoder.
        device: Torch device.
        seed: Seed shared by every fold, so what varies across folds is the held-out subject and
            not the initialisation.
        delta: Margin ``Delta``, to be taken from the tuning run rather than copied from the paper.
        lr: Learning rate, likewise.
        sim_max: Where ``max_kl`` is taken, likewise.
        exclude_subjects: Subjects dropped entirely, meant for the ones whose labels were seen while
            selecting the hyperparameters. Keeping them would make the out-of-fold error optimistic
            with respect to that selection.

    Returns:
        Mapping from method name to its out-of-fold session-level predictions.
    """
    X, meta, F = load_data()
    if exclude_subjects:
        keep = ~meta.subject.isin(exclude_subjects).values
        meta = meta[keep].reset_index(drop=True)
        X, F = X[keep], F[keep]
        print(f"Excluidos {len(exclude_subjects)} sujetos usados para ajustar hiperparametros: "
              f"{sorted(exclude_subjects)}")
    subjects = sorted(meta.subject.unique())
    y = meta.age.values.astype(float)
    groups = meta.subject.values
    print(f"LOSO sobre {len(subjects)} sujetos | {len(meta)} ventanas | descriptor {F.shape}")

    shuffled_path = FEATURES.with_name(f"expert_features_P_diverso_shuffled_s{seed}.npy")
    if "B3" in methods:
        # B3: same descriptor, permuted across windows. Isolates the geometry from the loss shape.
        # Regenerated every run so the seed actually reaches the permutation.
        rng = np.random.default_rng(seed)
        np.save(shuffled_path, F[rng.permutation(len(F))])

    cache = EmbeddingCache(X, device, random_seed=seed)
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
                    emb = cache.get(pretrain_fold(subject, epochs, FEATURES, "expclr", seed,
                                                  delta, lr, sim_max))
                    Xtr = np.hstack([emb[train], F[train]])
                    Xte = np.hstack([emb[test], F[test]])
                scaler, probe = fit_probe(Xtr, y[train], groups[train])
                pred_test = probe.predict(scaler.transform(Xte))
            else:                                   # métodos con encoder
                if method == "ExpCLR":
                    ckpt = pretrain_fold(subject, epochs, FEATURES, "expclr", seed, delta, lr,
                                         sim_max)
                elif method == "B3":
                    ckpt = pretrain_fold(subject, epochs, shuffled_path, "shuffled", seed,
                                         delta, lr, sim_max)
                elif method == "B2":
                    ckpt = None                     # encoder aleatorio
                else:
                    raise ValueError(f"metodo desconocido: {method}")
                emb = cache.get(ckpt)
                scaler, probe = fit_probe(emb[train], y[train], groups[train])
                pred_test = probe.predict(scaler.transform(emb[test]))

            preds[method].append(aggregate_to_sessions(meta[test], y[test], pred_test))

        print(f"  [{i:2d}/{len(subjects)}] {subject}  {time.time()-t0:5.1f}s", flush=True)

    stats = cache.stats
    print(f"embeddings: {stats['misses']} extracciones, {stats['hits']} reutilizadas", flush=True)
    return {m: pd.concat(v, ignore_index=True) for m, v in preds.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: ExpCLR con P_diverso bajo LOSO.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epocas de preentrenamiento; la perdida converge hacia la 50.")
    parser.add_argument("--methods", nargs="+",
                        default=["ExpCLR", "B0", "B1", "B2", "B7"],
                        help="ExpCLR, B0 media, B1 Ridge sobre descriptor, B2 encoder aleatorio, "
                             "B3 descriptor permutado, B7 embedding+descriptor.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla compartida por los 45 folds: lo que varia entre folds es el "
                             "sujeto excluido, no la inicializacion.")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Margen Delta. El 1.0 del paper solo es valido si la media de "
                             "(1 - s_ij) del descriptor vale 1; con P_diverso no es asi. "
                             "Tomar el valor de tune_expclr.py.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate de Adam. El paper lo optimiza por dataset (Tab. 9), "
                             "asi que tomarlo tambien de tune_expclr.py.")
    parser.add_argument("--sim_max", choices=["train", "batch"], default="train",
                        help="Donde se toma max_kl ||f_k - f_l||. El paper permite ambas "
                             "(Sec. 3.4) y cambia la escala de s_ij, luego interactua con Delta. "
                             "Tomarlo de tune_expclr.py.")
    parser.add_argument("--exclude_subjects", nargs="*", default=[],
                        help="Sujetos que se excluyen del LOSO por haberse usado para ajustar "
                             "hiperparametros. Ver validation_subjects en best_config.json.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Lee delta, lr, sim_max y los sujetos de validacion del "
                             "best_config.json que escribe tune_expclr.py, en vez de a mano.")
    args = parser.parse_args()

    if args.config:
        # Reading the tuning output wholesale avoids transcribing four settings by hand, which is
        # how a tuned sim_max silently failed to reach the final run before.
        cfg = json.loads(args.config.read_text())
        args.delta, args.lr = cfg["delta"], cfg["lr"]
        args.sim_max = cfg.get("sim_max", args.sim_max)
        args.exclude_subjects = cfg["validation_subjects"]
        print(f"config de {args.config}: delta={args.delta} lr={args.lr} "
              f"sim_max={args.sim_max} | {len(args.exclude_subjects)} sujetos excluidos")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | epocas: {args.epochs} | metodos: {args.methods}")

    results = run_loso(args.methods, args.epochs, device, args.seed, args.delta, args.lr,
                       args.sim_max, args.exclude_subjects)

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
