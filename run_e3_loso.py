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
from checkpoint_naming import expclr_checkpoint_name
from train_expclr import checkpoint_is_reusable

DATA = Path("data/processed/all_all")
FEATURES = Path("data/processed/expert_features/expert_features_P_diverso.npy")
OUT = Path("save/e3_results")


def load_data() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    X = np.load(DATA / "processed_windows.npy", mmap_mode="r")
    meta = pd.read_csv(DATA / "processed_metadata.csv")
    F = np.load(FEATURES)
    if not (len(X) == len(meta) == len(F)):
        raise ValueError(f"desalineados: X={len(X)}, meta={len(meta)}, F={len(F)}")
    return X, meta, F


def pretrain_fold(subject: str, epochs: int, features: Path, tag: str, seed: int = 42,
                  delta: float = 1.0, lr: float = 5e-3, sim_max: str = "train",
                  tau: float = 1.0, tuning_excluded: list[str] | None = None) -> Path:
    # The fold encoder must not see the hyperparameter-tuning subjects either;
    # a short hash of the full exclusion list keeps configurations with
    # different tuning sets from sharing (and overwriting) checkpoint files.
    excl = sorted({subject, *(tuning_excluded or [])})
    excl_tag = ""
    if len(excl) > 1:
        excl_tag = "_x" + hashlib.md5("_".join(excl).encode()).hexdigest()[:6]
    fold_id = f"loso_{tag}_e{epochs}_s{seed}_{sim_max}{excl_tag}_{subject}"
    ckpt = Path("save/models") / expclr_checkpoint_name(
        "all", "all", fold_id, "P_diverso",
        batch_size=64, lr=lr, temperature=tau, delta=delta)
    expected = {"delta": delta, "lr": lr, "temperature": tau, "loss_on": "projection",
                "sim_max": sim_max, "dropout": 0.0, "num_epochs": epochs, "seed": seed,
                "exclude_subjects": excl}
    if checkpoint_is_reusable(ckpt, expected):
        return ckpt
    cmd = [sys.executable, "src/train_expclr.py",
           "--data_path", str(DATA), "--expert_features", str(features),
           "--descriptor", "P_diverso", "--zone", "all", "--frequency", "all",
           "--fold_id", fold_id, "--num_epochs", str(epochs), "--seed", str(seed),
           "--delta", str(delta), "--lr", str(lr), "--temperature", str(tau),
           "--loss_on", "projection", "--dropout", "0.0", "--sim_max", sim_max,
           "--exclude_subjects", *excl]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not ckpt.exists():
        raise RuntimeError(f"fallo el preentrenamiento de {subject}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
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

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}


def run_loso(methods: list[str], epochs: int, device, seed: int = 42, delta: float = 1.0,
             lr: float = 5e-3, sim_max: str = "train",
             exclude_subjects: list[str] | None = None,
             tau: float = 1.0) -> dict[str, pd.DataFrame]:
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

    # B3 permutes the FULL on-disk descriptor so the subprocess (which reloads
    # the complete dataset) stays row-aligned; the artifact is derived data and
    # lives under save/, not next to the real descriptors.
    shuffled_path = OUT / f"expert_features_P_diverso_shuffled_s{seed}.npy"
    if "B3" in methods:
        rng = np.random.default_rng(seed)
        F_full = np.load(FEATURES)
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(shuffled_path, F_full[rng.permutation(len(F_full))])

    cache = EmbeddingCache(X, device, random_seed=seed)
    preds: dict[str, list[pd.DataFrame]] = {m: [] for m in methods}
    for i, subject in enumerate(subjects, 1):
        t0 = time.time()
        test = meta.subject.values == subject
        train = ~test
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue

        for method in methods:
            if method == "B0":
                pred_test = np.full(test.sum(), y[train].mean())
            elif method in ("B1", "B7"):
                if method == "B1":
                    Xtr, Xte = F[train], F[test]
                else:
                    emb = cache.get(pretrain_fold(subject, epochs, FEATURES, "expclr", seed,
                                                  delta, lr, sim_max, tau,
                                                  tuning_excluded=exclude_subjects))
                    Xtr = np.hstack([emb[train], F[train]])
                    Xte = np.hstack([emb[test], F[test]])
                scaler, probe = fit_probe(Xtr, y[train], groups[train])
                pred_test = probe.predict(scaler.transform(Xte))
            else:
                if method == "ExpCLR":
                    ckpt = pretrain_fold(subject, epochs, FEATURES, "expclr", seed, delta, lr,
                                         sim_max, tau, tuning_excluded=exclude_subjects)
                elif method == "B3":
                    ckpt = pretrain_fold(subject, epochs, shuffled_path, "shuffled", seed,
                                         delta, lr, sim_max, tau,
                                         tuning_excluded=exclude_subjects)
                elif method == "B2":
                    ckpt = None
                else:
                    raise ValueError(f"metodo desconocido: {method}")
                emb = cache.get(ckpt)
                scaler, probe = fit_probe(emb[train], y[train], groups[train])
                pred_test = probe.predict(scaler.transform(emb[test]))
                knn_scaler, knn = fit_knn_probe(emb[train], y[train])
                knn_pred = knn.predict(knn_scaler.transform(emb[test]))
                preds.setdefault(f"{method}_KNN", []).append(
                    aggregate_to_sessions(meta[test], y[test], knn_pred))

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
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Temperatura del minado de negativos duros. El paper declara que su "
                             "propio 1.0 no es optimo (App. B.4: 'other tau < 1.0 could improve "
                             "the performance slightly'), asi que tomarlo de tune_expclr.py.")
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
        cfg = json.loads(args.config.read_text())
        args.delta, args.lr = cfg["delta"], cfg["lr"]
        args.sim_max = cfg.get("sim_max", args.sim_max)
        args.tau = cfg.get("tau", args.tau)
        args.exclude_subjects = cfg["validation_subjects"]
        print(f"config de {args.config}: delta={args.delta} lr={args.lr} tau={args.tau} "
              f"sim_max={args.sim_max} | {len(args.exclude_subjects)} sujetos excluidos")

    if not args.exclude_subjects:
        print("[AVISO] LOSO sin --config ni --exclude_subjects: los sujetos usados para "
              "ajustar hiperparametros entraran en la evaluacion (sesgo optimista).")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | epocas: {args.epochs} | metodos: {args.methods}")

    results = run_loso(args.methods, args.epochs, device, args.seed, args.delta, args.lr,
                       args.sim_max, args.exclude_subjects, args.tau)

    rows = []
    for method, sessions in results.items():
        sessions.to_csv(OUT / f"predicciones_{method}.csv", index=False)
        m = session_metrics(sessions)
        low, high = bootstrap_ci(sessions, "mae")
        rows.append({"metodo": method, **m, "mae_ci_low": low, "mae_ci_high": high,
                     **subject_metrics(sessions)})
    summary = pd.DataFrame(rows).sort_values("mae")
    summary.to_csv(OUT / "resumen_metodos.csv", index=False)

    session_cols = ["metodo", "n_sessions", "mae", "mae_ci_low", "mae_ci_high", "r2", "rho"]
    print("\n=== nivel sesion, out-of-fold (cifra principal) ===")
    print(summary[session_cols].round(3).to_string(index=False))
    print("\n=== nivel sujeto, formula de master_table.py (comparable con E0-E2) ===")
    print(summary[["metodo", "nrmse_subject", "rmse_subject", "r2_subject"]]
          .round(3).to_string(index=False))
    print("  (promedia las visitas de cada nino, por eso no es la cifra principal)")

    gaps = [(m, summary.loc[summary.metodo == m, "mae"].iloc[0],
             summary.loc[summary.metodo == f"{m}_KNN", "mae"].iloc[0])
            for m in results if f"{m}_KNN" in results.keys()]
    if gaps:
        print("\n=== diferencia lineal vs KNN(k=1), MAE en meses ===")
        for method, linear, knn in sorted(gaps, key=lambda g: g[1]):
            print(f"  {method:10s} lineal {linear:5.2f} | KNN {knn:5.2f} | diferencia {knn-linear:+5.2f}")

    if "ExpCLR" in results and "B1" in results:
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
