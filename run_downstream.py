import json
import os
import subprocess
import re
import argparse
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from checkpoint_naming import (
    ae_checkpoint_name,
    checkpoint_is_reusable,
    expclr_checkpoint_name,
    mae_checkpoint_name,
    simclr_checkpoint_name,
    triplet_checkpoint_name,
    vae_checkpoint_name,
    interfusion_checkpoint_name,
)
from build_expert_features import DESCRIPTORS, descriptor_path
from train_vae import BETA_CONVENTION
from montage import MONTAGE_SIDECAR, load_channel_names, resolve_processed_montage

# Configuration of experiments
# Methods to run for each evaluation mode
LINEAR_PROBE_METHODS = ["PCA", "SimCLR", "AE", "MAE", "TripletLoss", "VAE", "InterFusion",
                        "ExpCLR"]
FINE_TUNING_METHODS = ["supervised", "SimCLR", "AE", "MAE", "TripletLoss", "VAE", "InterFusion",
                       "ExpCLR"]

# The descriptor the contrastive loss compares against. The narrow one is used because it
# is the dense one: the alpha peak of the specparam fit is undefined wherever the fit finds
# no peak, and the loss compares every pair of the batch, so a window described by imputed
# values still occupies a row of the geometry.
EXPCLR_DESCRIPTOR = "P_madurativo"
EXPCLR_BATCH_SIZE = 64
EXPCLR_LR = 0.005
EXPCLR_TAU = 1.0
EXPCLR_DELTA = 1.0
EXPCLR_SIM_MAX = "train"
EXPCLR_EPOCHS = 100
EXPCLR_LR_GAMMA = 0.99
EXPCLR_EMBEDDING_SIZE = 128
# Folds differ in which subject is held out, not in how the encoder starts, so the seed is
# held fixed across them and the fold-to-fold spread measures only the split.
EXPCLR_SEED = 42
EXPCLR_FEATURES = descriptor_path(EXPCLR_DESCRIPTOR)

EXPCLR_VARIANTS = {
    "ExpCLR": {"descriptor": EXPCLR_DESCRIPTOR, "features": EXPCLR_FEATURES},
}


def load_expclr_tuning(config_dir):
    """Replaces the paper's defaults with a tuned configuration, where one exists.

    Args:
        config_dir (str): Directory holding one ``best_config_<descriptor>.json`` per
            descriptor.

    Raises:
        ValueError: If a configuration records that nothing beat the random encoder, in
            which case running the folds would only reproduce that result.
    """
    for name, variant in EXPCLR_VARIANTS.items():
        path = os.path.join(config_dir, f"best_config_{variant['descriptor']}.json")
        if not os.path.exists(path):
            print(f"[WARNING] No tuned config for {name} at {path}; using the paper's defaults "
                  f"(delta={EXPCLR_DELTA}, lr={EXPCLR_LR}, tau={EXPCLR_TAU})")
            continue
        with open(path) as fh:
            cfg = json.load(fh)
        if cfg.get("beats_random_baseline") is False:
            raise ValueError(
                f"{path} records that no configuration beat the random encoder for "
                f"{variant['descriptor']}. Re-running the folds would only reproduce that."
            )
        variant.update(delta=cfg["delta"], lr=cfg["lr"], tau=cfg["tau"], sim_max=cfg["sim_max"])
        print(f"  {name}: delta={cfg['delta']} lr={cfg['lr']} tau={cfg['tau']} "
              f"sim_max={cfg['sim_max']} (MAE {cfg['mae']:.2f} vs "
              f"{cfg['random_baseline_mae']:.2f} for the random encoder)")


def expclr_hparams(method):
    """Returns the hyperparameters of an ExpCLR variant.

    Args:
        method (str): Variant name.

    Returns:
        tuple: (delta, learning rate, temperature, where the maximum distance is taken).
    """
    v = EXPCLR_VARIANTS[method]
    return (v.get("delta", EXPCLR_DELTA), v.get("lr", EXPCLR_LR),
            v.get("tau", EXPCLR_TAU), v.get("sim_max", EXPCLR_SIM_MAX))

# Neighbour strategies and the directory each one lives in. A zone other than the
# whole head reads the index built from that zone's channels: the positive of a
# frontal encoder is the window a frontal montage calls nearest, not the one the
# whole head does. `lag0` is the same-session index built with exclude_lag=0, so
# the window next to the anchor is eligible: an upper bound on how much the
# temporal-contiguity shortcut alone can buy.
_NIDX_BASE = {
    "nbr":     "neighbor_index",
    "xsubj":   "neighbor_index_crosssubj",
    "diffage": "neighbor_index_diffage",
    "lag0":    "neighbor_index_lag0",
}

WHOLE_HEAD_ZONE = "all"

_METRIC_OF_SUFFIX = {"cosine": "cosine", "wasser": "wasserstein", "riemann": "riemann"}
_TAG_OF_STRATEGY = {"nbr": "nbr", "xsubj": "xs", "diffage": "da", "lag0": "lag0"}

SIMCLR_VARIANT_NAMES = ("SimCLR",) + tuple(
    f"SimCLR-{strategy}-{suffix}"
    for strategy in _NIDX_BASE
    for suffix in _METRIC_OF_SUFFIX
)


def neighbor_index_dir(strategy, zone):
    """Returns the index directory of a neighbour strategy for one zone.

    Args:
        strategy (str): Key of :data:`_NIDX_BASE`.
        zone (str): Head zone of the run; the whole head keeps the plain name.

    Returns:
        str: Path of the index directory.
    """
    base = os.path.join("data", "processed", _NIDX_BASE[strategy])
    return base if zone == WHOLE_HEAD_ZONE else f"{base}_{zone}"


def build_simclr_variants(zone):
    """Builds the SimCLR variant table for one zone.

    The neighbour index is resolved per zone, so a variant name means "nearest
    window according to this zone" in every run. The tag is not zone-dependent
    because the checkpoint name already carries the zone.

    Args:
        zone (str): Head zone of the run.

    Returns:
        dict: Variant name to its pre-training flags and checkpoint tag.
    """
    variants = {"SimCLR": {"flags": [], "tag": ""}}
    for strategy in _NIDX_BASE:
        index_dir = neighbor_index_dir(strategy, zone)
        for suffix, metric in _METRIC_OF_SUFFIX.items():
            variants[f"SimCLR-{strategy}-{suffix}"] = {
                "flags": ["--positives", "neighbor",
                          "--neighbor_metric", metric,
                          "--neighbor_index_dir", index_dir],
                "tag": f"{_TAG_OF_STRATEGY[strategy]}{suffix}",
            }
    return variants


def check_neighbor_indices(variants, selected, zone, data_path):
    """Fails before training if a neighbour index is missing or foreign.

    The index records the montage it was built from, so pairing windows by the
    wrong electrodes is caught here instead of after hours of pre-training.

    Args:
        variants (dict): Output of :func:`build_simclr_variants`.
        selected (list): Variant names about to run.
        zone (str): Head zone of the run.
        data_path (str): Windows array the run evaluates.

    Raises:
        SystemExit: If an index is absent, predates the montage fix, or
            describes another montage.
    """
    needed = {spec["flags"][spec["flags"].index("--neighbor_index_dir") + 1]
              for name in selected
              for spec in [variants[name]]
              if "--neighbor_index_dir" in spec["flags"]}
    if not needed:
        return

    n_channels = np.load(data_path, mmap_mode="r").shape[1]
    expected = resolve_processed_montage(data_path, n_channels)
    for index_dir in sorted(needed):
        if not os.path.isdir(index_dir):
            raise SystemExit(
                f"[ERROR] Zone {zone!r} needs the neighbour index {index_dir}, which does "
                "not exist. Build it with src/build_neighbor_index.py on that zone.")
        sidecar = os.path.join(index_dir, MONTAGE_SIDECAR)
        if not os.path.exists(sidecar):
            raise SystemExit(
                f"[ERROR] {index_dir} records no montage, so it predates the montage fix "
                "and may pair windows by the wrong electrodes. Rebuild it.")
        if load_channel_names(sidecar) != expected:
            raise SystemExit(
                f"[ERROR] {index_dir} was built from another montage than {data_path}. "
                f"Rebuild the index for zone {zone!r}.")
        print(f"[INFO] Neighbour index for zone {zone!r}: {index_dir}", flush=True)

def parse_output(output):
    """
    Parses the stdout of downstream.py to extract metrics and subject-level predictions.
    Returns: (nrmse, r2, rmse, subject_avgs)
    where subject_avgs is a list of tuples: [(y_true_mean, y_pred_mean), ...]
    """
    # Search for metrics in "Subject-Avg" format
    nrmse_match = re.search(r"Test nRMSE \(Subject-Avg\)=([\d.]+)", output)
    r2_match = re.search(r"Test R2 \(Subject-Avg\)=([\d.-]+)", output)
    rmse_match = re.search(r"Test RMSE \(Subject-Avg\)=([\d.]+)", output)

    nrmse = float(nrmse_match.group(1)) if nrmse_match else None
    r2 = float(r2_match.group(1)) if r2_match else None
    rmse = float(rmse_match.group(1)) if rmse_match else None

    if nrmse is None or r2 is None or rmse is None:
        print("[WARNING] Could not parse metrics from output.")
        print(output)

    # Parse per-subject average predictions for global R² calculation
    # Format: SUBJECT_AVG_PRED: y_true_mean y_pred_mean
    subject_avgs = []
    for line in output.split('\n'):
        if line.startswith('SUBJECT_AVG_PRED:'):
            parts = line.split(': ')[1].split()
            if len(parts) == 2:
                subject_avgs.append((float(parts[0]), float(parts[1])))

    return nrmse, r2, rmse, subject_avgs

def config_data_paths(zone, frequency):
    """Resolves the processed-data paths for a zone/frequency configuration.

    Args:
        zone (str): Head zone (all, frontal, central, occipital, parietal).
        frequency (str): Frequency band (all, theta, alpha, beta, gamma).

    Returns:
        tuple: (windows .npy path, metadata .csv path) under data/processed/<zone>_<freq>/.
    """
    cfg_dir = os.path.join("data", "processed", f"{zone}_{frequency}")
    return (os.path.join(cfg_dir, "processed_windows.npy"),
            os.path.join(cfg_dir, "processed_metadata.csv"))


def run_pretraining(method, target, zone, frequency, test_subjects, fold_id, no_skip=False,
                    allow_legacy=False, seed=42, simclr_flags=None, simclr_tag="",
                    vae_beta=1.0, vae_free_bits=0.0):
    """
    Constructs and runs a single call to pretraining script, excluding test subjects.
    Returns the path to the trained model.

    Args:
        method: Pre-training method (SimCLR, AE, TripletLoss)
        target: Target for TripletLoss (ignored for SimCLR and AE)
        zone: Brain zone
        frequency: Frequency band
        test_subjects: List of subjects to exclude
        fold_id: Unique fold identifier for naming the model
        no_skip: If True, forces retraining even if the model exists
    """
    simclr_flags = list(simclr_flags or [])
    if method in ["PCA", "supervised"]:
        # These methods do not require pre-training
        return None

    # What varies across ExpCLR folds is the held-out subject, not the initialisation, so
    # the seed stays fixed and the spread between folds measures only the split.
    eff_seed = EXPCLR_SEED if method in EXPCLR_VARIANTS else seed

    exclude_sorted = sorted(str(s) for s in test_subjects)
    expected = {
        "zone": zone,
        "frequency": frequency,
        "exclude_subjects": exclude_sorted,
        "seed": eff_seed,
    }

    # Determine the model filename based on the method
    if method == "SimCLR":
        eff_fold_id = f"{fold_id}_{simclr_tag}" if simclr_tag else fold_id
        model_filename = simclr_checkpoint_name(zone, frequency, eff_fold_id)
        # The orchestrator never passes --aug_mode, and positives/metric come
        # from the variant flags; pin them so a manually trained checkpoint
        # with a different augmentation cannot be reused silently.
        is_neighbor = "--positives" in simclr_flags
        nbr_metric = (simclr_flags[simclr_flags.index("--neighbor_metric") + 1]
                      if "--neighbor_metric" in simclr_flags else None)
        # neighbor_index_dir is the only thing separating the nbr and lag0
        # variants under the same metric, so the gate has to compare it.
        nbr_dir = (simclr_flags[simclr_flags.index("--neighbor_index_dir") + 1]
                   if "--neighbor_index_dir" in simclr_flags else None)
        expected.update({
            "method": "SimCLR",
            "fold_id": eff_fold_id,
            "aug_mode": "legacy",
            "positives": "neighbor" if is_neighbor else "augment",
            "neighbor_metric": nbr_metric,
            "neighbor_index_dir": nbr_dir,
            # The epoch count is neither in the filename nor derivable from it, so
            # a shorter run would otherwise pass for a full one.
            "num_epochs": 100,
        })
    elif method == "AE":
        model_filename = ae_checkpoint_name(zone, frequency, fold_id)
        expected.update({"method": "AE", "fold_id": fold_id, "lr": 1e-3})
    elif method == "MAE":
        model_filename = mae_checkpoint_name(zone, frequency, fold_id)
        expected.update({"method": "MAE", "fold_id": fold_id, "lr": 1e-3})
    elif method == "TripletLoss":
        model_filename = triplet_checkpoint_name(target, zone, frequency, fold_id)
        expected.update({"method": "TripletLoss", "fold_id": fold_id, "target": target})
    elif method == "VAE":
        model_filename = vae_checkpoint_name(
            method, zone, frequency, fold_id,
            beta=vae_beta, prior="standard", free_bits=vae_free_bits,
        )
        expected.update({
            "method": method,
            "fold_id": fold_id,
            "beta": vae_beta,
            # Two checkpoints can carry the same beta under different objectives,
            # so the convention is part of what has to match to reuse one.
            "beta_convention": BETA_CONVENTION,
            "prior": "standard",
            "free_bits": vae_free_bits,
            "lr": 1e-3,
            # The code width changes the model but not its filename.
            "latent_dim": 128,
        })
        if allow_legacy:
            # Legacy VAE checkpoints share this filename pattern but their beta is
            # in the pre-rescaling scale: the same name, a different objective.
            print(f"  > [REUSE] {method}: legacy reuse disabled for the VAE "
                  f"(incompatible beta scale).")
            allow_legacy = False
    elif method == "InterFusion":
        model_filename = interfusion_checkpoint_name(zone, frequency, fold_id)
        # The filename does not encode the architecture, so two variants would
        # share it; pin the widths the sidecar records and downstream reads back.
        expected.update({
            "method": "InterFusion",
            "fold_id": fold_id,
            "z_dim": 4,
            "rnn_hidden": 500,
            "dense_hidden": 500,
            "flow_levels": 20,
            "epochs": 20,
            "pretrain_epochs": 20,
            "lr": 1e-3,
        })
    elif method in EXPCLR_VARIANTS:
        variant = EXPCLR_VARIANTS[method]
        delta, lr, tau, sim_max = expclr_hparams(method)
        model_filename = expclr_checkpoint_name(
            zone, frequency, fold_id, variant["descriptor"],
            batch_size=EXPCLR_BATCH_SIZE, lr=lr, temperature=tau, delta=delta,
        )
        expected.update({
            "method": "ExpCLR",
            "fold_id": fold_id,
            "descriptor": variant["descriptor"],
            "delta": delta,
            "lr": lr,
            "temperature": tau,
            # Neither of these reaches the filename, and both change the encoder: where
            # the largest descriptor distance is taken, and how long it trained.
            "sim_max": sim_max,
            "num_epochs": EXPCLR_EPOCHS,
            "lr_gamma": EXPCLR_LR_GAMMA,
            # Neither width reaches the filename, and both change what the weights mean.
            "embedding_size": EXPCLR_EMBEDDING_SIZE,
            "descriptor_dim": len(DESCRIPTORS[variant["descriptor"]]),
            # The loss shapes the encoder output, which is what downstream evaluates.
            "loss_on": "projection",
        })
    else:
        print(f"[WARNING] Pretraining not implemented for method: {method}")
        return None

    model_path = os.path.join("save/models", model_filename)

    # Reuse only a checkpoint whose sidecar matches the intended configuration
    if not no_skip and checkpoint_is_reusable(model_path, expected, allow_legacy=allow_legacy):
        print(f"  > Reusable model found: {model_filename}. Skipping pretraining.")
        return model_path

    # Build command based on the method
    if method == "SimCLR":
        command = [
            "python", "src/train_simclr.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", eff_fold_id,
            "--data_path", os.path.dirname(config_data_paths(zone, frequency)[0]),
        ] + simclr_flags + [
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]
    elif method == "AE":
        command = [
            "python", "src/train_auto.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]
    elif method == "MAE":
        command = [
            "python", "src/train_mae.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--mask-ratio", "0.3",
            "--block-size", "25",  # 100ms @ 250Hz
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]
    elif method == "InterFusion":
        command = [
            "python", "src/train_interfusion.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]
    elif method == "VAE":
        command = [
            "python", "src/train_vae.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]
    elif method == "TripletLoss":
        command = [
            "python", "src/train_triplet_loss.py",
            "--target", target,
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]
    elif method in EXPCLR_VARIANTS:
        variant = EXPCLR_VARIANTS[method]
        delta, lr, tau, sim_max = expclr_hparams(method)
        win_path, _ = config_data_paths(zone, frequency)
        command = [
            "python", "src/train_expclr.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--data_path", os.path.dirname(win_path),
            "--expert_features", variant["features"],
            "--descriptor", variant["descriptor"],
            "--batch_size", str(EXPCLR_BATCH_SIZE),
            "--num_epochs", str(EXPCLR_EPOCHS),
            "--lr", str(lr),
            "--temperature", str(tau),
            "--delta", str(delta),
            "--sim_max", sim_max,
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]

    if method in ("AE", "MAE", "VAE", "InterFusion"):
        win_path, meta_path = config_data_paths(zone, frequency)
        command += ["--data-path", win_path, "--meta-path", meta_path]
    if method == "VAE":
        command += ["--beta", str(vae_beta), "--free-bits", str(vae_free_bits)]

    command += ["--seed", str(eff_seed)]

    print(f"  > Running pretraining command: {' '.join(command)}")
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(f"  > Pretraining completed successfully")

        # The model was already saved with the correct name including fold_id
        model_path = os.path.join("save/models", model_filename)

        if os.path.exists(model_path):
            print(f"  > Model saved as: {model_filename}")
            return model_path
        else:
            print(f"[WARNING] Expected model not found at {model_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Pretraining command failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None

def run_downstream_experiment(method, eval_mode, target, zone, frequency, model_path, seed, test_subjects, fold_id=None, downstream_method=None):
    """
    Constructs and runs a single call to downstream.py.
    Returns: (nrmse, r2, rmse, subject_avgs)
    """
    win_path, meta_path = config_data_paths(zone, frequency)
    command = [
        "python", "src/downstream.py",
        "--method", downstream_method or method,
        "--target", target,
        "--eval_mode", eval_mode,
        "--zone", zone,
        "--frequency", frequency,
        "--seed", str(seed),
        "--data_path", win_path,
        "--meta_path", meta_path,
    ]

    # PCA/supervised have model_path=None
    if model_path is not None:
        command.extend(["--model_path", model_path])

    if test_subjects:
        command.extend(["--test_subjects"] + [str(s) for s in test_subjects])

    if fold_id:
        command.extend(["--fold_id", fold_id])

    print(f"  > Running downstream: {method} | {target} | {eval_mode}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Downstream command failed")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None, None, None, []
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return None, None, None, []


def execute_fold(fold_idx, train_subjects, test_subjects, args, targets, eval_modes, target_subject_dict):
    """
    Executes a complete fold:
    1. Pre-trains SimCLR and AE (once each)
    2. Pre-trains TripletLoss for each target
    3. Evaluates all methods with all combinations of target and eval_mode

    Args:
        target_subject_dict: Dictionary {target: [subjects with valid data for that target]}

    Returns:
        List of fold results
    """
    fold_results = []
    fold_id = f"fold{fold_idx}"
    seed = args.base_seed + fold_idx

    selected = set(args.methods) if getattr(args, "methods", None) else None
    def _use(m):
        return selected is None or m in selected

    simclr_variants = list(getattr(args, "simclr_variants", None) or ["SimCLR"]) if _use("SimCLR") else []
    def _expand(mlist):
        out = []
        for m in mlist:
            if m == "SimCLR":
                out.extend(simclr_variants)
            else:
                out.append(m)
        return out
    lp_methods = _expand([m for m in LINEAR_PROBE_METHODS if _use(m)])
    ft_methods = _expand([m for m in FINE_TUNING_METHODS if _use(m)])

    print(f"\n{'='*80}", flush=True)
    print(f"FOLD {fold_idx+1}: Train subjects ({len(train_subjects)}): {train_subjects}", flush=True)
    print(f"         Test subjects ({len(test_subjects)}): {test_subjects}", flush=True)
    print(f"{'='*80}", flush=True)

    # Dictionary to store pre-trained models
    pretrained_models = {}
    pretrain_seed = args.base_seed + fold_idx

    # ========================================================================
    # PHASE 1: PRE-TRAINING (considering dependencies)
    # ========================================================================
    print(f"\n[PHASE 1] Pre-training models for fold {fold_idx+1}...", flush=True)

    # 1.1 Pre-train each requested SimCLR variant (once, independent of target)
    variant_specs = build_simclr_variants(args.zone)
    for variant in simclr_variants:
        spec = variant_specs[variant]
        print(f"\n  Pre-training {variant} (self-supervised)...", flush=True)
        pretrained_models[variant] = run_pretraining(
            method="SimCLR",
            target=None,
            zone=args.zone,
            frequency=args.frequency,
            test_subjects=test_subjects,
            fold_id=fold_id,
            no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
            simclr_flags=spec["flags"], simclr_tag=spec["tag"],
        )

    # 1.2 Pre-train AE (once, independent of target)
    if _use("AE"):
        print(f"\n  Pre-training AE (self-supervised)...", flush=True)
        pretrained_models["AE"] = run_pretraining(
            method="AE",
            target=None,
            zone=args.zone,
            frequency=args.frequency,
            test_subjects=test_subjects,
            fold_id=fold_id,
            no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed
        )
    else:
        pretrained_models["AE"] = None

    # 1.3 Pre-train MAE (once, independent of target)
    if _use("MAE"):
        print(f"\n  Pre-training MAE (masked self-supervised)...", flush=True)
        pretrained_models["MAE"] = run_pretraining(
            method="MAE",
            target=None,
            zone=args.zone,
            frequency=args.frequency,
            test_subjects=test_subjects,
            fold_id=fold_id,
            no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed
        )
    else:
        pretrained_models["MAE"] = None

    # 1.4 Pre-train VAE (once, independent of target)
    if _use("VAE"):
        print(f"\n  Pre-training VAE (variational self-supervised, beta={args.vae_beta})...", flush=True)
        pretrained_models["VAE"] = run_pretraining(
            method="VAE",
            target=None,
            zone=args.zone,
            frequency=args.frequency,
            test_subjects=test_subjects,
            fold_id=fold_id,
            no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
            vae_beta=args.vae_beta, vae_free_bits=args.vae_free_bits,
        )
    else:
        pretrained_models["VAE"] = None

    # 1.5 Pre-train InterFusion (once, independent of target)
    if _use("InterFusion"):
        print(f"\n  Pre-training InterFusion (hierarchical HVAE)...", flush=True)
        pretrained_models["InterFusion"] = run_pretraining(
            method="InterFusion",
            target=None,
            zone=args.zone,
            frequency=args.frequency,
            test_subjects=test_subjects,
            fold_id=fold_id,
            no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
        )
    else:
        pretrained_models["InterFusion"] = None

    # 1.55 Pre-train ExpCLR (once, independent of target)
    for variant in EXPCLR_VARIANTS:
        if _use(variant):
            print(f"\n  Pre-training {variant} on the {EXPCLR_VARIANTS[variant]['descriptor']} "
                  f"descriptor...", flush=True)
            pretrained_models[variant] = run_pretraining(
                method=variant,
                target=None,
                zone=args.zone,
                frequency=args.frequency,
                test_subjects=test_subjects,
                fold_id=fold_id,
                no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
            )
        else:
            pretrained_models[variant] = None

    # 1.6 Pre-train TripletLoss for each target (depends on target)
    for target_idx, target in enumerate(targets):
        if not _use("TripletLoss"):
            pretrained_models[f"TripletLoss_{target}"] = None
            continue
        print(f"\n  Pre-training TripletLoss for target '{target}'...", flush=True)

        # Filter test_subjects to include only those with valid data for this target
        valid_test_subjects = [s for s in test_subjects if s in target_subject_dict[target]]

        if not valid_test_subjects:
            print(f"    [SKIP] No test subjects have valid data for target '{target}'")
            pretrained_models[f"TripletLoss_{target}"] = None
            continue

        triplet_model = run_pretraining(
            method="TripletLoss",
            target=target,
            zone=args.zone,
            frequency=args.frequency,
            test_subjects=valid_test_subjects,
            fold_id=fold_id,
            no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed
        )
        pretrained_models[f"TripletLoss_{target}"] = triplet_model

    # ========================================================================
    # PHASE 2: DOWNSTREAM EVALUATION (all combinations)
    # ========================================================================
    print(f"\n[PHASE 2] Downstream evaluation for fold {fold_idx+1}...", flush=True)

    experiment_counter = 0
    methods_by_mode = {"linear_probe": lp_methods, "fine_tuning": ft_methods}
    total_experiments = len(targets) * sum(
        len(methods_by_mode[mode]) for mode in eval_modes
    )

    for target in targets:
        # Filter train and test subjects for this specific target
        valid_train_subjects = [s for s in train_subjects if s in target_subject_dict[target]]
        valid_test_subjects = [s for s in test_subjects if s in target_subject_dict[target]]

        if not valid_test_subjects:
            print(f"\n[SKIP] No test subjects have valid data for target '{target}'. Skipping all experiments for this target.")
            continue

        print(f"\nTarget '{target}': {len(valid_train_subjects)} train subjects, {len(valid_test_subjects)} test subjects")

        for eval_mode in eval_modes:
            methods = methods_by_mode[eval_mode]

            for method in methods:
                experiment_counter += 1
                print(f"\n  [{experiment_counter}/{total_experiments}] Evaluating: {method} | {target} | {eval_mode}")

                # Determine which pre-trained model to use
                downstream_method = method
                if method in SIMCLR_VARIANT_NAMES:
                    model_path = pretrained_models.get(method)
                    downstream_method = "SimCLR"
                elif method in EXPCLR_VARIANTS:
                    model_path = pretrained_models.get(method)
                    downstream_method = "ExpCLR"
                elif method == "AE":
                    model_path = pretrained_models["AE"]
                elif method == "MAE":
                    model_path = pretrained_models["MAE"]
                elif method == "VAE":
                    model_path = pretrained_models["VAE"]
                elif method == "InterFusion":
                    model_path = pretrained_models["InterFusion"]
                elif method == "TripletLoss":
                    model_path = pretrained_models[f"TripletLoss_{target}"]
                else:  # PCA o supervised
                    model_path = None

                # Verify that the model exists (except for PCA and supervised)
                if method not in ["PCA", "supervised"] and model_path is None:
                    print(f"    [SKIP] Pre-trained model not available")
                    continue

                # Run downstream with the filtered subjects for this target
                nrmse, r2, rmse, subject_avgs = run_downstream_experiment(
                    method=method,
                    eval_mode=eval_mode,
                    target=target,
                    zone=args.zone,
                    frequency=args.frequency,
                    model_path=model_path,
                    seed=seed,
                    test_subjects=valid_test_subjects,
                    fold_id=fold_id,
                    downstream_method=downstream_method,
                )

                if nrmse is not None and r2 is not None and rmse is not None:
                    result_entry = {
                        "fold": fold_idx,
                        "method": method,
                        "eval_mode": eval_mode,
                        "target": target,
                        "nRMSE": nrmse,
                        "R2": r2,
                        "RMSE": rmse,
                        "subject_avgs": subject_avgs  # List of (y_true, y_pred) per-subject averages
                    }

                    fold_results.append(result_entry)
                    print(f"    ✓ nRMSE={nrmse:.4f}, R2={r2:.4f}, RMSE={rmse:.2f}", flush=True)
                else:
                    print(f"    [ERROR] Failed to get metrics")

    print(f"\n{'='*80}", flush=True)
    print(f"FOLD {fold_idx+1} COMPLETED: Collected {len(fold_results)} results", flush=True)
    print(f"{'='*80}", flush=True)

    return fold_results


def parse_fold_ranges(fold_ranges_arg):
    """
    Parse fold ranges from command line argument.

    Args:
        fold_ranges_arg: List of strings in format "target:start:end"

    Returns:
        Dictionary {target: (start, end)} or None if not provided
    """
    if fold_ranges_arg is None:
        return None

    fold_ranges_dict = {}
    for range_str in fold_ranges_arg:
        parts = range_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid fold_range format: {range_str}. Expected 'target:start:end'")

        target, start, end = parts
        fold_ranges_dict[target] = (int(start), int(end))

    return fold_ranges_dict


def main(args):
    # A missing or foreign neighbour index is worth an immediate stop: the run
    # would otherwise pre-train for hours before pairing windows by electrodes
    # that are not the ones being evaluated.
    check_neighbor_indices(
        build_simclr_variants(args.zone), args.simclr_variants, args.zone,
        config_data_paths(args.zone, args.frequency)[0],
    )

    # Load metadata to get subject IDs for cross-validation
    meta_path = args.meta_path if args.meta_path else config_data_paths(args.zone, args.frequency)[1]
    meta_df = pd.read_csv(meta_path)
    print(f"[INFO] Using metadata: {meta_path}", flush=True)

    # Filter subjects independently for each target
    targets = args.targets
    target_subject_dict = {}

    for target in targets:
        if target in meta_df.columns:
            valid_mask = ~meta_df[target].isna()
            meta_df_target = meta_df[valid_mask]
            target_subjects = sorted(meta_df_target['subject'].unique().tolist())
            target_subject_dict[target] = target_subjects
        else:
            print(f"[WARNING] Target '{target}' not found in metadata columns")
            target_subject_dict[target] = []

    # Parse fold ranges per target
    fold_ranges_dict = parse_fold_ranges(args.fold_ranges)

    # For the CV configuration, use the union of all subjects
    # (each experiment will be filtered according to the specific target)
    all_subjects_set = set()
    for subjects in target_subject_dict.values():
        all_subjects_set.update(subjects)
    unique_subjects = sorted(list(all_subjects_set))
    n_subjects = len(unique_subjects)

    print(f"\n{'='*80}", flush=True)
    print(f"PIPELINE CONFIGURATION", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Targets: {targets}")
    for target, subjects in target_subject_dict.items():
        print(f"  - {target}: {len(subjects)} subjects with valid data")
    print(f"Evaluation modes: {args.eval_modes}")
    print(f"Zone: {args.zone}")
    print(f"Frequency: {args.frequency}")
    print(f"Total subjects (union of all targets): {n_subjects}")
    print(f"Subject IDs: {unique_subjects}")
    print(f"CV Strategy: {args.cv_strategy}", flush=True)
    if fold_ranges_dict:
        print(f"Fold ranges per target: {fold_ranges_dict}", flush=True)

    # Configure cross-validation
    if args.cv_strategy == "kfold":
        # K-Fold uses the union of all subjects
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.base_seed)
        cv_splits = list(kf.split(unique_subjects))
        print(f"Number of folds: {args.n_folds}", flush=True)

        # If a fold range is specified, filter
        if args.fold_range:
            start_fold, end_fold = args.fold_range
            cv_splits = cv_splits[start_fold:end_fold]
            print(f"Processing folds: {start_fold} to {end_fold-1}", flush=True)

        print(f"{'='*80}\n", flush=True)

        # Run each fold
        all_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            # Adjust fold_idx if processing a range
            actual_fold_idx = fold_idx + (args.fold_range[0] if args.fold_range else 0)

            train_subjects = [unique_subjects[i] for i in train_idx]
            test_subjects = [unique_subjects[i] for i in test_idx]

            fold_results = execute_fold(
                fold_idx=actual_fold_idx,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
                args=args,
                targets=targets,
                eval_modes=args.eval_modes,
                target_subject_dict=target_subject_dict
            )

            all_results.extend(fold_results)

    elif args.cv_strategy == "leave_one_out":
        # Leave-One-Out is done independently for each target
        print(f"\n{'='*80}")
        print("LEAVE-ONE-OUT CONFIGURATION")
        print(f"{'='*80}")

        all_results = []

        for target in targets:
            target_subjects = target_subject_dict[target]
            n_target_subjects = len(target_subjects)

            print(f"\nTarget '{target}': {n_target_subjects} folds (one per subject)")

            # Create LOO splits for this target
            loo = LeaveOneOut()
            loo_splits = list(loo.split(target_subjects))

            # If a fold range is specified for this target, filter
            if fold_ranges_dict and target in fold_ranges_dict:
                start_fold, end_fold = fold_ranges_dict[target]
                loo_splits = loo_splits[start_fold:end_fold]
                print(f"  Processing folds: {start_fold} to {end_fold-1}")
            elif fold_ranges_dict:
                # If fold_ranges exist but not for this target, skip
                print(f"  [SKIP] No fold range specified for target '{target}'")
                continue

            # Run each fold para este target
            for fold_idx, (train_idx, test_idx) in enumerate(loo_splits):
                # Adjust fold_idx if processing a range
                if fold_ranges_dict and target in fold_ranges_dict:
                    actual_fold_idx = fold_idx + fold_ranges_dict[target][0]
                else:
                    actual_fold_idx = fold_idx

                train_subjects = [target_subjects[i] for i in train_idx]
                test_subjects = [target_subjects[i] for i in test_idx]

                # For LOO, only evaluate the current target in this fold
                fold_results = execute_fold(
                    fold_idx=actual_fold_idx,
                    train_subjects=train_subjects,
                    test_subjects=test_subjects,
                    args=args,
                    targets=[target],  # Solo evaluar este target
                    eval_modes=args.eval_modes,
                    target_subject_dict=target_subject_dict
                )

                all_results.extend(fold_results)

    else:
        raise ValueError(f"Invalid cv_strategy: {args.cv_strategy}")

    if not all_results:
        print("\n[ERROR] No results were collected. Exiting.")
        return

    # ========================================================================
    # AGGREGATION AND SAVING OF RESULTS
    # ========================================================================
    print(f"\n{'='*80}", flush=True)
    print("AGGREGATING RESULTS", flush=True)
    print(f"{'='*80}", flush=True)

    # Extract subject_avgs and create DataFrame for metrics
    df_raw_data = []
    subject_avgs_by_experiment = {}

    for result in all_results:
        # Convert subject_avgs to strings for saving in CSV
        subject_avgs_str = ';'.join([f"{y_true},{y_pred}" for y_true, y_pred in result['subject_avgs']])

        # Save basic metrics + subject_avgs
        row = {
            'fold': result['fold'],
            'method': result['method'],
            'eval_mode': result['eval_mode'],
            'target': result['target'],
            'nRMSE': result['nRMSE'],
            'R2': result['R2'],
            'RMSE': result['RMSE'],
            'subject_avgs': subject_avgs_str  # New field
        }
        df_raw_data.append(row)

        # Accumulate subject_avgs per experiment
        key = (result['method'], result['eval_mode'], result['target'])
        if key not in subject_avgs_by_experiment:
            subject_avgs_by_experiment[key] = []
        subject_avgs_by_experiment[key].extend(result['subject_avgs'])

    df_raw = pd.DataFrame(df_raw_data)

    # Aggregate results by method, eval_mode and target
    df_agg = df_raw.groupby(['method', 'eval_mode', 'target']).agg(
        nRMSE_mean=('nRMSE', 'mean'),
        nRMSE_std=('nRMSE', 'std'),
        R2_mean=('R2', 'mean'),
        R2_std=('R2', 'std'),
        RMSE_mean=('RMSE', 'mean'),
        RMSE_std=('RMSE', 'std')
    ).reset_index()

    # Calculate global R² and nRMSE from per-subject averages (unbiased by window count)
    r2_global_list = []
    nrmse_global_list = []

    for _, row in df_agg.iterrows():
        key = (row['method'], row['eval_mode'], row['target'])
        if key in subject_avgs_by_experiment and len(subject_avgs_by_experiment[key]) > 1:
            avgs = subject_avgs_by_experiment[key]
            y_true = np.array([a[0] for a in avgs])
            y_pred = np.array([a[1] for a in avgs])

            # R² global
            r2_global = r2_score(y_true, y_pred)
            r2_global_list.append(r2_global)

            # nRMSE global (normalised by std of actual values across all subjects)
            rmse_global = np.sqrt(np.mean((y_true - y_pred) ** 2))
            std_global = np.std(y_true)
            nrmse_global = rmse_global if std_global < 1e-8 else rmse_global / std_global
            nrmse_global_list.append(nrmse_global)
        else:
            # If <= 1 subject or no data, use original metrics
            r2_global_list.append(row['R2_mean'])
            nrmse_global_list.append(row['nRMSE_mean'])

    df_agg['R2_global'] = r2_global_list
    df_agg['nRMSE_global'] = nrmse_global_list
    print(f"\n[INFO] R² and nRMSE global calculated from subject-level averages (unbiased by window count)")

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate suffix based on fold_ranges_dict or fold_range
    if args.cv_strategy == "kfold" and args.fold_range:
        # For kfold with --fold_range
        start, end = args.fold_range
        fold_suffix = f"_folds{start}-{end-1}"
    elif fold_ranges_dict:
        # For LOO with --fold_ranges
        fold_suffix_parts = []
        for target, (start, end) in sorted(fold_ranges_dict.items()):
            fold_suffix_parts.append(f"{target}_{start}-{end-1}")
        fold_suffix = "_" + "_".join(fold_suffix_parts)
    else:
        fold_suffix = ""

    raw_csv_path = os.path.join(args.save_dir, f'downstream_raw_results_{args.cv_strategy}{fold_suffix}.csv')
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"[INFO] Raw results saved to: {raw_csv_path}")

    agg_csv_path = os.path.join(args.save_dir, f'downstream_agg_results_{args.cv_strategy}{fold_suffix}.csv')
    df_agg.to_csv(agg_csv_path, index=False)
    print(f"[INFO] Aggregated results saved to: {agg_csv_path}")

    print("\n--- Aggregated Results ---")
    print(df_agg.to_string())

    print(f"\n{'='*80}", flush=True)
    print("PIPELINE FINISHED SUCCESSFULLY", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a pipeline of downstream experiments with cross-validation."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["age", "cit_36mo"],
        choices=["age", "cit_36mo"],
        help="Target variables for the downstream tasks."
    )
    parser.add_argument(
        "--eval_modes",
        nargs="+",
        default=["linear_probe", "fine_tuning"],
        choices=["linear_probe", "fine_tuning"],
        help="Evaluation modes to run."
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="all",
        help="Head zone data (all, frontal, etc.)"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="all",
        help="Frequency band (all, alpha, etc.)"
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Path to metadata CSV file. If omitted, uses data/processed/{zone}_{frequency}/."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/downstream_results",
        help="Directory to save the final CSV and plot."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=1234,
        help="Base seed for reproducibility."
    )
    parser.add_argument(
        "--cv_strategy",
        type=str,
        default="kfold",
        choices=["kfold", "leave_one_out"],
        help="Cross-validation strategy: kfold or leave_one_out."
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds for k-fold cross-validation (ignored for leave_one_out)."
    )
    parser.add_argument(
        "--fold_ranges",
        nargs="+",
        type=str,
        default=None,
        metavar="TARGET:START:END",
        help="Process specific fold ranges per target (for LOO). Format: 'age:0:10 cit_36mo:5:15'. Useful for parallelization."
    )
    parser.add_argument(
        "--fold_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Process specific fold range for kfold CV. Format: --fold_range 0 2 (processes folds 0 and 1). Useful for parallelization."
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Force retraining even if model already exists."
    )
    parser.add_argument(
        "--allow_legacy",
        action="store_true",
        help="Allow reusing checkpoints whose sidecar is a backfilled legacy record."
    )
    parser.add_argument(
        "--vae_beta",
        type=float,
        default=1.0,
        help="KL weight for the VAE pretraining, under the strict gaussian ELBO "
             "where the reconstruction carries a factor of one half: beta=1 is "
             "the textbook VAE. train_vae.py rescales it to 2*beta/(C*T)."
    )
    parser.add_argument(
        "--vae_free_bits",
        type=float,
        default=0.0,
        help="Per-dimension KL floor (nats) for VAE pretraining."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=str,
        default=None,
        help="Optional subset of methods to run (e.g., AE MAE). Filters the default "
             "linear-probe/fine-tuning method lists; if omitted, all supported methods run."
    )
    parser.add_argument(
        "--expclr_config",
        type=str,
        default=None,
        help="Directory holding the tuned ExpCLR configurations. Without it the paper's "
             "defaults are used (delta=1, tau=1)."
    )
    parser.add_argument(
        "--simclr_variants",
        nargs="+",
        type=str,
        default=["SimCLR"],
        choices=list(SIMCLR_VARIANT_NAMES),
        help="Which SimCLR variants to run as first-class methods (default: just the "
             "standard 'SimCLR' control). E.g. --simclr_variants SimCLR SimCLR-nbr-cosine."
    )
    args = parser.parse_args()
    if args.expclr_config:
        load_expclr_tuning(args.expclr_config)
    main(args)