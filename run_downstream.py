import os
import subprocess
import re
import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score

# Configuration of experiments
# Methods to run for each evaluation mode
LINEAR_PROBE_METHODS = ["PCA", "SimCLR", "AE", "MAE", "TripletLoss", "VAE", "CVAE", "CVAE-SP", "ExpCLR"]
FINE_TUNING_METHODS = ["supervised", "SimCLR", "AE", "MAE", "TripletLoss", "VAE", "CVAE", "CVAE-SP", "ExpCLR"]

# ExpCLR (E3) hyperparameters. They are kept here as constants because run_pretraining must
# rebuild the checkpoint filename that src/train_expclr.py writes, character for character.
# Values follow Nonnenmacher et al. (ICML 2022): tau = 1, Delta = 1, batch = 64, lr = 5e-3.
EXPCLR_DESCRIPTOR = "P_full"
EXPCLR_BATCH_SIZE = 64
EXPCLR_LR = 0.005
EXPCLR_TAU = 1.0
EXPCLR_DELTA = 1.0
EXPCLR_FEATURES = "data/processed/expert_features/expert_features_P_full.npy"

# ExpCLR variants, following the same registry pattern as SIMCLR_VARIANTS below: each is a
# first-class "method" with its own pre-training and its own result rows. What varies is the
# expert descriptor, so no extra tag is needed: src/train_expclr.py already embeds the
# descriptor label in the checkpoint name, which keeps the variants from overwriting each other.
# "ExpCLR" (P_full, the 106 curated features) is the reference; the others are descriptor ablations.
EXPCLR_VARIANTS = {
    "ExpCLR":              {"descriptor": EXPCLR_DESCRIPTOR, "features": EXPCLR_FEATURES},
    "ExpCLR-diverso":      {"descriptor": "P_diverso",
                            "features": "data/processed/expert_features/expert_features_P_diverso.npy"},
    "ExpCLR-madurativo":   {"descriptor": "P_madurativo",
                            "features": "data/processed/expert_features/expert_features_P_madurativo.npy"},
}

# SimCLR variants. Each is a first-class "method": it gets its own pre-training
# (SimCLR encoder with the given train_simclr.py flags) and its own result rows
# (labelled with the variant name). "tag" disambiguates the saved .pth so variants
# never overwrite each other. "SimCLR" (empty flags/tag) is the standard control.
# The registry is extensible: add augmentation variants (e.g. --aug_mode) the same way.
# Neighbor-positive index directories. The pairing STRATEGY is encoded by the directory
# (each holds neighbor_index_<metric>.npy); the metric selects the file inside it.
_NIDX_SESSION   = "data/processed/neighbor_index"            # same subject + same session (main)
_NIDX_CROSSSUBJ = "data/processed/neighbor_index_crosssubj"  # same age, different subject
_NIDX_DIFFAGE   = "data/processed/neighbor_index_diffage"    # same subject, different age


def _nbr(metric, index_dir):
    """SimCLR flags for a neighbor-positive variant with a given metric and index dir."""
    return ["--positives", "neighbor", "--neighbor_metric", metric, "--neighbor_index_dir", index_dir]


SIMCLR_VARIANTS = {
    "SimCLR":                 {"flags": [], "tag": ""},                             # standard control
    # same-session neighbor positives (main experiment)
    "SimCLR-nbr-cosine":      {"flags": _nbr("cosine", _NIDX_SESSION),      "tag": "nbrcosine"},
    "SimCLR-nbr-wasser":      {"flags": _nbr("wasserstein", _NIDX_SESSION), "tag": "nbrwasser"},
    "SimCLR-nbr-riemann":     {"flags": _nbr("riemann", _NIDX_SESSION),     "tag": "nbrriemann"},
    # ablation A: cross-subject, same age (developmental signal vs subject identity)
    "SimCLR-xsubj-cosine":    {"flags": _nbr("cosine", _NIDX_CROSSSUBJ),      "tag": "xscosine"},
    "SimCLR-xsubj-wasser":    {"flags": _nbr("wasserstein", _NIDX_CROSSSUBJ), "tag": "xswasser"},
    "SimCLR-xsubj-riemann":   {"flags": _nbr("riemann", _NIDX_CROSSSUBJ),     "tag": "xsriemann"},
    # ablation B: same subject, different age (breaks the age==session shortcut)
    "SimCLR-diffage-cosine":  {"flags": _nbr("cosine", _NIDX_DIFFAGE),      "tag": "dacosine"},
    "SimCLR-diffage-wasser":  {"flags": _nbr("wasserstein", _NIDX_DIFFAGE), "tag": "dawasser"},
    "SimCLR-diffage-riemann": {"flags": _nbr("riemann", _NIDX_DIFFAGE),     "tag": "dariemann"},
}

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
    """Builds the processed data/metadata paths for a zone x frequency config.

    Mirrors the layout produced by run_pipeline.py: one folder per config at
    ``data/processed/{zone}_{frequency}/`` holding processed_windows.npy and
    processed_metadata.csv.

    Args:
        zone (str): Brain zone (e.g., "all", "frontal").
        frequency (str): Frequency band (e.g., "all", "alpha").

    Returns:
        tuple[str, str]: (windows_path, metadata_path).
    """
    cfg_dir = os.path.join("data", "processed", f"{zone}_{frequency}")
    return (os.path.join(cfg_dir, "processed_windows.npy"),
            os.path.join(cfg_dir, "processed_metadata.csv"))


VAE_FAMILY = ("VAE", "CVAE", "CVAE-SP")


def run_pretraining(method, target, zone, frequency, test_subjects, fold_id, no_skip=False, vae_beta=0.003,
                    vae_prior="standard", vae_free_bits=0.0, cvae_cond_dim=16, vae_conditional=False,
                    simclr_flags=None, simclr_tag=""):
    """
    Constructs and runs a single call to pretraining script, excluding test subjects.
    Returns the path to the trained model.

    Args:
        method: Pre-training method (SimCLR, AE, MAE, VAE, TripletLoss)
        target: Target for TripletLoss (ignored for SimCLR and AE)
        zone: Brain zone
        frequency: Frequency band
        test_subjects: List of subjects to exclude
        fold_id: Unique fold identifier for naming the model
        no_skip: If True, forces retraining even if the model exists
        vae_beta: KL weight for the VAE (encoded in its model filename)
        simclr_flags: Extra flags for train_simclr.py that select a SimCLR variant
            (e.g. ["--positives", "neighbor", "--neighbor_metric", "cosine"]).
        simclr_tag: Short tag appended to the SimCLR model filename so different
            variants never overwrite each other's checkpoint.
    """
    simclr_flags = list(simclr_flags or [])
    if method in ["PCA", "supervised"]:
        # These methods do not require pre-training
        return None

    # Determine the model filename based on the method
    if method == "SimCLR":
        # Embed the variant tag INSIDE the fold_id handed to train_simclr.py: that script
        # builds its own .pth name from --fold_id, so the tag must travel through fold_id
        # for the saved file to match what we look for here (one distinct checkpoint per
        # variant, no collisions).
        eff_fold_id = f"{fold_id}_{simclr_tag}" if simclr_tag else fold_id
        model_filename = f"SimCLR_{zone}_{frequency}_{eff_fold_id}_batch_512_lr_0.001_wd_0.0001_temperature_0.05.pth"
    elif method == "AE":
        model_filename = f"AE_{zone}_{frequency}_{fold_id}_hidden128_e100.pth"
    elif method == "MAE":
        model_filename = f"MAE_{zone}_{frequency}_{fold_id}_hidden128_mask30_block25_e100.pth"
    elif method == "TripletLoss":
        model_filename = f"Triplet_{target}_{zone}_{frequency}_{fold_id}_emb128_m0.4.pth"
    elif method in VAE_FAMILY:
        # Must mirror the model_name built in src/train_vae.py exactly (the label is
        # forwarded as --tag so the filename prefix equals the method/result label).
        model_filename = (
            f"{method}_{zone}_{frequency}_{fold_id}"
            f"_hidden128_beta{vae_beta}_prior{vae_prior}_fb{vae_free_bits}_e100.pth"
        )
    elif method in EXPCLR_VARIANTS:
        # Must mirror the checkpoint name built in src/train_expclr.py exactly. The descriptor
        # label is what distinguishes one variant's checkpoint from another's.
        model_filename = (
            f"ExpCLR_{zone}_{frequency}_{fold_id}_{EXPCLR_VARIANTS[method]['descriptor']}"
            f"_batch_{EXPCLR_BATCH_SIZE}_lr_{EXPCLR_LR}_tau_{EXPCLR_TAU}_delta_{EXPCLR_DELTA}.pth"
        )
    else:
        print(f"[WARNING] Pretraining not implemented for method: {method}")
        return None

    model_path = os.path.join("save/models", model_filename)

    # Check if the model already exists
    if not no_skip and os.path.exists(model_path):
        print(f"  > Model already exists: {model_filename}. Skipping pretraining.")
        return model_path

    # Build command based on the method
    if method == "SimCLR":
        # Point SimCLR at the same zone x frequency data folder that downstream reads
        # (its default is the empty data/processed/5_s), then append the variant flags.
        simclr_data_dir = os.path.join("data", "processed", f"{zone}_{frequency}")
        command = [
            "python", "src/train_simclr.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", eff_fold_id,
            "--data_path", simclr_data_dir,
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
        expclr_data_dir = os.path.join("data", "processed", f"{zone}_{frequency}")
        variant = EXPCLR_VARIANTS[method]
        command = [
            "python", "src/train_expclr.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--data_path", expclr_data_dir,
            "--expert_features", variant["features"],
            "--descriptor", variant["descriptor"],
            "--batch_size", str(EXPCLR_BATCH_SIZE),
            "--lr", str(EXPCLR_LR),
            "--temperature", str(EXPCLR_TAU),
            "--delta", str(EXPCLR_DELTA),
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]
    elif method in VAE_FAMILY:
        command = [
            "python", "src/train_vae.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--tag", method,
            "--exclude_subjects"
        ] + [str(s) for s in test_subjects]

    # Point the generative train scripts at the zone x frequency data (they share
    # --data-path/--meta-path); otherwise they fall back to the empty data/processed/5_s.
    if method in ("AE", "MAE") or method in VAE_FAMILY:
        win_path, meta_path = config_data_paths(zone, frequency)
        command += ["--data-path", win_path, "--meta-path", meta_path]
    if method in VAE_FAMILY:
        command += [
            "--beta", str(vae_beta),
            "--prior", vae_prior,
            "--free-bits", str(vae_free_bits),
        ]
    if vae_conditional:
        command += ["--conditional", "--cond-dim", str(cvae_cond_dim)]

    print(f"  > Running pretraining command: {' '.join(command)}")
    try:
        result = subprocess.run(
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

def run_downstream_experiment(method, eval_mode, target, zone, frequency, model_path, seed, test_subjects, fold_id=None, downstream_method=None, cond_dim=16):
    """
    Constructs and runs a single call to downstream.py.
    Returns: (nrmse, r2, rmse, subject_avgs)

    Args:
        method: Result label (may be a SimCLR variant name, e.g. "SimCLR-nbr-cosine").
        downstream_method: The architecture passed to downstream.py's --method (one of
            its fixed choices). For SimCLR variants this is "SimCLR"; defaults to `method`.
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

    # The CVAE backbone must be rebuilt with the same condition-embedding width.
    if (downstream_method or method) == "CVAE":
        command.extend(["--cond-dim", str(cond_dim)])

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

    # Optional method subset (e.g., --methods AE MAE VAE) to bound cost.
    selected = set(args.methods) if getattr(args, "methods", None) else None
    def _use(m):
        return selected is None or m in selected

    # SimCLR expands into the requested variants (control + neighbor positives, ...).
    # Each variant is treated as its own method downstream; "SimCLR" in the base lists
    # is replaced by the selected variant names.
    simclr_variants = list(getattr(args, "simclr_variants", None) or ["SimCLR"]) if _use("SimCLR") else []
    # ExpCLR expands the same way: "ExpCLR" in the base lists is replaced by the selected
    # descriptor variants, each of which becomes its own result label.
    expclr_variants = list(getattr(args, "expclr_variants", None) or ["ExpCLR"]) if _use("ExpCLR") else []
    def _expand(mlist):
        out = []
        for m in mlist:
            if m == "SimCLR":
                out.extend(simclr_variants)
            elif m == "ExpCLR":
                out.extend(expclr_variants)
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

    # ========================================================================
    # PHASE 1: PRE-TRAINING (considering dependencies)
    # ========================================================================
    print(f"\n[PHASE 1] Pre-training models for fold {fold_idx+1}...", flush=True)

    # 1.1 Pre-train each selected SimCLR variant (once each, independent of target)
    for variant in simclr_variants:
        spec = SIMCLR_VARIANTS[variant]
        # Each variant's flags are self-contained (including --neighbor_index_dir for
        # neighbor variants), so the pairing strategy/metric is fully specified here.
        print(f"\n  Pre-training {variant} (self-supervised)...", flush=True)
        pretrained_models[variant] = run_pretraining(
            method="SimCLR", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip,
            simclr_flags=spec["flags"], simclr_tag=spec["tag"],
        )

    # 1.2 Pre-train AE (once, independent of target)
    if _use("AE"):
        print(f"\n  Pre-training AE (self-supervised)...", flush=True)
        pretrained_models["AE"] = run_pretraining(
            method="AE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip,
        )
    else:
        pretrained_models["AE"] = None

    # 1.3 Pre-train MAE (once, independent of target)
    if _use("MAE"):
        print(f"\n  Pre-training MAE (masked self-supervised)...", flush=True)
        pretrained_models["MAE"] = run_pretraining(
            method="MAE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip,
        )
    else:
        pretrained_models["MAE"] = None

    # 1.4 Pre-train VAE (once, independent of target). Baseline: standard N(0,I) prior.
    if _use("VAE"):
        print(f"\n  Pre-training VAE (variational self-supervised, beta={args.vae_beta})...", flush=True)
        pretrained_models["VAE"] = run_pretraining(
            method="VAE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip,
            vae_beta=args.vae_beta, vae_prior="standard", vae_free_bits=args.vae_free_bits,
            vae_conditional=False,
        )
    else:
        pretrained_models["VAE"] = None

    # 1.4b Pre-train CVAE (age-conditioned encoder + conditional/rich prior). This is the
    # rich-prior configuration; CVAE-SP below isolates the prior with the same encoder.
    if _use("CVAE"):
        print(f"\n  Pre-training CVAE (age-conditioned, conditional/rich prior, "
              f"beta={args.vae_beta})...", flush=True)
        pretrained_models["CVAE"] = run_pretraining(
            method="CVAE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip,
            vae_beta=args.vae_beta, vae_prior="conditional",
            vae_free_bits=args.vae_free_bits, cvae_cond_dim=args.cvae_cond_dim,
            vae_conditional=True,
        )
    else:
        pretrained_models["CVAE"] = None

    # 1.4c Pre-train CVAE-SP (age-conditioned encoder + STANDARD prior). Ablation row:
    # same conditional architecture as CVAE but N(0,I) prior, to isolate the rich prior's
    # marginal effect within the conditional model.
    if _use("CVAE-SP"):
        print(f"\n  Pre-training CVAE-SP (age-conditioned, standard prior, "
              f"beta={args.vae_beta})...", flush=True)
        pretrained_models["CVAE-SP"] = run_pretraining(
            method="CVAE-SP", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip,
            vae_beta=args.vae_beta, vae_prior="standard",
            vae_free_bits=args.vae_free_bits, cvae_cond_dim=args.cvae_cond_dim,
            vae_conditional=True,
        )
    else:
        pretrained_models["CVAE-SP"] = None

    # 1.4d Pre-train ExpCLR (E3): contrastive learning guided by the continuous expert
    # descriptor instead of augmented views (Nonnenmacher et al., ICML 2022).
    # One pre-training per requested descriptor variant, each with its own checkpoint.
    for variant in expclr_variants:
        print(f"\n  Pre-training {variant} (descriptor={EXPCLR_VARIANTS[variant]['descriptor']}, "
              f"tau={EXPCLR_TAU}, delta={EXPCLR_DELTA})...", flush=True)
        pretrained_models[variant] = run_pretraining(
            method=variant, target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip,
        )
    for variant in EXPCLR_VARIANTS:
        pretrained_models.setdefault(variant, None)

    # 1.5 Pre-train TripletLoss for each target (depends on target)
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

        pretrained_models[f"TripletLoss_{target}"] = run_pretraining(
            method="TripletLoss", target=target, zone=args.zone, frequency=args.frequency,
            test_subjects=valid_test_subjects, fold_id=fold_id, no_skip=args.no_skip,
        )

    # ========================================================================
    # PHASE 2: DOWNSTREAM EVALUATION (all combinations)
    # ========================================================================
    print(f"\n[PHASE 2] Downstream evaluation for fold {fold_idx+1}...", flush=True)

    experiment_counter = 0
    total_experiments = len(targets) * (len(lp_methods) + len(ft_methods))

    for target in targets:
        # Filter train and test subjects for this specific target
        valid_train_subjects = [s for s in train_subjects if s in target_subject_dict[target]]
        valid_test_subjects = [s for s in test_subjects if s in target_subject_dict[target]]

        if not valid_test_subjects:
            print(f"\n[SKIP] No test subjects have valid data for target '{target}'. Skipping all experiments for this target.")
            continue

        print(f"\nTarget '{target}': {len(valid_train_subjects)} train subjects, {len(valid_test_subjects)} test subjects")

        for eval_mode in eval_modes:
            # Select methods based on eval_mode (respecting the optional --methods subset)
            if eval_mode == "linear_probe":
                methods = lp_methods
            else:  # fine_tuning
                methods = ft_methods

            for method in methods:
                experiment_counter += 1
                print(f"\n  [{experiment_counter}/{total_experiments}] Evaluating: {method} | {target} | {eval_mode}")

                # Determine which pre-trained model to use. SimCLR variants keep their
                # variant name as the result label but evaluate with downstream --method SimCLR.
                downstream_method = method
                if method in SIMCLR_VARIANTS:
                    model_path = pretrained_models.get(method)
                    downstream_method = "SimCLR"
                elif method == "AE":
                    model_path = pretrained_models["AE"]
                elif method == "MAE":
                    model_path = pretrained_models["MAE"]
                elif method == "VAE":
                    model_path = pretrained_models["VAE"]
                elif method == "CVAE":
                    model_path = pretrained_models["CVAE"]
                elif method == "CVAE-SP":
                    # Same CVAE architecture, standard prior; evaluate as --method CVAE.
                    model_path = pretrained_models["CVAE-SP"]
                    downstream_method = "CVAE"
                elif method in EXPCLR_VARIANTS:
                    # Descriptor variants keep their own result label but share the encoder
                    # architecture, so downstream.py always sees --method ExpCLR.
                    model_path = pretrained_models.get(method)
                    downstream_method = "ExpCLR"
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
                    cond_dim=args.cvae_cond_dim,
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
    # Load metadata to get subject IDs for cross-validation. Default to the metadata of the
    # selected zone x frequency config; an explicit --meta_path overrides it.
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
        "--methods",
        nargs="+",
        type=str,
        default=None,
        help="Optional subset of methods to run (e.g., AE MAE VAE). Filters the default "
             "linear-probe/fine-tuning method lists; if omitted, all supported methods run."
    )
    parser.add_argument(
        "--vae_beta",
        type=float,
        default=0.003,
        help="KL weight for the VAE pretraining (must match train_vae.py's beta formatting)."
    )
    parser.add_argument(
        "--vae_free_bits",
        type=float,
        default=0.0,
        help="Per-dimension KL floor (nats) for VAE/CVAE pretraining (anti-collapse)."
    )
    parser.add_argument(
        "--cvae_cond_dim",
        type=int,
        default=16,
        help="Width of the CVAE session-age condition embedding (train and eval must match)."
    )
    parser.add_argument(
        "--simclr_variants",
        nargs="+",
        type=str,
        default=["SimCLR"],
        choices=list(SIMCLR_VARIANTS.keys()),
        help="Which SimCLR variants to run as first-class methods (default: just the "
             "standard 'SimCLR' control). E.g. --simclr_variants SimCLR SimCLR-nbr-cosine."
    )
    parser.add_argument(
        "--expclr_variants",
        nargs="+",
        type=str,
        default=["ExpCLR"],
        choices=list(EXPCLR_VARIANTS.keys()),
        help="Which ExpCLR descriptor variants to run as first-class methods (default: just "
             "'ExpCLR', the P_full reference). E.g. --expclr_variants ExpCLR-diverso."
    )
    parser.add_argument(
        "--neighbor_index_dir",
        type=str,
        default="data/processed/neighbor_index",
        help="Directory with neighbor_index_<metric>.npy, used by SimCLR neighbor-positive variants."
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
    args = parser.parse_args()
    main(args)