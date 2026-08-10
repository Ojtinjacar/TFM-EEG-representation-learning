import os
import subprocess
import re
import argparse
import json
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
)

# Configuration of experiments
# Methods to run for each evaluation mode
LINEAR_PROBE_METHODS = ["PCA", "SimCLR", "AE", "MAE", "TripletLoss", "VAE", "CVAE", "CVAE-SP", "ExpCLR"]
FINE_TUNING_METHODS = ["supervised", "SimCLR", "AE", "MAE", "TripletLoss", "VAE", "CVAE", "CVAE-SP", "ExpCLR"]

EXPCLR_DESCRIPTOR = "P_full"
EXPCLR_BATCH_SIZE = 64
EXPCLR_LR = 0.005
EXPCLR_TAU = 1.0
EXPCLR_DELTA = 1.0
EXPCLR_FEATURES = "data/processed/expert_features/expert_features_P_full.npy"

EXPCLR_VARIANTS = {
    "ExpCLR":              {"descriptor": EXPCLR_DESCRIPTOR, "features": EXPCLR_FEATURES},
    "ExpCLR-diverso":      {"descriptor": "P_diverso",
                            "features": "data/processed/expert_features/expert_features_P_diverso.npy"},
    "ExpCLR-madurativo":   {"descriptor": "P_madurativo",
                            "features": "data/processed/expert_features/expert_features_P_madurativo.npy"},
}


def load_expclr_tuning(config_dir):
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
              f"{cfg['random_baseline_mae']:.2f} del encoder aleatorio)")


def expclr_hparams(method):
    v = EXPCLR_VARIANTS[method]
    return (v.get("delta", EXPCLR_DELTA), v.get("lr", EXPCLR_LR),
            v.get("tau", EXPCLR_TAU), v.get("sim_max", "train"))

_NIDX_SESSION   = "data/processed/neighbor_index"
_NIDX_CROSSSUBJ = "data/processed/neighbor_index_crosssubj"
_NIDX_DIFFAGE   = "data/processed/neighbor_index_diffage"


def _nbr(metric, index_dir):
    return ["--positives", "neighbor", "--neighbor_metric", metric, "--neighbor_index_dir", index_dir]


SIMCLR_VARIANTS = {
    "SimCLR":                 {"flags": [], "tag": ""},
    "SimCLR-nbr-cosine":      {"flags": _nbr("cosine", _NIDX_SESSION),      "tag": "nbrcosine"},
    "SimCLR-nbr-wasser":      {"flags": _nbr("wasserstein", _NIDX_SESSION), "tag": "nbrwasser"},
    "SimCLR-nbr-riemann":     {"flags": _nbr("riemann", _NIDX_SESSION),     "tag": "nbrriemann"},
    "SimCLR-xsubj-cosine":    {"flags": _nbr("cosine", _NIDX_CROSSSUBJ),      "tag": "xscosine"},
    "SimCLR-xsubj-wasser":    {"flags": _nbr("wasserstein", _NIDX_CROSSSUBJ), "tag": "xswasser"},
    "SimCLR-xsubj-riemann":   {"flags": _nbr("riemann", _NIDX_CROSSSUBJ),     "tag": "xsriemann"},
    "SimCLR-diffage-cosine":  {"flags": _nbr("cosine", _NIDX_DIFFAGE),      "tag": "dacosine"},
    "SimCLR-diffage-wasser":  {"flags": _nbr("wasserstein", _NIDX_DIFFAGE), "tag": "dawasser"},
    "SimCLR-diffage-riemann": {"flags": _nbr("riemann", _NIDX_DIFFAGE),     "tag": "dariemann"},
}

def parse_output(output):
    """
    Parses the stdout of downstream.py to extract metrics and predictions.
    Returns: (nrmse, r2, rmse, subject_avgs, session_avgs) where subject_avgs is
    [(y_true_mean, y_pred_mean), ...] and session_avgs is
    [(subject, y_true, y_pred_mean), ...].
    """
    # Search for metrics in "Subject-Avg" format. 'nan' is a legal value (e.g.
    # a fully degenerate target): it must parse, not silently drop the fold.
    nrmse_match = re.search(r"Test nRMSE \(Subject-Avg\)=([\d.]+|nan)", output)
    r2_match = re.search(r"Test R2 \(Subject-Avg\)=(-?[\d.]+|nan)", output)
    rmse_match = re.search(r"Test RMSE \(Subject-Avg\)=([\d.]+|nan)", output)

    nrmse = float(nrmse_match.group(1)) if nrmse_match else None
    r2 = float(r2_match.group(1)) if r2_match else None
    rmse = float(rmse_match.group(1)) if rmse_match else None

    if nrmse is None or r2 is None or rmse is None:
        print("[WARNING] Could not parse metrics from output.")
        print(output)

    # Parse per-subject average predictions for global R² calculation
    # Format: SUBJECT_AVG_PRED: y_true_mean y_pred_mean
    subject_avgs = []
    session_avgs = []
    for line in output.split('\n'):
        if line.startswith('SUBJECT_AVG_PRED:'):
            parts = line.split(': ')[1].split()
            if len(parts) == 2:
                subject_avgs.append((float(parts[0]), float(parts[1])))
        elif line.startswith('SESSION_AVG_PRED:'):
            parts = line.split(': ')[1].split()
            if len(parts) == 3:
                session_avgs.append((parts[0], float(parts[1]), float(parts[2])))

    return nrmse, r2, rmse, subject_avgs, session_avgs

def config_data_paths(zone, frequency):
    cfg_dir = os.path.join("data", "processed", f"{zone}_{frequency}")
    return (os.path.join(cfg_dir, "processed_windows.npy"),
            os.path.join(cfg_dir, "processed_metadata.csv"))


VAE_FAMILY = ("VAE", "CVAE", "CVAE-SP")


def run_pretraining(method, target, zone, frequency, test_subjects, fold_id, no_skip=False, vae_beta=1.0,
                    vae_prior="standard", vae_free_bits=0.0, cvae_cond_dim=16, vae_conditional=False,
                    simclr_flags=None, simclr_tag="", allow_legacy=False, seed=42):
    simclr_flags = list(simclr_flags or [])
    if method in ["PCA", "supervised"]:
        # These methods do not require pre-training
        return None

    # ExpCLR deliberately keeps one fixed seed across folds so that only the
    # excluded subject varies between LOSO encoders.
    eff_seed = 42 if method in EXPCLR_VARIANTS else seed

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
        expected.update({
            "method": "SimCLR",
            "fold_id": eff_fold_id,
            "aug_mode": "legacy",
            "positives": "neighbor" if is_neighbor else "augment",
            "neighbor_metric": nbr_metric,
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
    elif method in VAE_FAMILY:
        model_filename = vae_checkpoint_name(
            method, zone, frequency, fold_id,
            beta=vae_beta, prior=vae_prior, free_bits=vae_free_bits,
        )
        expected.update({
            "method": method,
            "fold_id": fold_id,
            "beta": vae_beta,
            "prior": vae_prior,
            "free_bits": vae_free_bits,
            "conditional": bool(vae_conditional),
            "lr": 1e-3,
        })
        if allow_legacy:
            # Legacy VAE-family checkpoints are provably collapsed AND their
            # filename beta is in the pre-F1 code scale: same name, different
            # objective. Never reuse them.
            print(f"  > [REUSE] {method}: legacy reuse disabled for the VAE "
                  f"family (collapsed checkpoints, incompatible beta scale).")
            allow_legacy = False
    elif method in EXPCLR_VARIANTS:
        delta, lr, tau, _ = expclr_hparams(method)
        model_filename = expclr_checkpoint_name(
            zone, frequency, fold_id, EXPCLR_VARIANTS[method]["descriptor"],
            batch_size=EXPCLR_BATCH_SIZE, lr=lr, temperature=tau, delta=delta,
        )
        expected.update({"fold_id": fold_id, "descriptor": EXPCLR_VARIANTS[method]["descriptor"]})
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
        delta, lr, tau, sim_max = expclr_hparams(method)
        command = [
            "python", "src/train_expclr.py",
            "--zone", zone,
            "--frequency", frequency,
            "--fold_id", fold_id,
            "--data_path", expclr_data_dir,
            "--expert_features", variant["features"],
            "--descriptor", variant["descriptor"],
            "--batch_size", str(EXPCLR_BATCH_SIZE),
            "--lr", str(lr),
            "--temperature", str(tau),
            "--delta", str(delta),
            "--sim_max", sim_max,
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

    command += ["--seed", str(eff_seed)]

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
        return None, None, None, [], []
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return None, None, None, [], []


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
    fold_failures = []
    pretrain_seed = args.base_seed + fold_idx

    # ========================================================================
    # PHASE 1: PRE-TRAINING (considering dependencies)
    # ========================================================================
    print(f"\n[PHASE 1] Pre-training models for fold {fold_idx+1}...", flush=True)

    for variant in simclr_variants:
        spec = SIMCLR_VARIANTS[variant]
        print(f"\n  Pre-training {variant} (self-supervised)...", flush=True)
        pretrained_models[variant] = run_pretraining(
            method="SimCLR", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
            simclr_flags=spec["flags"], simclr_tag=spec["tag"],
        )

    # 1.2 Pre-train AE (once, independent of target)
    if _use("AE"):
        print(f"\n  Pre-training AE (self-supervised)...", flush=True)
        pretrained_models["AE"] = run_pretraining(
            method="AE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
        )
    else:
        pretrained_models["AE"] = None

    # 1.3 Pre-train MAE (once, independent of target)
    if _use("MAE"):
        print(f"\n  Pre-training MAE (masked self-supervised)...", flush=True)
        pretrained_models["MAE"] = run_pretraining(
            method="MAE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
        )
    else:
        pretrained_models["MAE"] = None

    if _use("VAE"):
        print(f"\n  Pre-training VAE (variational self-supervised, beta={args.vae_beta})...", flush=True)
        pretrained_models["VAE"] = run_pretraining(
            method="VAE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
            vae_beta=args.vae_beta, vae_prior="standard", vae_free_bits=args.vae_free_bits,
            vae_conditional=False,
        )
    else:
        pretrained_models["VAE"] = None

    if _use("CVAE"):
        print(f"\n  Pre-training CVAE (age-conditioned, conditional/rich prior, "
              f"beta={args.vae_beta})...", flush=True)
        pretrained_models["CVAE"] = run_pretraining(
            method="CVAE", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
            vae_beta=args.vae_beta, vae_prior="conditional",
            vae_free_bits=args.vae_free_bits, cvae_cond_dim=args.cvae_cond_dim,
            vae_conditional=True,
        )
    else:
        pretrained_models["CVAE"] = None

    if _use("CVAE-SP"):
        print(f"\n  Pre-training CVAE-SP (age-conditioned, standard prior, "
              f"beta={args.vae_beta})...", flush=True)
        pretrained_models["CVAE-SP"] = run_pretraining(
            method="CVAE-SP", target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
            vae_beta=args.vae_beta, vae_prior="standard",
            vae_free_bits=args.vae_free_bits, cvae_cond_dim=args.cvae_cond_dim,
            vae_conditional=True,
        )
    else:
        pretrained_models["CVAE-SP"] = None

    for variant in expclr_variants:
        print(f"\n  Pre-training {variant} (descriptor={EXPCLR_VARIANTS[variant]['descriptor']}, "
              f"tau={EXPCLR_TAU}, delta={EXPCLR_DELTA})...", flush=True)
        pretrained_models[variant] = run_pretraining(
            method=variant, target=None, zone=args.zone, frequency=args.frequency,
            test_subjects=test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
        )
    for variant in EXPCLR_VARIANTS:
        pretrained_models.setdefault(variant, None)

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
            test_subjects=valid_test_subjects, fold_id=fold_id, no_skip=args.no_skip, allow_legacy=args.allow_legacy, seed=pretrain_seed,
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
            if eval_mode == "linear_probe":
                methods = lp_methods
            else:  # fine_tuning
                methods = ft_methods

            for method in methods:
                experiment_counter += 1
                print(f"\n  [{experiment_counter}/{total_experiments}] Evaluating: {method} | {target} | {eval_mode}")

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
                    model_path = pretrained_models["CVAE-SP"]
                    downstream_method = "CVAE"
                elif method in EXPCLR_VARIANTS:
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
                nrmse, r2, rmse, subject_avgs, session_avgs = run_downstream_experiment(
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

                if subject_avgs:
                    result_entry = {
                        "fold": fold_idx,
                        "method": method,
                        "eval_mode": eval_mode,
                        "target": target,
                        "nRMSE": np.nan if nrmse is None else nrmse,
                        "R2": np.nan if r2 is None else r2,
                        "RMSE": np.nan if rmse is None else rmse,
                        "subject_avgs": subject_avgs,  # List of (y_true, y_pred) per-subject averages
                        "session_avgs": session_avgs,  # List of (subject, y_true, y_pred) per session
                    }

                    fold_results.append(result_entry)
                    print(f"    ✓ nRMSE={result_entry['nRMSE']:.4f}, R2={result_entry['R2']:.4f}, "
                          f"RMSE={result_entry['RMSE']:.2f}", flush=True)
                else:
                    fold_failures.append((method, eval_mode, target))
                    print(f"    [ERROR] Failed to get metrics/predictions")

    print(f"\n{'='*80}", flush=True)
    print(f"FOLD {fold_idx+1} COMPLETED: Collected {len(fold_results)} results", flush=True)
    if fold_failures:
        print(f"FOLD {fold_idx+1} FAILURES ({len(fold_failures)}): {fold_failures}", flush=True)
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
        session_avgs_str = ';'.join([f"{subj},{y_true},{y_pred}" for subj, y_true, y_pred in result.get('session_avgs', [])])

        # Save basic metrics + subject_avgs
        row = {
            'fold': result['fold'],
            'method': result['method'],
            'eval_mode': result['eval_mode'],
            'target': result['target'],
            'nRMSE': result['nRMSE'],
            'R2': result['R2'],
            'RMSE': result['RMSE'],
            'subject_avgs': subject_avgs_str,
            'session_avgs': session_avgs_str
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
            nrmse_global = np.nan if std_global < 1e-8 else rmse_global / std_global
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
        default=1.0,
        help="KL weight for the VAE pretraining in canonical ELBO units "
             "(beta=1 is the standard VAE; train_vae.py rescales internally)."
    )
    parser.add_argument(
        "--vae_free_bits",
        type=float,
        default=0.0,
        help="Per-dimension KL floor (nats) for VAE/CVAE pretraining."
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
        "--expclr_config",
        type=str,
        default=None,
        help="Directory with tune_expclr.py's best_config_<descriptor>.json files. Each ExpCLR "
             "variant then pre-trains with its own tuned delta/lr/tau instead of the paper's "
             "defaults, which degenerate on this descriptor (see docs/expclr.md)."
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
    parser.add_argument(
        "--allow_legacy",
        action="store_true",
        help="Allow reusing checkpoints whose sidecar is a backfilled legacy record."
    )
    args = parser.parse_args()
    if args.expclr_config:
        print(f"Hiperparametros de ExpCLR tuneados, desde {args.expclr_config}:")
        load_expclr_tuning(args.expclr_config)
    main(args)
