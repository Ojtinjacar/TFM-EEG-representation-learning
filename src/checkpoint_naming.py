"""Single source of truth for pretraining checkpoint names and sidecar metadata.

Every train script builds its checkpoint filename through the functions below,
and the orchestrators (run_downstream.py, run_pipeline.py, run_e3_loso.py)
resolve the expected filename through the same functions, so the two sides can
never drift apart.

Each checkpoint <name>.pth is accompanied by a sidecar <name>_config.json that
records the configuration that produced it (most importantly the excluded test
subjects and the seed). ``checkpoint_is_reusable`` validates a checkpoint
against the configuration a caller is about to train with, replacing blind
reuse-by-file-existence.
"""

import json
import os

import numpy as np


def _fold_suffix(fold_id):
    return f"_{fold_id}" if fold_id else ""


def simclr_checkpoint_name(zone, frequency, fold_id, batch_size=512, lr=1e-3,
                           weight_decay=1e-4, temperature=0.05):
    """Returns the SimCLR checkpoint filename.

    Args:
        zone (str): Head zone of the processed data.
        frequency (str): Frequency band of the processed data.
        fold_id (str | None): Fold identifier, may embed a variant tag.
        batch_size (int): Contrastive batch size.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        temperature (float): NT-Xent temperature.

    Returns:
        str: Checkpoint filename (no directory).
    """
    return (f"SimCLR_{zone}_{frequency}{_fold_suffix(fold_id)}"
            f"_batch_{batch_size}_lr_{lr}_wd_{weight_decay}"
            f"_temperature_{temperature}.pth")


def ae_checkpoint_name(zone, frequency, fold_id, hidden_size=128, epochs=100):
    """Returns the deterministic autoencoder checkpoint filename."""
    return (f"AE_{zone}_{frequency}{_fold_suffix(fold_id)}"
            f"_hidden{hidden_size}_e{epochs}.pth")


def mae_checkpoint_name(zone, frequency, fold_id, hidden_size=128,
                        mask_ratio=0.3, block_size=25, epochs=100):
    """Returns the masked autoencoder checkpoint filename."""
    block_suffix = f"_block{block_size}" if block_size else ""
    return (f"MAE_{zone}_{frequency}{_fold_suffix(fold_id)}"
            f"_hidden{hidden_size}_mask{int(mask_ratio * 100)}"
            f"{block_suffix}_e{epochs}.pth")


def triplet_checkpoint_name(target, zone, frequency, fold_id,
                            embedding_size=128, margin=0.4):
    """Returns the triplet-loss checkpoint filename."""
    return (f"Triplet_{target}_{zone}_{frequency}{_fold_suffix(fold_id)}"
            f"_emb{embedding_size}_m{margin}.pth")


def vae_checkpoint_name(tag, zone, frequency, fold_id, beta, prior, free_bits,
                        hidden_size=128, epochs=100):
    """Returns the VAE-family checkpoint filename.

    Args:
        tag (str): Method tag ("VAE", "CVAE" or "CVAE-SP").
        zone (str): Head zone of the processed data.
        frequency (str): Frequency band of the processed data.
        fold_id (str | None): Fold identifier.
        beta (float): KL weight exactly as passed on the CLI.
        prior (str): Latent prior name ("standard" | "conditional").
        free_bits (float): Per-dimension KL floor.
        hidden_size (int): Encoder width and latent size.
        epochs (int): Training epochs.

    Returns:
        str: Checkpoint filename (no directory).
    """
    return (f"{tag}_{zone}_{frequency}{_fold_suffix(fold_id)}"
            f"_hidden{hidden_size}_beta{beta}_prior{prior}"
            f"_fb{free_bits}_e{epochs}.pth")


def expclr_checkpoint_name(zone, frequency, fold_id, descriptor, batch_size,
                           lr, temperature, delta):
    """Returns the ExpCLR checkpoint filename."""
    return (f"ExpCLR_{zone}_{frequency}{_fold_suffix(fold_id)}_{descriptor}"
            f"_batch_{batch_size}_lr_{lr}_tau_{temperature}"
            f"_delta_{delta}.pth")


def sidecar_path(checkpoint_path):
    """Returns the sidecar JSON path for a checkpoint path."""
    return str(checkpoint_path).replace(".pth", "_config.json")


def write_sidecar(checkpoint_path, config):
    """Writes the sidecar JSON next to a checkpoint.

    Args:
        checkpoint_path (str): Path of the saved ``.pth`` file.
        config (dict): JSON-serializable training configuration. Callers must
            include at least ``exclude_subjects`` (sorted list of str) and
            ``seed`` (int or None).
    """
    with open(sidecar_path(checkpoint_path), "w") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)


def checkpoint_is_reusable(checkpoint_path, expected, allow_legacy=False):
    """Decides whether an existing checkpoint matches the intended config.

    A checkpoint is reusable only if it exists, has a sidecar, and every key in
    ``expected`` matches the recorded value (floats compared with np.isclose).

    Sidecars with ``legacy: true`` were backfilled for checkpoints trained
    before sidecars existed. They are refused unless ``allow_legacy`` is True,
    and even then ``exclude_subjects`` must be recorded and match; other
    expected keys absent from a legacy sidecar are skipped.

    Args:
        checkpoint_path (str): Path of the ``.pth`` file.
        expected (dict): Configuration the caller is about to train with.
        allow_legacy (bool): Accept backfilled legacy sidecars.

    Returns:
        bool: True if the checkpoint can be reused safely.
    """
    checkpoint_path = str(checkpoint_path)
    name = os.path.basename(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return False
    sc = sidecar_path(checkpoint_path)
    if not os.path.exists(sc):
        print(f"[REUSE] {name}: no sidecar; not reusable.")
        return False
    with open(sc) as fh:
        recorded = json.load(fh)

    if recorded.get("legacy", False):
        if not allow_legacy:
            print(f"[REUSE] {name}: legacy checkpoint; refused without allow_legacy.")
            return False
        if "exclude_subjects" not in recorded:
            print(f"[REUSE] {name}: legacy sidecar lacks exclude_subjects; not reusable.")
            return False
        keys = [k for k in expected if k in recorded]
        if "exclude_subjects" in expected and "exclude_subjects" not in keys:
            print(f"[REUSE] {name}: expected exclude_subjects but sidecar lacks it.")
            return False
    else:
        keys = list(expected.keys())

    for key in keys:
        value = expected[key]
        if key not in recorded:
            print(f"[REUSE] {name}: sidecar missing key '{key}'; not reusable.")
            return False
        if isinstance(value, float) and isinstance(recorded[key], (int, float)):
            if not np.isclose(recorded[key], value):
                print(f"[REUSE] {name}: '{key}' mismatch "
                      f"(recorded={recorded[key]}, expected={value}).")
                return False
        elif recorded[key] != value:
            print(f"[REUSE] {name}: '{key}' mismatch "
                  f"(recorded={recorded[key]!r}, expected={value!r}).")
            return False
    return True
