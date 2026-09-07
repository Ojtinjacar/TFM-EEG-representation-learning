"""Resolution and loading of trained encoders.

Finds the checkpoint that belongs to a (method, fold, zone, frequency) and loads
its backbone with the weights frozen. The checkpoint alone does not say what it
was trained on, so the sidecar written next to it is what decides the match: a
name that merely looks right is refused rather than guessed at.

Used by the latent-space maps, which need an encoder before they can represent
anything.
"""

import glob
import json
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import run_downstream as rd
from checkpoint_naming import sidecar_path
from downstream import BACKBONE_BY_METHOD
from interfusion import InterFusionEEG


class CheckpointNotFound(FileNotFoundError):
    """Raised when no checkpoint matches a requested method and fold."""


def expected_sidecar(method, fold_id, zone, frequency):
    """Returns the sidecar fields that identify the checkpoint of a method.

    Args:
        method (str): Method name, including the SimCLR and ExpCLR variants.
        fold_id (str): Fold identifier, e.g. ``fold0``.
        zone (str): Head zone the encoder trained on.
        frequency (str): Frequency band.

    Returns:
        dict: Fields a matching sidecar must record.
    """
    expected = {"zone": zone, "frequency": frequency, "fold_id": fold_id}
    simclr = rd.build_simclr_variants(zone)
    if method in simclr:
        tag = simclr[method]["tag"]
        expected["method"] = "SimCLR"
        # A variant carries its tag inside the fold id: fold0_xscosine.
        expected["fold_id"] = f"{fold_id}_{tag}" if tag else fold_id
    elif method in rd.EXPCLR_VARIANTS:
        # The variants differ in the descriptor, which the filename shares otherwise.
        expected["method"] = "ExpCLR"
        expected["descriptor"] = rd.EXPCLR_VARIANTS[method]["descriptor"]
    else:
        expected["method"] = method
    return expected


def find_checkpoint(method, fold_id, model_dirs, zone, frequency):
    """Locates the checkpoint a method wrote for one fold, zone and band.

    The sidecar decides rather than the filename. A name cannot tell two zones of equal
    channel count apart, and several checkpoints of one fold can differ in something it does
    not carry at all, so an ambiguous match is refused instead of resolved alphabetically:
    loading the wrong one would measure a different encoder under the right label.

    Args:
        method (str): Method name, including variants.
        fold_id (str): Fold identifier.
        model_dirs (list): Directories to search, in priority order.
        zone (str): Head zone the encoder must have trained on.
        frequency (str): Frequency band it must have trained on.

    Returns:
        str: Path to the checkpoint.

    Raises:
        CheckpointNotFound: If nothing matches, or if more than one does.
    """
    expected = expected_sidecar(method, fold_id, zone, frequency)
    for directory in model_dirs:
        matches = []
        for sidecar in sorted(glob.glob(os.path.join(directory, "*_config.json"))):
            checkpoint = sidecar.replace("_config.json", ".pth")
            if not os.path.exists(checkpoint):
                continue
            with open(sidecar) as fh:
                recorded = json.load(fh)
            if all(recorded.get(key) == value for key, value in expected.items()):
                matches.append(checkpoint)
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise CheckpointNotFound(
                f"'{method}' {fold_id} on {zone}/{frequency} matches {len(matches)} "
                f"checkpoints in {directory}: {[os.path.basename(m) for m in matches]}. "
                "They differ in something their sidecars do not distinguish, so pick one "
                "with --model_dirs rather than have it chosen alphabetically."
            )
    raise CheckpointNotFound(
        f"No checkpoint for '{method}' {fold_id} on zone {zone!r}, band {frequency!r} in "
        f"{model_dirs}. Checkpoints without a sidecar are not eligible."
    )


def load_backbone(method, model_path, x_dim, window, embedding_size, sfreq, device):
    """Rebuilds a frozen backbone exposing ``get_embedding``.

    Args:
        method (str): Method name.
        model_path (str): Checkpoint path.
        x_dim (int): Number of channels.
        window (int): Samples per window.
        embedding_size (int): Embedding width for the legacy encoders.
        sfreq (int): Sampling frequency.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Backbone in eval mode.

    Raises:
        ValueError: If the method has no frozen-backbone mapping.
    """
    if method == "InterFusion":
        with open(sidecar_path(model_path)) as fh:
            sidecar = json.load(fh)
        backbone = InterFusionEEG(
            x_dim=x_dim, window=window, z_dim=sidecar["z_dim"],
            strides=tuple(sidecar.get("strides", (2, 1, 2, 1, 2, 2, 2))),
            rnn_hidden=sidecar["rnn_hidden"],
            dense_hidden=sidecar.get("dense_hidden", 500),
            flow_levels=sidecar["flow_levels"],
            embedding_stats=sidecar.get("embedding_stats", "mean"),
        ).to(device)
    else:
        base = "SimCLR" if method in rd.build_simclr_variants("all") else method
        base = "ExpCLR" if method in rd.EXPCLR_VARIANTS else base
        model_class = BACKBONE_BY_METHOD.get(base)
        if model_class is None:
            raise ValueError(f"Method '{method}' has no frozen-backbone mapping")
        backbone = model_class(
            input_size=window, hidden_size=embedding_size, n_channels=x_dim,
            sfreq=sfreq, lstm_hidden_size=embedding_size // 2,
        ).to(device)

    backbone.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    backbone.eval()
    return backbone
