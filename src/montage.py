"""Single source of truth for the electrode montage and its regions of interest.

The recording carries 109 electrodes; the four regions cover 70 of them and the
remaining 39 belong to none. ``postprocessing.py`` therefore selects channels by
**membership**, keeping the montage order, so the processed array starts at E3
and ends at Cz rather than at E1.

Reconstructing those names by truncating the montage list to the channel count is
wrong and silently so: it drops 25 of the selected channels, admits 25 that are
absent, and leaves zero positions in agreement. Processed datasets therefore
record their own ``channel_names.txt`` and consumers read it instead of guessing.
"""

import os

PRESET_ZONES = {
    "central": ['E35', 'E29', 'E13', 'E6', 'E112', 'E111', 'E110', 'E41', 'E36', 'E30',
                'E7', 'E106', 'E105', 'E104', 'E103', 'E47', 'E42', 'E37', 'E31', 'Cz',
                'E80', 'E87', 'E93', 'E98', 'E54', 'E55', 'E79'],
    "frontal": ['E33', 'E34', 'E27', 'E23', 'E18', 'E16', 'E10', 'E3', 'E123', 'E116',
                'E122', 'E28', 'E24', 'E19', 'E11', 'E4', 'E124', 'E117', 'E20', 'E12',
                'E5', 'E118'],
    "parietal": ['E52', 'E53', 'E61', 'E62', 'E78', 'E86', 'E60', 'E67', 'E72', 'E77',
                 'E85', 'E92', 'E59', 'E91'],
    "occipital": ['E66', 'E71', 'E76', 'E84', 'E70', 'E75', 'E83'],
}

ROIS = ("frontal", "central", "parietal", "occipital")

MONTAGE_SIDECAR = "channel_names.txt"


class MontageError(RuntimeError):
    """Raised when channel names cannot be matched to the array they describe."""


def load_channel_names(path):
    """Returns the channel names of a montage file, one per line."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def select_channel_indices(zones, all_channels):
    """Returns the indices of the channels belonging to the given zones.

    Selection is by membership and preserves the montage order, so the result is
    strictly increasing and the zone order of the argument is irrelevant.

    Args:
        zones (list): Region names to keep.
        all_channels (list): Channel names of the full montage, in recording order.

    Returns:
        list: Indices into the channel axis, sorted ascending.

    Raises:
        MontageError: If a zone is not one of the four regions.
    """
    selected = set()
    for zone in zones:
        if zone not in PRESET_ZONES:
            raise MontageError(
                f"Zone {zone!r} is not one of {sorted(PRESET_ZONES)}.")
        selected.update(PRESET_ZONES[zone])
    return [i for i, ch in enumerate(all_channels) if ch in selected]


def roi_indices(channels, n_channels):
    """Maps each region of interest to its channels of the recording.

    Args:
        channels (list): Channel names of the montage, in recording order.
        n_channels (int): Number of channels of the array to describe.

    Returns:
        dict: Region name to channel indices.

    Raises:
        MontageError: If the names do not account for the channels of the array,
            or if a region ends up with no channel of its own.
    """
    if len(channels) != n_channels:
        raise MontageError(
            f"{len(channels)} channel names for an array of {n_channels} channels: the "
            "regions are resolved by position, so both must describe the same montage.")
    indices = {roi: [i for i, ch in enumerate(channels) if ch in set(PRESET_ZONES[roi])]
               for roi in ROIS}
    empty = [roi for roi, idx in indices.items() if not idx]
    if empty:
        raise MontageError(
            f"No channel of the montage belongs to {empty}. The regions span the whole "
            "descriptor and cannot be built from a montage that misses one.")
    return indices


def resolve_processed_montage(data_path, n_channels, channels_txt=None):
    """Returns the channel names of a processed dataset.

    Reads the ``channel_names.txt`` written next to ``processed_windows.npy``.
    ``channels_txt`` overrides it, and is validated the same way: a montage of a
    different size is an error, never something to truncate.

    Args:
        data_path (str): Path of the ``.npy`` whose montage is needed.
        n_channels (int): Channel count of that array.
        channels_txt (str | None): Explicit montage file to use instead.

    Returns:
        list: Channel names, one per channel of the array, in the same order.

    Raises:
        MontageError: If the montage is missing or does not match the array.
    """
    path = channels_txt or os.path.join(os.path.dirname(data_path), MONTAGE_SIDECAR)
    if not os.path.exists(path):
        raise MontageError(
            f"{data_path} has {n_channels} channels but no montage next to it "
            f"({path} is missing), and the names cannot be guessed from the full "
            "montage: the selection keeps channels by region membership, not by "
            "position. Regenerate the dataset with src/postprocessing.py, which now "
            f"writes {MONTAGE_SIDECAR} alongside the windows.")
    channels = load_channel_names(path)
    if len(channels) != n_channels:
        raise MontageError(
            f"{path} lists {len(channels)} channels but {data_path} has {n_channels}. "
            "Truncating would misalign every region; regenerate the dataset instead.")
    return channels
