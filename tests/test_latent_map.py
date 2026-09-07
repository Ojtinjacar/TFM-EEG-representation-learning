"""Tests for the latent-space maps."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from latent_map import (  # noqa: E402
    COLOURINGS,
    ProjectionError,
    colour_values,
    draw_grid,
    project,
)


@pytest.fixture
def latent():
    """A space with structure: 8 subjects, 4 sessions each, 10 windows per session."""
    rng = np.random.default_rng(0)
    rows, subjects, ages, blocks = [], [], [], []
    for s in range(8):
        centre = rng.normal(size=6) * 3
        for age in (6, 9, 16, 36):
            for _ in range(10):
                rows.append(centre + age * 0.1 + rng.normal(scale=0.3, size=6))
                subjects.append(f"S{s:02d}")
                ages.append(age)
                blocks.append(1)
    meta = pd.DataFrame({"subject": subjects, "age": ages, "block": blocks})
    return np.array(rows), meta


def test_every_projection_returns_two_dimensions_and_states_its_settings(latent):
    embeddings, _ = latent
    for technique in ("tsne", "pca"):
        coords, caption = project(embeddings, technique)
        assert coords.shape == (len(embeddings), 2)
        assert technique in caption and "random_state=42" in caption


def test_an_unknown_projection_is_refused(latent):
    embeddings, _ = latent
    with pytest.raises(ProjectionError, match="Unknown projection"):
        project(embeddings, "mds")


def test_every_colouring_returns_one_value_per_window(latent):
    embeddings, meta = latent
    features = pd.DataFrame({"total_power_frontal": np.arange(len(meta), dtype=float)})

    for kind in COLOURINGS:
        values, label, categorical = colour_values(kind, meta, embeddings, features)
        assert len(values) == len(meta), kind
        assert isinstance(label, str) and label
        assert isinstance(categorical, bool)


def test_band_power_without_features_says_so(latent):
    embeddings, meta = latent
    with pytest.raises(ValueError, match="band_power needs"):
        colour_values("band_power", meta, embeddings, features=None)


def test_an_unknown_colouring_is_refused(latent):
    embeddings, meta = latent
    with pytest.raises(ValueError, match="Unknown colouring"):
        colour_values("temperament", meta, embeddings)


def test_the_grid_draws_one_panel_per_method_and_colouring(latent, tmp_path):
    embeddings, meta = latent
    coords, caption = project(embeddings, "pca")
    colours = {k: colour_values(k, meta, embeddings) for k in ("age", "subject")}
    panels = {"AE": (coords, colours), "SimCLR": (coords, colours)}

    out = tmp_path / "grid.png"
    draw_grid(panels, ["age", "subject"], caption, str(out))
    assert out.exists() and out.stat().st_size > 0
