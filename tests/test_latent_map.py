"""Tests for the latent-space maps and the measures that back them."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from latent_geometry import subject_identifiability  # noqa: E402
from latent_map import (  # noqa: E402
    COLOURINGS,
    ProjectionError,
    colour_values,
    draw_grid,
    geometry_metrics,
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


def test_a_faithful_projection_is_trusted_and_a_random_one_is_not(latent):
    """Without the negative control, the measure could return a plausible number on anything."""
    embeddings, meta = latent
    rng = np.random.default_rng(1)

    faithful = geometry_metrics(embeddings, embeddings[:, :2], meta)
    scrambled = geometry_metrics(embeddings, rng.normal(size=(len(embeddings), 2)), meta)

    assert faithful["trustworthiness"] > scrambled["trustworthiness"]
    assert scrambled["trustworthiness"] < 0.8


def test_identity_is_recovered_when_encoded_and_not_when_absent(latent):
    embeddings, meta = latent
    rng = np.random.default_rng(2)
    sessions = meta[["subject", "age"]].astype(str).agg("_".join, axis=1).to_numpy()

    encoded = subject_identifiability(embeddings, meta["subject"], sessions)
    noise = subject_identifiability(rng.normal(size=embeddings.shape), meta["subject"],
                                    sessions)

    assert encoded["times_chance"] > 3
    assert noise["times_chance"] < 2
    assert encoded["chance"] == pytest.approx(1 / 8)


def test_a_subject_with_one_session_is_skipped_rather_than_counted_as_a_miss():
    rng = np.random.default_rng(3)
    subjects = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 5)
    sessions = np.array(["A1"] * 5 + ["A2"] * 5 + ["B1"] * 5 + ["B2"] * 5 + ["C1"] * 5)
    result = subject_identifiability(rng.normal(size=(25, 4)), subjects, sessions)

    # C has a single session, so it cannot appear in its own training split.
    assert result["n_scored"] == 20
    assert result["chance"] == pytest.approx(1 / 3)


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


def test_the_age_silhouette_is_higher_when_age_separates(latent):
    embeddings, meta = latent
    rng = np.random.default_rng(4)
    by_age = np.array([[a * 1.0, 0, 0, 0, 0, 0] for a in meta["age"]], dtype=float)
    by_age += rng.normal(scale=0.1, size=by_age.shape)

    separated = geometry_metrics(by_age, by_age[:, :2], meta)["silhouette_age"]
    mixed = geometry_metrics(rng.normal(size=embeddings.shape),
                             rng.normal(size=(len(meta), 2)), meta)["silhouette_age"]
    assert separated > mixed
