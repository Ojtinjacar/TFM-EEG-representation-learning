"""Supervised evaluation of a tabular descriptor against a developmental target.

Two baselines in this project answer the same question with different features: how much of
a child's development is already in the spectrum, before any network is involved. One reads
the FOOOF measures of ``apsd_baseline.py``, the other the expert descriptor of
``expert_baseline.py``. Everything between the feature matrix and the reported number is the
same for both, and it lives here, so that "evaluated the same way" is a fact of the code
rather than something two files have to keep agreeing on by hand.

Three decisions are worth stating because they are what makes these numbers comparable with
the rest of the project.

**The row is the unit the target varies over.** Age changes between the visits of a child,
so a row is a session and its label is the age of that visit; averaging the spectrum of the
four visits and labelling it with the first one measures something else entirely. An
intelligence quotient measured once does not change between visits, so there a row is a
child and the visits are averaged into it.

**The partition is the one of the whole cohort.** Folds are built over every subject before
anyone is dropped for lacking a target, which is what lets fold 3 here and fold 3 of
``run_downstream.py`` hold out the same children.

**Nothing fitted on the training side may see the test side.** The scaler and the ridge
penalty come from ``fit_probe``, which refits inside each inner split; the medians that fill
the gaps come from the training rows alone.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

from eval_expclr import fit_probe
from folds import BASE_SEED, N_FOLDS, canonical_subject_folds

SUBJECT_COL = "subject"
SESSION_COL = "age"

#: Row units a target can be evaluated at.
SESSION_LEVEL = "session"
SUBJECT_LEVEL = "subject"


class BaselineError(RuntimeError):
    """Raised when a baseline cannot be evaluated as requested."""


def natural_level(meta, target, subject_col=SUBJECT_COL, session_col=SESSION_COL):
    """Returns the row unit a target should be evaluated at.

    Decided from the data rather than from the name of the target: one that changes between
    the visits of a child has to keep them apart, and one that does not is a property of the
    child and is averaged into a single row.

    Args:
        meta (pd.DataFrame): Window metadata, one row per window.
        target (str): Name of the target column.
        subject_col (str): Column holding the subject identifier.
        session_col (str): Column holding the visit.

    Returns:
        str: :data:`SESSION_LEVEL` or :data:`SUBJECT_LEVEL`.

    Raises:
        BaselineError: If the target or the subject column is absent.
    """
    for column in (target, subject_col):
        if column not in meta.columns:
            raise BaselineError(f"The metadata carries no {column!r} column.")
    if session_col not in meta.columns:
        return SUBJECT_LEVEL

    varies = meta.groupby(subject_col)[target].nunique(dropna=True).max()
    return SESSION_LEVEL if varies is not None and varies > 1 else SUBJECT_LEVEL


def aggregate_rows(features, meta, target, level, method="mean",
                   subject_col=SUBJECT_COL, session_col=SESSION_COL):
    """Aggregates per-window features into the rows a target is evaluated over.

    Args:
        features (pd.DataFrame | np.ndarray): One row per window, aligned by position with
            ``meta``.
        meta (pd.DataFrame): Window metadata, one row per window.
        target (str): Name of the target column.
        level (str): :data:`SESSION_LEVEL` or :data:`SUBJECT_LEVEL`.
        method (str): ``mean`` or ``median``; missing values are skipped either way.
        subject_col (str): Column holding the subject identifier.
        session_col (str): Column holding the visit.

    Returns:
        tuple: ``(rows, feature_cols)``. ``rows`` carries the grouping keys, the aggregated
        features and the target, with the rows lacking a target already dropped.

    Raises:
        BaselineError: If the lengths disagree, if the aggregator is unknown, or if nothing
            survives the drop.
    """
    if method not in ("mean", "median"):
        raise BaselineError(f"Unknown aggregation {method!r}; use 'mean' or 'median'.")
    if level not in (SESSION_LEVEL, SUBJECT_LEVEL):
        raise BaselineError(f"Unknown level {level!r}.")

    frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    if len(frame) != len(meta):
        raise BaselineError(
            f"{len(frame)} feature rows for {len(meta)} windows: the descriptor is indexed "
            "by position, so a mismatch means the two describe different datasets."
        )

    feature_cols = list(frame.columns)
    keys = [subject_col] + ([session_col] if level == SESSION_LEVEL else [])
    # The age is both a key and a target, so it is asked for once: a duplicated column
    # cannot be grouped on.
    carried = list(dict.fromkeys(keys + [target]))
    combined = pd.concat(
        [frame.reset_index(drop=True), meta[carried].reset_index(drop=True)], axis=1
    )

    grouped = combined.groupby(keys, as_index=False)
    rows = getattr(grouped[feature_cols], method)()
    if target not in keys:
        # The target is constant within a group by construction, so it is carried over
        # rather than aggregated: averaging it would invent a label nobody was measured at.
        rows = rows.merge(grouped[target].first(), on=keys, how="inner")
    rows = rows.dropna(subset=[target])

    if rows.empty:
        raise BaselineError(f"No row is left with a value for {target!r}.")
    return rows.reset_index(drop=True), feature_cols


def impute_with_train_medians(features, train):
    """Fills the gaps of a descriptor with the column medians of the training split.

    Ridge cannot take a missing value. The medians come from the training side alone: taken
    from the whole set they would let the held-out child inform the values it is then scored
    on. Ported from ``run_expclr_folds.py``, which already did this correctly.

    Args:
        features (np.ndarray): Feature matrix, shape (n_rows, n_features).
        train (np.ndarray): Boolean mask selecting the training rows.

    Returns:
        tuple: ``(filled, imputed_fraction)``, the matrix with no missing values left and
        the fraction of cells that had to be filled.

    Raises:
        BaselineError: If a column is missing throughout the training split, which leaves
            nothing to fill it with.
    """
    missing = np.isnan(features)
    empty = missing[train].all(axis=0)
    if empty.any():
        raise BaselineError(
            f"{int(empty.sum())} feature columns are missing across the whole training "
            "split, so there is no median to fill them with."
        )
    medians = np.nanmedian(features[train], axis=0)
    filled = np.where(missing, medians, features)
    return filled, float(missing.mean()) if missing.size else 0.0


def compute_metrics(y_true, y_pred):
    """Returns RMSE, R2 and nRMSE of a set of predictions.

    R2 and nRMSE need the target to vary: with a single row, or with every row sharing the
    same label, they are undefined and reported as such rather than as zero.

    Args:
        y_true (np.ndarray): Observed values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: ``{"RMSE", "R2", "nRMSE"}``.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    spread = float(np.std(y_true))
    if len(y_true) < 2 or spread <= 1e-8:
        return {"RMSE": rmse, "R2": np.nan, "nRMSE": np.nan}
    return {"RMSE": rmse, "R2": float(r2_score(y_true, y_pred)), "nRMSE": rmse / spread}


def _fold_assignments(rows, all_subjects, cv_strategy, n_folds, base_seed,
                      fold_range=None, subject_col=SUBJECT_COL):
    """Returns the held-out subjects of each fold, with their absolute fold number.

    Args:
        rows (pd.DataFrame): Rows to be evaluated, one per session or per subject.
        all_subjects (list[str]): Every subject of the cohort, sorted.
        cv_strategy (str): ``kfold`` or ``leave_one_out``.
        n_folds (int): Splits of the k-fold.
        base_seed (int): Seed of the shuffle.
        fold_range (tuple[int, int] | None): Half-open range of folds to run.
        subject_col (str): Column holding the subject identifier.

    Returns:
        list[tuple[int, list[str]]]: Pairs of absolute fold index and held-out subjects.

    Raises:
        BaselineError: If the strategy is unknown.
    """
    if cv_strategy == "kfold":
        # Over the whole cohort, so that a fold holds out the same children here as it does
        # in run_downstream.py, whether or not they have a value for this target.
        assignments = list(enumerate(canonical_subject_folds(all_subjects, n_folds, base_seed)))
    elif cv_strategy == "leave_one_out":
        # One subject at a time, and only among those this target actually has, which is
        # what run_downstream.py does too.
        target_subjects = sorted(rows[subject_col].unique())
        assignments = [(i, [target_subjects[test[0]]])
                       for i, (_, test) in enumerate(LeaveOneOut().split(target_subjects))]
    else:
        raise BaselineError(f"Unknown cv strategy {cv_strategy!r}.")

    if fold_range is not None:
        start, end = fold_range
        assignments = assignments[start:end]
    return assignments


def run_cv(rows, feature_cols, target, all_subjects, cv_strategy="kfold", n_folds=N_FOLDS,
           base_seed=BASE_SEED, fold_range=None, subject_col=SUBJECT_COL, verbose=True):
    """Evaluates a descriptor out of fold and returns the per-fold results.

    Args:
        rows (pd.DataFrame): Output of :func:`aggregate_rows`.
        feature_cols (list[str]): Columns of ``rows`` holding the features.
        target (str): Name of the target column.
        all_subjects (list[str]): Every subject of the cohort, sorted.
        cv_strategy (str): ``kfold`` or ``leave_one_out``.
        n_folds (int): Splits of the k-fold.
        base_seed (int): Seed of the shuffle.
        fold_range (tuple[int, int] | None): Half-open range of folds to run.
        subject_col (str): Column holding the subject identifier.
        verbose (bool): Whether to report each fold as it closes.

    Returns:
        tuple: ``(results, predictions)``, one row per fold and one row per evaluated
        session or subject.

    Raises:
        BaselineError: If no fold could be evaluated.
    """
    X = rows[feature_cols].to_numpy(dtype=float)
    y = rows[target].to_numpy(dtype=float)
    subjects = rows[subject_col].astype(str).to_numpy()

    results, predictions = [], []
    for fold_idx, held_out in _fold_assignments(
        rows, all_subjects, cv_strategy, n_folds, base_seed, fold_range, subject_col
    ):
        test = np.isin(subjects, [str(s) for s in held_out])
        train = ~test
        if not test.any() or not train.any():
            # A fold can hold out only children without a value for this target. Saying so
            # is better than reporting a fold that never ran.
            if verbose:
                print(f"  [SKIP] fold {fold_idx}: no rows on one side for {target!r}",
                      flush=True)
            continue

        filled, imputed = impute_with_train_medians(X, train)
        scaler, probe = fit_probe(filled[train], y[train], subjects[train])
        y_pred = probe.predict(scaler.transform(filled[test]))

        metrics = compute_metrics(y[test], y_pred)
        results.append({
            "fold": fold_idx, "target": target,
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            "n_test_subjects": int(len(set(subjects[test]))),
            "alpha": float(probe.alpha), "imputed_frac": imputed,
            **metrics,
        })
        fold_predictions = rows.loc[test, [c for c in rows.columns if c not in feature_cols]].copy()
        fold_predictions["fold"] = fold_idx
        fold_predictions["y_true"] = y[test]
        fold_predictions["y_pred"] = y_pred
        predictions.append(fold_predictions)

        if verbose:
            print(f"  fold {fold_idx}: n_test={int(test.sum())} RMSE={metrics['RMSE']:.3f} "
                  f"R2={metrics['R2']:.3f} alpha={probe.alpha:g}", flush=True)

    if not results:
        raise BaselineError(f"No fold could be evaluated for {target!r}.")
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def aggregate_metrics(results, predictions, target, method, level):
    """Summarises the folds of one target into a single row.

    Both readings are reported because they answer different questions. The per-fold mean
    weighs every fold equally and its spread says how much the fold composition matters; the
    pooled figure is computed over every out-of-fold prediction at once and is the one to
    quote, since a fold holding two or three children gives an unstable R2.

    Args:
        results (pd.DataFrame): Per-fold results from :func:`run_cv`.
        predictions (pd.DataFrame): Out-of-fold predictions from :func:`run_cv`.
        target (str): Name of the target column.
        method (str): Name of the baseline, carried into the table.
        level (str): Row unit the target was evaluated at.

    Returns:
        dict: One summary row.
    """
    pooled = compute_metrics(predictions["y_true"], predictions["y_pred"])
    summary = {"target": target, "method": method, "level": level,
               "n_folds": int(results["fold"].nunique()),
               "n_rows": int(len(predictions)),
               "n_subjects": int(predictions["subject"].nunique()),
               "imputed_frac_mean": float(results["imputed_frac"].mean())}
    for name in ("RMSE", "R2", "nRMSE"):
        summary[f"{name}_mean"] = float(results[name].mean())
        summary[f"{name}_std"] = float(results[name].std())
        summary[f"{name}_pooled"] = pooled[name]
    return summary


def fold_suffix(cv_strategy, fold_range=None, fold_ranges_dict=None):
    """Builds the filename suffix that records which folds a run covered.

    Same convention as ``run_downstream.py``, so that shards written by different nodes sit
    next to each other without overwriting.

    Args:
        cv_strategy (str): ``kfold`` or ``leave_one_out``.
        fold_range (tuple[int, int] | None): Half-open range, k-fold only.
        fold_ranges_dict (dict | None): Half-open range per target, leave-one-out only.

    Returns:
        str: The suffix, empty when the run covered everything.
    """
    if cv_strategy == "kfold" and fold_range:
        return f"_folds{fold_range[0]}-{fold_range[1] - 1}"
    if cv_strategy == "leave_one_out" and fold_ranges_dict:
        parts = [f"{t}_{s}-{e - 1}" for t, (s, e) in sorted(fold_ranges_dict.items())]
        return "_" + "_".join(parts)
    return ""


def parse_fold_ranges(fold_ranges_arg):
    """Parses ``TARGET:START:END`` arguments into a dictionary.

    Args:
        fold_ranges_arg (list[str] | None): Arguments as given on the command line.

    Returns:
        dict: Target to half-open ``(start, end)``; empty when nothing was given.

    Raises:
        BaselineError: If an argument is malformed.
    """
    ranges = {}
    for item in fold_ranges_arg or []:
        parts = item.split(":")
        if len(parts) != 3:
            raise BaselineError(f"Expected TARGET:START:END, got {item!r}.")
        target, start, end = parts
        try:
            ranges[target] = (int(start), int(end))
        except ValueError as exc:
            raise BaselineError(f"Fold range of {item!r} is not a pair of integers.") from exc
    return ranges


def write_outputs(prefix, save_dir, raw, agg, predictions_by_target, cv_strategy, suffix):
    """Writes the three tables a baseline run produces.

    Args:
        prefix (str): Filename prefix identifying the baseline.
        save_dir (str): Directory the tables are written to.
        raw (pd.DataFrame): Per-fold results of every target.
        agg (pd.DataFrame): One summary row per target.
        predictions_by_target (dict): Target to its out-of-fold predictions.
        cv_strategy (str): Recorded in the filenames.
        suffix (str): Output of :func:`fold_suffix`.

    Returns:
        list[str]: Paths written.
    """
    os.makedirs(save_dir, exist_ok=True)
    written = []

    raw_path = os.path.join(save_dir, f"{prefix}_raw_results_{cv_strategy}{suffix}.csv")
    raw.to_csv(raw_path, index=False)
    written.append(raw_path)

    agg_path = os.path.join(save_dir, f"{prefix}_agg_results_{cv_strategy}{suffix}.csv")
    agg.to_csv(agg_path, index=False)
    written.append(agg_path)

    for target, frame in predictions_by_target.items():
        path = os.path.join(
            save_dir, f"{prefix}_predictions_{target}_{cv_strategy}{suffix}.csv"
        )
        frame.to_csv(path, index=False)
        written.append(path)

    for path in written:
        print(f"[INFO] Written: {path}", flush=True)
    return written


def evaluate_targets(features, meta, targets, prefix, save_dir, method_name,
                     cv_strategy="kfold", n_folds=N_FOLDS, base_seed=BASE_SEED,
                     fold_range=None, fold_ranges_dict=None, aggregation="mean",
                     subject_col=SUBJECT_COL, verbose=True):
    """Runs the whole evaluation for every target and writes the tables.

    Args:
        features (pd.DataFrame): One row per window, aligned by position with ``meta``.
        meta (pd.DataFrame): Window metadata, one row per window.
        targets (list[str]): Targets to evaluate.
        prefix (str): Filename prefix identifying the baseline.
        save_dir (str): Directory the tables are written to.
        method_name (str): Name of the baseline, carried into the tables.
        cv_strategy (str): ``kfold`` or ``leave_one_out``.
        n_folds (int): Splits of the k-fold.
        base_seed (int): Seed of the shuffle.
        fold_range (tuple[int, int] | None): Half-open range of folds, k-fold only.
        fold_ranges_dict (dict | None): Half-open range per target, leave-one-out only.
        aggregation (str): ``mean`` or ``median``, how windows are pooled into a row.
        subject_col (str): Column holding the subject identifier.
        verbose (bool): Whether to report each fold as it closes.

    Returns:
        tuple: ``(raw, agg, predictions_by_target)``.

    Raises:
        BaselineError: If no target could be evaluated.
    """
    all_subjects = sorted(meta[subject_col].astype(str).unique())
    raw_frames, summaries, predictions_by_target = [], [], {}

    for target in targets:
        if target not in meta.columns:
            print(f"[WARNING] The metadata carries no {target!r} column; skipping.",
                  flush=True)
            continue
        if cv_strategy == "leave_one_out" and fold_ranges_dict and target not in fold_ranges_dict:
            # Same rule as run_downstream.py: with explicit ranges, a target without one is
            # someone else's shard, not this node's work.
            print(f"  [SKIP] No fold range specified for target {target!r}", flush=True)
            continue

        level = natural_level(meta, target, subject_col)
        rows, feature_cols = aggregate_rows(
            features, meta, target, level, aggregation, subject_col
        )
        if verbose:
            print(f"\n[{target}] level={level}, {len(rows)} rows, "
                  f"{rows[subject_col].nunique()} subjects, {len(feature_cols)} features",
                  flush=True)

        target_range = fold_range
        if cv_strategy == "leave_one_out" and fold_ranges_dict:
            target_range = fold_ranges_dict[target]

        results, predictions = run_cv(
            rows, feature_cols, target, all_subjects, cv_strategy, n_folds, base_seed,
            target_range, subject_col, verbose,
        )
        results["method"], results["level"] = method_name, level
        raw_frames.append(results)
        summaries.append(aggregate_metrics(results, predictions, target, method_name, level))
        predictions_by_target[target] = predictions

    if not raw_frames:
        raise BaselineError("No target could be evaluated.")

    raw = pd.concat(raw_frames, ignore_index=True)
    agg = pd.DataFrame(summaries)
    suffix = fold_suffix(cv_strategy, fold_range, fold_ranges_dict)
    write_outputs(prefix, save_dir, raw, agg, predictions_by_target, cv_strategy, suffix)
    return raw, agg, predictions_by_target
