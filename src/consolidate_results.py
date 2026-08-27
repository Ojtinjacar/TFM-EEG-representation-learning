"""Pools the fold results of every run into one comparison base.

Each run writes ``save/<run>/results/<variant>_<zone>_<band>_<target>_fold<k>.csv``, so
every dimension that tells two results apart is already in the name and in the row. This
reads that tree, splits the packed variant label into the dimensions a query actually asks
about, and writes one folder per method.

Two tables come out of it, because the questions are answered at different levels. The
fold table carries what the pipeline computed, which is a metric taken *within* each
subject and then averaged. The subject table unpacks the per-subject averages, which is
what pooling over the whole cohort needs; that pooled figure is the one the writeup
reports, so a notebook that reads ``r2`` from the fold table is reading something else.

Nothing here records which machine produced a result. The work is split across machines by
method, but the split is an operational detail: the same campaign run on one machine or on
four has to give the same base.

Run from the repository root::

    python src/consolidate_results.py --source save
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict

ZONES = ("occipital", "parietal", "frontal", "central", "all")

# The metric suffix each run wrote, and the name it stands for.
_METRICS = {"cosine": "cosine", "wasser": "wasserstein", "riemann": "riemann"}
_STRATEGIES = ("nbr", "xsubj", "diffage", "lag0")

# The descriptor each ExpCLR variant compares against. The variant label alone does not
# say it, and it is what tells two of these runs apart.
_EXPCLR_DESCRIPTOR = {
    "ExpCLR": "P_madurativo",
    "ExpCLR-full": "P_full",
    "ExpCLR-aper": "P_aper",
}

# The convention a result file follows. Anything else under results/ is not a result: the
# pipeline also drops aggregate files there when it closes a zone, and pooling those would
# count every fold twice.
RESULT_NAME = re.compile(
    r"^(?P<variant>.+?)_(?P<zone>" + "|".join(ZONES) + r")"
    r"_(?P<frequency>[^_]+)_(?P<target>.+)_fold(?P<fold>\d+)\.csv$"
)

FOLD_COLUMNS = ["family", "variant", "strategy", "metric", "descriptor", "zone",
                "frequency", "target", "eval_mode", "fold", "nrmse", "r2", "rmse",
                "campaign", "source_file"]
SUBJECT_COLUMNS = ["family", "variant", "strategy", "metric", "descriptor", "zone",
                   "frequency", "target", "eval_mode", "fold", "subject_index",
                   "y_true", "y_pred", "campaign"]


class ConsolidationError(RuntimeError):
    """Raised when a result file cannot be attributed to a run."""


def is_result_file(name: str) -> bool:
    """Says whether a filename is a per-fold result of the convention.

    Args:
        name (str): Basename of the file.

    Returns:
        bool: True if the name carries variant, zone, band, target and fold.
    """
    return RESULT_NAME.match(name) is not None


def normalise(method: str) -> dict:
    """Splits a packed method label into the dimensions a query asks about.

    Args:
        method (str): Raw ``method`` value of the row.

    Returns:
        dict: ``family``, ``variant``, ``strategy``, ``metric`` and ``descriptor``.
    """
    out = {"family": method, "variant": method,
           "strategy": "", "metric": "", "descriptor": ""}

    if method.startswith("InterFusion"):
        out["family"] = "InterFusion"
    elif method in _EXPCLR_DESCRIPTOR:
        out["family"] = "ExpCLR"
        out["descriptor"] = _EXPCLR_DESCRIPTOR[method]
    elif method.startswith("SimCLR"):
        out["family"] = "SimCLR"
        parts = method.split("-")
        if len(parts) == 3 and parts[1] in _STRATEGIES:
            out["strategy"] = parts[1]
            out["metric"] = _METRICS.get(parts[2], parts[2])
        else:
            # The control: its positive is a synthetic perturbation, not a real window.
            out["strategy"] = "augment"
    return out


def read_rows(path: str, campaign: str):
    """Reads one result file into normalised fold and subject rows.

    The zone and the band are read from the row and never inferred from where the file
    sits: a result that depends on its own path is a result somebody can move and change.

    Args:
        path (str): Path of the CSV.
        campaign (str): Run the file belongs to.

    Yields:
        tuple: ``(fold_row, subject_rows)`` per row of the file.

    Raises:
        ConsolidationError: If a row lacks its zone or band, or carries no per-subject
            averages.
    """
    with open(path) as fh:
        for row in csv.DictReader(fh):
            method = row["method"]
            zone, frequency = row.get("zone"), row.get("frequency")
            if zone not in ZONES or not frequency:
                raise ConsolidationError(
                    f"{path}: {method} fold {row.get('fold')} records zone={zone!r} "
                    f"frequency={frequency!r}. Both belong in the row; a result whose zone "
                    "has to be guessed from its directory is not attributable."
                )

            common = dict(normalise(method), zone=zone, frequency=frequency,
                          target=row["target"], eval_mode=row["eval_mode"],
                          fold=int(row["fold"]), campaign=campaign)

            fold_row = dict(common, nrmse=row["nRMSE"], r2=row["R2"], rmse=row["RMSE"],
                            source_file=os.path.basename(path))

            packed = row.get("subject_avgs", "")
            if not packed.strip():
                raise ConsolidationError(
                    f"{path}: {method} fold {row['fold']} carries no subject averages, "
                    "so nothing can be pooled from it."
                )
            subject_rows = []
            for i, pair in enumerate(p for p in packed.split(";") if p.strip()):
                y_true, y_pred = pair.split(",")
                subject_rows.append(dict(common, subject_index=i,
                                         y_true=y_true, y_pred=y_pred))
            yield fold_row, subject_rows


def collect(source: str):
    """Reads every run under the source tree.

    Args:
        source (str): Directory holding ``<run>/results/*.csv``.

    Returns:
        tuple: ``(fold_rows, subject_rows, manifest_entries)``.
    """
    folds, subjects, manifest = [], [], []
    for campaign in sorted(os.listdir(source)):
        results_dir = os.path.join(source, campaign, "results")
        if not os.path.isdir(results_dir):
            continue
        for name in sorted(f for f in os.listdir(results_dir) if is_result_file(f)):
            path = os.path.join(results_dir, name)
            n = 0
            for fold_row, subject_rows in read_rows(path, campaign):
                folds.append(fold_row)
                subjects.extend(subject_rows)
                n += 1
            manifest.append({
                "campaign": campaign, "file": name, "rows": n,
                "md5": hashlib.md5(open(path, "rb").read()).hexdigest(),
            })
    return folds, subjects, manifest


def deduplicate(folds: list, subjects: list):
    """Drops rows that describe the same run of the same fold twice.

    The convention should make this impossible, since a result names every dimension that
    tells it apart. It is kept as a net: if it ever fires, two runs are writing the same
    name and the count is what makes that visible instead of silent.

    Args:
        folds (list): Fold rows.
        subjects (list): Subject rows.

    Returns:
        tuple: ``(folds, subjects, n_dropped)``.
    """
    def key(r):
        return (r["variant"], r["zone"], r["frequency"], r["target"],
                r["eval_mode"], r["fold"])

    seen, kept_folds = set(), []
    for r in folds:
        if key(r) in seen:
            continue
        seen.add(key(r))
        kept_folds.append(r)

    # The subject rows need the same treatment and not a membership test against the folds
    # kept: every copy of a repeated fold shares its key, so a filter by key keeps them all
    # and the cohort silently grows past its size.
    seen_subjects, kept_subjects = set(), []
    for r in subjects:
        k = key(r) + (r["subject_index"],)
        if k in seen_subjects:
            continue
        seen_subjects.add(k)
        kept_subjects.append(r)
    return kept_folds, kept_subjects, len(folds) - len(kept_folds)


def keep_complete(folds: list, subjects: list, n_folds: int):
    """Drops runs that never covered every fold.

    A run left half done is not a weaker version of a finished one: pooling it gives a
    figure over a fraction of the cohort that reads like the real thing. While a campaign is
    still executing, most of what is on disk is exactly that.

    Args:
        folds (list): Fold rows.
        subjects (list): Subject rows.
        n_folds (int): Folds a complete run has.

    Returns:
        tuple: ``(folds, subjects, dropped)`` where ``dropped`` maps the incomplete runs
            to how many folds they had.
    """
    def group(r):
        return (r["variant"], r["zone"], r["frequency"], r["target"], r["eval_mode"])

    counts = defaultdict(set)
    for r in folds:
        counts[group(r)].add(r["fold"])
    incomplete = {g: len(f) for g, f in counts.items() if len(f) != n_folds}

    kept_folds = [r for r in folds if group(r) not in incomplete]
    kept_subjects = [r for r in subjects if group(r) not in incomplete]
    return kept_folds, kept_subjects, incomplete


def write_csv(path: str, rows: list, columns: list):
    """Writes rows to a CSV with a fixed column order.

    Args:
        path (str): Destination file.
        rows (list): Rows as dicts.
        columns (list): Column order.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_by_method(out_dir: str, folds: list):
    """Writes one folder per method, one file per zone inside it.

    Args:
        out_dir (str): Root of the ``results`` tree.
        folds (list): Fold rows.

    Returns:
        int: Number of files written.
    """
    grouped = defaultdict(list)
    for r in folds:
        grouped[(r["variant"], r["zone"])].append(r)
    for (variant, zone), rows in sorted(grouped.items()):
        write_csv(os.path.join(out_dir, "by_method", variant, f"{zone}.csv"),
                  sorted(rows, key=lambda r: (r["target"], r["eval_mode"], r["fold"])),
                  FOLD_COLUMNS)
    return len(grouped)


def main(args):
    """Consolidates the results of every run and reports what was written.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    folds, subjects, manifest = collect(args.source)
    if not folds:
        raise SystemExit(
            f"[ERROR] No result files under {args.source}/<run>/results/. Expected names "
            "like SimCLR-xsubj-cosine_occipital_all_age_fold0.csv."
        )
    folds, subjects, dropped = deduplicate(folds, subjects)
    folds, subjects, incomplete = keep_complete(folds, subjects, args.n_folds)

    if incomplete:
        print(f"[WARN] {len(incomplete)} runs dropped for being incomplete:")
        for (variant, zone, _, target, mode), n in sorted(incomplete.items()):
            print(f"  {variant} / {zone} / {target} / {mode}: {n} of {args.n_folds} folds")

    if not folds:
        raise SystemExit(
            f"[ERROR] Every run under {args.source} is short of {args.n_folds} folds; "
            "nothing complete to consolidate."
        )

    write_csv(os.path.join(args.out, "results_folds.csv"), folds, FOLD_COLUMNS)
    write_csv(os.path.join(args.out, "results_subjects.csv"), subjects, SUBJECT_COLUMNS)
    n_files = write_by_method(args.out, folds)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    campaigns = sorted({r["campaign"] for r in folds})
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as fh:
        json.dump({"repo_commit": commit, "source": args.source,
                   "campaigns": campaigns,
                   "fold_rows": len(folds), "subject_rows": len(subjects),
                   "duplicate_fold_rows_dropped": dropped,
                   "incomplete_runs_dropped": {"/".join(g): n
                                               for g, n in sorted(incomplete.items())},
                   "files": manifest}, fh, indent=2, sort_keys=True)

    variants = {r["variant"] for r in folds}
    print(f"{len(folds)} fold rows, {len(subjects)} subject rows, "
          f"{dropped} duplicates dropped")
    print(f"{len(variants)} methods over {n_files} method/zone files")
    print(f"campaigns: {', '.join(campaigns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default="save",
                        help="Directory holding <run>/results/*.csv.")
    parser.add_argument("--out", type=str, default="results",
                        help="Directory to write the consolidated base to.")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Folds a complete run has; anything short of it is dropped.")
    main(parser.parse_args())
