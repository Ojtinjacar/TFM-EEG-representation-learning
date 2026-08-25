"""Pools the fold results of every run into one comparison base.

Results were written by launches that each chose their own output directory, so the zone
of a run lives in the path rather than in the rows, and some method labels carry the zone
inside them. This normalises both: it recovers the zone, splits the packed variant label
into the dimensions a query actually asks about, and writes one folder per method.

Two tables come out of it, because the questions are answered at different levels. The
fold table carries what the pipeline computed, which is a metric taken *within* each
subject and then averaged. The subject table unpacks the per-subject averages, which is
what pooling over the whole cohort needs; that pooled figure is the one the writeup
reports, so a notebook that reads ``r2`` from the fold table is reading something else.

Run from the repository root::

    python src/consolidate_results.py --source <dir with <machine>/<save path>/*.csv>
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
WHOLE_HEAD = "all"

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

FOLD_COLUMNS = ["family", "variant", "strategy", "metric", "descriptor", "zone",
                "frequency", "target", "eval_mode", "fold", "nrmse", "r2", "rmse",
                "canonical", "campaign", "machine", "source_dir", "source_file"]
SUBJECT_COLUMNS = ["family", "variant", "strategy", "metric", "descriptor", "zone",
                   "frequency", "target", "eval_mode", "fold", "subject_index",
                   "y_true", "y_pred", "canonical", "campaign", "machine", "source_dir"]


class ConsolidationError(RuntimeError):
    """Raised when a result file cannot be attributed to a run."""


def zone_of(source_dir: str, row: dict, method: str) -> str:
    """Recovers the zone of a result row.

    Newer runs carry it as a column; older ones only in the output directory, and the
    InterFusion launches also glued it onto the method label.

    Args:
        source_dir (str): Directory the file came from, relative to ``save``.
        row (dict): The row as read, which may carry a ``zone`` column.
        method (str): Raw method label of the row.

    Returns:
        str: One of :data:`ZONES`.

    Raises:
        ConsolidationError: If no zone can be recovered.
    """
    if row.get("zone") in ZONES:
        return row["zone"]
    for zone in ZONES:
        if method.endswith(f"_{zone}"):
            return zone
    parts = [p for p in source_dir.split("/") if p]
    if parts and parts[-1] in ZONES:
        return parts[-1]
    for zone in ZONES:
        if zone != WHOLE_HEAD and parts and parts[-1].endswith(f"_{zone}"):
            return zone
    # Everything else was a whole-montage run: those launches never named the zone
    # because there was only one.
    return WHOLE_HEAD


def normalise(method: str, zone: str) -> dict:
    """Splits a packed method label into the dimensions a query asks about.

    Args:
        method (str): Raw ``method`` value of the row.
        zone (str): Zone already recovered for the row.

    Returns:
        dict: ``family``, ``variant``, ``strategy``, ``metric`` and ``descriptor``.
    """
    variant = method
    for z in ZONES:
        if variant.endswith(f"_{z}"):
            variant = variant[: -len(f"_{z}")]
            break

    out = {"family": variant, "variant": variant,
           "strategy": "", "metric": "", "descriptor": ""}

    if variant.startswith("InterFusion"):
        out["family"] = "InterFusion"
    elif variant in _EXPCLR_DESCRIPTOR:
        out["family"] = "ExpCLR"
        out["descriptor"] = _EXPCLR_DESCRIPTOR[variant]
    elif variant.startswith("SimCLR"):
        out["family"] = "SimCLR"
        parts = variant.split("-")
        if len(parts) == 3 and parts[1] in _STRATEGIES:
            out["strategy"] = parts[1]
            out["metric"] = _METRICS.get(parts[2], parts[2])
        else:
            # The control: its positive is a synthetic perturbation, not a real window.
            out["strategy"] = "augment"
    return out


def read_rows(path: str, machine: str, source_dir: str):
    """Reads one result file into normalised fold and subject rows.

    Args:
        path (str): Path of the CSV.
        machine (str): Machine the file came from.
        source_dir (str): Directory of the file, relative to ``save``.

    Yields:
        tuple: ``(fold_row, subject_rows)`` per row of the file.

    Raises:
        ConsolidationError: If a row carries no per-subject averages.
    """
    with open(path) as fh:
        for row in csv.DictReader(fh):
            method = row["method"]
            zone = zone_of(source_dir, row, method)
            names = normalise(method, zone)
            common = dict(names, zone=zone,
                          frequency=row.get("frequency") or "all",
                          target=row["target"], eval_mode=row["eval_mode"],
                          fold=int(row["fold"]))

            fold_row = dict(common, nrmse=row["nRMSE"], r2=row["R2"], rmse=row["RMSE"],
                            machine=machine, source_dir=source_dir,
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
                subject_rows.append(dict(common, subject_index=i, y_true=y_true,
                                         y_pred=y_pred, machine=machine,
                                         source_dir=source_dir))
            yield fold_row, subject_rows


def collect(source: str):
    """Walks the extracted tree and normalises everything under it.

    Args:
        source (str): Directory holding ``<machine>/<save path>/*.csv``.

    Returns:
        tuple: ``(fold_rows, subject_rows, manifest_entries)``.
    """
    folds, subjects, manifest = [], [], []
    for machine in sorted(os.listdir(source)):
        machine_dir = os.path.join(source, machine)
        if not os.path.isdir(machine_dir) or len(machine) != 1:
            continue
        for root, _, files in os.walk(machine_dir):
            for name in sorted(f for f in files
                               if f.startswith("downstream_raw_results_kfold")):
                path = os.path.join(root, name)
                source_dir = os.path.relpath(root, machine_dir)
                n = 0
                for fold_row, subject_rows in read_rows(path, machine, source_dir):
                    folds.append(fold_row)
                    subjects.extend(subject_rows)
                    n += 1
                manifest.append({
                    "machine": machine, "source_dir": source_dir, "file": name,
                    "rows": n,
                    "md5": hashlib.md5(open(path, "rb").read()).hexdigest(),
                })
    return folds, subjects, manifest


def deduplicate(folds: list, subjects: list):
    """Drops rows repeated because two machines produced the same fold.

    The whole-head neighbour sweep was split across machines with one fold computed on
    both, and counting it twice would tighten the spread without adding evidence.

    Args:
        folds (list): Fold rows.
        subjects (list): Subject rows.

    Returns:
        tuple: ``(folds, subjects, n_dropped)``.
    """
    def key(r):
        return (r["variant"], r["zone"], r["frequency"], r["target"],
                r["eval_mode"], r["fold"], r["source_dir"])

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
    figure over a fraction of the cohort that reads like the real thing. One of these hid
    for a whole campaign, a variant that fold 0 still carried and later folds did not.

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


def campaign_of(source_dir: str) -> str:
    """Names the campaign a result directory belongs to.

    A campaign wrote one directory per zone, so the zone has to come off the name or each
    zone would look like a campaign of its own, and a rule that prefers the widest campaign
    would have nothing to prefer.

    Args:
        source_dir (str): Directory of the file, relative to ``save``.

    Returns:
        str: Campaign name, without the zone.
    """
    name = source_dir.replace("save/", "").strip("/")
    parts = name.split("/")
    if len(parts) > 1 and parts[-1] in ZONES:
        return "/".join(parts[:-1])
    for zone in ZONES:
        if name.endswith(f"_{zone}"):
            return name[: -len(f"_{zone}")]
    return name


def mark_canonical(folds: list, subjects: list):
    """Picks one provenance per run, leaving the rest available but not default.

    The whole montage was covered by several campaigns, so twelve of its runs exist two to
    four times over. They are all real, and one of the pairs is what measures the spread of
    a re-execution, but a query that groups by method and zone without noticing would pool
    the same subject four times and read it as a larger cohort.

    The winner is the campaign that covers the most zones for that variant, because that is
    the one a comparison across regions has to use; ties break by name so the choice is
    stable between runs.

    Args:
        folds (list): Fold rows.
        subjects (list): Subject rows.

    Returns:
        int: Number of runs that had more than one provenance.
    """
    zones_per = defaultdict(set)
    for r in folds:
        zones_per[(r["variant"], campaign_of(r["source_dir"]))].add(r["zone"])

    provenances = defaultdict(set)
    for r in folds:
        provenances[(r["variant"], r["zone"], r["frequency"],
                     r["target"], r["eval_mode"])].add(r["source_dir"])

    winner = {}
    for run, dirs in provenances.items():
        winner[run] = max(sorted(dirs),
                          key=lambda d: len(zones_per[(run[0], campaign_of(d))]))

    def run_of(r):
        return (r["variant"], r["zone"], r["frequency"], r["target"], r["eval_mode"])

    for rows in (folds, subjects):
        for r in rows:
            r["campaign"] = campaign_of(r["source_dir"])
            r["canonical"] = winner[run_of(r)] == r["source_dir"]
    return sum(1 for dirs in provenances.values() if len(dirs) > 1)


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
    """Consolidates the extracted results and reports what was written.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    folds, subjects, manifest = collect(args.source)
    folds, subjects, dropped = deduplicate(folds, subjects)
    folds, subjects, incomplete = keep_complete(folds, subjects, args.n_folds)
    multi = mark_canonical(folds, subjects)

    if incomplete:
        print(f"[WARN] {len(incomplete)} runs dropped for being incomplete:")
        for (variant, zone, _, target, mode), n in sorted(incomplete.items()):
            print(f"  {variant} / {zone} / {target} / {mode}: {n} of {args.n_folds} folds")

    write_csv(os.path.join(args.out, "results_folds.csv"), folds, FOLD_COLUMNS)
    write_csv(os.path.join(args.out, "results_subjects.csv"), subjects, SUBJECT_COLUMNS)
    n_files = write_by_method(args.out, folds)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    with open(os.path.join(args.out, "MANIFEST.json"), "w") as fh:
        json.dump({"repo_commit": commit, "extracted_from": args.source,
                   "fold_rows": len(folds), "subject_rows": len(subjects),
                   "duplicate_fold_rows_dropped": dropped,
                   "runs_with_several_provenances": multi,
                   "incomplete_runs_dropped": {"/".join(g): n
                                               for g, n in sorted(incomplete.items())},
                   "files": manifest}, fh, indent=2, sort_keys=True)

    variants = {r["variant"] for r in folds}
    print(f"{len(folds)} fold rows, {len(subjects)} subject rows, "
          f"{dropped} duplicates dropped")
    print(f"{len(variants)} methods over {n_files} method/zone files")
    print(f"{multi} runs have several provenances; use canonical=True for one of each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, required=True,
                        help="Directory holding <machine>/<save path>/*.csv.")
    parser.add_argument("--out", type=str, default="results",
                        help="Directory to write the consolidated base to.")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Folds a complete run has; anything short of it is dropped.")
    main(parser.parse_args())
