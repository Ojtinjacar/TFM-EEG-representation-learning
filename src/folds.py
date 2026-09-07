"""Single source of truth for the subject-level partition of the cohort.

Every experiment in this project has to split the same way or its numbers cannot be put
next to another's. The rule is one line long, but it was written out in three places, and a
test already exists to catch the drift between two of them. Whoever needs the partition
imports it from here instead of rebuilding it.

The split is made over **every** subject of the cohort, sorted, before anyone is filtered
out for lacking a target or for having been used to choose hyperparameters. Filtering
afterwards keeps a subject in the fold it was assigned to; filtering first would shift
everyone else and quietly break the correspondence between two runs.
"""

from sklearn.model_selection import KFold

N_FOLDS = 10
BASE_SEED = 1234


def canonical_subject_folds(subjects, n_folds=N_FOLDS, base_seed=BASE_SEED):
    """Splits the subjects into folds, in the partition the whole project uses.

    Args:
        subjects (list[str]): Every subject of the cohort. Sort them before calling; the
            partition depends on the order, so an unsorted list gives a different one.
        n_folds (int): Number of splits.
        base_seed (int): Seed of the shuffle.

    Returns:
        list[list[str]]: Held-out subjects of each fold, in fold order.

    Raises:
        ValueError: If there are fewer subjects than folds, which scikit-learn reports in
            terms of samples rather than of subjects.
    """
    if len(subjects) < n_folds:
        raise ValueError(
            f"{len(subjects)} subjects cannot be split into {n_folds} folds."
        )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    return [[subjects[i] for i in test_idx] for _, test_idx in kf.split(subjects)]
