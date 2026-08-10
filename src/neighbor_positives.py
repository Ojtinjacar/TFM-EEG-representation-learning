"""Muestreo de positivos por vecino más cercano para SSL contrastivo sobre EEG.

En lugar de generar el positivo con una perturbación sintética (augmentation ciego), esta
estrategia toma como positivo la **ventana real más cercana** a un ancla dentro de su misma
sesión (y, por defecto, su misma actividad/bloque), según una de tres distancias:

- ``cosine``: sobre el vector de features 24-D z-scoreado intra-sujeto (perfil espectral);
- ``wasserstein``: sobre el PSD normalizado por ROI (forma espectral);
- ``riemann``: sobre la matriz de covarianza espacial SPD (patrón entre canales).

Es un muestreo de positivos por vecino, en la línea de NNCLR (Dwibedi et al., ICCV 2021,
arXiv:2104.14548). Novedad para EEG y propuesta del TFM: aquí el vecino se busca en el espacio
de features de entrada (no en un espacio de embeddings aprendido), por lo que su fiabilidad
depende de estas distancias fijas. Restringir el vecino a la misma actividad mitiga el riesgo de
falso positivo (positivos que cruzarían de tarea).

Este módulo NO modifica la pérdida. ``NeighborPositiveDataset`` entrega, por ítem, el ancla y
dos vistas ``(anchor, view1, view2)`` compatibles con el pipeline heredado
(``CIMCYCDataset`` en ``train_simclr.py``); con ``k > 1`` positivos, la integración en la
pérdida (expandir a pares o usar una pérdida multi-positivo tipo SupCon) queda a cargo del
llamador.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

VALID_METRICS = ("cosine", "wasserstein", "riemann")
_EPS = 1e-30


def _session_distance_matrix(
    metric: str,
    gpos: np.ndarray,
    reprs: Mapping[str, object],
) -> np.ndarray:
    """Matriz de distancias (n x n) entre las ventanas de un grupo, según ``metric``.

    Args:
        metric: 'cosine' | 'wasserstein' | 'riemann'.
        gpos: posiciones globales (índices en meta_epochs / X) de las ventanas del grupo.
        reprs: representaciones precomputadas. Claves requeridas por métrica:
            'cosine' -> 'feat_z' (N, D); 'wasserstein' -> 'psd_norm' (N, n_roi, n_freq);
            'riemann' -> 'cov' (M, C, C) y 'gpos_to_cov' (dict posición-global -> fila en cov).

    Returns:
        Matriz simétrica (n, n) de distancias con diagonal 0.

    Raises:
        ValueError: si ``metric`` no es válido o falta una representación.
    """
    n = len(gpos)
    if metric == "cosine":
        feat_z = np.asarray(reprs["feat_z"])
        v = feat_z[gpos].astype(np.float64)
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + _EPS)
        d = 1.0 - vn @ vn.T
        np.fill_diagonal(d, 0.0)
        return d
    if metric == "wasserstein":
        from eda_utils import wasserstein_fourier_distance  # noqa: PLC0415

        psd = np.asarray(reprs["psd_norm"])[gpos]
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dij = float(wasserstein_fourier_distance(psd[i], psd[j]).mean())
                d[i, j] = d[j, i] = dij
        return d
    if metric == "riemann":
        from pyriemann.geometry.distance import pairwise_distance  # noqa: PLC0415

        cov = reprs["cov"]
        g2c = reprs["gpos_to_cov"]
        covs = np.stack([np.asarray(cov[g2c[int(g)]], dtype=np.float64) for g in gpos])
        d = pairwise_distance(covs, metric="riemann")
        np.fill_diagonal(d, 0.0)
        return d
    raise ValueError(f"metric no válida: {metric!r} (usar una de {VALID_METRICS})")


def build_neighbor_positive_table(
    reprs: Mapping[str, object],
    meta_q: pd.DataFrame,
    metric: str,
    *,
    k: int = 2,
    same_activity: bool = True,
    exclude_lag: int = 0,
) -> pd.DataFrame:
    """Construye la tabla de vecinos positivos por ventana, según una distancia.

    Para cada ventana de calidad (ancla) busca sus ``k`` vecinos más cercanos dentro de su misma
    sesión ``(subject, age)`` y, si ``same_activity``, de su mismo ``block``, excluyendo el propio
    ancla y (opcionalmente) los vecinos con ``|lag| <= exclude_lag``.

    Args:
        reprs: representaciones precomputadas (ver :func:`_session_distance_matrix`).
        meta_q: metadatos de ventanas de calidad; su índice es la posición global en X.
            Debe contener columnas ``subject, age, epoch_index, block``.
        metric: 'cosine' | 'wasserstein' | 'riemann'.
        k: número máximo de vecinos positivos por ancla.
        same_activity: si True, el vecino debe compartir ``block`` con el ancla.
        exclude_lag: excluye vecinos con ``|epoch_index_ancla - epoch_index_vecino| <= exclude_lag``.

    Returns:
        DataFrame con una fila por (ancla, vecino): columnas ``subject, age, block, metric,
        anchor_gpos, anchor_epoch, neigh_gpos, neigh_epoch, lag, dist, rank`` (``rank`` 1 = más
        cercano). Las anclas sin vecino válido no generan filas.

    Raises:
        ValueError: si ``metric`` no es válido.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric no válida: {metric!r} (usar una de {VALID_METRICS})")

    group_cols = ["subject", "age"] + (["block"] if same_activity else [])
    rows: list[dict] = []
    for keys, grp in meta_q.groupby(group_cols):
        grp = grp.sort_values("epoch_index")
        gpos = grp.index.values.astype(int)
        ep = grp["epoch_index"].values
        blk = grp["block"].values
        n = len(gpos)
        if n < 2:
            continue
        d = _session_distance_matrix(metric, gpos, reprs)
        for i in range(n):
            lags = np.abs(ep - ep[i])
            mask = np.ones(n, bool)
            mask[i] = False
            if exclude_lag > 0:
                mask &= lags > exclude_lag
            cand = np.where(mask)[0]
            if len(cand) == 0:
                continue
            order = cand[np.argsort(d[i, cand])[:k]]
            for rank, j in enumerate(order, start=1):
                rows.append(
                    {
                        "subject": grp["subject"].iloc[i],
                        "age": int(grp["age"].iloc[i]),
                        "block": int(blk[i]),
                        "metric": metric,
                        "anchor_gpos": int(gpos[i]),
                        "anchor_epoch": int(ep[i]),
                        "neigh_gpos": int(gpos[j]),
                        "neigh_epoch": int(ep[j]),
                        "lag": int(lags[j]),
                        "dist": float(d[i, j]),
                        "rank": rank,
                    }
                )
    cols = ["subject", "age", "block", "metric", "anchor_gpos", "anchor_epoch",
            "neigh_gpos", "neigh_epoch", "lag", "dist", "rank"]
    return pd.DataFrame(rows, columns=cols)


def neighbor_index_array(table: pd.DataFrame, n_total: int, k: int) -> np.ndarray:
    """Convierte la tabla de vecinos en un array (n_total, k) de índices globales.

    Args:
        table: salida de :func:`build_neighbor_positive_table`.
        n_total: número total de ventanas (longitud de X), para dimensionar el array.
        k: número de columnas de vecinos por ancla.

    Returns:
        Array int (n_total, k) donde la entrada es la posición global del vecino de ese rango, o
        -1 si no existe. Filas de anclas sin vecinos quedan a -1.
    """
    idx = np.full((n_total, k), -1, dtype=np.int64)
    for _, r in table.iterrows():
        rk = int(r["rank"]) - 1
        if rk < k:
            idx[int(r["anchor_gpos"]), rk] = int(r["neigh_gpos"])
    return idx


class NeighborPositiveDataset:
    """Dataset de positivos por vecino, compatible con el pipeline SimCLR heredado.

    Cada ítem devuelve ``(anchor, view1, view2)`` donde ``view2`` es una ventana **vecina real**
    (positivo por distancia) y ``view1`` es el ancla o un augmentation del ancla. Imita las
    convenciones de ``CIMCYCDataset`` (tensores ``torch.FloatTensor`` de forma ``(n_canales,
    n_muestras)``), de modo que el ``train_loader`` heredado (``for anchor, aug1, aug2 in ...``)
    lo consume sin cambios.

    Args:
        X: tensor/array ``(N, C, T)`` con las ventanas EEG.
        neighbor_index: array ``(N, k)`` de posiciones globales de vecinos (-1 si no hay), p. ej.
            de :func:`neighbor_index_array`.
        augment: callable opcional ``tensor(C,T) -> tensor(C,T)`` para aumentar ``view1`` (y el
            fallback). Si es None, ``view1`` es el ancla sin cambios.
        fallback: qué hacer cuando un ancla no tiene vecino válido: ``'duplicate'`` (view2 = ancla,
            o su augmentation si ``augment`` no es None) o ``'skip_none'`` (view2 = None).
        seed: semilla para la elección aleatoria del vecino entre los disponibles.
    """

    def __init__(
        self,
        X,
        neighbor_index: np.ndarray,
        *,
        augment: Callable | None = None,
        fallback: str = "duplicate",
        seed: int = 42,
    ) -> None:
        import torch  # noqa: PLC0415

        self._torch = torch
        self.X = X if isinstance(X, torch.Tensor) else torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.neighbor_index = np.asarray(neighbor_index, dtype=np.int64)
        if self.neighbor_index.shape[0] != len(self.X):
            raise ValueError("neighbor_index desalineado con X")
        if fallback not in ("duplicate", "skip_none"):
            raise ValueError("fallback debe ser 'duplicate' o 'skip_none'")
        self.augment = augment
        self.fallback = fallback
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def coverage(self) -> float:
        """Fracción de anclas con al menos un vecino válido."""
        return float((self.neighbor_index >= 0).any(axis=1).mean())

    def __getitem__(self, i: int):
        anchor = self.X[i]
        view1 = self.augment(anchor) if self.augment is not None else anchor.clone()
        neighs = self.neighbor_index[i]
        neighs = neighs[neighs >= 0]
        if len(neighs) == 0:
            if self.fallback == "skip_none":
                return anchor, view1, None
            view2 = self.augment(anchor) if self.augment is not None else anchor.clone()
            return anchor, view1, view2
        j = int(neighs[self._rng.integers(len(neighs))])
        return anchor, view1, self.X[j]
