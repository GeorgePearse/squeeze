from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from squeeze.layouts import optimize_layout_aligned_euclidean
from squeeze.sparse import arr_intersect as intersect1d
from squeeze.sparse import arr_union as union1d
from squeeze.spectral import spectral_layout
from squeeze.umap_ import UMAP, make_epochs_per_sample

if TYPE_CHECKING:
    import scipy.sparse

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


@numba.njit(parallel=True)
def in1d(arr: np.ndarray, test_set: np.ndarray) -> np.ndarray:
    """Check whether elements of arr are in test_set.

    Parameters
    ----------
    arr : ndarray
        Input array to test.
    test_set : array-like
        Values to test against.

    Returns
    -------
    ndarray of bool
        Boolean array of same shape as arr indicating element membership in test_set.

    """
    test_set = set(test_set)
    result = np.empty(arr.shape[0], dtype=np.bool_)
    for i in numba.prange(arr.shape[0]):
        if arr[i] in test_set:
            result[i] = True
        else:
            result[i] = False

    return result


def invert_dict(d: dict[Any, Any]) -> dict[Any, Any]:
    """Invert a dictionary by swapping keys and values.

    Parameters
    ----------
    d : dict
        Dictionary to invert.

    Returns
    -------
    dict
        Dictionary with keys and values swapped.

    """
    return {value: key for key, value in d.items()}


@numba.njit()
def procrustes_align(
    embedding_base: np.ndarray,
    embedding_to_align: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Align embedding_to_align to embedding_base using Procrustes analysis.

    Parameters
    ----------
    embedding_base : ndarray of shape (n_samples, n_components)
        The base embedding to align to.
    embedding_to_align : ndarray of shape (n_samples, n_components)
        The embedding to be aligned.
    anchors : tuple of ndarrays
        A tuple of two arrays containing indices of anchor points in each embedding.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        The aligned embedding.

    """
    subset1 = embedding_base[anchors[0]]
    subset2 = embedding_to_align[anchors[1]]
    M = subset2.T @ subset1
    U, _S, V = np.linalg.svd(M)
    R = U @ V
    return embedding_to_align @ R


def expand_relations(
    relation_dicts: list[dict[int, int]],
    window_size: int = 3,
) -> np.ndarray:
    """Expand relation dictionaries to include nearby timepoints within a window.

    Parameters
    ----------
    relation_dicts : list of dict
        List of dictionaries mapping indices between consecutive timepoints.
    window_size : int, optional
        Size of window to expand relations, by default 3.

    Returns
    -------
    ndarray of shape (n_timepoints+1, 2*window_size+1, max_n_samples)
        Expanded relation mappings including windowed relationships.

    """
    max_n_samples = (
        max(
            [max(d.keys()) for d in relation_dicts]
            + [max(d.values()) for d in relation_dicts],
        )
        + 1
    )
    result = np.full(
        (len(relation_dicts) + 1, 2 * window_size + 1, max_n_samples),
        -1,
        dtype=np.int32,
    )
    reverse_relation_dicts = [invert_dict(d) for d in relation_dicts]
    for i in range(result.shape[0]):
        for j in range(window_size):
            result_index = (window_size) + (j + 1)
            if i + j + 1 >= len(relation_dicts):
                result[i, result_index] = np.full(max_n_samples, -1, dtype=np.int32)
            else:
                mapping = np.arange(max_n_samples)
                for k in range(j + 1):
                    mapping = np.array(
                        [relation_dicts[i + k].get(n, -1) for n in mapping],
                    )
                result[i, result_index] = mapping

        for j in range(0, -window_size, -1):
            result_index = (window_size) + (j - 1)
            if i + j - 1 < 0:
                result[i, result_index] = np.full(max_n_samples, -1, dtype=np.int32)
            else:
                mapping = np.arange(max_n_samples)
                for k in range(0, j - 1, -1):
                    mapping = np.array(
                        [reverse_relation_dicts[i + k - 1].get(n, -1) for n in mapping],
                    )
                result[i, result_index] = mapping

    return result


@numba.njit()
def build_neighborhood_similarities(
    graphs_indptr: numba.typed.List,
    graphs_indices: numba.typed.List,
    relations: np.ndarray,
) -> np.ndarray:
    """Build similarity weights based on neighborhood overlap across timepoints.

    Parameters
    ----------
    graphs_indptr : typed list of ndarray
        List of graph index pointers (CSR format) for each timepoint.
    graphs_indices : typed list of ndarray
        List of graph indices (CSR format) for each timepoint.
    relations : ndarray of shape (n_timepoints, window_size, max_n_samples)
        Relation mappings between timepoints.

    Returns
    -------
    ndarray of shape (n_timepoints, window_size, max_n_samples)
        Jaccard similarity weights based on neighborhood overlap.

    """
    result = np.zeros(relations.shape, dtype=np.float32)
    center_index = (relations.shape[1] - 1) // 2
    for i in range(relations.shape[0]):
        base_graph_indptr = graphs_indptr[i]
        base_graph_indices = graphs_indices[i]
        for j in range(relations.shape[1]):
            if i + j - center_index < 0 or i + j - center_index >= len(graphs_indptr):
                continue

            comparison_graph_indptr = graphs_indptr[i + j - center_index]
            comparison_graph_indices = graphs_indices[i + j - center_index]
            for k in range(relations.shape[2]):
                comparison_index = relations[i, j, k]
                if comparison_index < 0:
                    continue

                raw_base_graph_indices = base_graph_indices[
                    base_graph_indptr[k] : base_graph_indptr[k + 1]
                ].copy()
                base_indices = relations[i, j][
                    raw_base_graph_indices[raw_base_graph_indices < relations.shape[2]]
                ]
                base_indices = base_indices[base_indices >= 0]
                comparison_indices = comparison_graph_indices[
                    comparison_graph_indptr[comparison_index] : comparison_graph_indptr[
                        comparison_index + 1
                    ]
                ]
                comparison_indices = comparison_indices[
                    in1d(comparison_indices, relations[i, j])
                ]

                intersection_size = intersect1d(base_indices, comparison_indices).shape[
                    0
                ]
                union_size = union1d(base_indices, comparison_indices).shape[0]

                if union_size > 0:
                    result[i, j, k] = intersection_size / union_size
                else:
                    result[i, j, k] = 1.0

    return result


def get_nth_item_or_val(iterable_or_val: Any, n: int) -> Any:
    """Get the nth item from an iterable or return the value if not iterable.

    Parameters
    ----------
    iterable_or_val : iterable or scalar
        Either an iterable (list, tuple, ndarray) or a scalar value.
    n : int
        Index to retrieve if iterable_or_val is iterable.

    Returns
    -------
    value
        The nth element if iterable, otherwise the input value itself.

    Raises
    ------
    ValueError
        If the type is not recognized.

    """
    if iterable_or_val is None:
        return None
    if type(iterable_or_val) in (list, tuple, np.ndarray):
        return iterable_or_val[n]
    if type(iterable_or_val) in (int, float, bool, None):
        return iterable_or_val
    msg = "Unrecognized parameter type"
    raise ValueError(msg)


PARAM_NAMES = (
    "n_neighbors",
    "n_components",
    "metric",
    "metric_kwds",
    "n_epochs",
    "learning_rate",
    "init",
    "min_dist",
    "spread",
    "set_op_mix_ratio",
    "local_connectivity",
    "repulsion_strength",
    "negative_sample_rate",
    "transform_queue_size",
    "angular_rp_forest",
    "target_n_neighbors",
    "target_metric",
    "target_metric_kwds",
    "target_weight",
    "unique",
)


def set_aligned_params(
    new_params: dict[str, Any],
    existing_params: dict[str, Any],
    n_models: int,
    param_names: tuple[str, ...] = PARAM_NAMES,
) -> dict[str, Any]:
    """Update existing parameters with new parameters for aligned UMAP.

    Parameters
    ----------
    new_params : dict
        New parameter values to add.
    existing_params : dict
        Existing parameter dictionary to update.
    n_models : int
        Number of existing models.
    param_names : tuple, optional
        Parameter names to update, by default PARAM_NAMES.

    Returns
    -------
    dict
        Updated parameter dictionary.

    """
    for param in param_names:
        if param in new_params:
            if isinstance(existing_params[param], list):
                existing_params[param].append(new_params[param])
            elif isinstance(existing_params[param], tuple):
                existing_params[param] = existing_params[param] + (new_params[param],)
            elif isinstance(existing_params[param], np.ndarray):
                existing_params[param] = np.append(
                    existing_params[param],
                    new_params[param],
                )
            elif new_params[param] != existing_params[param]:
                existing_params[param] = (existing_params[param],) * n_models + (
                    new_params[param],
                )

    return existing_params


@numba.njit()
def init_from_existing_internal(
    previous_embedding: np.ndarray,
    weights_indptr: np.ndarray,
    weights_indices: np.ndarray,
    weights_data: np.ndarray,
    relation_dict: numba.typed.Dict,
) -> np.ndarray:
    """Initialize new embedding from existing embedding using weighted neighbors.

    Parameters
    ----------
    previous_embedding : ndarray of shape (n_prev_samples, n_components)
        The existing embedding from a previous timepoint.
    weights_indptr : ndarray
        CSR format index pointer array for the graph.
    weights_indices : ndarray
        CSR format indices array for the graph.
    weights_data : ndarray
        CSR format data array for the graph.
    relation_dict : numba.typed.Dict
        Dictionary mapping new indices to previous indices.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        Initialized embedding for new data.

    """
    n_samples = weights_indptr.shape[0] - 1
    n_features = previous_embedding.shape[1]
    result = np.zeros((n_samples, n_features), dtype=np.float32)

    for i in range(n_samples):
        if i in relation_dict:
            result[i] = previous_embedding[relation_dict[i]]
        else:
            normalisation = 0.0
            for idx in range(weights_indptr[i], weights_indptr[i + 1]):
                j = weights_indices[idx]
                if j in relation_dict:
                    normalisation += weights_data[idx]
                    result[i] += (
                        weights_data[idx] * previous_embedding[relation_dict[j]]
                    )
            if normalisation == 0:
                # Initialize with uniform random values in range [-10, 10]
                # Using a simple approach compatible with numba
                for k in range(n_features):
                    # Generate pseudo-random value using index-based seeding
                    # This is a simple deterministic approach for initialization
                    result[i, k] = ((i * n_features + k) % 20) - 10.0
            else:
                result[i] /= normalisation

    return result


def init_from_existing(
    previous_embedding: np.ndarray,
    graph: scipy.sparse.spmatrix,
    relations: dict[int, int],
) -> np.ndarray:
    """Initialize new embedding from existing embedding.

    Parameters
    ----------
    previous_embedding : ndarray of shape (n_prev_samples, n_components)
        The existing embedding from a previous timepoint.
    graph : sparse matrix
        The k-neighbor graph for the new data.
    relations : dict
        Dictionary mapping new indices to previous indices.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        Initialized embedding for new data.

    """
    typed_relations = numba.typed.Dict.empty(numba.types.int32, numba.types.int32)
    for key, val in relations.items():
        typed_relations[np.int32(key)] = np.int32(val)
    return init_from_existing_internal(
        previous_embedding,
        graph.indptr,
        graph.indices,
        graph.data,
        typed_relations,
    )


class AlignedUMAP(BaseEstimator):
    """Aligned UMAP for multiple datasets with temporal or batch relationships.

    AlignedUMAP extends UMAP to handle multiple related datasets (e.g., time series
    or batch data) by jointly optimizing their embeddings with alignment constraints.
    This ensures that related points across datasets are embedded near each other.

    Parameters
    ----------
    n_neighbors : int or list of int, optional
        The number of neighbors to consider for each point. Can be a single value
        or a list of values (one per dataset), by default 15.
    n_components : int, optional
        The dimension of the embedding space. Must be constant across datasets, by default 2.
    metric : str or callable or list, optional
        The metric to use for computing distances in high dimensional space, by default "euclidean".
    metric_kwds : dict or list of dict, optional
        Keyword arguments for the metric function, by default None.
    n_epochs : int or list of int, optional
        The number of training epochs, by default None.
    learning_rate : float or list of float, optional
        The initial learning rate for optimization, by default 1.0.
    init : str, optional
        How to initialize the embedding, by default "spectral".
    alignment_regularisation : float, optional
        Weight of the alignment penalty, by default 1e-2.
    alignment_window_size : int, optional
        Number of datasets on each side to align with, by default 3.
    min_dist : float or list of float, optional
        The minimum distance between embedded points, by default 0.1.
    spread : float or list of float, optional
        The effective scale of embedded points, by default 1.0.
    low_memory : bool, optional
        Whether to use a lower memory but more computationally expensive approach, by default False.
    set_op_mix_ratio : float or list of float, optional
        Interpolation between fuzzy union and fuzzy intersection, by default 1.0.
    local_connectivity : float or list of float, optional
        Number of nearest neighbors assumed to be locally connected, by default 1.0.
    repulsion_strength : float or list of float, optional
        Weighting of negative samples in low dimensional embedding, by default 1.0.
    negative_sample_rate : int or list of int, optional
        Number of negative samples per positive sample, by default 5.
    transform_queue_size : float, optional
        Size of the queue for transform operations, by default 4.0.
    a : float, optional
        More specific parameter controlling embedding, by default None.
    b : float, optional
        More specific parameter controlling embedding, by default None.
    random_state : int or RandomState, optional
        Random state for reproducibility, by default None.
    angular_rp_forest : bool or list of bool, optional
        Whether to use angular random projection forest, by default False.
    target_n_neighbors : int, optional
        Number of neighbors for target (supervised) embedding, by default -1.
    target_metric : str or callable, optional
        Metric for target space in supervised dimension reduction, by default "categorical".
    target_metric_kwds : dict, optional
        Keyword arguments for target metric, by default None.
    target_weight : float, optional
        Weight of supervised target in embedding, by default 0.5.
    transform_seed : int, optional
        Random seed for transform operations, by default 42.
    force_approximation_algorithm : bool, optional
        Force use of approximate nearest neighbors, by default False.
    verbose : bool, optional
        Whether to print progress messages, by default False.
    unique : bool or list of bool, optional
        Whether to consider only unique data points, by default False.

    """

    def __init__(
        self,
        n_neighbors: int | list[int] | tuple[int, ...] = 15,
        n_components: int = 2,
        metric: str | Any | list[Any] | tuple[Any, ...] = "euclidean",
        metric_kwds: dict[str, Any] | list[dict[str, Any]] | None = None,
        n_epochs: int | list[int] | tuple[int, ...] | None = None,
        learning_rate: float | list[float] | tuple[float, ...] = 1.0,
        init: str = "spectral",
        alignment_regularisation: float = 1.0e-2,
        alignment_window_size: int = 3,
        min_dist: float | list[float] | tuple[float, ...] = 0.1,
        spread: float | list[float] | tuple[float, ...] = 1.0,
        low_memory: bool = False,
        set_op_mix_ratio: float | list[float] | tuple[float, ...] = 1.0,
        local_connectivity: float | list[float] | tuple[float, ...] = 1.0,
        repulsion_strength: float | list[float] | tuple[float, ...] = 1.0,
        negative_sample_rate: int | list[int] | tuple[int, ...] = 5,
        transform_queue_size: float = 4.0,
        a: float | None = None,
        b: float | None = None,
        random_state: int | np.random.RandomState | None = None,
        angular_rp_forest: bool | list[bool] | tuple[bool, ...] = False,
        target_n_neighbors: int = -1,
        target_metric: str = "categorical",
        target_metric_kwds: dict[str, Any] | None = None,
        target_weight: float = 0.5,
        transform_seed: int = 42,
        force_approximation_algorithm: bool = False,
        verbose: bool = False,
        unique: bool | list[bool] | tuple[bool, ...] = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds

        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate
        self.alignment_regularisation = alignment_regularisation
        self.alignment_window_size = alignment_window_size

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique

        self.a = a
        self.b = b

    def fit(
        self,
        X: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
        y: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray | None = None,
        **fit_params: Any,
    ) -> AlignedUMAP:
        """Fit aligned UMAP on multiple related datasets.

        Parameters
        ----------
        X : list of arrays
            List of datasets to jointly embed. Each array should be of shape
            (n_samples_i, n_features).
        y : list of arrays, optional
            Optional list of target arrays for supervised dimension reduction,
            by default None.
        **fit_params : dict
            Additional fit parameters. Must include 'relations' - a list of
            dictionaries mapping indices between consecutive datasets.

        Returns
        -------
        self
            The fitted AlignedUMAP instance.

        Raises
        ------
        ValueError
            If 'relations' is not provided in fit_params, or if dimensions
            don't match expectations.

        """
        if "relations" not in fit_params:
            msg = "Aligned UMAP requires relations between data to be specified"
            raise ValueError(
                msg,
            )

        self.dict_relations_ = fit_params["relations"]
        assert type(self.dict_relations_) in (list, tuple)
        assert type(X) in (list, tuple, np.ndarray)
        assert (len(X) - 1) == (len(self.dict_relations_))

        if y is not None:
            assert type(y) in (list, tuple, np.ndarray)
            assert (len(y) - 1) == (len(self.dict_relations_))
        else:
            y = [None] * len(X)

        # We need n_components to be constant or this won't work
        if type(self.n_components) in (list, tuple, np.ndarray):
            msg = "n_components must be a single integer, and cannot vary"
            raise ValueError(msg)

        self.n_models_ = len(X)

        # Store raw data to avoid accessing private members later
        self.raw_data_ = [check_array(X[n]) for n in range(len(X))]

        if self.n_epochs is None:
            self.n_epochs = 200

        n_epochs = self.n_epochs

        self.mappers_ = [
            UMAP(
                n_neighbors=get_nth_item_or_val(self.n_neighbors, n),
                min_dist=get_nth_item_or_val(self.min_dist, n),
                n_epochs=get_nth_item_or_val(self.n_epochs, n),
                repulsion_strength=get_nth_item_or_val(self.repulsion_strength, n),
                learning_rate=get_nth_item_or_val(self.learning_rate, n),
                init=self.init,
                spread=get_nth_item_or_val(self.spread, n),
                negative_sample_rate=get_nth_item_or_val(self.negative_sample_rate, n),
                local_connectivity=get_nth_item_or_val(self.local_connectivity, n),
                set_op_mix_ratio=get_nth_item_or_val(self.set_op_mix_ratio, n),
                unique=get_nth_item_or_val(self.unique, n),
                n_components=self.n_components,
                metric=self.metric,
                metric_kwds=self.metric_kwds,
                low_memory=self.low_memory,
                random_state=self.random_state,
                angular_rp_forest=self.angular_rp_forest,
                transform_queue_size=self.transform_queue_size,
                target_n_neighbors=self.target_n_neighbors,
                target_metric=self.target_metric,
                target_metric_kwds=self.target_metric_kwds,
                target_weight=self.target_weight,
                transform_seed=self.transform_seed,
                force_approximation_algorithm=self.force_approximation_algorithm,
                verbose=self.verbose,
                a=self.a,
                b=self.b,
            ).fit(X[n], y[n])
            for n in range(self.n_models_)
        ]

        window_size = fit_params.get("window_size", self.alignment_window_size)
        relations = expand_relations(self.dict_relations_, window_size)

        indptr_list = numba.typed.List.empty_list(numba.types.int32[::1])
        indices_list = numba.typed.List.empty_list(numba.types.int32[::1])
        heads = numba.typed.List.empty_list(numba.types.int32[::1])
        tails = numba.typed.List.empty_list(numba.types.int32[::1])
        epochs_per_samples = numba.typed.List.empty_list(numba.types.float64[::1])

        for mapper in self.mappers_:
            indptr_list.append(mapper.graph_.indptr)
            indices_list.append(mapper.graph_.indices)
            heads.append(mapper.graph_.tocoo().row)
            tails.append(mapper.graph_.tocoo().col)
            epochs_per_samples.append(
                make_epochs_per_sample(mapper.graph_.tocoo().data, n_epochs),
            )

        rng_state_transform = np.random.RandomState(self.transform_seed)
        regularisation_weights = build_neighborhood_similarities(
            indptr_list,
            indices_list,
            relations,
        )
        first_init = spectral_layout(
            self.raw_data_[0],
            self.mappers_[0].graph_,
            self.n_components,
            rng_state_transform,
        )
        expansion = 10.0 / np.abs(first_init).max()
        first_embedding = (first_init * expansion).astype(
            np.float32,
            order="C",
        )

        embeddings = numba.typed.List.empty_list(numba.types.float32[:, ::1])
        embeddings.append(first_embedding)
        for i in range(1, self.n_models_):
            next_init = spectral_layout(
                self.raw_data_[i],
                self.mappers_[i].graph_,
                self.n_components,
                rng_state_transform,
            )
            expansion = 10.0 / np.abs(next_init).max()
            next_embedding = (next_init * expansion).astype(
                np.float32,
                order="C",
            )
            anchor_data = relations[i][window_size - 1]
            left_anchors = anchor_data[anchor_data >= 0]
            right_anchors = np.where(anchor_data >= 0)[0]
            embeddings.append(
                procrustes_align(
                    embeddings[-1],
                    next_embedding,
                    np.vstack([left_anchors, right_anchors]),
                ),
            )

        seed_triplet = rng_state_transform.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64,
        )
        self.embeddings_ = optimize_layout_aligned_euclidean(
            embeddings,
            embeddings,
            heads,
            tails,
            n_epochs,
            epochs_per_samples,
            regularisation_weights,
            relations,
            seed_triplet,
            lambda_=self.alignment_regularisation,
            move_other=True,
        )

        for i, embedding in enumerate(self.embeddings_):
            disconnected_vertices = (
                np.array(self.mappers_[i].graph_.sum(axis=1)).flatten() == 0
            )
            embedding[disconnected_vertices] = np.full(self.n_components, np.nan)

        return self

    def fit_transform(
        self,
        X: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
        y: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray | None = None,
        **fit_params: Any,
    ) -> list[np.ndarray]:
        """Fit aligned UMAP and return the embeddings.

        Parameters
        ----------
        X : list of arrays
            List of datasets to jointly embed.
        y : list of arrays, optional
            Optional list of target arrays for supervised dimension reduction,
            by default None.
        **fit_params : dict
            Additional fit parameters. Must include 'relations'.

        Returns
        -------
        list of arrays
            List of embeddings, one for each input dataset.

        """
        self.fit(X, y, **fit_params)
        return self.embeddings_

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params: Any,
    ) -> None:
        """Add a new dataset to an existing aligned UMAP embedding.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            New dataset to add to the aligned embedding.
        y : array, optional
            Optional target array for supervised dimension reduction, by default None.
        **fit_params : dict
            Additional fit parameters. Must include 'relations' - a dictionary
            mapping new dataset indices to the most recent existing dataset.

        Returns
        -------
        self
            The updated AlignedUMAP instance with the new dataset added.

        Raises
        ------
        ValueError
            If 'relations' is not provided or if n_components varies.

        """
        if "relations" not in fit_params:
            msg = "Aligned UMAP requires relations between data to be specified"
            raise ValueError(
                msg,
            )

        new_dict_relations = fit_params["relations"]
        assert isinstance(new_dict_relations, dict)

        X = check_array(X)

        self.__dict__ = set_aligned_params(fit_params, self.__dict__, self.n_models_)

        # We need n_components to be constant or this won't work
        if type(self.n_components) in (list, tuple, np.ndarray):
            msg = "n_components must be a single integer, and cannot vary"
            raise ValueError(msg)

        if self.n_epochs is None:
            self.n_epochs = 200

        n_epochs = self.n_epochs

        new_mapper = UMAP(
            n_neighbors=get_nth_item_or_val(self.n_neighbors, self.n_models_),
            min_dist=get_nth_item_or_val(self.min_dist, self.n_models_),
            n_epochs=get_nth_item_or_val(self.n_epochs, self.n_models_),
            repulsion_strength=get_nth_item_or_val(
                self.repulsion_strength,
                self.n_models_,
            ),
            learning_rate=get_nth_item_or_val(self.learning_rate, self.n_models_),
            init=self.init,
            spread=get_nth_item_or_val(self.spread, self.n_models_),
            negative_sample_rate=get_nth_item_or_val(
                self.negative_sample_rate,
                self.n_models_,
            ),
            local_connectivity=get_nth_item_or_val(
                self.local_connectivity,
                self.n_models_,
            ),
            set_op_mix_ratio=get_nth_item_or_val(self.set_op_mix_ratio, self.n_models_),
            unique=get_nth_item_or_val(self.unique, self.n_models_),
            n_components=self.n_components,
            metric=self.metric,
            metric_kwds=self.metric_kwds,
            low_memory=self.low_memory,
            random_state=self.random_state,
            angular_rp_forest=self.angular_rp_forest,
            transform_queue_size=self.transform_queue_size,
            target_n_neighbors=self.target_n_neighbors,
            target_metric=self.target_metric,
            target_metric_kwds=self.target_metric_kwds,
            target_weight=self.target_weight,
            transform_seed=self.transform_seed,
            force_approximation_algorithm=self.force_approximation_algorithm,
            verbose=self.verbose,
            a=self.a,
            b=self.b,
        ).fit(X, y)

        self.n_models_ += 1
        self.mappers_ += [new_mapper]
        self.raw_data_.append(X)

        self.dict_relations_ += [new_dict_relations]

        window_size = fit_params.get("window_size", self.alignment_window_size)
        new_relations = expand_relations(self.dict_relations_, window_size)

        indptr_list = numba.typed.List.empty_list(numba.types.int32[::1])
        indices_list = numba.typed.List.empty_list(numba.types.int32[::1])
        heads = numba.typed.List.empty_list(numba.types.int32[::1])
        tails = numba.typed.List.empty_list(numba.types.int32[::1])
        epochs_per_samples = numba.typed.List.empty_list(numba.types.float64[::1])

        for i, mapper in enumerate(self.mappers_):
            indptr_list.append(mapper.graph_.indptr)
            indices_list.append(mapper.graph_.indices)
            heads.append(mapper.graph_.tocoo().row)
            tails.append(mapper.graph_.tocoo().col)
            if i == len(self.mappers_) - 1:
                epochs_per_samples.append(
                    make_epochs_per_sample(mapper.graph_.tocoo().data, n_epochs),
                )
            else:
                epochs_per_samples.append(
                    np.full(mapper.embedding_.shape[0], n_epochs + 1, dtype=np.float64),
                )

        new_regularisation_weights = build_neighborhood_similarities(
            indptr_list,
            indices_list,
            new_relations,
        )

        # TODO: We can likely make this more efficient and not recompute each time
        inv_dict_relations = invert_dict(new_dict_relations)

        new_embedding = init_from_existing(
            self.embeddings_[-1],
            new_mapper.graph_,
            inv_dict_relations,
        )

        self.embeddings_.append(new_embedding)

        rng_state_transform = np.random.RandomState(self.transform_seed)
        seed_triplet = rng_state_transform.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64,
        )
        self.embeddings_ = optimize_layout_aligned_euclidean(
            self.embeddings_,
            self.embeddings_,
            heads,
            tails,
            n_epochs,
            epochs_per_samples,
            new_regularisation_weights,
            new_relations,
            seed_triplet,
            lambda_=self.alignment_regularisation,
        )
