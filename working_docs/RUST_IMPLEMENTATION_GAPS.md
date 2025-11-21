# Rust Implementation Gaps & Analysis

## 1. API Surface & Integration
*   **Rust Exposure:** The Rust backend exposes a single class `HnswIndex` with methods `new`, `query`, `update`, and `neighbor_graph`.
*   **Python Wrapper:** `umap/hnsw_wrapper.py` acts as a bridge, converting data to `np.float32` and passing it to the Rust backend.
*   **Discrepancy:** The Rust API mimics the expected interface of an HNSW index (accepting `m`, `ef_construction`, `random_state`), but **silently ignores** these critical structural parameters during initialization.

## 2. Missing Algorithms
*   **The HNSW Algorithm:** The core Hierarchical Navigable Small World graph algorithm is **missing**. The `query` method performs a linear scan over all data points (brute-force), making it unsuitable for large datasets.
*   **Sparse Matrix Support:** The Rust backend has **zero support for sparse matrices**. It requires dense vectors (`Vec<f32>`), whereas Python's `umap/sparse.py` and `pynndescent` efficiently handle sparse inputs (CSR/CSC).
*   **Distance Metrics:** The Rust implementation supports a tiny fraction of the metrics available in Python:
    *   **Rust:** `euclidean` (l2), `manhattan` (l1), `cosine`, `chebyshev` (linfinity), `hamming`.
    *   **Python (Missing in Rust):** `minkowski`, `braycurtis`, `canberra`, `mahalanobis`, `wminkowski`, `seuclidean`, `haversine`, `jaccard`, `dice`, `russellrao`, `kulsinski`, `rogerstanimoto`, `sokalmichener`, `sokalsneath`, `yule`.

## 3. Parameter Parity
The following parameters passed from Python are **ignored** or unsupported by the Rust backend:

| Parameter | Python Usage | Rust Implementation |
| :--- | :--- | :--- |
| `m` (n_neighbors) | Controls HNSW graph connectivity | **Ignored** (Brute force has no graph) |
| `ef_construction` | Controls build accuracy/speed trade-off | **Ignored** |
| `ef` (query time) | Controls search accuracy/speed trade-off | **Ignored** |
| `random_state` | Seed for stochastic search | **Ignored** (Search is deterministic) |
| `metric_kwds` | Arguments for complex metrics (e.g., `p`) | **Unsupported** |

## 4. Performance Benchmarks (Assessment)
*   **Theoretical:**
    *   **Python (Pynndescent):** Scaling is roughly $O(N \log N)$ for index construction and $O(\log N)$ for queries.
    *   **Rust (Current):** Scaling is $O(N^2)$ for index construction (neighbor graph) and $O(N)$ for queries.
*   **Practical Impact:** The Rust backend will be strictly **slower** than the Python/Numba implementation for any dataset size where HNSW is typically useful ($N > \sim 5000$). It provides no performance benefit and will likely cause the application to hang on large datasets.

## 5. Documentation Gaps
*   **Misleading Naming:** The Rust crate and struct are named `_hnsw_backend` and `HnswIndex`, which is misleading given the brute-force implementation.
*   **No Feature Flags:** There is no indication in `Cargo.toml` or documentation that this is a "debug" or "fallback" backend.
*   **Missing Docstrings:** The Rust code lacks documentation explaining its limitations or intended use case as a baseline verifier.
