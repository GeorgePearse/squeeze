# UMAP (Uniform Manifold Approximation and Projection)

UMAP is a nonlinear dimensionality reduction algorithm that builds a **weighted neighbor graph** in high dimensions and then learns a low-dimensional embedding that best preserves that graph.

If you want the full mathematical deep dive, see: [How UMAP Works](../how_umap_works.md).

## What UMAP is “trying to keep the same”

UMAP is easiest to interpret as preserving **neighbor relationships** (local structure), while still producing a globally coherent layout because neighborhoods must connect consistently.

```mermaid
flowchart LR
  X["High-D points"] --> KNN["kNN search<br/>(n_neighbors)"]
  KNN --> W["Edge weights<br/>(distance → membership strength)"]
  W --> G["Fuzzy neighbor graph<br/>(a weighted graph)"]
  G --> Y0["Initialize embedding<br/>(often spectral)"]
  Y0 --> OPT["Optimize low-D layout<br/>to match graph edges"]
  OPT --> Y["Low-D embedding"]
```

## Phase 1: Build a fuzzy neighbor graph

UMAP starts with a k-nearest-neighbor graph and converts distances into **edge strengths** (think: “how confident are we that i and j are neighbors?”).

### Visual intuition: the same points, two different neighborhood scales

Small `n_neighbors` builds a *more local* graph; large `n_neighbors` builds a *more connected* graph.

#### Small `n_neighbors` (mostly within local groups)

```mermaid
graph LR
  subgraph A["Local group A"]
    A1((A1)) --- A2((A2))
    A2 --- A3((A3))
    A3 --- A4((A4))
  end
  subgraph B["Local group B"]
    B1((B1)) --- B2((B2))
    B2 --- B3((B3))
    B3 --- B4((B4))
  end
```

#### Large `n_neighbors` (adds more cross-group connectivity)

```mermaid
graph LR
  subgraph A["Local group A"]
    A1((A1)) --- A2((A2))
    A2 --- A3((A3))
    A3 --- A4((A4))
  end
  subgraph B["Local group B"]
    B1((B1)) --- B2((B2))
    B2 --- B3((B3))
    B3 --- B4((B4))
  end

  A2 --- B2
  A3 --- B3
```

## Phase 2: Lay out the graph in low dimensions

UMAP then learns positions \(y_i\) so that:

- edges with **high** membership strength become **short distances** (attraction),
- non-edges are kept apart (repulsion, often implemented with negative sampling).

### Visual intuition: attraction vs repulsion

```mermaid
flowchart LR
  E["Graph edge (i, j)<br/>high membership"] --> A["Attraction: pull i and j together"]
  N["Non-edge / sampled negative"] --> R["Repulsion: push apart to prevent collapse"]
```

## The two knobs you feel the most

### `n_neighbors` (local scale)

```mermaid
flowchart TD
  NNsmall["Small n_neighbors"] --> NN1["Emphasizes very local neighborhoods"]
  NNsmall --> NN2["Sharper, tighter clusters"]
  NNlarge["Large n_neighbors"] --> NN3["More global connectivity"]
  NNlarge --> NN4["Smoother, more continuous structure"]
```

### `min_dist` (how tightly points can pack)

```mermaid
flowchart TD
  MDsmall["Small min_dist"] --> MD1["Allows very tight packing"]
  MDsmall --> MD2["Denser clusters; more separation"]
  MDlarge["Large min_dist"] --> MD3["Enforces spacing"]
  MDlarge --> MD4["Clusters look 'fluffier'"]
```

## Step-by-step summary

```mermaid
flowchart TD
  A["Input data X"] --> B["Compute kNN graph"]
  B --> C["Convert distances → edge strengths"]
  C --> D["Combine directed neighborhoods<br/>into a fuzzy graph"]
  D --> E["Initialize Y"]
  E --> F["Optimize Y to match edge strengths"]
  F --> G["Return embedding Y"]
```

## How to interpret UMAP plots safely

- **Local neighborhoods are meaningful**: if points are close, they were likely neighbors in high-D.
- **Global distances are somewhat meaningful**, but still not a strict metric—don’t over-interpret exact between-cluster spacing.

