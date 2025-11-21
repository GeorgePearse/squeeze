# Multi-View Cross-Filter Visualization: Explore Multiple Perspectives

**Status:** Feature Specification
**Date:** November 4, 2025
**Target Release:** Phase 3 (Months 7-12)

---

## Executive Summary

**Multi-View Cross-Filtering** enables:
- **Side-by-side comparison** of 4, 6, 8, or 16 scatter plots
- **Synchronized selection** across all views
- **Real-time filtering** - select in one plot, see highlighted everywhere
- **Method comparison** - compare UMAP, t-SNE, PCA, PaCMAP side-by-side
- **Parameter exploration** - see effect of changing n_neighbors, min_dist, etc.
- **Feature exploration** - color each plot by different genes/features
- **Interactive discovery** - find patterns across multiple perspectives

This transforms exploration from "look at one embedding" → "explore embedding space interactively across multiple methods/features/parameters simultaneously."

---

## Part 1: Core Concept

### 1.1 Basic Multi-View Grid

```python
from umap.visualize import MultiViewScatterplot

# Create 4 embeddings with different parameters
embeddings = {
    "UMAP (n=15)": umap.UMAP(n_neighbors=15).fit_transform(X),
    "UMAP (n=30)": umap.UMAP(n_neighbors=30).fit_transform(X),
    "t-SNE": tsne.TSNE().fit_transform(X),
    "PCA": pca.PCA(n_components=2).fit_transform(X),
}

# Display in 2×2 grid
multi = MultiViewScatterplot(
    embeddings=embeddings,
    labels=y,
    layout="2x2",  # or "2x3", "2x4", "3x3", "4x2", etc.
)

multi.show()
```

**Result:**
```
┌─────────────────┬─────────────────┐
│ UMAP (n=15)     │ UMAP (n=30)     │
│                 │                 │
│  ••  •• •  •••  │  ••  ••  • •••  │
│  •• •    •   •• │  ••  •  •• •••  │
├─────────────────┼─────────────────┤
│ t-SNE           │ PCA             │
│                 │                 │
│  •• •  •  •  •  │ • • • • • • •   │
│  •  • •• ••  •  │  •  • • • • •   │
└─────────────────┴─────────────────┘

Select points in any view → All views highlight the same points
```

### 1.2 Cross-Filtering in Action

```python
# User selects a region in "UMAP (n=15)"
# → All views immediately highlight those points
# → Can explore which method/parameter preserves that structure
```

---

## Part 2: Use Cases

### 2.1 Method Comparison

```python
from umap.visualize import MethodComparison

comparison = MethodComparison(
    X=X,
    y=y,
    methods=[
        ("UMAP", {}),
        ("t-SNE", {"perplexity": 30}),
        ("PCA", {}),
        ("PaCMAP", {}),
        ("PHATE", {}),
        ("TriMap", {}),
    ],
    layout="2x3",
)

comparison.show()

# Each method runs automatically, displays side-by-side
# Select a cluster in one → see how other methods organize it
```

**Use case:** Published comparison papers (e.g., "We compared 6 methods on this dataset")

### 2.2 Parameter Exploration

```python
from umap.visualize import ParameterExploration

params = ParameterExploration(
    X=X,
    y=y,
    method="UMAP",
    parameters={
        "n_neighbors": [5, 15, 30, 50],
        "min_dist": [0.01, 0.1, 0.5],
    },
    layout="auto",  # 2x4 grid for 8 combinations
)

params.show()

# See effect of each parameter combination
# Select points in one → see how they move across parameters
```

**Use case:** Parameter tuning - understand what each parameter does

### 2.3 Feature Exploration (Single-Cell)

```python
from umap.visualize import FeatureExploration
import scanpy as sc

adata = sc.read_h5ad("data.h5ad")
X_umap = adata.obsm['X_umap']

# Color each plot by different genes
genes = ["CD4", "CD8", "FOXP3", "IL2", "TNF", "IFNG", "IL10", "CTLA4"]

exploration = FeatureExploration(
    X_embedded=X_umap,
    features=adata.X,  # Gene expression matrix
    feature_names=adata.var_names,
    selected_features=genes,
    layout="2x4",
)

exploration.show()

# Each plot shows the same embedding, colored by different genes
# Select high-expressing region in one gene → see correlated expression
```

**Use case:** Gene co-expression analysis, finding related features

### 2.4 Time-Series Analysis

```python
from umap.visualize import TimeSeriesExploration

# Embeddings at different time points
embeddings_by_timepoint = {
    "Day 0": umap.UMAP().fit_transform(X[y_time == 0]),
    "Day 1": umap.UMAP().fit_transform(X[y_time == 1]),
    "Day 2": umap.UMAP().fit_transform(X[y_time == 2]),
    "Day 7": umap.UMAP().fit_transform(X[y_time == 7]),
}

timeseries = TimeSeriesExploration(
    embeddings=embeddings_by_timepoint,
    labels=y_celltype,
    layout="2x2",
)

timeseries.show()

# See how cell populations evolve over time
# Track a specific population across timepoints
```

**Use case:** Developmental biology, disease progression, time-course experiments

### 2.5 Batch Integration Comparison

```python
from umap.visualize import BatchIntegrationComparison

# Compare different batch correction methods
methods = {
    "Original (with batch)": X_original,
    "Harmony": X_harmony,
    "ComBat": X_combat,
    "Seurat v3": X_seurat,
    "scanorama": X_scanorama,
}

embeddings = {
    name: umap.UMAP().fit_transform(data)
    for name, data in methods.items()
}

comparison = BatchIntegrationComparison(
    embeddings=embeddings,
    labels=y_celltype,
    batch_labels=batch_labels,
    layout="2x3",
)

comparison.show()

# See which method best integrates batches
# Select a cell type → see if it's consistent across integration methods
```

---

## Part 3: Implementation Architecture

### 3.1 MultiViewScatterplot Class

```python
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np

class MultiViewScatterplot:
    """Display multiple scatter plots with cross-filtering."""

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        layout: str = "auto",
        width: int = 1200,
        height: int = 1200,
        point_size: float = 5,
        color_by: Optional[str] = None,
        title: str = "Multi-View Comparison",
    ):
        """
        Parameters
        ----------
        embeddings : Dict[str, np.ndarray]
            Dictionary mapping view names to (n_samples, 2) embeddings
        labels : np.ndarray
            Optional categorical labels for coloring
        metadata : Dict
            Additional metadata for each point
        layout : str
            "auto", "2x2", "2x3", "2x4", "3x3", "4x2", etc.
        """
        self.embeddings = embeddings
        self.labels = labels
        self.metadata = metadata or {}
        self.layout = self._parse_layout(layout, len(embeddings))
        self.width = width
        self.height = height
        self.point_size = point_size
        self.color_by = color_by
        self.title = title

        # Validate inputs
        n_points = list(embeddings.values())[0].shape[0]
        for name, emb in embeddings.items():
            if emb.shape[0] != n_points:
                raise ValueError(f"All embeddings must have same n_samples. "
                               f"{name} has {emb.shape[0]}, expected {n_points}")

        self._selected_indices = set()
        self._callbacks = {}

    def _parse_layout(self, layout: str, n_views: int) -> Tuple[int, int]:
        """Parse layout string to (rows, cols)."""
        if layout == "auto":
            # Automatically choose grid
            if n_views <= 4:
                return (2, 2)
            elif n_views <= 6:
                return (2, 3)
            elif n_views <= 8:
                return (2, 4)
            elif n_views <= 9:
                return (3, 3)
            else:
                return (4, 4)

        parts = layout.split("x")
        return (int(parts[0]), int(parts[1]))

    def show(self):
        """Display in Jupyter."""
        from IPython.display import HTML
        html = self._generate_html()
        return HTML(html)

    def save_html(self, filename: str):
        """Save as standalone HTML."""
        html = self._generate_html()
        with open(filename, 'w') as f:
            f.write(html)

    def _generate_html(self) -> str:
        """Generate multi-view HTML with cross-filtering."""
        rows, cols = self.layout
        cell_width = self.width // cols
        cell_height = self.height // rows

        # Generate individual plots
        plots_html = []
        for i, (name, embedding) in enumerate(self.embeddings.items()):
            plot = Scatterplot(
                embedding,
                labels=self.labels,
                title=name,
                width=cell_width,
                height=cell_height,
                point_size=self.point_size,
            )
            plots_html.append((name, plot._to_json()))

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.title}</title>
            <script src="https://unpkg.com/regl-scatterplot@latest/umd/regl-scatterplot.js"></script>
            <style>
                body {{
                    margin: 0;
                    font-family: sans-serif;
                    background: #f5f5f5;
                }}
                #grid {{
                    display: grid;
                    grid-template-columns: repeat({cols}, 1fr);
                    gap: 10px;
                    padding: 10px;
                }}
                .plot-container {{
                    background: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    position: relative;
                }}
                .plot-title {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(255,255,255,0.9);
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 12px;
                    z-index: 10;
                }}
                canvas {{
                    display: block;
                    width: 100%;
                    height: 100%;
                }}
                #info {{
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    max-width: 300px;
                }}
            </style>
        </head>
        <body>
            <div id="grid">
                {self._generate_plot_containers(cols, rows)}
            </div>

            <div id="info">
                <strong>{self.title}</strong><br>
                <span id="selection-count">Selected: 0 points</span><br>
                <button onclick="clearSelection()">Clear Selection</button><br>
                <button onclick="exportSelection()">Export Selection</button>
            </div>

            <script>
                const scatterplots = {{}};
                const selectedIndices = new Set();
                const allData = {self._get_all_data_json()};

                // Create scatterplots
                {self._generate_initialization_js()}

                // Cross-filtering
                function onSelection(plotName, indices) {{
                    selectedIndices.clear();
                    indices.forEach(i => selectedIndices.add(i));

                    // Update all plots
                    Object.keys(scatterplots).forEach(name => {{
                        const colors = scatterplots[name].data.colors.map((c, i) => {{
                            if (selectedIndices.has(i)) {{
                                return [c[0], c[1], c[2], 1.0];  // Full opacity
                            }} else {{
                                return [c[0], c[1], c[2], 0.1];  // Fade out
                            }}
                        }});
                        scatterplots[name].updateData({{opacityBy: colors}});
                    }});

                    updateSelectionInfo();
                }}

                function clearSelection() {{
                    selectedIndices.clear();
                    Object.keys(scatterplots).forEach(name => {{
                        const colors = scatterplots[name].data.colors.map(c =>
                            [c[0], c[1], c[2], 0.8]
                        );
                        scatterplots[name].updateData({{opacityBy: colors}});
                    }});
                    updateSelectionInfo();
                }}

                function updateSelectionInfo() {{
                    document.getElementById('selection-count').textContent =
                        `Selected: ${{selectedIndices.size}} points`;
                }}

                function exportSelection() {{
                    const indices = Array.from(selectedIndices);
                    const csv = indices.join('\\n');
                    const blob = new Blob([csv], {{type: 'text/csv'}});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'selection.csv';
                    a.click();
                }}

                // Listen for selections in each plot
                {self._generate_selection_listeners()}
            </script>
        </body>
        </html>
        """
        return html

    def _generate_plot_containers(self, cols: int, rows: int) -> str:
        """Generate HTML containers for each plot."""
        html = ""
        for i, name in enumerate(self.embeddings.keys()):
            html += f"""
            <div class="plot-container">
                <canvas id="canvas-{i}"></canvas>
                <div class="plot-title">{name}</div>
            </div>
            """
        return html

    def _get_all_data_json(self) -> str:
        """Get all plot data as JSON."""
        import json
        data = {}
        for name, embedding in self.embeddings.items():
            plot = Scatterplot(
                embedding,
                labels=self.labels,
            )
            data[name] = plot._to_json()
        return json.dumps(data)

    def _generate_initialization_js(self) -> str:
        """Generate JavaScript to initialize all plots."""
        js = ""
        for i, name in enumerate(self.embeddings.keys()):
            js += f"""
            scatterplots['{name}'] = createScatterplot({{
                canvas: document.getElementById('canvas-{i}'),
                data: allData['{name}'],
            }});
            """
        return js

    def _generate_selection_listeners(self) -> str:
        """Generate JavaScript selection listeners."""
        js = ""
        for i, name in enumerate(self.embeddings.keys()):
            js += f"""
            scatterplots['{name}'].subscribe('select', (indices) => {{
                onSelection('{name}', indices);
            }});
            """
        return js
```

### 3.2 Convenience Classes

```python
class MethodComparison(MultiViewScatterplot):
    """Compare multiple DR methods."""

    def __init__(self, X, y=None, methods=None, **kwargs):
        embeddings = {}
        for method_name, params in methods:
            if method_name.lower() == "umap":
                from umap import UMAP
                reducer = UMAP(**params)
            elif method_name.lower() == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(**params)
            # ... etc for other methods

            embeddings[method_name] = reducer.fit_transform(X)

        super().__init__(embeddings=embeddings, labels=y, **kwargs)


class ParameterExploration(MultiViewScatterplot):
    """Explore effect of parameter changes."""

    def __init__(self, X, y=None, method="UMAP", parameters=None, **kwargs):
        embeddings = {}

        # Generate all parameter combinations
        from itertools import product
        param_combinations = product(*parameters.values())

        for combo in param_combinations:
            param_dict = dict(zip(parameters.keys(), combo))
            label = " ".join(f"{k}={v}" for k, v in param_dict.items())

            if method.lower() == "umap":
                from umap import UMAP
                reducer = UMAP(**param_dict)

            embeddings[label] = reducer.fit_transform(X)

        super().__init__(embeddings=embeddings, labels=y, **kwargs)
```

---

## Part 4: Advanced Features

### 4.1 Linked Statistics

```python
# Show statistics for selected points
class MultiViewWithStats(MultiViewScatterplot):
    """Multi-view with linked statistics panel."""

    def _generate_html(self) -> str:
        # Base HTML
        html = super()._generate_html()

        # Add statistics panel
        stats_panel = """
        <div id="stats-panel">
            <h3>Selected Point Statistics</h3>
            <div id="stat-count">Count: -</div>
            <div id="stat-mean-x">Mean X: -</div>
            <div id="stat-mean-y">Mean Y: -</div>
            <div id="stat-labels">Labels: -</div>
        </div>
        """

        # Add JavaScript to update stats
        js = """
        function updateStats() {
            const count = selectedIndices.size;
            const xs = Array.from(selectedIndices).map(i => allData['points'][i][0]);
            const ys = Array.from(selectedIndices).map(i => allData['points'][i][1]);
            const meanX = xs.length ? xs.reduce((a,b) => a+b)/xs.length : 0;
            const meanY = ys.length ? ys.reduce((a,b) => a+b)/ys.length : 0;

            document.getElementById('stat-count').textContent = `Count: ${count}`;
            document.getElementById('stat-mean-x').textContent = `Mean X: ${meanX.toFixed(3)}`;
            document.getElementById('stat-mean-y').textContent = `Mean Y: ${meanY.toFixed(3)}`;
        }
        """

        return html
```

### 4.2 Export Filtered Subsets

```python
# When user selects points, export filtered data
def export_selection(self, filename: str):
    """Export selected points and their metadata."""
    indices = np.array(list(self._selected_indices))

    # Get original data for selected points
    X_selected = self.X[indices]
    y_selected = self.labels[indices] if self.labels is not None else None

    # Save
    np.save(f"{filename}_X.npy", X_selected)
    if y_selected is not None:
        np.save(f"{filename}_y.npy", y_selected)

    print(f"Exported {len(indices)} selected points")
```

### 4.3 Animated Transitions

```python
# Smoothly transition between different parameter sets
def animate_parameter_sweep(self, parameter_name: str, values: List):
    """Animate through parameter values."""
    # Each frame: show embedding with different parameter value
    # Cross-filtering tracks how structure changes
    pass
```

---

## Part 5: Use Case: Complete Single-Cell Analysis

```python
from umap.visualize import SingleCellAnalysisDashboard
import scanpy as sc

# Load data
adata = sc.read_h5ad("pbmc_68k.h5ad")

# Create comprehensive dashboard
dashboard = SingleCellAnalysisDashboard(
    adata=adata,
    method_comparison=[
        ("UMAP n=15", {"n_neighbors": 15}),
        ("UMAP n=30", {"n_neighbors": 30}),
        ("t-SNE", {}),
        ("PCA", {}),
    ],
    feature_exploration=["CD4", "CD8", "FOXP3", "IL2", "TNF"],
    batch_visualization=["Batch A", "Batch B", "Batch C"],
    layout="auto",
)

dashboard.show()

# User can:
# - Select a cluster in one method, see how other methods organize it
# - Select high-CD4-expressing region, see which cells are selected
# - Select cells from Batch A, see if they separate by batch
# - All selections synchronized across all views
```

---

## Part 6: Comparison to Single-View

| Capability | Single View | Multi-View |
|-----------|-----------|-----------|
| See one embedding | ✓ | ✓ |
| Compare methods | Manual | Automatic |
| Find parameter effects | Guess/iterate | Visual exploration |
| Validate structure | One perspective | Multiple perspectives |
| Understand trade-offs | Not possible | Immediate |
| Publication-ready | Easy | With slight more work |

---

## Part 7: Performance & Scalability

### For 100k points in 2×2 grid:
- **Memory**: ~100MB (4 plots × 25MB each)
- **Rendering**: 60fps
- **Interaction**: Instant selection highlighting

### For 1M points with 8 views:
- **Memory**: ~2GB (8 plots × 250MB each)
- **Rendering**: 30fps
- **Interaction**: ~100ms latency

**Optimization:**
- Use LOD rendering for large point counts
- Chunk data if >1M points
- Use indexed selection for fast highlighting

---

## Part 8: Implementation Timeline

### Phase 3 (Week 1-2)
- [ ] MultiViewScatterplot core class
- [ ] Basic 2×2, 2×3, 2×4 layouts
- [ ] Cross-filtering synchronization
- [ ] HTML generation

### Phase 3 (Week 3-4)
- [ ] MethodComparison convenience class
- [ ] ParameterExploration convenience class
- [ ] Statistics panel
- [ ] Export selections

### Phase 3 (Week 5+)
- [ ] FeatureExploration for biology
- [ ] TimeSeriesExploration for tracking
- [ ] Animation between parameter sweeps
- [ ] Interactive parameter adjustment

---

## Summary

Multi-View Cross-Filtering transforms UMAP from **"show one embedding"** to **"explore embedding space interactively across multiple perspectives simultaneously"**.

**Key benefits:**
1. **Method comparison** - See trade-offs instantly
2. **Parameter tuning** - Understand effect of each parameter
3. **Feature exploration** - Find correlations and patterns
4. **Publication-ready** - Interactive figures for papers
5. **Discovery** - Find unexpected patterns across views
6. **Validation** - Confirm structure is real, not algorithm artifact

This positions UMAP as the go-to tool for exploratory data analysis in biology, NLP, and data science.
