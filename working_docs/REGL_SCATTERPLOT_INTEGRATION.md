# Regl-Scatterplot Integration: High-Performance WebGL Visualization

**Status:** Technical Specification
**Date:** November 4, 2025
**Target Release:** Phase 3 (Months 7-12)

---

## Executive Summary

**regl-scatterplot** is a WebGL-based visualization library optimized for:
- **Speed**: Render 1M+ points at 60fps
- **Interactivity**: Hover, click, zoom, pan, select
- **Export**: Save as PNG, SVG, or interactive HTML
- **Mobile**: Works on tablets and phones
- **Integration**: Simple TypeScript/Python API

UMAP should provide first-class support for regl-scatterplot as the default high-performance visualization backend.

---

## Part 1: Why Regl-Scatterplot?

### Current Limitations of Matplotlib/Plotly

| Feature | Matplotlib | Plotly | regl-scatterplot |
|---------|-----------|--------|------------------|
| **Points** | <100k slow | <100k ok | 1M+ fast |
| **FPS** | Static (no animation) | 30 fps | 60 fps |
| **Interactivity** | Limited | Good | Excellent |
| **Mobile** | No | Yes | Yes |
| **Zoom/Pan** | Basic | Good | Instant |
| **Performance Tuning** | No | No | Point size, opacity control |
| **Custom Shaders** | No | No | Yes |
| **Export** | Good | Good | PNG/SVG/HTML |
| **Learning Curve** | Easy | Easy | Medium |

### Ideal Use Cases

**Perfect for regl-scatterplot:**
- Single-cell RNA-seq (100k-1M cells)
- Large image datasets (millions of embeddings)
- Real-time interactive exploration
- Mobile/web-based tools
- Publication-quality interactive figures

**Fallback to Plotly:**
- Smaller datasets (<50k points)
- Need simple Python API
- Don't need maximum performance

**Use Matplotlib:**
- Static publication figures
- Custom styling/annotations
- Integrated into scripts

---

## Part 2: Core Integration API

### 2.1 Basic Visualization

```python
from umap.visualize import Scatterplot

# Create interactive plot
plot = Scatterplot(
    X_embedded,
    labels=y,
    title="UMAP Embedding",
    width=1000,
    height=800,
)

# Display in Jupyter
plot.show()

# Export to standalone HTML
plot.save_html("embedding.html")
```

### 2.2 Configuration

```python
plot = Scatterplot(
    X_embedded,
    labels=y,
    # Coloring
    color_by="labels",           # or column name
    categorical_colors=True,
    palette="Set1",              # matplotlib colormap
    # Sizing
    point_size=10,
    point_size_scale=1.0,
    # Opacity
    point_opacity=0.8,
    point_opacity_scale=1.0,
    # Hover behavior
    hover_info=["index", "label", "value"],
    hover_html=True,
    # Annotations
    annotate_clusters=False,
    show_legend=True,
    # Performance
    max_points_render=1_000_000,
    use_lod=True,              # Level-of-detail rendering
)

plot.show()
```

### 2.3 Interactive Callbacks

```python
# Add selection handler
def on_select(indices):
    print(f"Selected points: {indices}")

plot.on_select(on_select)

# Add hover handler
def on_hover(index, data):
    print(f"Hovering over point {index}: {data}")

plot.on_hover(on_hover)

# Show in Jupyter
plot.show()
```

### 2.4 Advanced Features

```python
# Multiple data attributes per point
plot = Scatterplot(
    X_embedded,
    metadata={
        "sample_id": sample_ids,
        "batch": batch_labels,
        "quality": quality_scores,
        "expression": gene_expression,
    }
)

# Hover shows all metadata
plot.show()

# Coloring by continuous values
plot = Scatterplot(
    X_embedded,
    color_by=gene_expression,
    colorscale="viridis",
    vmin=0,
    vmax=10,
)

plot.show()

# Sizing by continuous values
plot = Scatterplot(
    X_embedded,
    labels=y,
    size_by=quality_scores,
    size_range=(5, 50),
)

plot.show()
```

---

## Part 3: Implementation Architecture

### 3.1 Python Layer

```python
# umap/visualize/scatterplot.py

from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Union
import numpy as np
import pandas as pd

@dataclass
class ScatterplotConfig:
    """Configuration for regl-scatterplot visualization."""

    # Data
    width: int = 1000
    height: int = 800
    background_color: str = "#ffffff"

    # Coloring
    color_by: Union[str, np.ndarray] = None
    categorical_colors: bool = True
    palette: str = "Set1"
    colorscale: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # Sizing
    point_size: float = 10
    point_size_scale: float = 1.0
    size_by: Optional[np.ndarray] = None
    size_range: tuple = (5, 50)

    # Opacity
    point_opacity: float = 0.8
    point_opacity_scale: float = 1.0
    opacity_by: Optional[np.ndarray] = None

    # Hover
    hover_info: List[str] = None
    hover_html: bool = True

    # Annotations
    annotate_clusters: bool = False
    show_legend: bool = True
    title: str = "Embedding"

    # Performance
    max_points_render: int = 1_000_000
    use_lod: bool = True
    point_outline: bool = False
    point_outline_width: float = 1.0


class Scatterplot:
    """High-performance WebGL scatter plot using regl-scatterplot."""

    def __init__(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[ScatterplotConfig] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            (n_samples, 2) or (n_samples, 3) embedding coordinates
        labels : np.ndarray
            Optional categorical labels for coloring
        metadata : Dict
            Additional data to show on hover
        config : ScatterplotConfig
            Configuration object
        **kwargs
            Configuration overrides
        """
        self.X = X
        self.labels = labels
        self.metadata = metadata or {}

        # Build config
        if config is None:
            config = ScatterplotConfig(**kwargs)
        else:
            # Override with kwargs
            for key, value in kwargs.items():
                setattr(config, key, value)

        self.config = config
        self._process_data()
        self._callbacks = {}

    def _process_data(self):
        """Process data for visualization."""
        # Normalize coordinates to [0, 1]
        self.X_norm = self._normalize_coordinates(self.X)

        # Process colors
        self.colors = self._compute_colors()

        # Process sizes
        self.sizes = self._compute_sizes()

        # Process opacities
        self.opacities = self._compute_opacities()

    def _normalize_coordinates(self, X: np.ndarray) -> np.ndarray:
        """Normalize coordinates to [0, 1] range."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        # Avoid division by zero
        X_range = X_max - X_min
        X_range[X_range == 0] = 1

        return (X - X_min) / X_range

    def _compute_colors(self) -> np.ndarray:
        """Compute RGBA colors for each point."""
        n_points = len(self.X)

        if self.config.color_by is None and self.labels is None:
            # Default: gray
            colors = np.full((n_points, 4), [0.5, 0.5, 0.5, self.config.point_opacity])

        elif isinstance(self.config.color_by, str):
            # Color by metadata column
            if self.config.color_by in self.metadata:
                values = self.metadata[self.config.color_by]
                if np.issubdtype(values.dtype, np.number):
                    # Continuous: use colorscale
                    colors = self._continuous_to_rgba(values)
                else:
                    # Categorical: use palette
                    colors = self._categorical_to_rgba(values)
            else:
                raise ValueError(f"Metadata column '{self.config.color_by}' not found")

        elif isinstance(self.config.color_by, np.ndarray):
            # Array of values
            if np.issubdtype(self.config.color_by.dtype, np.number):
                colors = self._continuous_to_rgba(self.config.color_by)
            else:
                colors = self._categorical_to_rgba(self.config.color_by)

        elif self.labels is not None:
            # Color by labels
            colors = self._categorical_to_rgba(self.labels)

        else:
            # Fallback to gray
            colors = np.full((n_points, 4), [0.5, 0.5, 0.5, self.config.point_opacity])

        return colors

    def _continuous_to_rgba(self, values: np.ndarray) -> np.ndarray:
        """Convert continuous values to RGBA colors."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import get_cmap

        cmap = get_cmap(self.config.colorscale)
        norm = Normalize(vmin=self.config.vmin or values.min(),
                        vmax=self.config.vmax or values.max())

        normalized = norm(values)
        rgba = cmap(normalized)

        # Set alpha
        rgba[:, 3] = self.config.point_opacity

        return rgba

    def _categorical_to_rgba(self, labels: np.ndarray) -> np.ndarray:
        """Convert categorical labels to RGBA colors."""
        from matplotlib.colors import ListedColormap
        from matplotlib.cm import get_cmap

        unique_labels = np.unique(labels)
        cmap = get_cmap(self.config.palette)

        n_colors = len(unique_labels)
        colors_normalized = cmap(np.linspace(0, 1, n_colors))

        # Map labels to colors
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        rgba = np.array([colors_normalized[label_to_idx[label]] for label in labels])

        # Set alpha
        rgba[:, 3] = self.config.point_opacity

        return rgba

    def _compute_sizes(self) -> np.ndarray:
        """Compute point sizes."""
        n_points = len(self.X)

        if self.config.size_by is None:
            return np.full(n_points, self.config.point_size)

        # Normalize size_by to size_range
        size_min, size_max = self.config.size_range
        s_min = self.config.size_by.min()
        s_max = self.config.size_by.max()

        if s_max == s_min:
            return np.full(n_points, size_min)

        normalized = (self.config.size_by - s_min) / (s_max - s_min)
        return size_min + normalized * (size_max - size_min)

    def _compute_opacities(self) -> np.ndarray:
        """Compute point opacities."""
        n_points = len(self.X)

        if self.config.opacity_by is None:
            return np.full(n_points, self.config.point_opacity)

        # Normalize to [0, 1]
        o_min = self.config.opacity_by.min()
        o_max = self.config.opacity_by.max()

        if o_max == o_min:
            return np.full(n_points, self.config.point_opacity)

        return (self.config.opacity_by - o_min) / (o_max - o_min) * self.config.point_opacity

    def _to_json(self) -> Dict:
        """Convert to JSON-serializable format for JavaScript."""
        data = {
            "points": self.X_norm.tolist(),
            "colors": self.colors.tolist(),
            "sizes": self.sizes.tolist(),
            "opacities": self.opacities.tolist(),
            "metadata": {},
        }

        # Add metadata for hover
        for key, values in self.metadata.items():
            data["metadata"][key] = values.tolist()

        return data

    def show(self):
        """Display in Jupyter notebook."""
        from IPython.display import HTML
        html = self._generate_html()
        return HTML(html)

    def save_html(self, filename: str):
        """Save as standalone HTML file."""
        html = self._generate_html()
        with open(filename, 'w') as f:
            f.write(html)
        print(f"Saved to {filename}")

    def _generate_html(self) -> str:
        """Generate HTML with embedded regl-scatterplot."""
        data_json = json.dumps(self._to_json())

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.config.title}</title>
            <script src="https://unpkg.com/regl-scatterplot@latest/umd/regl-scatterplot.js"></script>
            <style>
                body {{
                    margin: 0;
                    overflow: hidden;
                    font-family: sans-serif;
                }}
                #canvas {{
                    display: block;
                }}
                #info {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <canvas id="canvas"></canvas>
            <div id="info">
                <strong>{self.config.title}</strong><br>
                Points: {len(self.X)}<br>
                <span id="hover-info"></span>
            </div>

            <script>
                const data = {data_json};

                // Create scatterplot
                const canvas = document.getElementById('canvas');
                const scatterplot = createScatterplot({{
                    canvas: canvas,
                    width: {self.config.width},
                    height: {self.config.height},
                    background: '{self.config.background_color}',
                }});

                // Set data
                scatterplot.draw({{
                    points: data.points,
                    colorBy: data.colors,
                    sizeBy: data.sizes,
                    opacityBy: data.opacities,
                }});

                // Handle hover
                scatterplot.subscribe('hover', (hoverPoints) => {{
                    if (hoverPoints.length > 0) {{
                        const idx = hoverPoints[0];
                        const info = `Point: ${{idx}}`;
                        document.getElementById('hover-info').textContent = info;
                    }}
                }});

                // Handle selection
                scatterplot.subscribe('select', (selectedPoints) => {{
                    console.log('Selected:', selectedPoints);
                }});

                // Handle resize
                window.addEventListener('resize', () => {{
                    scatterplot.setSize(window.innerWidth, window.innerHeight);
                }});
            </script>
        </body>
        </html>
        """
        return html

    def on_select(self, callback: Callable[[List[int]], None]):
        """Register callback for point selection."""
        self._callbacks['select'] = callback

    def on_hover(self, callback: Callable[[int, Dict], None]):
        """Register callback for point hover."""
        self._callbacks['hover'] = callback
```

---

## Part 4: Integration Points

### 4.1 Default in Jupyter

```python
# When user calls plot() in Jupyter, default to regl-scatterplot
from umap.visualize import plot

# Automatically uses regl-scatterplot for large datasets
fig = plot(X_umap, labels=y)

# Falls back to Plotly for small datasets
fig = plot(X_umap_small, labels=y)
```

### 4.2 UMAP Output Integration

```python
from umap import UMAP

reducer = UMAP()
X_embedded = reducer.fit_transform(X)

# Built-in visualization
fig = reducer.plot(X_embedded, labels=y)
fig.show()
```

### 4.3 Interactive Notebooks

```python
# Jupyter extension for UMAP with regl-scatterplot
from umap.jupyter import EmbeddingExplorer

explorer = EmbeddingExplorer(X_embedded, metadata=adata.obs)
explorer.show()

# Features:
# - Interactive plot
# - Hover to see metadata
# - Click to select points
# - Export selections
# - Linked plots (embedding + original data)
```

---

## Part 5: Performance Considerations

### 5.1 Level-of-Detail (LOD) Rendering

```python
# For 1M+ points, use LOD
plot = Scatterplot(
    X_embedded,
    use_lod=True,  # Automatically reduces points when zoomed out
    lod_threshold=1000,  # Use LOD when >1000 points visible
)

plot.show()
```

### 5.2 Memory Optimization

```python
# Use 32-bit floats instead of 64-bit
X_embedded_32 = X_embedded.astype(np.float32)

plot = Scatterplot(X_embedded_32)
plot.show()  # Smaller HTML file size
```

### 5.3 Chunked Rendering

```python
# For extremely large datasets, render in chunks
if len(X_embedded) > 1_000_000:
    # Render first 500k points
    plot1 = Scatterplot(X_embedded[:500_000])
    plot1.save_html("embedding_part1.html")

    # Render next 500k points
    plot2 = Scatterplot(X_embedded[500_000:])
    plot2.save_html("embedding_part2.html")
```

---

## Part 6: Advanced Features

### 6.1 Custom Color Mapping

```python
# Map continuous values to custom color function
plot = Scatterplot(
    X_embedded,
    color_by=expression_values,
    color_function=lambda x: {
        "red": max(0, x),
        "green": max(0, -x),
        "blue": abs(x),
        "alpha": 0.8,
    }
)
```

### 6.2 Linked Plots

```python
# Create linked plots: embedding + original features
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scattergl"}, {"type": "bar"}]]
)

# Left: regl-scatterplot rendered as image in Plotly
fig.add_trace(go.Scattergl(x=X_embedded[:, 0], y=X_embedded[:, 1]), row=1, col=1)

# Right: feature values for selected point
selected_feature_values = X[selected_point]
fig.add_trace(go.Bar(y=selected_feature_values), row=1, col=2)

fig.show()
```

### 6.3 Animation

```python
# Show embedding building process
from umap.visualize import animate_embedding_construction

animate_embedding_construction(
    reducer,
    X,
    output_file="embedding_construction.mp4",
    fps=30,
)
```

---

## Part 7: Comparison with Alternatives

| Feature | regl-scatterplot | Plotly | Matplotlib | Vispy | Napari |
|---------|-----------------|--------|-----------|-------|--------|
| **WebGL** | Yes | Limited | No | Yes | Yes |
| **1M+ points** | Yes (60fps) | No (slow) | No | Yes | Yes |
| **Interactive** | Excellent | Good | No | Good | Yes |
| **Export** | HTML, PNG | HTML | PNG, PDF | Images | N/A |
| **Python API** | Good | Excellent | Excellent | Fair | Good |
| **Mobile** | Yes | Yes | No | No | No |
| **Learning Curve** | Medium | Easy | Easy | Hard | Medium |
| **Customization** | Shaders | Limited | Unlimited | High | Limited |

---

## Part 8: Implementation Timeline

### Phase 3 (Week 1-2)
- [ ] Basic Scatterplot class
- [ ] Color, size, opacity mapping
- [ ] HTML generation and saving
- [ ] Jupyter integration

### Phase 3 (Week 3-4)
- [ ] Hover information
- [ ] Selection callbacks
- [ ] LOD rendering
- [ ] Documentation and examples

### Phase 3 (Week 5+)
- [ ] Linked plots
- [ ] Custom shaders
- [ ] Animation support
- [ ] Advanced interactivity

---

## Part 9: Configuration Examples

### Single-Cell RNA-seq

```python
from umap.visualize import Scatterplot
import scanpy as sc

adata = sc.read_h5ad("data.h5ad")
X_umap = adata.obsm['X_umap']

plot = Scatterplot(
    X_umap,
    color_by=adata.obs['cell_type'],
    hover_info=['cell_type', 'batch', 'nUMI'],
    metadata=adata.obs,
    title="scRNA-seq UMAP",
)

plot.show()
```

### Gene Expression

```python
plot = Scatterplot(
    X_umap,
    color_by=adata.var['BRCA1'],  # Gene expression
    colorscale="viridis",
    size_by=adata.obs['nUMI'],
    point_size=5,
    title="BRCA1 Expression",
)

plot.show()
```

### NLP Embeddings

```python
plot = Scatterplot(
    X_umap,
    labels=document_topics,
    hover_info=['title', 'author', 'date'],
    metadata={
        'title': titles,
        'author': authors,
        'date': dates,
    },
    title="Document Embeddings",
)

plot.show()
```

---

## Part 10: Export Options

### Save as HTML (Default)

```python
plot.save_html("embedding.html")
# Result: Standalone HTML file (~2-5MB for 1M points)
# Share via email, upload to web server, embed in reports
```

### Export as PNG

```python
plot.save_png("embedding.png", dpi=150)
# Result: Static image, good for papers
```

### Export Selections

```python
# Export selected points
selected_indices = plot.get_selection()
selected_data = X_embedded[selected_indices]
np.save("selected_points.npy", selected_data)
```

---

## Summary

**regl-scatterplot should be the default visualization for UMAP** because:

1. **Performance**: 1M+ points at 60fps vs Plotly's struggles with 100k
2. **Interactivity**: Better UX for exploration and discovery
3. **Web-native**: Perfect for cloud-based platforms
4. **Modern**: WebGL is the standard for data visualization
5. **Export**: Create shareable interactive HTML figures
6. **Mobile**: Works on tablets and phones

**Integration path:**
- Phase 3: Basic support (current plan)
- Add to default visualization toolchain
- Make it the primary choice for large datasets
- Build community visualizations around it
