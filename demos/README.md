# Squeeze Interactive Demos

Interactive web-based visualizations of dimensionality reduction algorithms using [regl-scatterplot](https://github.com/flekschas/regl-scatterplot).

## Features

- **WebGL-accelerated scatterplots** - Smooth rendering of 100k+ points
- **Multiple datasets** - MNIST Digits, Fashion MNIST, CIFAR-10, 20 Newsgroups
- **Algorithm comparison** - UMAP, t-SNE, PaCMAP, PCA, MDS, PHATE
- **Interactive controls** - Point size, opacity, color schemes
- **Hover information** - See details for individual points

## Quick Start

### 1. Install dependencies

```bash
cd demos
npm install
```

### 2. Generate embeddings

First, make sure you have Squeeze installed:

```bash
# From the project root
pip install -e .
```

Then generate the embedding data:

```bash
npm run generate
# or
python scripts/generate_embeddings.py
```

### 3. Start the dev server

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

## Project Structure

```
demos/
├── public/
│   └── data/           # Generated embedding JSON files
├── scripts/
│   └── generate_embeddings.py  # Python script to create embeddings
├── src/
│   ├── main.js         # Main application code
│   └── styles.css      # Styles
├── index.html          # Main HTML page
├── package.json        # Node.js dependencies
├── vite.config.js      # Vite configuration
└── README.md           # This file
```

## Adding New Datasets

1. Add a loader function in `scripts/generate_embeddings.py`
2. Add the dataset config to `DATASETS` in `src/main.js`
3. Run `npm run generate` to create the embeddings

## Building for Production

```bash
npm run build
```

The built files will be in `dist/`.

## Data Format

Embedding files are JSON with this structure:

```json
{
  "points": [[x1, y1], [x2, y2], ...],
  "labels": [0, 1, 2, ...],
  "metadata": {
    "dataset": "digits",
    "algorithm": "umap",
    "n_points": 1797,
    "n_classes": 10,
    "compute_time": 7.25
  }
}
```

## Technologies

- [regl-scatterplot](https://github.com/flekschas/regl-scatterplot) - WebGL scatterplot
- [Vite](https://vitejs.dev/) - Build tool
- [Squeeze](https://github.com/GeorgePearse/umap) - Dimensionality reduction
