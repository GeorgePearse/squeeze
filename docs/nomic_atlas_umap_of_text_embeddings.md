UMAP of Text Embeddings with Nomic Atlas
=======================

[Nomic Atlas](https://atlas.nomic.ai/) is a platform for interactively visualizing and exploring massive datasets. It automates the creation of embeddings and 2D coordinate projections using UMAP.

![UMAP interactive visualization with Nomic Atlas](https://assets.nomicatlas.com/airline-reviews-umap.gif)
<!-- width: 600 -->

Nomic Atlas automatically generates embeddings for your data and allows you to explore large datasets in a web browser. Atlas provides:

* In-browser analysis of your UMAP data with the [Atlas Analyst](https://docs.nomic.ai/atlas/data-maps/atlas-analyst)
* Vector search over your UMAP data using the [Nomic API](https://docs.nomic.ai/atlas/data-maps/guides/vector-search-over-your-data)
* Interactive features like zooming, recoloring, searching, and filtering in the [Nomic Atlas data map](https://docs.nomic.ai/atlas/data-maps/controls)
* Scalability for millions of data points
* Rich information display on hover
* Shareable UMAPs via URL links to your embeddings and data maps in Atlas

This example demonstrates how to use [Nomic Atlas](https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/using-umap-with-atlas) to create interactive maps of text using embeddings and UMAP.

## Setup


1. Get the required python packages with `pip instll nomic pandas`
2. Get a Nomic API key [here](https://atlas.nomic.ai/cli-login)
3. Run `nomic login nk-...` in a terminal window or use the following code:

```python3
import nomic
nomic.login('nk-...')



```

Download Example Data
--------------------

```python3
import pandas as pd

# Example data
df = pd.read_csv("https://docs.nomic.ai/singapore_airlines_reviews.csv")

```

## Create Atlas Dataset


```python3
from nomic import AtlasDataset
dataset = AtlasDataset("airline-reviews-data")

```

## Upload to Atlas


```python3
dataset.add_data(df)

```

## Create Data Map


We specify the `text` field from `df` as the field to create embeddings from. We choose some standard UMAP parameters as well.

```python3
from nomic.data_inference import ProjectionOptions

# model="umap" is how you choose UMAP in Nomic Atlas
# You can adjust n_neighbors, min_dist,
# and n_epochs as you would with the UMAP library.
atlas_map = dataset.create_index(
    indexed_field='text',
    projection=ProjectionOptions(
      model="umap",
      n_neighbors=20,
      min_dist=0.01,
      n_epochs=200
  )
)

print(f"Explore your interactive map at: {atlas_map.map_link}")

```

Your map will be available in your [Atlas Dashboard](https://atlas.nomic.ai/data).
