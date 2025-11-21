import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

import squeeze as sqz

reducer = sqz.UMAP(random_state=42)
embedding = reducer.fit_transform(digits.data)

import base64
from io import BytesIO

from PIL import Image


def embeddable_image(data):
    img_data = 255 - 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode="L").resize((64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format="png")
    for_encoding = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(for_encoding).decode()


from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, output_file, show

output_file("basic_usage_bokeh_example.html")

digits_df = pd.DataFrame(embedding, columns=("x", "y"))
digits_df["digit"] = [str(x) for x in digits.target]
digits_df["image"] = list(map(embeddable_image, digits.images))

datasource = ColumnDataSource(digits_df)
color_mapping = CategoricalColorMapper(
    factors=[str(9 - x) for x in digits.target_names],
    palette=Spectral10,
)

plot_figure = figure(
    title="UMAP projection of the Digits dataset",
    plot_width=600,
    plot_height=600,
    tools=("pan, wheel_zoom, reset"),
)

plot_figure.add_tools(
    HoverTool(
        tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Digit:</span>
        <span style='font-size: 18px'>@digit</span>
    </div>
</div>
""",
    ),
)

plot_figure.circle(
    "x",
    "y",
    source=datasource,
    color={"field": "digit", "transform": color_mapping},
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4,
)
show(plot_figure)
