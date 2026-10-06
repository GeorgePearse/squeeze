# Benchmark datasets

## Digits

The default dataset is `sklearn.datasets.load_digits`: all **1,797 samples**, each
with **64 raw pixel features**, and ten digit classes. It is small enough for the
repository's dense methods and is the baseline for repeated comparisons.

## Fashion-MNIST

The optional [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
uses the official **10,000-image test split**, not the training split. The standard
benchmark selects **2,000 images**, balanced at **200 per class**, without replacement
using NumPy seed 42. Each 28×28 image becomes 784 raw float64 pixel features.
There is no feature standardization or PCA preprocessing in this benchmark.

Class order: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker,
Bag, Ankle boot. Labels determine stratification and evaluation only.

### Download and integrity

The loader downloads the official compressed IDX image and label files over HTTPS
from repository revision `b2617bb6d3ffa2e429640350f613e3291e10b141`. It validates
the published compressed-file MD5 checksums, IDX headers, payload lengths, and
class IDs before use. The default cache is `~/.cache/squeeze/fashion-mnist`.
Raw images are not committed to Squeeze.

| File | Published MD5 |
| --- | --- |
| `t10k-images-idx3-ubyte.gz` | `bef4ecab320f06d8554ea6380940ec79` |
| `t10k-labels-idx1-ubyte.gz` | `bb300cfdad3c16e7a12a480ee83cd310` |

`--samples` accepts multiples of ten from 100 through 10,000. Large samples are
opt-in because several algorithms and quality metrics are quadratic. `--cache-dir`
selects a different cache. The protocol retains selected sample indices, source
revision, checksums, and data/label hashes.

The saved sample comes from the standard test partition, but repeated optimization
against it makes it an exploratory benchmark, not an untouched held-out evaluation.
See [reproduction commands](benchmarking.md).
