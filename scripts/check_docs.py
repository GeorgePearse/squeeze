"""Execute maintained documentation examples and check their API/artifact contracts."""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_examples() -> list[tuple[str, str]]:
    """Validate API coverage and signatures, returning executable examples."""
    sys.path.insert(0, str(ROOT))
    import squeeze  # noqa: PLC0415 - resolve the checkout before importing

    docs = ROOT / "docs"
    reference = (docs / "api.md").read_text()
    for name in squeeze.__all__:
        if not re.search(rf"\b{re.escape(name)}\b", reference):
            message = f"Public export missing from API reference: {name}"
            raise ValueError(message)

    examples = []
    for path in sorted(docs.rglob("*.md")):
        content = path.read_text()
        for name, signature in re.findall(
            r"```text\nsqueeze\.(\w+)([^\n]+)\n```",
            content,
        ):
            actual = str(inspect.signature(getattr(squeeze, name)))
            if actual != signature:
                message = f"Stale signature in {path}: {name}{signature} != {actual}"
                raise ValueError(message)
        for match in re.finditer(
            r"^```python\n(.*?)^```$",
            content,
            re.MULTILINE | re.DOTALL,
        ):
            line = content[: match.start()].count("\n") + 1
            examples.append((f"{path.relative_to(ROOT)}:{line}", match.group(1)))

    return examples


def check_assets() -> None:
    """Require site figures and CSVs to match the saved benchmark evidence."""
    docs = ROOT / "docs"
    fashion = "working_docs/heatmap_refresh/fashion-mnist/"
    assets = {
        "digits.png": "metrics_heatmap.png",
        "digits.csv": "metrics_results.csv",
        "fashion-mnist.png": fashion + "metrics_heatmap.png",
        "fashion-mnist.csv": fashion + "metrics_results.csv",
    }
    for name, original in assets.items():
        if (docs / "assets/benchmarks" / name).read_bytes() != (
            ROOT / original
        ).read_bytes():
            message = f"Stale documentation benchmark asset: {name}"
            raise ValueError(message)


def main() -> None:
    """Fail on a stale API signature, benchmark asset, or broken Python example."""
    examples = collect_examples()
    check_assets()
    if not examples:
        message = "No executable documentation examples found"
        raise ValueError(message)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[name] = "1"
    for location, code in examples:
        print(f"Checking {location}", flush=True)  # noqa: T201 - CLI progress
        with tempfile.TemporaryDirectory(prefix="squeeze-doc-example-") as directory:
            subprocess.run(  # noqa: S603 - repository-authored examples
                [sys.executable, "-c", code],
                check=True,
                cwd=directory,
                env=env,
                timeout=180,
            )
    print(  # noqa: T201 - CLI summary
        f"Passed {len(examples)} examples, API contracts, and benchmark assets.",
    )


if __name__ == "__main__":
    main()
