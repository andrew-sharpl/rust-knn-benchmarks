# rust-knn-benchmarks

Benchmark notebook comparing [`rust-knn`](https://pypi.org/project/rust-knn/) against scikit-learn's `KNeighborsClassifier` across training sizes, dimensionalities, and real datasets.

## Run

```bash
uv sync
uv run jupyter notebook notebooks/benchmark.ipynb
```

To run benchmarks from the terminal instead of the notebook:

```bash
uv run python -m benchmark scalability      # Section 2
uv run python -m benchmark dimensionality   # Section 3
```

Results are written to `results/` (gitignored).
