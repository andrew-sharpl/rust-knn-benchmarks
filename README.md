# rust-knn benchmarks

A benchmark study comparing [rust-knn](https://github.com/andrew-sharpl/rust-knn), a k-nearest-neighbors classifier written in Rust and exposed to Python via PyO3, against scikit-learn's `KNeighborsClassifier`.

**[Read the full write-up →](https://andrew-sharpl.github.io/rust-knn-benchmarks/)**

## What this is

rust-knn implements brute-force and KD-tree nearest-neighbor search in Rust, parallelized with rayon, behind a small scikit-learn-like Python API. This repo measures how its prediction latency compares to scikit-learn across two axes, training-set size and dimensionality, on both synthetic sweeps and five real datasets.

The goal isn't to claim a clean win. It's to map precisely where a Rust implementation helps and where it doesn't, and to understand why.

## What the benchmarks show

- **Small datasets:** rust-knn is the fastest option, several times quicker than scikit-learn, since there's too little data for scikit-learn's vectorized kernels to amortize their overhead.
- **Larger or structured low-dimensional data:** scikit-learn's mature, heavily tuned KD-tree prunes better and pulls ahead.
- **High dimensionality:** scikit-learn's vectorized brute-force scan wins, though rust-knn's KD-tree degrades more gracefully than scikit-learn's own.

The clearest path forward is SIMD in the distance loop: the high-dimensional gap traces directly to scikit-learn computing distances through a vectorized, cache-blocked BLAS routine where rust-knn currently uses a scalar loop. That work is in progress.

## Methodology

Median of 7 runs per configuration, 3 warmup runs discarded, fixed seed, `time.perf_counter()`, only `predict` timed. Every implementation is checked for prediction agreement against scikit-learn before its timing is reported. Full details are in the write-up.

## Running it yourself

```bash
uv sync
uv run jupyter notebook notebooks/benchmark.ipynb
```

The notebook reads cached results from `results/` by default (`RELOAD = True`). Since `results/` is gitignored, a fresh clone won't have the CSVs — set `RELOAD = False` on first run to generate them, then flip back. Absolute numbers depend on hardware; the curve shapes and crossover points are the portable part.
