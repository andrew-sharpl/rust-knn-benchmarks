import time
import numpy as np
import pandas as pd
from importlib.metadata import version as _pkg_version
from sklearn.neighbors import KNeighborsClassifier
import knn
from itertools import product


def rust_knn_version() -> str:
    """Version of the installed rust-knn package (e.g. '0.1.3')."""
    return _pkg_version("rust-knn")

_METRIC_DICT = {
    "euclidean": knn.Metric.Euclidean,
    "manhattan": knn.Metric.Manhattan,
    "cosine": knn.Metric.Cosine,
}

_ALGORITHM_DICT = {
    "bruteforce": knn.Algorithm.BruteForce,
    "kdtree": knn.Algorithm.KdTree,
}

_SKLEARN_ALGO = {"bruteforce": "brute", "kdtree": "kd_tree"}


def _metric_enum(name: str) -> knn.Metric:
    if name not in _METRIC_DICT:
        raise ValueError(f"unknown metric: {name!r}. valid: {list(_METRIC_DICT)}")
    return _METRIC_DICT[name]


def _algorithm_enum(name: str) -> knn.Algorithm:
    if name not in _ALGORITHM_DICT:
        raise ValueError(f"unknown algorithm: {name!r}. valid: {list(_ALGORITHM_DICT)}")
    return _ALGORITHM_DICT[name]


def make_model(library: str, algorithm: str, k: int, metric: str, X_train, y_train):
    if library == "rust":
        model = knn.KnnClassifier(
            k,
            metric=_metric_enum(metric),
            algorithm=_algorithm_enum(algorithm),
        )
    elif library == "sklearn":
        model = KNeighborsClassifier(
            n_neighbors=k,
            algorithm=_SKLEARN_ALGO[algorithm],
            metric=metric,
        )
    else:
        raise ValueError(f"unknown library: {library}")
    model.fit(X_train, y_train)
    return model


def time_predict(model, X_test: np.ndarray) -> tuple[float, list[int]]:
    """Time a single .predict() call. Returns (elapsed_seconds, predictions)."""
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    elapsed = time.perf_counter() - t0
    return elapsed, list(preds)


def _time_all_combos(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    metric: str,
    runs: int,
    warmup: int,
    n_train: int,
    dim: int,
    n_queries: int,
) -> list[dict]:
    """Time all four library/algorithm combos and verify predictions agree.

    Returns a list of row dicts (one per combo). Rows for combos that failed
    to fit have median_s=NaN and error=<message>.
    Raises AssertionError if successful combos disagree on predictions.
    """
    rows = []
    # Keyed by (library, algorithm) so we can compare predictions across combos after the loop.
    predictions_by_combo: dict[tuple[str, str], list[int]] = {}

    for library, algorithm in product(["rust", "sklearn"], ["bruteforce", "kdtree"]):
        row = {
            "n_train": n_train,
            "dim": dim,
            "n_queries": n_queries,
            "k": k,
            "metric": metric,
            "library": library,
            "algorithm": algorithm,
            "median_s": np.nan,
            "error": None,
        }

        # fit() can raise — e.g. KD-tree + cosine is unsupported in rust-knn.
        try:
            model = make_model(library, algorithm, k, metric, X_train, y_train)
        except Exception as e:
            row["error"] = str(e)
            rows.append(row)
            continue

        # Warm up caches / CPU freq before measured runs; discard these timings.
        for _ in range(warmup):
            model.predict(X_test)

        elapsed_times = []
        for i in range(runs):
            elapsed, preds = time_predict(model, X_test)
            elapsed_times.append(elapsed)
            # Predictions are deterministic across runs — stash run #0 for the correctness check.
            if i == 0:
                predictions_by_combo[(library, algorithm)] = preds

        row["median_s"] = float(np.median(elapsed_times))
        rows.append(row)

    # Compare every successful combo against the first; allow up to 1% mismatch
    # due to tie-breaking when k is small and distances collide.
    combos = list(predictions_by_combo.items())
    if combos:
        (ref_library, ref_algorithm), ref_preds = combos[0]
        total = len(ref_preds)
        for (library, algorithm), preds in combos[1:]:
            mismatches = sum(1 for a, b in zip(ref_preds, preds) if a != b)
            match_rate = 1.0 - mismatches / total
            if match_rate < 0.99:
                raise AssertionError(
                    f"predictions disagree: {library}/{algorithm} matches "
                    f"{ref_library}/{ref_algorithm} on {mismatches}/{total} "
                    f"({match_rate:.1%}, below 99% threshold) "
                    f"(n_train={n_train}, dim={dim}, k={k}, metric={metric})"
                )
            if mismatches > 0:
                print(f"NOTE: {library}/{algorithm} vs {ref_library}/{ref_algorithm}: "
                      f"{mismatches}/{total} mismatches ({match_rate:.1%}, within tolerance)")

    return rows


def run_single(
    n_train: int,
    dim: int,
    n_queries: int,
    k: int = 3,
    metric: str = "euclidean",
    n_classes: int = 3,
    seed: int = 42,
    runs: int = 7,
    warmup: int = 1,
) -> pd.DataFrame:
    """Benchmark all library/algorithm combinations on a single (n_train, dim) config.

    Returns a long-format DataFrame, one row per combo.
    Columns: n_train, dim, n_queries, k, metric, library, algorithm, median_s, error.
    Rows for combos that failed to fit have median_s=NaN and error=<message>.
    Raises AssertionError if successful combos disagree on predictions.
    """
    # Fixed seed so re-runs produce identical data; only the implementation varies.
    rng = np.random.default_rng(seed)
    X_train = rng.random((n_train, dim))
    y_train = rng.integers(0, n_classes, size=n_train).astype(np.int64)
    X_test = rng.random((n_queries, dim))

    rows = _time_all_combos(
        X_train, y_train, X_test,
        k=k, metric=metric, runs=runs, warmup=warmup,
        n_train=n_train, dim=dim, n_queries=n_queries,
    )
    return pd.DataFrame(rows)


def run_scalability_sweep(
    dim: int = 10,
    sizes: list[int] = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000],
    n_queries: int = 500,
    k: int = 3,
    metric: str = "euclidean",
    seed: int = 42,
    runs: int = 7,
    warmup: int = 1,
) -> pd.DataFrame:
    """Fix dimension, vary training size.

    Returns a DataFrame with 4 rows per size (each combo of library/algorithm).
    """
    all_frames = []
    for n_train in sizes:
        print(f"scalability: n_train={n_train}, dim={dim}")
        df = run_single(
            n_train=n_train,
            dim=dim,
            n_queries=n_queries,
            k=k,
            metric=metric,
            seed=seed,
            runs=runs,
            warmup=warmup,
        )
        all_frames.append(df)
    return pd.concat(all_frames, ignore_index=True)


def run_dimensionality_sweep(
    n_train: int = 50_000,
    dims: list[int] = [2, 5, 10, 50, 100, 500],
    n_queries: int = 500,
    k: int = 3,
    metric: str = "euclidean",
    seed: int = 42,
    runs: int = 7,
    warmup: int = 1,
) -> pd.DataFrame:
    """Fix training size, vary dimensionality.

    Returns a DataFrame with 4 rows per dim (each combo of library/algorithm).
    """
    all_frames = []
    for d in dims:
        print(f"dimensionality: n_train={n_train}, dim={d}")
        df = run_single(
            n_train=n_train,
            dim=d,
            n_queries=n_queries,
            k=k,
            metric=metric,
            seed=seed,
            runs=runs,
            warmup=warmup,
        )
        all_frames.append(df)
    return pd.concat(all_frames, ignore_index=True)


def _load_digits():
    from sklearn.datasets import load_digits
    data = load_digits()
    return data.data.astype(np.float64), data.target.astype(np.int64), "digits"  # type: ignore[attr-defined]


def _load_covertype():
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype()
    # Labels are 1-indexed (1-7); rust-knn expects 0-indexed
    return data.data.astype(np.float64), (data.target - 1).astype(np.int64), "covertype"  # type: ignore[attr-defined]


def _load_mnist():
    from sklearn.datasets import fetch_openml
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    return data.data.astype(np.float64), data.target.astype(np.int64), "mnist"  # type: ignore[attr-defined]


def _load_california_housing():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    # Target is continuous (median house value); bin into 5 equal-frequency classes
    # so KNN classification applies. Binning done on the full target before split.
    y = pd.qcut(data.target, q=5, labels=False).astype(np.int64)  # type: ignore[attr-defined]
    return data.data.astype(np.float64), y, "california_housing"  # type: ignore[attr-defined]


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return data.data.astype(np.float64), data.target.astype(np.int64), "breast_cancer"  # type: ignore[attr-defined]


def run_real_datasets(
    n_queries: int = 500,
    k: int = 3,
    metric: str = "euclidean",
    runs: int = 7,
    warmup: int = 1,
) -> pd.DataFrame:
    """Benchmark all four combos on real datasets.

    Datasets: Digits, Breast Cancer, Covertype, California Housing, MNIST.
    Each dataset is split: last n_queries rows become the test set, the rest
    is training. Returns a long-format DataFrame with a 'dataset' column.
    """
    loaders = [
        _load_breast_cancer,
        _load_digits,
        _load_california_housing,
        _load_covertype,
        _load_mnist,
    ]
    all_frames = []
    for loader in loaders:
        X, y, name = loader()
        n_train = X.shape[0] - n_queries
        dim = X.shape[1]
        X_train, X_test = X[:-n_queries], X[-n_queries:]
        y_train = y[:-n_queries]
        print(f"real: {name} ({n_train} train, {dim} dim)")
        rows = _time_all_combos(
            X_train, y_train, X_test,
            k=k, metric=metric, runs=runs, warmup=warmup,
            n_train=n_train, dim=dim, n_queries=n_queries,
        )
        for row in rows:
            row["dataset"] = name
        all_frames.append(pd.DataFrame(rows))
    return pd.concat(all_frames, ignore_index=True)
