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

    # Compare every successful combo against the first; mismatch means a real bug or tie-breaking difference.
    combos = list(predictions_by_combo.items())
    if combos:
        (ref_library, ref_algorithm), ref_preds = combos[0]
        for (library, algorithm), preds in combos[1:]:
            if preds != ref_preds:
                raise AssertionError(
                    f"predictions disagree: {library}/{algorithm} != "
                    f"{ref_library}/{ref_algorithm} "
                    f"(n_train={n_train}, dim={dim}, k={k}, metric={metric})"
                )

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
