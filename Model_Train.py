#!/usr/bin/env python3
"""Reusable feature definitions and deterministic feature transforms."""

from __future__ import annotations

import json
import math
from typing import Iterable
import numpy as np


HISTORY_WINDOWS_DAYS = (7, 15, 30, 45, 60)

REFLECTANCE_BANDS = (
    "Band1_Red",
    "Band2_NIR",
    "Band3_Blue",
    "Band4_Green",
    "Band5_SWIR1",
    "Band6_SWIR2",
    "Band7_SWIR3",
)

QUALITY_FEATURES = (
    "Albedo_Quality",
    "BRDF_Albedo_Uncertainty",
    "Percent_Inputs",
    "Percent_Snow",
    "qa_tier",
)

IGBP_CODES = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16)
IGBP_ONE_HOT_FEATURES = tuple(f"IGBP_{code}" for code in IGBP_CODES)


def history_window_features(
    windows: Iterable[int] = HISTORY_WINDOWS_DAYS,
) -> tuple[str, ...]:
    """Return finite-window statistics for each band.  The target date is excluded from every history window.  The implementation that reads the time series must preserve this rule across year boundaries.
    """
    return tuple(
        f"hist{window}_{band}_{stat}"
        for window in windows
        for band in REFLECTANCE_BANDS
        for stat in ("mean", "std", "count", "max")
    )


def recency_features() -> tuple[str, ...]:
    """Return one last-valid value and one recency value per band"""

    return tuple(
        name
        for band in REFLECTANCE_BANDS
        for name in (f"last_{band}", f"days_since_last_{band}")
    )


HISTORY_FEATURES = history_window_features() + recency_features()

COMMON_FEATURES = (
    tuple(f"target_{band}" for band in REFLECTANCE_BANDS)
    + QUALITY_FEATURES
    + ("doy_sin", "doy_cos")
    + HISTORY_FEATURES
)


PERIODIC_PROXY_FEATURES = (
    "proxy_lat_scaled",
    "proxy_lon_sin",
    "proxy_lon_cos",
    "proxy_phase16_sin",
    "proxy_phase16_cos",
    *(f"proxy_phase16_is_{phase:02d}" for phase in range(16)),
    "proxy_track_plus_h1_sin",
    "proxy_track_plus_h1_cos",
    "proxy_track_minus_h1_sin",
    "proxy_track_minus_h1_cos",
    "proxy_track_plus_h2_sin",
    "proxy_track_plus_h2_cos",
    "proxy_track_minus_h2_sin",
    "proxy_track_minus_h2_cos",
)


MODEL_FEATURES = COMMON_FEATURES + ("Local_Solar_Noon",) + PERIODIC_PROXY_FEATURES

MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 8,
    "eta": 0.02,
    "max_bin": 64,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "gamma": 0.0,
    "seed": 42,
    "verbosity": 1,
}

NUM_BOOST_ROUND = 4000


def feature_names(include_land_cover: bool = False) -> tuple[str, ...]:
    """Return the ordered names of the feature columns"""

    features = MODEL_FEATURES
    if include_land_cover:
        features += IGBP_ONE_HOT_FEATURES
    return features


def periodic_proxy_features(
    latitude: np.ndarray | float,
    longitude: np.ndarray | float,
    days_since_1970: np.ndarray | int,
) -> dict[str, np.ndarray]:
    """Generate deterministic spatial and periodic proxy features.
    ``days_since_1970`` is an integer day index.  The phase repeats every 16
    days and is represented by cyclic, one-hot, and harmonic columns.
    """

    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.deg2rad(np.asarray(longitude, dtype=np.float64))
    day_index = np.asarray(days_since_1970, dtype=np.int64)
    phase = np.mod(day_index, 16)
    phase_angle = 2.0 * math.pi * phase / 16.0

    values: dict[str, np.ndarray] = {
        "proxy_lat_scaled": lat / 90.0,
        "proxy_lon_sin": np.sin(lon),
        "proxy_lon_cos": np.cos(lon),
        "proxy_phase16_sin": np.sin(phase_angle),
        "proxy_phase16_cos": np.cos(phase_angle),
    }
    for value in range(16):
        values[f"proxy_phase16_is_{value:02d}"] = (phase == value).astype(np.float64)
    for harmonic in (1, 2):
        plus = harmonic * (phase_angle + lon)
        minus = harmonic * (phase_angle - lon)
        values[f"proxy_track_plus_h{harmonic}_sin"] = np.sin(plus)
        values[f"proxy_track_plus_h{harmonic}_cos"] = np.cos(plus)
        values[f"proxy_track_minus_h{harmonic}_sin"] = np.sin(minus)
        values[f"proxy_track_minus_h{harmonic}_cos"] = np.cos(minus)
    return {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}


def history_features_for_day(
    reflectance: np.ndarray,
    valid_mask: np.ndarray,
    dates: np.ndarray,
    target_index: int,
    windows: Iterable[int] = HISTORY_WINDOWS_DAYS,
) -> dict[str, np.ndarray]:
    """Generate target-day historical statistics from aligned time series.

    ``reflectance`` has shape ``(time, pixels, bands)`` and follows
    ``REFLECTANCE_BANDS``.  ``dates`` contains one calendar date per time
    step.  Windows use calendar-day boundaries, exclude the target date, and
    ignore invalid values consistently for all statistics.
    """
    values = np.asarray(reflectance, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    window_values_days = tuple(int(window) for window in windows)
    if not window_values_days or any(window < 1 for window in window_values_days):
        raise ValueError("windows must contain positive day lengths")
    if values.ndim != 3 or values.shape[2] != len(REFLECTANCE_BANDS):
        raise ValueError("reflectance must have shape (time, pixels, seven bands)")
    date_values = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    if date_values.ndim != 1 or date_values.shape[0] != values.shape[0]:
        raise ValueError("dates must have one value for each time step")
    if np.any(np.diff(date_values) < 0):
        raise ValueError("dates must be sorted in ascending order")
    if valid.shape not in (values.shape[:2], values.shape):
        raise ValueError("valid_mask must have shape (time, pixels) or reflectance shape")
    if not 0 <= target_index < values.shape[0]:
        raise IndexError("target_index is outside the time axis")

    finite = np.isfinite(values)
    if valid.ndim == 2:
        finite &= valid[:, :, None]
    else:
        finite &= valid
    target_ordinal = date_values[target_index]
    before_target = date_values < target_ordinal

    result: dict[str, np.ndarray] = {}
    for window in window_values_days:
        in_window = before_target & (date_values >= target_ordinal - int(window))
        window_values = np.where(finite[in_window], values[in_window], np.nan)
        for band_index, band in enumerate(REFLECTANCE_BANDS):
            band_values = window_values[:, :, band_index]
            finite_band = np.isfinite(band_values)
            count = np.sum(finite_band, axis=0, dtype=np.float32)
            safe_values = np.where(finite_band, band_values, 0.0)
            total = np.sum(safe_values, axis=0, dtype=np.float64)
            total2 = np.sum(safe_values * safe_values, axis=0, dtype=np.float64)
            mean = np.full(values.shape[1], np.nan, dtype=np.float32)
            std = np.full(values.shape[1], np.nan, dtype=np.float32)
            has_value = count > 0
            mean[has_value] = (total[has_value] / count[has_value]).astype(np.float32)
            variance = total2[has_value] / count[has_value] - mean[has_value] ** 2
            std[has_value] = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
            maximum = np.full(values.shape[1], np.nan, dtype=np.float32)
            if band_values.shape[0] > 0:
                maximum[has_value] = np.max(
                    np.where(finite_band, band_values, -np.inf), axis=0
                )[has_value]
            result[f"hist{window}_{band}_mean"] = mean
            result[f"hist{window}_{band}_std"] = std
            result[f"hist{window}_{band}_count"] = count
            result[f"hist{window}_{band}_max"] = maximum

    recency_window = max(window_values_days)
    recency_mask = before_target & (
        date_values >= target_ordinal - recency_window
    )
    recency_indices = np.flatnonzero(recency_mask)
    for band_index, band in enumerate(REFLECTANCE_BANDS):
        last_value = np.full(values.shape[1], np.nan, dtype=np.float32)
        days_since = np.full(values.shape[1], np.nan, dtype=np.float32)
        unresolved = np.ones(values.shape[1], dtype=bool)
        for source_index in recency_indices[::-1]:
            available = unresolved & finite[source_index, :, band_index]
            last_value[available] = values[source_index, available, band_index]
            days_since[available] = target_ordinal - date_values[source_index]
            unresolved[available] = False
            if not unresolved.any():
                break
        result[f"last_{band}"] = last_value
        result[f"days_since_last_{band}"] = days_since
    return result


def build_feature_matrix(
    columns: dict[str, object],
    *,
    latitude: object,
    longitude: object,
    days_since_1970: object,
    doy: object | None = None,
    include_land_cover: bool = False,
) -> np.ndarray:
    """Build the ordered feature matrix from aligned input columns.

    ``columns`` contains target-day reflectance, quality, historical, recency,
    and local-solar-noon columns.  If ``doy`` is supplied, its cyclic columns
    are generated here; otherwise ``doy_sin`` and ``doy_cos`` must already be
    present in ``columns``.
    """
    arrays = {name: np.asarray(value) for name, value in columns.items()}
    if doy is not None:
        day_of_year = np.asarray(doy, dtype=np.float64)
        arrays["doy_sin"] = np.sin(2.0 * math.pi * day_of_year / 366.0)
        arrays["doy_cos"] = np.cos(2.0 * math.pi * day_of_year / 366.0)
    arrays.update(periodic_proxy_features(latitude, longitude, days_since_1970))

    names = feature_names(include_land_cover=include_land_cover)
    missing = [name for name in names if name not in arrays]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:5]}")
    try:
        matrix = np.column_stack(
            [np.asarray(arrays[name], dtype=np.float32) for name in names]
        )
    except ValueError as exc:
        raise ValueError("All feature columns must be one-dimensional and aligned") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("The generated feature matrix has an invalid shape")
    return matrix


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    include_land_cover: bool = False,
    device: str = "cuda",
    nthread: int = 10,
):
    """Train the regression model on the supplied feature matrix and labels."""
    import xgboost as xgb

    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels, dtype=np.float32)
    names = feature_names(include_land_cover=include_land_cover)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} feature columns, got {matrix.shape}")
    if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
        raise ValueError("labels must be one-dimensional and aligned with features")
    training_matrix = xgb.QuantileDMatrix(
        matrix,
        label=target,
        max_bin=int(MODEL_PARAMS["max_bin"]),
        feature_names=list(names),
    )
    params = dict(MODEL_PARAMS)
    params["device"] = device
    params["nthread"] = int(nthread)
    return xgb.train(
        params=params,
        dtrain=training_matrix,
        num_boost_round=NUM_BOOST_ROUND,
    )


def predict_model(
    model,
    features: np.ndarray,
    *,
    include_land_cover: bool = False,
) -> np.ndarray:
    """Generate predictions using the same ordered feature specification."""
    import xgboost as xgb

    matrix = np.asarray(features, dtype=np.float32)
    names = feature_names(include_land_cover=include_land_cover)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} feature columns, got {matrix.shape}")
    prediction_matrix = xgb.DMatrix(matrix, feature_names=list(names))
    return np.asarray(model.predict(prediction_matrix), dtype=np.float32)


def feature_manifest() -> dict[str, object]:
    """Return a serializable summary of the feature definitions"""

    return {
        "schema": "sif_feature_definitions_v1",
        "history_windows_days": list(HISTORY_WINDOWS_DAYS),
        "reflectance_bands": list(REFLECTANCE_BANDS),
        "history_statistics": ["mean", "std", "count", "max"],
        "recency_statistics": ["last_valid", "days_since_last_valid"],
        "feature_counts": {
            "base": len(feature_names()),
            "with_land_cover": len(feature_names(include_land_cover=True)),
        },
    }


def validate_feature_spec() -> None:
    """Check that feature names and generated proxy columns remain consistent"""
    
    if len(feature_names()) != 198:
        raise AssertionError("Unexpected base feature dimension")
    if len(feature_names(include_land_cover=True)) != 212:
        raise AssertionError("Unexpected feature dimension with land-cover categories")
    if set(PERIODIC_PROXY_FEATURES) != set(
        periodic_proxy_features(0.0, 0.0, 0).keys()
    ):
        raise AssertionError("Proxy feature names and generated values are inconsistent")


if __name__ == "__main__":
    validate_feature_spec()
    print(feature_manifest())
