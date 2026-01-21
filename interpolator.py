# interpolator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


class TimeOutOfRangeError(ValueError):
    pass


class SpatialOutOfRangeError(ValueError):
    pass


@dataclass(frozen=True)
class GridAxes:
    lst_axis: np.ndarray
    lat_axis: np.ndarray
    alt_axis: np.ndarray


def default_axes() -> GridAxes:
    return GridAxes(
        lst_axis=np.linspace(0, 23.66666667, 72),
        lat_axis=np.linspace(-87.5, 87.5, 36),
        alt_axis=np.linspace(100, 980, 45),
    )


class DensityInterpolator:
    """
    Query ROPE output at a single timestamp and spatial coordinate.

    Density source priority:
      1) res["density_mean"] if present
      2) res["meta_density"] otherwise

    Uncertainty:
      If res["density_std"] exists (same shape as density), query() ALSO returns "sigma".

    time_mode:
      - "hold_next_hour": use next model hour (ceil), no time interpolation
      - "interp_time": linear interpolation between bracketing snapshots
    """

    def __init__(
        self,
        res: Dict[str, Any],
        axes: Optional[GridAxes] = None,
        bounds_error: bool = False,
        fill_value: float = np.nan,
    ):
        if "window_df" not in res or "datetime" not in res["window_df"].columns:
            raise KeyError("res must contain 'window_df' with a 'datetime' column")

        # Prefer density_mean if present, else meta_density
        if "density_mean" in res:
            self.dens = np.asarray(res["density_mean"])
        else:
            if "meta_density" not in res:
                raise KeyError("res must contain 'meta_density' or 'density_mean'")
            self.dens = np.asarray(res["meta_density"])

        self.times = pd.to_datetime(res["window_df"]["datetime"]).reset_index(drop=True)

        if self.dens.ndim != 4 or self.dens.shape[1:] != (72, 36, 45):
            raise ValueError(f"density must have shape (T,72,36,45). Got {self.dens.shape}")

        if len(self.times) != self.dens.shape[0]:
            raise ValueError(
                f"Time alignment mismatch: len(window_df)={len(self.times)} vs density T={self.dens.shape[0]}"
            )

        # Optional sigma field
        self.sigma = None
        if "density_std" in res:
            self.sigma = np.asarray(res["density_std"])
            if self.sigma.shape != self.dens.shape:
                raise ValueError(
                    f"density_std must have same shape as density. dens={self.dens.shape}, std={self.sigma.shape}"
                )

        self.axes = axes if axes is not None else default_axes()
        self.bounds_error = bool(bounds_error)
        self.fill_value = fill_value

        # Bounds
        self._lst_min, self._lst_max = float(self.axes.lst_axis.min()), float(self.axes.lst_axis.max())
        self._lat_min, self._lat_max = float(self.axes.lat_axis.min()), float(self.axes.lat_axis.max())
        self._alt_min, self._alt_max = float(self.axes.alt_axis.min()), float(self.axes.alt_axis.max())
        self._t_min = self.times.iloc[0]
        self._t_max = self.times.iloc[-1]

    def bounds(self) -> Dict[str, float]:
        return {
            "lst_min": self._lst_min,
            "lst_max": self._lst_max,
            "lat_min": self._lat_min,
            "lat_max": self._lat_max,
            "alt_km_min": self._alt_min,
            "alt_km_max": self._alt_max,
            "time_min": self._t_min,
            "time_max": self._t_max,
        }

    # ---------- helpers ----------
    def _validate_spatial(self, lst: float, lat: float, alt_km: float) -> None:
        if not (self._lst_min <= lst <= self._lst_max):
            raise SpatialOutOfRangeError(
                f"LST out of bounds: {lst} not in [{self._lst_min}, {self._lst_max}]"
            )
        if not (self._lat_min <= lat <= self._lat_max):
            raise SpatialOutOfRangeError(
                f"lat out of bounds: {lat} not in [{self._lat_min}, {self._lat_max}]"
            )
        if not (self._alt_min <= alt_km <= self._alt_max):
            raise SpatialOutOfRangeError(
                f"alt_km out of bounds: {alt_km} not in [{self._alt_min}, {self._alt_max}]"
            )

    def _point(self, lst: float, lat: float, alt_km: float) -> np.ndarray:
        return np.array([[float(lst), float(lat), float(alt_km)]], dtype=np.float64)

    def _spatial_value(self, field_t: np.ndarray, point: np.ndarray) -> float:
        f = RegularGridInterpolator(
            (self.axes.lst_axis, self.axes.lat_axis, self.axes.alt_axis),
            field_t,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )
        return float(f(point)[0])

    def _bracket_indices(self, when: pd.Timestamp) -> tuple[int, int]:
        i1 = int(np.searchsorted(self.times.values, np.datetime64(when)))
        i0 = i1 - 1
        return i0, i1

    # ---------- main API ----------
    def query(
        self,
        when: Union[str, pd.Timestamp],
        lst: float,
        lat: float,
        alt_km: float,
        time_mode: str = "hold_next_hour",
    ) -> Dict[str, Any]:
        """
        Returns density at (when, lst, lat, alt_km).
        If density_std exists in res, ALSO returns sigma automatically.

        time_mode:
          - "hold_next_hour"
          - "interp_time"
        """
        when = pd.to_datetime(when)

        if time_mode not in ("hold_next_hour", "interp_time"):
            raise ValueError("time_mode must be 'hold_next_hour' or 'interp_time'")

        if when < self._t_min or when > self._t_max:
            raise TimeOutOfRangeError(
                f"Requested time {when} outside [{self._t_min}, {self._t_max}]"
            )

        self._validate_spatial(float(lst), float(lat), float(alt_km))
        point = self._point(lst, lat, alt_km)

        i0, i1 = self._bracket_indices(when)
        t0, t1 = self.times.iloc[i0], self.times.iloc[i1]

        # ---- hold_next_hour ----
        if time_mode == "hold_next_hour":
            use_i = i0 if when == t0 else i1

            dens_val = self._spatial_value(self.dens[use_i], point)

            out = {
                "datetime_requested": when,
                "datetime_used": self.times.iloc[use_i],
                "density": float(dens_val),
                "t_index": int(use_i),
                "time_mode": "hold_next_hour",
            }

            if self.sigma is not None:
                sig_val = self._spatial_value(self.sigma[use_i], point)
                out["sigma"] = float(sig_val)

            return out

        # ---- interp_time ----
        w = float((when - t0) / (t1 - t0))  # 0..1

        d0 = self._spatial_value(self.dens[i0], point)
        d1 = self._spatial_value(self.dens[i1], point)
        dens_val = (1.0 - w) * d0 + w * d1

        out = {
            "datetime": when,
            "density": float(dens_val),
            "t_index_left": int(i0),
            "t_index_right": int(i1),
            "datetime_left": t0,
            "datetime_right": t1,
            "time_weight_right": w,
            "time_mode": "interp_time",
        }

        if self.sigma is not None:
            s0 = self._spatial_value(self.sigma[i0], point)
            s1 = self._spatial_value(self.sigma[i1], point)
            sig_val = (1.0 - w) * s0 + w * s1
            out["sigma"] = float(sig_val)

        return out
