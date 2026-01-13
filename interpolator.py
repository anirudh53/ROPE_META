# interpolater.py
# Query ROPE output at a SINGLE timestamp and spatial coordinate.
#
# Two time handling modes:
#   1) time_mode="hold_next_hour"
#        - choose the NEXT available model time (ceil) and DO NOT interpolate in time
#        - spatial interpolation only
#
#   2) time_mode="interp_time"
#        - linear interpolation in time between bracketing model snapshots
#        - spatial interpolation at both times, then time interpolation
#
# Usage:
#   from rope1 import ROPE
#   from interpolater import DensityInterpolator
#
#   rope = ROPE(device="cuda")
#   res  = rope.run("2024-02-09 00:00:00", horizon=120)
#
#   q = DensityInterpolator(res)
#   v1 = q.query("2024-02-10 06:30:00", lst=10.5, lat=25.0, alt_km=400.0, time_mode="hold_next_hour")
#   v2 = q.query("2024-02-10 06:30:00", lst=10.5, lat=25.0, alt_km=400.0, time_mode="interp_time")

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


class TimeOutOfRangeError(ValueError):
    """Raised when the requested query time is outside the propagated ROPE window."""
    pass


class SpatialOutOfRangeError(ValueError):
    """Raised when the requested spatial coordinate is outside the ROPE grid bounds."""
    pass


@dataclass(frozen=True)
class GridAxes:
    lst_axis: np.ndarray
    lat_axis: np.ndarray
    alt_axis: np.ndarray


def default_axes() -> GridAxes:
    # Must match your COAE grid definition/order
    return GridAxes(
        lst_axis=np.linspace(0, 23.66666667, 72),
        lat_axis=np.linspace(-87.5, 87.5, 36),
        alt_axis=np.linspace(100, 980, 45),
    )


class DensityInterpolator:
    """
    Single-time spatial query on ROPE output.

    Requires:
      res["meta_density"] : (T,72,36,45)
      res["window_df"]["datetime"] : length T aligned to meta_density

    time_mode:
      - "hold_next_hour": use next model hour (ceil) only (no time interpolation)
      - "interp_time": linear interpolation in time between bracketing snapshots
    """

    def __init__(
        self,
        res: Dict[str, Any],
        axes: Optional[GridAxes] = None,
        bounds_error: bool = False,
        fill_value: float = np.nan,
    ):
        if "meta_density" not in res:
            raise KeyError("res must contain 'meta_density'")
        if "window_df" not in res or "datetime" not in res["window_df"].columns:
            raise KeyError("res must contain 'window_df' with a 'datetime' column")

        self.dens = np.asarray(res["meta_density"])
        self.times = pd.to_datetime(res["window_df"]["datetime"]).reset_index(drop=True)

        if self.dens.ndim != 4 or self.dens.shape[1:] != (72, 36, 45):
            raise ValueError(f"meta_density must have shape (T,72,36,45). Got {self.dens.shape}")

        if len(self.times) != self.dens.shape[0]:
            raise ValueError(
                f"Time alignment mismatch: len(window_df)={len(self.times)} vs meta_density T={self.dens.shape[0]}"
            )

        self.axes = axes if axes is not None else default_axes()
        self.bounds_error = bool(bounds_error)
        self.fill_value = fill_value

        # spatial bounds
        self._lst_min, self._lst_max = float(self.axes.lst_axis.min()), float(self.axes.lst_axis.max())
        self._lat_min, self._lat_max = float(self.axes.lat_axis.min()), float(self.axes.lat_axis.max())
        self._alt_min, self._alt_max = float(self.axes.alt_axis.min()), float(self.axes.alt_axis.max())

        # time bounds
        self._t_min = self.times.iloc[0]
        self._t_max = self.times.iloc[-1]

    # -------------------------
    # helpers
    # -------------------------
    def _validate_spatial(self, lst: float, lat: float, alt_km: float) -> None:
        if not (self._lst_min <= lst <= self._lst_max):
            raise SpatialOutOfRangeError(
                "Requested LST is outside the ROPE grid bounds.\n"
                f"  Requested LST: {lst}\n"
                f"  Available LST: [{self._lst_min}  -  {self._lst_max}]"
            )
        if not (self._lat_min <= lat <= self._lat_max):
            raise SpatialOutOfRangeError(
                "Requested latitude is outside the ROPE grid bounds.\n"
                f"  Requested lat: {lat}\n"
                f"  Available lat: [{self._lat_min}  -  {self._lat_max}]"
            )
        if not (self._alt_min <= alt_km <= self._alt_max):
            raise SpatialOutOfRangeError(
                "Requested altitude is outside the ROPE grid bounds.\n"
                f"  Requested alt_km: {alt_km}\n"
                f"  Available alt_km: [{self._alt_min}  -  {self._alt_max}]"
            )

    def _point(self, lst: float, lat: float, alt_km: float) -> np.ndarray:
        return np.array([[float(lst), float(lat), float(alt_km)]], dtype=np.float64)

    def _spatial_value(self, dens_t: np.ndarray, point: np.ndarray) -> float:
        f = RegularGridInterpolator(
            (self.axes.lst_axis, self.axes.lat_axis, self.axes.alt_axis),
            dens_t,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )
        return float(f(point)[0])

    def _bracket_indices(self, when: pd.Timestamp) -> tuple[int, int]:
        """
        Returns (i0,i1) such that times[i0] <= when <= times[i1], with i1=i0+1.
        Assumes when is inside range.
        """
        i1 = int(np.searchsorted(self.times.values, np.datetime64(when)))
        i0 = i1 - 1
        return i0, i1

    # -------------------------
    # public API
    # -------------------------
    def query(
        self,
        when: Union[str, pd.Timestamp],
        lst: float,
        lat: float,
        alt_km: float,
        time_mode: str = "interp_time",
    ) -> Dict[str, Any]:
        """
        Returns density at (when, lst, lat, alt_km) using selected time handling.

        time_mode:
          - "hold_next_hour": density at next propagated hour (ceil), spatial interpolation only
          - "interp_time": time interpolation between bracketing hours, spatial + time interpolation

        Raises
        ------
        TimeOutOfRangeError
            If `when` is outside [window_df.datetime.min(), window_df.datetime.max()].
        SpatialOutOfRangeError
            If (lst, lat, alt_km) is outside the grid bounds.
        """
        when = pd.to_datetime(when)

        if time_mode not in ("hold_next_hour", "interp_time"):
            raise ValueError("time_mode must be 'hold_next_hour' or 'interp_time'")

        # No clamping: validate time strictly
        if when < self._t_min or when > self._t_max:
            raise TimeOutOfRangeError(
                "Requested time is outside the propagated ROPE window.\n"
                f"  Requested: {when}\n"
                f"  Available: [{self._t_min}  -  {self._t_max}]\n"
                "Please run ROPE with a start_datetime/horizon that covers the requested time."
            )

        # No clamping: validate spatial strictly
        self._validate_spatial(float(lst), float(lat), float(alt_km))
        point = self._point(lst, lat, alt_km)

        # Inside range
        i0, i1 = self._bracket_indices(when)
        t0, t1 = self.times.iloc[i0], self.times.iloc[i1]

        # Case 1: hold-next-hour (ceil)
        if time_mode == "hold_next_hour":
            # if exactly at a model time, use that; else use the next one
            use_i = i0 if when == t0 else i1
            v = self._spatial_value(self.dens[use_i], point)
            return {
                "datetime_requested": when,
                "datetime_used": self.times.iloc[use_i],
                "density": v,
                "t_index": int(use_i),
                "time_mode": "hold_next_hour",
            }

        # Case 2: interpolate time axis as well (linear)
        w = float((when - t0) / (t1 - t0))  # 0..1
        v0 = self._spatial_value(self.dens[i0], point)
        v1 = self._spatial_value(self.dens[i1], point)
        v = (1.0 - w) * v0 + w * v1

        return {
            "datetime": when,
            "density": v,
            "t_index_left": int(i0),
            "t_index_right": int(i1),
            "datetime_left": t0,
            "datetime_right": t1,
            "time_weight_right": w,
            "time_mode": "interp_time",
        }

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
