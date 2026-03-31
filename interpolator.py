from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


class TimeOutOfRangeError(ValueError):
    pass


class SpatialOutOfRangeError(ValueError):
    """Kept for backward compatibility; this interpolator now WARNs (does not raise) on spatial OOB."""
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

    Spatial out-of-bounds handling (UPDATED):
      - LST is treated as CYCLIC with period 24 hr:
          -> lst = 24.0  behaves like 0.0
          -> lst = 23.8  interpolates across the seam with wrapped 0.0
          -> lst = -0.2  behaves like 23.8
      - lat below/above grid, or alt below grid minimum:
          -> emits a warning, returns 0.0 (and sigma=0.0 if available)
      - alt > 2000 km:
          -> emits a warning, returns 0.0 immediately

    Altitude handling (when spatially in-bounds for lat and alt >= alt_min):
      - alt_km <= 980          : standard RegularGridInterpolator (within grid)
      - 980 < alt_km <= 2000   : exponential extrapolation using top two grid levels
      - alt_km > 2000          : returns 0.0 immediately
    """

    ALT_HARD_ZERO_KM: float = 2000.0
    LST_PERIOD_HR: float = 24.0

    def __init__(
        self,
        res: Dict[str, Any],
        axes: Optional[GridAxes] = None,
        fill_value: float = np.nan,
    ):
        if "window_df" not in res or "datetime" not in res["window_df"].columns:
            raise KeyError("res must contain 'window_df' with a 'datetime' column")

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

        self.sigma = None
        if "density_std" in res:
            self.sigma = np.asarray(res["density_std"])
            if self.sigma.shape != self.dens.shape:
                raise ValueError(
                    f"density_std must have same shape as density. dens={self.dens.shape}, std={self.sigma.shape}"
                )

        self.axes = axes if axes is not None else default_axes()
        self.fill_value = fill_value

        self._lst_min = float(self.axes.lst_axis.min())
        self._lst_max = float(self.axes.lst_axis.max())
        self._lat_min = float(self.axes.lat_axis.min())
        self._lat_max = float(self.axes.lat_axis.max())
        self._alt_min = float(self.axes.alt_axis.min())
        self._alt_grid_top = float(self.axes.alt_axis.max())
        self._alt_max = self.ALT_HARD_ZERO_KM

        self._tail_z0 = float(self.axes.alt_axis[-2])
        self._tail_z1 = float(self.axes.alt_axis[-1])

        self._t_min = self.times.iloc[0]
        self._t_max = self.times.iloc[-1]

        # Extend cyclic LST axis with 24.0, and later repeat first slice at end.
        self._lst_axis_ext = np.append(self.axes.lst_axis, self.LST_PERIOD_HR)

    def bounds(self) -> Dict[str, float]:
        return {
            "lst_min": self._lst_min,
            "lst_max": self._lst_max,
            "lat_min": self._lat_min,
            "lat_max": self._lat_max,
            "alt_km_min": self._alt_min,
            "alt_km_max": self._alt_max,
            "alt_km_grid_top": self._alt_grid_top,
            "time_min": self._t_min,
            "time_max": self._t_max,
        }

    # ---------- helpers ----------

    def _spatial_oob_warning(self, msg: str) -> None:
        warnings.warn(msg, category=RuntimeWarning, stacklevel=3)

    def _wrap_lst(self, lst: float) -> float:
        """
        Wrap LST into [0, 24).
        Examples:
            24.0  -> 0.0
            23.8  -> 23.8
            -0.2  -> 23.8
            48.0  -> 0.0
        """
        return float(lst % self.LST_PERIOD_HR)

    def _spatial_is_in_bounds(self, lst: float, lat: float, alt_km: float) -> bool:
        """
        LST is cyclic, so it is always accepted if finite.
        lat / alt keep the same OOB behavior as before.
        """
        if not np.isfinite(lst):
            self._spatial_oob_warning(
                f"[DensityInterpolator] LST is not finite ({lst}). Returning 0."
            )
            return False

        if not (self._lat_min <= lat <= self._lat_max):
            self._spatial_oob_warning(
                f"[DensityInterpolator] lat out of bounds ({lat}); expected [{self._lat_min}, {self._lat_max}]. Returning 0."
            )
            return False

        if alt_km < self._alt_min:
            self._spatial_oob_warning(
                f"[DensityInterpolator] alt_km below grid minimum ({alt_km} < {self._alt_min}). Returning 0."
            )
            return False

        if alt_km > self.ALT_HARD_ZERO_KM:
            self._spatial_oob_warning(
                f"[DensityInterpolator] alt_km exceeds hard-zero ceiling ({alt_km} > {self.ALT_HARD_ZERO_KM}). Returning 0."
            )
            return False

        return True

    def _point(self, lst: float, lat: float, alt_km: float) -> np.ndarray:
        lst_wrapped = self._wrap_lst(lst)
        return np.array([[lst_wrapped, float(lat), float(alt_km)]], dtype=np.float64)

    def _extend_field_cyclic_lst(self, field_t: np.ndarray) -> np.ndarray:
        """
        Extend field along LST by appending the first slice at 24.0,
        so interpolation wraps smoothly across the seam.
        Input shape:  (72, 36, 45)
        Output shape: (73, 36, 45)
        """
        return np.concatenate([field_t, field_t[0:1, :, :]], axis=0)

    def _make_interpolator(self, field_t: np.ndarray) -> RegularGridInterpolator:
        """
        Build a spatial interpolator for one time-slice of a field.
        Uses cyclic extension in LST.
        """
        field_ext = self._extend_field_cyclic_lst(field_t)

        return RegularGridInterpolator(
            (self._lst_axis_ext, self.axes.lat_axis, self.axes.alt_axis),
            field_ext,
            bounds_error=False,
            fill_value=self.fill_value,
        )

    def _spatial_value(self, field_t: np.ndarray, point: np.ndarray) -> float:
        """
        Return the spatially interpolated (or extrapolated) value at *point*.

        Altitude regime logic
        ---------------------
        alt <= 980 km          : RegularGridInterpolator (trilinear within grid)
        980 < alt <= 2000 km   : exponential extrapolation anchored at grid top
        alt > 2000 km          : 0.0 (hard zero)

        LST logic
        ---------
        LST is cyclic. The interpolator is built on an extended axis:
            [0, ..., 23.6667, 24.0]
        where the 24.0 slice is a repeat of the 0.0 slice.
        """
        lst = float(point[0, 0])
        lat = float(point[0, 1])
        alt_km = float(point[0, 2])

        # always wrap before interpolation
        lst = self._wrap_lst(lst)
        point_wrapped = np.array([[lst, lat, alt_km]], dtype=np.float64)

        if alt_km > self.ALT_HARD_ZERO_KM:
            return 0.0

        f = self._make_interpolator(field_t)

        if alt_km <= self._alt_grid_top:
            return float(f(point_wrapped)[0])

        # Exponential tail (980 < alt <= 2000)
        pt_top1 = np.array([[lst, lat, self._tail_z1]], dtype=np.float64)
        pt_top0 = np.array([[lst, lat, self._tail_z0]], dtype=np.float64)

        rho1 = float(f(pt_top1)[0])
        rho0 = float(f(pt_top0)[0])

        dz = self._tail_z1 - self._tail_z0

        if rho0 > 0.0 and rho1 > 0.0:
            log_ratio = np.log(rho0 / rho1)
            H = dz / log_ratio if log_ratio > 1e-30 else 100.0
        else:
            H = 100.0

        extrapolated = rho1 * np.exp(-(alt_km - self._tail_z1) / H)
        return float(extrapolated)

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
        Return density at (when, lst, lat, alt_km).

        If density_std exists in res, ALSO returns "sigma".
        """
        when = pd.to_datetime(when)

        if time_mode not in ("hold_next_hour", "interp_time"):
            raise ValueError("time_mode must be 'hold_next_hour' or 'interp_time'")

        if when < self._t_min or when > self._t_max:
            raise TimeOutOfRangeError(
                f"Requested time {when} outside [{self._t_min}, {self._t_max}]"
            )

        lst_f, lat_f, alt_f = float(lst), float(lat), float(alt_km)
        lst_wrapped = self._wrap_lst(lst_f)

        spatial_ok = self._spatial_is_in_bounds(lst_f, lat_f, alt_f)
        point = self._point(lst_f, lat_f, alt_f)

        i0, i1 = self._bracket_indices(when)
        t0, t1 = self.times.iloc[i0], self.times.iloc[i1]

        # ── hold_next_hour ────────────────────────────────────────────────────
        if time_mode == "hold_next_hour":
            use_i = i0 if when == t0 else i1

            if not spatial_ok:
                out = {
                    "datetime_requested": when,
                    "datetime_used": self.times.iloc[use_i],
                    "density": 0.0,
                    "t_index": int(use_i),
                    "time_mode": "hold_next_hour",
                    "alt_regime": self._alt_regime(alt_f),
                    "spatial_oob": True,
                    "lst_requested": lst_f,
                    "lst_used_wrapped": lst_wrapped,
                }
                if self.sigma is not None:
                    out["sigma"] = 0.0
                return out

            dens_val = self._spatial_value(self.dens[use_i], point)

            out = {
                "datetime_requested": when,
                "datetime_used": self.times.iloc[use_i],
                "density": float(dens_val),
                "t_index": int(use_i),
                "time_mode": "hold_next_hour",
                "alt_regime": self._alt_regime(alt_f),
                "lst_requested": lst_f,
                "lst_used_wrapped": lst_wrapped,
            }

            if self.sigma is not None:
                sig_val = self._spatial_value(self.sigma[use_i], point)
                out["sigma"] = float(sig_val)

            return out

        # ── interp_time ───────────────────────────────────────────────────────
        w = float((when - t0) / (t1 - t0))

        if not spatial_ok:
            out = {
                "datetime": when,
                "density": 0.0,
                "t_index_left": int(i0),
                "t_index_right": int(i1),
                "datetime_left": t0,
                "datetime_right": t1,
                "time_weight_right": w,
                "time_mode": "interp_time",
                "alt_regime": self._alt_regime(alt_f),
                "spatial_oob": True,
                "lst_requested": lst_f,
                "lst_used_wrapped": lst_wrapped,
            }
            if self.sigma is not None:
                out["sigma"] = 0.0
            return out

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
            "alt_regime": self._alt_regime(alt_f),
            "lst_requested": lst_f,
            "lst_used_wrapped": lst_wrapped,
        }

        if self.sigma is not None:
            s0 = self._spatial_value(self.sigma[i0], point)
            s1 = self._spatial_value(self.sigma[i1], point)
            sig_val = (1.0 - w) * s0 + w * s1
            out["sigma"] = float(sig_val)

        return out

    # ---------- utility ----------

    def _alt_regime(self, alt_km: float) -> str:
        if alt_km > self.ALT_HARD_ZERO_KM:
            return "hard_zero"
        if alt_km > self._alt_grid_top:
            return "exponential_extrapolation"
        return "grid_interpolation"
