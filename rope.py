# rope.py



from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata

import tensorflow as tf
from tensorflow import keras

# --- Your project imports ---
from ts_utils.custom_layers import PositionalEncoding
from ae_utils import utils as utils_cae
from ae_utils.attn_models import COAE


# ============================================================
# Optional: reduce TensorFlow log spam
# Set BEFORE TF does heavy initialization if you want maximum effect.
# You can comment this block out if you prefer logs.
# ============================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=ERROR only
tf.get_logger().setLevel("ERROR")


# ============================================================
# Small utils
# ============================================================
def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def _safe_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_keras_ensemble(model_dir: str, n_models: int = 5, custom_objects: Optional[dict] = None) -> List[keras.Model]:
    models: List[keras.Model] = []
    for i in range(1, n_models + 1):
        path = os.path.join(model_dir, f"best_model_{i}.keras")
        models.append(keras.models.load_model(path, compile=False, custom_objects=custom_objects))
    return models


def make_infer_fn(model: keras.Model):
    """
    Compile a stable inference function once per model.
    This avoids model.predict() retracing spam inside loops.
    """
    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)
    return infer


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ROPEConfig:
    latent_dim: int = 10
    seq_len: int = 3
    pad_hours_before: int = 2
    decode_batch_size: int = 1

    # driver_cols is determined automatically from stats_ts dim:
    # driver_dim = len(mu) - latent_dim
    #  - if driver_dim==6: [f10,kp,t1,t2,t3,t4]
    #  - if driver_dim==7: [f10,kp,t1,t2,t3,t4,doy]
    # If you ever add more features, extend mapping in _resolve_driver_cols().

    @property
    def base_driver_cols_6(self) -> List[str]:
        return ["f10", "kp", "t1", "t2", "t3", "t4"]

    @property
    def base_driver_cols_7(self) -> List[str]:
        return ["f10", "kp", "t1", "t2", "t3", "t4", "doy"]


# ============================================================
# Normalization using stats_ts
# ============================================================
class FeatureNormalizer:
    """
    stats_ts must provide:
      - stats_ts['mu']
      - stats_ts['sigma']
    """
    def __init__(self, stats_ts: Dict[str, Any], latent_dim: int):
        self.latent_dim = latent_dim
        self.mu = _to_numpy(stats_ts["mu"]).astype(np.float32)
        self.sigma = _to_numpy(stats_ts["sigma"]).astype(np.float32)

        if len(self.mu) != len(self.sigma):
            raise ValueError(f"mu/sigma mismatch: len(mu)={len(self.mu)} len(sigma)={len(self.sigma)}")

        self.total_dim = len(self.mu)
        self.driver_dim = self.total_dim - self.latent_dim
        if self.driver_dim <= 0:
            raise ValueError(f"Invalid stats dims: total_dim={self.total_dim}, latent_dim={self.latent_dim}")

    def norm_full(self, X: np.ndarray) -> np.ndarray:
        # X shape (..., total_dim)
        return ((X - self.mu) / self.sigma).astype(np.float32)

    def norm_driver(self, drv: np.ndarray) -> np.ndarray:
        # drv shape (..., driver_dim)
        return ((drv - self.mu[self.latent_dim:]) / self.sigma[self.latent_dim:]).astype(np.float32)

    def denorm_latents(self, lat_norm: np.ndarray) -> np.ndarray:
        # lat_norm shape (..., latent_dim)
        return (lat_norm * self.sigma[: self.latent_dim]) + self.mu[: self.latent_dim]


# ============================================================
# IC table interpolator: (F10,Kp) -> latent coeffs
# ============================================================
class ICTableInterpolator:
    def __init__(self, ic_table: pd.DataFrame):
        self.pts = ic_table[["F10", "Kp"]].values
        self.vals = ic_table.drop(columns=["F10", "Kp"]).values

    def get_latent_coeffs(self, f10: float, kp: float) -> np.ndarray:
        pred = griddata(self.pts, self.vals, np.array([[f10, kp]]), method="linear")
        if np.isnan(pred).any():
            pred = griddata(self.pts, self.vals, np.array([[f10, kp]]), method="nearest")
        return pred.flatten().astype(np.float32)


# ============================================================
# Build driver window + harmonics
# ============================================================
class DriverWindowBuilder:
    def build_internal(self, driver_df: pd.DataFrame, start_datetime, horizon: int, seq_len: int) -> pd.DataFrame:
        """
        Builds internal dataframe:
          history: seq_len-1 hours before start_datetime
          forecast: horizon hours starting at start_datetime
        total rows = (seq_len-1) + horizon
        """
        start_dt = pd.to_datetime(start_datetime)
        hist_start = start_dt - timedelta(hours=(seq_len - 1))
        end_dt = start_dt + timedelta(hours=(horizon - 1))

        # Create exact hourly timeline to avoid off-by-one / missing hours
        timeline = pd.date_range(hist_start, end_dt, freq="h")

        df = driver_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").reindex(timeline).reset_index().rename(columns={"index": "datetime"})

        # Ensure hour/doy exist
        df["hour"] = df["datetime"].dt.hour
        df["doy"] = df["datetime"].dt.dayofyear.astype(float)

        # Harmonics
        df["t1"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["t2"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["t3"] = np.sin(2 * np.pi * df["doy"] / 365.25)
        df["t4"] = np.cos(2 * np.pi * df["doy"] / 365.25)

        # continuous doy
        df["doy"] = df["doy"] + df["hour"] / 24

        # Guard: if any required drivers are missing due to gaps
        if df[["f10", "kp"]].isna().any().any():
            missing = df[df[["f10", "kp"]].isna().any(axis=1)][["datetime", "f10", "kp"]].head(10)
            raise ValueError(
                "Driver dataframe has missing f10/kp after reindexing to hourly timeline. "
                "Your SW file likely has gaps around this time.\n"
                f"First missing rows:\n{missing}"
            )

        return df

# ============================================================
# Sequence builder: X_init_norm and x_chunk
# ============================================================
class SequenceBuilder:
    def __init__(
        self,
        cfg: ROPEConfig,
        normalizer: FeatureNormalizer,
        ic_interp: ICTableInterpolator,
        driver_cols: List[str],
    ):
        self.cfg = cfg
        self.norm = normalizer
        self.ic = ic_interp
        self.driver_cols = driver_cols

        self.D = self.norm.total_dim
        self.K = self.cfg.latent_dim
        self.S = self.cfg.seq_len

        # sanity
        if len(self.driver_cols) != self.norm.driver_dim:
            raise ValueError(
                f"driver_cols length ({len(self.driver_cols)}) != stats driver_dim ({self.norm.driver_dim}). "
                f"driver_cols={self.driver_cols}"
            )
        if self.D != (self.K + len(self.driver_cols)):
            raise ValueError(f"Total dim mismatch: D={self.D} vs K+drivers={self.K + len(self.driver_cols)}")

    def build_X_init_norm(self, window_df: pd.DataFrame) -> np.ndarray:
        coeff_rows = []
        drv_rows = []

        for i in range(self.S):
            row = window_df.iloc[i]
            coeff = self.ic.get_latent_coeffs(row["f10"], row["kp"])  # (K,)
            drv = row[self.driver_cols].to_numpy(dtype=np.float32)    # (driver_dim,)
            coeff_rows.append(coeff)
            drv_rows.append(drv)

        coeff_mat = np.vstack(coeff_rows)   # (S,K)
        drv_mat = np.vstack(drv_rows)       # (S,driver_dim)
        X = np.hstack([coeff_mat, drv_mat]) # (S,D)

        if X.shape[1] != self.D:
            raise ValueError(f"X_init feature dim mismatch: got {X.shape[1]}, expected {self.D}")

        return self.norm.norm_full(X)

    def build_x_chunk(self, X_init_norm: np.ndarray, window_df: pd.DataFrame, device: str,horizon:int) -> torch.Tensor:
        H=horizon
        x_chunk = np.zeros((H, self.S, self.D), dtype=np.float32)
        x_chunk[0] = X_init_norm

        raw_drv = window_df[self.driver_cols].to_numpy(dtype=np.float32)

        for t in range(1, H):
            drv_norm = self.norm.norm_driver(raw_drv[t])
            row = np.zeros((self.D,), dtype=np.float32)
            row[self.K:] = drv_norm
            x_chunk[t] = np.vstack([x_chunk[t - 1][1:], row])

        return torch.tensor(x_chunk, dtype=torch.float32, device=device)


# ============================================================
# Dynamic rollout using compiled infer_fn (NO model.predict)
# ============================================================
class DynamicRollout:
    def __init__(self, latent_dim: int):
        self.K = latent_dim

    def run(self, infer_fn, x_chunk_np: np.ndarray, horizon: int) -> np.ndarray:
        """
        x_chunk_np: (H,S,D) numpy float32
        Returns: (H-1, K) normalized latent preds
        """
        inp = x_chunk_np[:1].copy()  # (1,S,D)
        preds = np.zeros((horizon - 1, self.K), dtype=np.float32)

        for t in range(1, horizon):
            p = infer_fn(tf.constant(inp)).numpy()   # (1,K)
            preds[t - 1] = p[0].astype(np.float32)

            if (t + 1) < horizon:
                inp[0, :-1] = inp[0, 1:]
                inp[0, -1, : self.K] = p[0]
                inp[0, -1, self.K :] = x_chunk_np[t + 1, 2, self.K :]

        return preds


# ============================================================
# Meta fusion
# ============================================================
class MetaFusion:
    def __init__(self, coeff_level: bool = True):
        self.coeff_level = coeff_level

    def fuse(self, meta_infer_fn, x_chunk_np_T: np.ndarray, all_preds: np.ndarray) -> np.ndarray:
        """
        x_chunk_np_T: (T,S,D) where T=H-1
        all_preds: (M,T,K)
        returns: (T,K)
        """
        M, T, K = all_preds.shape
        W = meta_infer_fn(tf.constant(x_chunk_np_T)).numpy()

        preds_TMK = all_preds.transpose(1, 0, 2)  # (T,M,K)

        if self.coeff_level:
            # expects W: (T,M,K)
            return np.sum(W * preds_TMK, axis=1).astype(np.float32)

        # expects W: (T,M)
        return np.sum(W[:, :, None] * preds_TMK, axis=1).astype(np.float32)


# ============================================================
# Decode latents -> density
# ============================================================
class LatentDecoder:
    def __init__(self, cae: COAE, stats_cae: Any, device: str, batch_size: int = 1):
        self.cae = cae
        self.stats_cae = stats_cae
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def decode(self, latent_series: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(latent_series, dtype=torch.float32, device=self.device)
        loader = DataLoader(x_tensor, batch_size=self.batch_size, shuffle=False)

        dens = []
        for x0 in loader:
            x_rec = self.cae.decoder(x0.to(self.device)).detach().cpu().numpy()
            xhat = np.power(10, utils_cae.denormalize(x_rec, self.stats_cae))
            dens.append(xhat)

        dens = np.concatenate(dens, axis=0).squeeze(1)
        return dens





# ============================================================
# ROPE main class  (horizon moved to run)
# ============================================================
class ROPE:
    """
    Usage:
      from rope1 import ROPE
      rope = ROPE(device="cuda")                # loads everything once
      res  = rope.run("2024-02-09 00:00:00", horizon=120)
    """

    def __init__(
        self,
        device: str = "cuda",

        # ---- Paths (keep defaults if your structure matches) ----
        driver_csv: str = "data/sw_celestrack_1957.csv",
        ic_table_csv: str = "data/IC_Table_modified.csv",

        # IMPORTANT: this MUST match your training feature set
        stats_ts_path: str = "data/stats_ts.pt",
        stats_cae_path: str = "data/stats_cae.pt",

        coae_config_yaml: str = "weights/finetuned_coae/config.yaml",
        coae_weights_pth: str = "weights/finetuned_coae/best_weights_1gpu.pth",

        lstm_dir: str = "Models/Storms/LSTM MODELS",
        gru_dir: str = "Models/Storms/GRU MODELS",
        transformer_dir: str = "Models/Storms/TRANSFORMER MODELS",
        meta_model_path: str = "Meta Models/MetaStormTunedBLa0.keras",

        # --- behavior ---
        use_xla: bool = False,   # set True only if you want XLA. False reduces compile spam.
    ):
        self.device = _safe_device(device)

        # NOTE: horizon is no longer stored in cfg; it is passed to run()
        self.cfg = ROPEConfig()

        # Control XLA (optional)
        tf.config.optimizer.set_jit(bool(use_xla))

        # ----------------------------
        # Load driver dataframe
        # ----------------------------
        self.driver_df = pd.read_csv(driver_csv)
        self.driver_df["datetime"] = pd.to_datetime(self.driver_df["datetime"])
        if "hour" not in self.driver_df.columns:
            self.driver_df["hour"] = self.driver_df["datetime"].dt.hour
        if "doy" not in self.driver_df.columns:
            self.driver_df["doy"] = self.driver_df["datetime"].dt.dayofyear.astype(float)

        # ----------------------------
        # Load IC table + interpolator
        # ----------------------------
        self.ic_table = pd.read_csv(ic_table_csv)
        self.ic_interp = ICTableInterpolator(self.ic_table)

        # ----------------------------
        # Load stats_ts / stats_cae
        # ----------------------------
        if not os.path.exists(stats_ts_path):
            raise FileNotFoundError(f"stats_ts_path not found: {stats_ts_path}")
        self.stats_ts = torch.load(stats_ts_path, map_location="cpu")

        if not os.path.exists(stats_cae_path):
            raise FileNotFoundError(f"stats_cae_path not found: {stats_cae_path}")
        try:
            self.stats_cae = torch.load(stats_cae_path, weights_only=True, map_location="cpu")
        except TypeError:
            self.stats_cae = torch.load(stats_cae_path, map_location="cpu")

        # ----------------------------
        # Normalizer
        # ----------------------------
        self.normalizer = FeatureNormalizer(self.stats_ts, latent_dim=self.cfg.latent_dim)
        self.driver_cols = self._resolve_driver_cols(self.normalizer.driver_dim)

        # ----------------------------
        # Load COAE
        # ----------------------------
        coae_cfg = _load_yaml(coae_config_yaml)
        self.cae = COAE(config=coae_cfg.get("model"))
        try:
            sd = torch.load(coae_weights_pth, weights_only=True, map_location="cpu")
        except TypeError:
            sd = torch.load(coae_weights_pth, map_location="cpu")
        self.cae.load_state_dict(sd)
        self.cae.to(self.device)
        self.cae.eval()

        # ----------------------------
        # Load base models (15)
        # ----------------------------
        self.lstm_models = load_keras_ensemble(lstm_dir, n_models=5, custom_objects=None)
        self.gru_models = load_keras_ensemble(gru_dir, n_models=5, custom_objects=None)
        self.transformer_models = load_keras_ensemble(
            transformer_dir,
            n_models=5,
            custom_objects={"PositionalEncoding": PositionalEncoding},
        )
        self.all_models = self.lstm_models + self.gru_models + self.transformer_models

        # Pre-compile inference fns (kills retracing)
        self.infer_fns = [make_infer_fn(m) for m in self.all_models]

        # ----------------------------
        # Load meta model + compiled infer
        # ----------------------------
        self.meta_model = keras.models.load_model(meta_model_path, compile=False)
        self.meta_infer_fn = make_infer_fn(self.meta_model)

        # ----------------------------
        # Pipeline components
        # ----------------------------
        self.window_builder = DriverWindowBuilder()
        self.seq_builder = SequenceBuilder(
            cfg=self.cfg,
            normalizer=self.normalizer,
            ic_interp=self.ic_interp,
            driver_cols=self.driver_cols,
        )
        self.rollout = DynamicRollout(latent_dim=self.cfg.latent_dim)
        self.fuser = MetaFusion(coeff_level=True)
        self.decoder = LatentDecoder(
            self.cae, self.stats_cae, device=self.device, batch_size=self.cfg.decode_batch_size
        )

    def _resolve_driver_cols(self, driver_dim: int) -> List[str]:
        if driver_dim == 6:
            return self.cfg.base_driver_cols_6
        if driver_dim == 7:
            return self.cfg.base_driver_cols_7

        raise ValueError(
            f"Unsupported driver_dim={driver_dim}. "
            f"Extend _resolve_driver_cols() if you added more features."
        )

    def run(
        self,
        start_datetime,
        horizon: int = 120,                       
        driver_df: Optional[pd.DataFrame] = None,
        decode_all: bool = False,
    ) -> Dict[str, Any]:

        H = int(horizon)
        df_drv = driver_df if driver_df is not None else self.driver_df

        # Build internal df: (seq_len-1 history) + (H forecast)
        internal_df = self.window_builder.build_internal(
            df_drv, start_datetime=start_datetime, horizon=H, seq_len=self.cfg.seq_len
        )

        # Forecast slice starts at index (seq_len-1)
        start_idx = self.cfg.seq_len - 1
        forecast_df = internal_df.iloc[start_idx : start_idx + H].reset_index(drop=True)

        # Build initial sequence from first seq_len rows of internal_df
        X_init_norm = self.seq_builder.build_X_init_norm(internal_df.iloc[: self.cfg.seq_len])

        # Build x_chunk for forecast horizon
        x_chunk = self.seq_builder.build_x_chunk(X_init_norm, forecast_df, device=self.device,horizon=H)

        # Convert once to numpy for TF
        x_chunk_np = x_chunk.detach().cpu().numpy().astype(np.float32)

        # Run base models -> (H-1,K) normalized latents
        base_list = []
        for infer_fn in self.infer_fns:
            preds = self.rollout.run(infer_fn, x_chunk_np, horizon=H)  # (H-1,K)
            base_list.append(preds)
        base_latents_norm = np.stack(base_list, axis=0).astype(np.float32)  # (M,H-1,K)

        # Meta fuse (normalized) -> (H-1,K)
        meta_in = x_chunk_np[: H - 1]  # (T,S,D)
        meta_latents_norm = self.fuser.fuse(self.meta_infer_fn, meta_in, base_latents_norm)  # (H-1,K)

        # De-normalize predicted latents (H-1,K)
        meta_latents_pred = self.normalizer.denorm_latents(meta_latents_norm).astype(np.float32)

        # Include t0 latent as first output so total = H
        init_lat_norm = X_init_norm[-1, : self.cfg.latent_dim]        # (K,)
        init_lat_phys = self.normalizer.denorm_latents(init_lat_norm) # (K,)
        meta_latents_full = np.vstack([init_lat_phys[None, :], meta_latents_pred])  # (H,K)

        # Decode ALL H steps so it aligns 1-to-1 with forecast_df
        meta_density = self.decoder.decode(meta_latents_full)

        out: Dict[str, Any] = {
            "window_df": forecast_df[["datetime", "f10", "kp"]].copy(),  # aligned length = H
            "meta_density": meta_density,                                # (H,72,36,45)
        }

        if decode_all:
            decoded_all = []
            for i in range(base_latents_norm.shape[0]):
                lat_phys_pred = self.normalizer.denorm_latents(base_latents_norm[i]).astype(np.float32)  # (H-1,K)
                lat_phys_full = np.vstack([init_lat_phys[None, :], lat_phys_pred])                       # (H,K)
                dens = self.decoder.decode(lat_phys_full)
                decoded_all.append(dens)
            out["decoded_all"] = np.stack(decoded_all, axis=0)  # (M,H,72,36,45)

        return out
