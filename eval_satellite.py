import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ts_utils import satellite as satmod
from rope import ROPE
from interpolator import DensityInterpolator



def compare_against_satellite(
    satellite_name: str,
    start_time: str,
    horizon: int = 120,
    device: str = "cuda",
    time_mode: str = "interp_time",
):
    # --- ROPE run + interpolator ---
    rope_obj = ROPE(device=device)
    res = rope_obj.run(start_time, horizon=horizon)  # if you want uncertainty, call with decode_all=True upstream
    q = DensityInterpolator(res)
    b = q.bounds()

    tmin = pd.Timestamp(b["time_min"])
    tmax = pd.Timestamp(b["time_max"])

    # --- load satellite months overlapping [tmin, tmax] ---
    months = []
    cur = pd.Timestamp(year=tmin.year, month=tmin.month, day=1)
    end = pd.Timestamp(year=tmax.year, month=tmax.month, day=1)
    while cur <= end:
        months.append((cur.year, cur.month))
        cur = cur + pd.offsets.MonthBegin(1)

    df_parts = []
    for yy, mm in months:
        df_parts.append(satmod.load_satellite_data(satellite=satellite_name, year=yy, month=mm))
    df_sat = pd.concat(df_parts, ignore_index=True)

    # --- build datetime + alt_km ---
    df_sat2 = df_sat.copy()
    df_sat2["datetime"] = pd.to_datetime(
        df_sat2["date"].astype(str) + " " + df_sat2["time"].astype(str),
        errors="coerce",
    )

    # alt in meters vs km (simple heuristic)
    alt_num = pd.to_numeric(df_sat2["alt"], errors="coerce")
    med_alt = float(np.nanmedian(alt_num.values))
    df_sat2["alt_km"] = alt_num / 1000.0 if (np.isfinite(med_alt) and med_alt > 2000) else alt_num

    # --- choose observed density column ---
    sat_lower = satellite_name.lower()
    if sat_lower == "swarm":
        obs_col = "swarm_density"
    elif sat_lower == "champ":
        obs_col = "champ_density"  # change if your CHAMP df uses a different name
    elif sat_lower == "grace-fo":
        obs_col = "grace_fo_density"  # change if needed
    else:
        raise ValueError(f"Unknown satellite_name='{satellite_name}'. Add its density column mapping.")

    # --- filter to bounds ---
    mask = (
        df_sat2["datetime"].between(tmin, tmax)
        & df_sat2["lst"].between(b["lst_min"], b["lst_max"])
        & df_sat2["lat"].between(b["lat_min"], b["lat_max"])
        & df_sat2["alt_km"].between(b["alt_km_min"], b["alt_km_max"])
    )

    df_filt = df_sat2.loc[mask].copy()
    df_filt = df_filt.dropna(subset=["datetime", "lst", "lat", "alt_km", obs_col])

    if len(df_filt) == 0:
        raise ValueError(f"No satellite points after filtering. Window: {tmin} → {tmax}")

    # --- interpolate ROPE on satellite track ---
    def _query_row(r):
        out = q.query(
            r["datetime"],
            lst=float(r["lst"]),
            lat=float(r["lat"]),
            alt_km=float(r["alt_km"]),
            time_mode=time_mode,
        )
        d = float(np.asarray(out["density"]).squeeze())

        # If sigma is available in res + interpolator, query() will include it
        s = None
        if isinstance(out, dict) and ("sigma" in out):
            s = float(np.asarray(out["sigma"]).squeeze())

        return pd.Series({"rope_density": d, "rope_sigma": s})

    tqdm.pandas()
    tmp = df_filt.progress_apply(_query_row, axis=1)
    df_filt["rope_density"] = tmp["rope_density"].values
    df_filt["rope_sigma"] = tmp["rope_sigma"].values

    # --- pack eval df ---
    df_eval = df_filt[["datetime", "lst", "lat", "alt_km", obs_col, "rope_density", "rope_sigma"]].copy()
    df_eval = df_eval.rename(columns={obs_col: "obs_density"}).sort_values("datetime")

    # --- plot 1: density comparison ---
    # --- plot 2: f10/kp from res['window_df'] ---
    # --- plot 3: uncertainty sigma (if present) ---
    has_sigma = df_eval["rope_sigma"].notna().any()

    fig, axes = plt.subplots(
        3 if has_sigma else 2,
        1,
        figsize=(14, 9 if has_sigma else 6.5),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1.3, 1.3] if has_sigma else [3, 1.3]},
    )

    ax0 = axes[0]
    ax0.plot(df_eval["datetime"], df_eval["obs_density"], label=f"{satellite_name} observed")
    ax0.plot(df_eval["datetime"], df_eval["rope_density"], label="ROPE (Meta)")

    if has_sigma:
        mu = df_eval["rope_density"].to_numpy()
        sig = df_eval["rope_sigma"].to_numpy()
        x = df_eval["datetime"].to_numpy()
        ax0.fill_between(x, mu - 2 * sig, mu + 2 * sig, alpha=0.2, label="ROPE ±2σ")

    ax0.set_xlabel("Time")
    ax0.set_ylabel("Density")
    ax0.set_title(f"{satellite_name} vs ROPE (Meta) | {tmin} → {tmax}")
    ax0.legend()

    # plot f10/kp (res window_df)
    ax1 = axes[1]
    wdf = res["window_df"].copy()
    wdf["datetime"] = pd.to_datetime(wdf["datetime"])
    wdf = wdf.sort_values("datetime")

    ax1.plot(wdf["datetime"], wdf["f10"], color="tab:blue", lw=2.5, label="F10")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("F10")

    ax1b = ax1.twinx()
    ax1b.plot(wdf["datetime"], wdf["kp"], color="tab:red", lw=2.0, linestyle="--", label="Kp")
    ax1b.set_ylabel("Kp")

    # combine legends from twin axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title("Celestrack Drivers")

    if has_sigma:
        ax2 = axes[2]
        ax2.plot(df_eval["datetime"], df_eval["rope_sigma"], label="σ (interpolated)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Sigma")
        ax2.set_title("ROPE uncertainty along satellite track")
        ax2.legend()

    plt.tight_layout()
    plt.show()

    return df_eval
