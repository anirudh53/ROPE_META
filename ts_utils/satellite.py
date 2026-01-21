import pandas as pd
import requests
from io import BytesIO, StringIO
import zipfile
import re


def load_satellite_data(satellite, year, month, version="v02"):
    sat = satellite.lower()

    satellite_info = {
        "grace": {
            "dir": "GRACE_data",
            "filename": f"GA_DNS_ACC_{year}_{month:02d}_{version}.zip",
            "columns": ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                        'grace_density', 'dens_mean', 'flag_dens', 'flag_dens_mean']
        },
        "champ": {
            "dir": "CHAMP_data",
            "filename": f"CH_DNS_ACC_{year}-{month:02d}_{version}.zip",
            "columns": ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                        'champ_density', 'dens_mean', 'flag_dens', 'flag_dens_mean']
        },
        "swarm": {
            "dir": "Swarm_data",
            "filename": f"SA_DNS_POD_{year}_{month:02d}_{version}.zip",
            "columns": ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                        'swarm_density', 'dens_mean', 'flag_dens', 'flag_dens_mean']
        },
        "grace-fo": {
            "dir": "GRACE-FO_data",
            "filename": f"GC_DNS_ACC_{year}_{month:02d}_{version}c.zip",
            "columns": ['date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
                        'grace_fo_density', 'dens_mean', 'flag_dens', 'flag_dens_mean']
        }
    }

    if sat not in satellite_info:
        raise ValueError(f"Unknown satellite: {sat}")

    info = satellite_info[sat]
    zip_filename = info["filename"]
    columns = info["columns"]

    # HTTPS URL (FTP is deprecated)
    url = f"https://thermosphere.tudelft.nl/data/data/version_02/{info['dir']}/{zip_filename}"

    try:
        # Download ZIP via HTTPS
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        zip_buffer = BytesIO(response.content)

        # Extract text file
        with zipfile.ZipFile(zip_buffer) as zf:
            txt_filename = zf.namelist()[0]
            raw = zf.read(txt_filename).decode("utf-8")

        # Fix missing dates using forward-fill
        fixed_lines = []
        current_date = None
        for line in raw.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            if re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
                current_date = parts[0]
                fixed_lines.append(" ".join(parts))
            else:
                fixed_lines.append(current_date + " " + " ".join(parts))

        # Read into DataFrame
        df = pd.read_csv(
            StringIO("\n".join(fixed_lines)),
            sep=r"\s+",
            names=columns
        )

        print(f"Loaded {sat.upper()} {year}-{month:02d}: {df.shape[0]} rows")
        return df

    except Exception as e:
        print(f"Failed to load {zip_filename}: {e}")
        return None
