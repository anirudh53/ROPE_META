import pandas as pd
from ftplib import FTP
from io import BytesIO, StringIO
import zipfile
import re

def load_satellite_data(satellite, year, month, version="v02"):
    sat = satellite.lower()

    # Satellite-specific settings
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

    ftp_host = "thermosphere.tudelft.nl"
    ftp_path = f"/version_02/{satellite_info[sat]['dir']}"
    zip_filename = satellite_info[sat]['filename']
    columns = satellite_info[sat]['columns']

    try:
        # Connect to FTP and read zip into memory
        ftp = FTP(ftp_host)
        ftp.login()
        ftp.cwd(ftp_path)
        zip_buffer = BytesIO()
        ftp.retrbinary(f"RETR {zip_filename}", zip_buffer.write)
        ftp.quit()

        # Extract and decode file
        zip_buffer.seek(0)
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
        df = pd.read_csv(StringIO("\n".join(fixed_lines)), sep=r"\s+", names=columns)
        print(f" Loaded {sat.upper()} {year}-{month:02d}: {df.shape[0]} rows")
        return df

    except Exception as e:
        print(f"Failed to load {zip_filename}: {e}")
        return None



# # EXAMPLE USAGE
# df_grace = load_satellite_data(satellite="grace", year=2004, month=1)
# df_champ = load_satellite_data(satellite="champ", year=2003, month=7)
# df_swarm = load_satellite_data(satellite="swarm", year=2017, month=3)
# df_grace_fo = load_satellite_data(satellite="grace-fo", year=2019, month=5)
