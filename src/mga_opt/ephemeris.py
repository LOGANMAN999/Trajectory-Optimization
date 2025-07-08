from pathlib import Path
import spiceypy as spice
from spiceypy import stypes
import pandas as pd
from .constants import MU
from scipy.interpolate import CubicHermiteSpline
import numpy as np


#spice.furnsh("data/de440.bsp")      # planetary ephemeris
#spice.furnsh("data/jup365.bsp")      # Jovian moons
#spice.furnsh("data/naif0012.tls")    # leapseconds


ROOT_DIR = Path(__file__).resolve().parents[2]   
DATA_DIR = ROOT_DIR / "data"                    

for kernel in ("de440.bsp", "jup365.bsp", "naif0012.tls"):
    spice.furnsh(str((DATA_DIR / kernel).resolve()))


bodies = {
    "Sun":     10,
    "Mercury": 199,
    "Venus":   299,
    "Earth":   399,
    "Mars":    4,
    "Jupiter": 599,
    "Saturn" : 6, 
    "Neptune" : 8, 
}

times_utc = pd.date_range("1989-08-01", periods=16790, freq="1D")
times_et  = [spice.utc2et(t.strftime("%Y-%m-%dT%H:%M:%S")) for t in times_utc]


rows = []
for t_utc, t_et in zip(times_utc, times_et):
    for name, naif_id in bodies.items():
        state, _ = spice.spkezr(str(naif_id), t_et, "ECLIPJ2000", "NONE", "10")  
        
        x,y,z,vx,vy,vz = state
        rows.append({
            "time": t_utc, "body": name,
            "x": x, "y": y, "z": z,
            "vx": vx, "vy": vy, "vz": vz
        })


ephem_df = pd.DataFrame(rows)


spice.kclear()


ephem_df["t_num"] = (ephem_df.time.astype(np.int64) // 10**9).astype(float)

t0 = ephem_df.t_num.min()
t1 = ephem_df.t_num.max()

t_anim = np.arange(t0, t1, 3600.0)

splines = {}
for b in bodies:
    dfb = ephem_df[ephem_df.body == b].sort_values("t_num")
    t  = dfb["t_num"].values
    for coord in ["x","y"]:
        vals = dfb[coord].values
        # approximate derivative from velocity columns
        deriv = dfb["v" + coord].values
        splines[(b,coord)] = CubicHermiteSpline(t, vals, deriv)

__all__ = ["splines", "t0", "bodies", "precompute"]

def precompute(cfg_path: str) -> None:
    """
    Called once from the CLI to force-load kernels, sample ephemeris,
    build splines, and pickle them to disk so later runs avoid SPICE I/O.
    """
    import pickle, os
    cache_file = Path("cache") / f"ephem_{Path(cfg_path).stem}.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    with cache_file.open("wb") as f:
        pickle.dump({"splines": splines, "t0": t0, "bodies": bodies, "MU": MU}, f)

    print(f"[ephemeris] cached splines → {cache_file}")