"""
Round-2 revision: Moran's I correlogram of presence residuals (1 - mean OOF score)
for the selected configuration. Uses the archived presence_oof_residuals.csv and the
same statistic and permutation scheme as paper1_revision_diagnostics.py.
Reports cumulative distance bands (0-d km, comparable with the 100-km value in the
manuscript) and annular distance classes (a conventional correlogram).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "revision_outputs" / "presence_oof_residuals.csv"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

df = pd.read_csv(RES)
lon = np.radians(df["longitude"].to_numpy())
lat = np.radians(df["latitude"].to_numpy())
dlon = lon[:, None] - lon[None, :]
dlat = lat[:, None] - lat[None, :]
a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
dist = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
z = df["residual"].to_numpy() - df["residual"].mean()
n = len(z)


def moran(w, zz):
    s0 = w.sum()
    return n / s0 * np.sum(w * zz[:, None] * zz[None, :]) / np.sum(zz ** 2)


rows = []
classes = [("cumulative", 0, 25), ("cumulative", 0, 50), ("cumulative", 0, 100), ("cumulative", 0, 200),
           ("annular", 0, 25), ("annular", 25, 50), ("annular", 50, 100), ("annular", 100, 200),
           ("annular", 200, 400)]
for kind, lo, hi in classes:
    w = ((dist > lo) & (dist <= hi)).astype(float)
    if w.sum() == 0:
        continue
    obs = moran(w, z)
    rng = np.random.default_rng(42)
    null = np.array([moran(w, rng.permutation(z)) for _ in range(9999)])
    p = (np.count_nonzero(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    rows.append({"band_type": kind, "lower_km": lo, "upper_km": hi, "weight_links": int(w.sum()),
                 "morans_i": obs, "expected_i": -1 / (n - 1), "permutation_p_two_sided": p})
out = pd.DataFrame(rows)
out.to_csv(TAB / "TableS9_moran_correlogram.csv", index=False)
print(out.round(4).to_string(index=False))
