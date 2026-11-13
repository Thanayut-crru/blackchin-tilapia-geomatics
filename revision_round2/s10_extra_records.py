"""
Round-2 revision: identify eligible 2019-2024 records omitted from the 38 modelled
records and build two augmented occurrence sets for a sensitivity analysis.

Grid and eligibility follow audit/01_occurrence_audit.py exactly:
  research grade, not captive, coordinate uncertainty <= 5 km or unknown,
  one record per 2.5-arcmin cell with cell = (int(lat / 0.0417), int(lon / 0.0417)).
Omitted cells are split into
  (a) cells whose retained record after de-duplication was from 2025-2026, so the
      later year filter removed the cell although 2019-2024 records existed there
      (processing-order omission), and
  (b) other cells with eligible records that were not in the original download.
One record per added cell is taken (earliest observation date).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAB = HERE / "tables"
OUT = HERE / "extra_records"
OUT.mkdir(exist_ok=True)
CELL = 0.0417


def cell(lat, lon):
    return (int(lat / CELL), int(lon / CELL))


clean = pd.read_csv(ROOT / "audit" / "output_audit" / "occurrence_clean.csv", encoding="utf-8-sig")
clean["cell"] = [cell(a, b) for a, b in zip(clean.latitude, clean.longitude)]
post_cells = set(clean.loc[clean.year > 2024, "cell"])
model = pd.read_csv(ROOT / "revision_outputs" / "occurrence_model_records.csv")
model_cells = set(cell(a, b) for a, b in zip(model.latitude, model.longitude))
assert len(model_cells) == len(model), "modelled records should occupy distinct audit cells"

x = pd.ExcelFile(ROOT / "audit" / "blackchin_audit_verified.xlsx")
lv, st = x.parse("Live records"), x.parse("Study 36")
inat = lv[(lv.source == "iNaturalist") & (~lv.observation_id.isin(st.observation_id))].copy()
inat["cell"] = [cell(a, b) for a, b in zip(inat.latitude, inat.longitude)]
inat["obs_date"] = pd.to_datetime(inat.date, utc=True)


def category(r):
    if r.inat_quality_current != "research":
        return "not research grade"
    if bool(r.captive):
        return "captive"
    if pd.notna(r.uncertainty_m) and r.uncertainty_m > 5000:
        return "uncertainty > 5 km"
    if r.cell in model_cells:
        return "cell already modelled"
    if r.cell in post_cells:
        return "omitted by processing order"
    return "not in original download"


inat["category"] = inat.apply(category, axis=1)
summary = inat.groupby("category").agg(observations=("observation_id", "size"), cells=("cell", "nunique"))
print(summary.to_string())
summary.to_csv(TAB / "TableS10_omitted_record_categories.csv")


# was the record present in the original (13 June 2026) GBIF download of Thailand records?
raw = pd.read_csv(ROOT / "audit" / "original_download" / "occurrences_thailand.csv")
raw_ids = set(pd.to_numeric(raw.gbif_id, errors="coerce").dropna().astype("int64"))
raw_xy = set(zip(raw.latitude.round(4), raw.longitude.round(4)))
gb = lv[lv.source == "GBIF"][["gbif_id", "observation_id"]].rename(columns={"gbif_id": "gbif_link"})
inat = inat.merge(gb, on="observation_id", how="left")
by_xy = pd.Series([(round(a, 4), round(b, 4)) in raw_xy for a, b in zip(inat.latitude, inat.longitude)],
                  index=inat.index)
inat["in_original_download"] = pd.to_numeric(inat.gbif_link, errors="coerce").isin(raw_ids) | by_xy
om = inat[inat.category == "omitted by processing order"]
print("omitted observations:", len(om), "| present in original download:", int(om.in_original_download.sum()),
      "| cells with a record in the original download:", om[om.in_original_download].cell.nunique(), "of",
      om.cell.nunique())


def pick(df):
    return (df.sort_values("obs_date").groupby("cell", as_index=False).first())


order = pick(om[om.in_original_download])
newdl = pick(inat[inat.category == "not in original download"])
print("added cells: processing order =", len(order), "| not in original download =", len(newdl))

base = model[["latitude", "longitude", "year"]].assign(added="original", source="archived 38")


def as_occ(df, label):
    return pd.DataFrame({"latitude": df.latitude, "longitude": df.longitude,
                         "year": df.obs_date.dt.year, "added": label,
                         "source": "iNaturalist " + df.observation_id.astype("int64").astype(str)})


plus9 = pd.concat([base, as_occ(order, "processing-order omission")], ignore_index=True)
plus16 = pd.concat([plus9, as_occ(newdl, "not in original download")], ignore_index=True)
plus9.to_csv(OUT / "occ_plus_order.csv", index=False)
added = pd.concat([as_occ(order, "processing-order omission"), as_occ(newdl, "not in original download")])
added = added.merge(inat[["observation_id", "province", "date"]].assign(
    source="iNaturalist " + inat.observation_id.astype("int64").astype(str)), on="source", how="left")
added.to_csv(TAB / "TableS10_added_records.csv", index=False)
print("sizes:", len(base), len(plus9), len(plus16))
print(added[["added", "source", "province", "year"]].to_string(index=False))
