"""
Build final Table S3 (record-level provenance) and the photo list from the authors'
re-query audit (ecologies/blackchin_audit_verified.xlsx, retrieved 2026-11-13).
The audit did NOT visually assess photographs; that column stays for the authors.
"""
import re
from pathlib import Path

import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAB = HERE / "tables"
AUDIT = ROOT / "audit" / "blackchin_audit_verified.xlsx"


def sheet_with_links(name):
    wb = openpyxl.load_workbook(AUDIT)
    ws = wb[name]
    rows = list(ws.iter_rows(values_only=True))
    hdr = list(rows[0])
    out = []
    for r in rows[1:]:
        rec = {}
        for h, v in zip(hdr, r):
            if isinstance(v, str) and v.startswith("=HYPERLINK("):
                m = re.match(r'=HYPERLINK\("([^"]+)"', v)
                v = m.group(1) if m else v
            rec[h] = v
        out.append(rec)
    return pd.DataFrame(out)


st = sheet_with_links("Study 36")
ph = sheet_with_links("Photos")
occ = pd.read_csv(ROOT / "revision_outputs" / "occurrence_model_records.csv")

photos = ph[ph.observation_id.isin(st.observation_id)].drop_duplicates("photo_id")
photos = photos[["observation_id", "photo_id", "photo_url", "photo_page", "image_license", "attribution"]]
assert len(photos) == int(st.photo_count.sum()), (len(photos), st.photo_count.sum())
photos.to_csv(TAB / "TableS3_photographs.csv", index=False, encoding="utf-8-sig")

cs = pd.DataFrame({
    "occ_id": st.study_occ_id.astype(int),
    "source_group": "Citizen science",
    "source": st.study_source,
    "record_id": st.study_source_id.astype("int64").astype(str),
    "inaturalist_observation_id": st.observation_id.astype("int64").astype(str),
    "observation_url": "https://www.inaturalist.org/observations/" + st.observation_id.astype("int64").astype(str),
    "gbif_url": [u if s == "GBIF" else "" for u, s in zip(st.source_url, st.study_source)],
    "date": st.date.astype(str).str.slice(0, 10),
    "province": st.province,
    "latitude": st.latitude, "longitude": st.longitude,
    "coordinate_uncertainty_m": st.uncertainty_m,
    "quality_grade_2026_11_13": st.inat_quality_current,
    "captive_flag": st.captive,
    "photo_count": st.photo_count.astype(int),
    "first_photo_url": st.photo_url,
    "photo_licence_first": st.image_license,
    "photo_attribution": st.image_attribution,
    "archive_match": ["source ID, date and coordinates match (delta %s m)" % d for d in st.coordinate_delta_m],
    "visual_check_by_authors": "",
})
lit = occ[occ.source_group != "Citizen science"]
lit_rows = pd.DataFrame({
    "occ_id": lit.occ_id.astype(int), "source_group": "Agency/media report", "source": "Literature",
    "record_id": "agency/media capture report", "inaturalist_observation_id": "", "observation_url": "",
    "gbif_url": "", "date": lit.year.astype(int).astype(str), "province": lit.state_province,
    "latitude": lit.latitude, "longitude": lit.longitude, "coordinate_uncertainty_m": lit.coord_uncertainty_m,
    "quality_grade_2026_11_13": "not applicable", "captive_flag": "", "photo_count": "",
    "first_photo_url": "", "photo_licence_first": "", "photo_attribution": "",
    "archive_match": "locality: " + lit.locality.astype(str), "visual_check_by_authors": "not applicable",
})
s3 = pd.concat([cs, lit_rows], ignore_index=True).sort_values("occ_id")
s3.to_csv(TAB / "TableS3_record_provenance.csv", index=False, encoding="utf-8-sig")
print(s3.shape, "photos:", len(photos))
print("photo_count range", cs.photo_count.min(), cs.photo_count.max(), "total", cs.photo_count.sum())
print("all research grade:", (cs.quality_grade_2026_11_13 == "research").all(), "| captive:", cs.captive_flag.any())
print(s3.head(3).T.to_string())
