"""
Round-2 revision: summarise the uncertainty-zone sensitivity runs (Table S7).
Reads uz_sensitivity/<tag>/ outputs from s2_uz_sensitivity.py and the archived
v1.1.0 fold metrics (read-only) for the reproduction check and for the other
pseudo-absence strategies of RF/M_SAP.
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "blackchin_tilapia_analysis" / "output"
SENS = HERE / "uz_sensitivity"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "st21", ROOT / "blackchin_tilapia_analysis" / "scripts" / "21_pooled_metrics_stats_v2.py")
st21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st21)

LABEL = {"bio01": "Annual mean temperature", "bio04": "Temperature seasonality",
         "bio07": "Annual temperature range", "bio10": "Mean temperature, warmest quarter",
         "bio11": "Mean temperature, coldest quarter", "bio12": "Annual precipitation",
         "bio15": "Precipitation seasonality", "bio16": "Precipitation, wettest quarter",
         "bio17": "Precipitation, driest quarter", "elevation": "Elevation",
         "A1_dist_waterway": "Distance to waterway", "A2_waterway_order": "Waterway order",
         "P1_pop_density": "Population density", "P2_dist_road": "Distance to road",
         "P3_road_density": "Road density", "P4_dist_urban": "Distance to urban area"}

tags = {"base": (0.2, 0.6, 10)}
for d in sorted(SENS.iterdir()):
    if d.is_dir() and d.name != "base":
        lo_hi, b = d.name.split("_b")
        lo, hi = lo_hi[1:].split("-")
        tags[d.name] = (int(lo) / 100, int(hi) / 100, float(b))

# reproduction check against the archived v1.1.0 results
fm_arch = pd.read_csv(OUT / "experiment_fold_metrics.csv")
arch = fm_arch[(fm_arch.algo == "RF") & (fm_arch.feat_set == "M_SAP")
               & (fm_arch.pa_method == "uncertainty_zone")].set_index(["rep", "fold"])
base = pd.read_csv(SENS / "base" / "fold_metrics.csv").set_index(["rep", "fold"])
chk = pd.DataFrame({m: (base[m] - arch[m]).abs() for m in ["auc_roc", "pr_auc", "brier", "tss"]})
print("Reproduction check (max |difference| vs archived RF/M_SAP/UZ):")
print(chk.max().to_string())
chk.max().to_frame("max_abs_difference").to_csv(TAB / "TableS8_reproduction_check.csv")


def exact_signflip(d):
    d = np.asarray(d)
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    return float((np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12).mean())


base_imp = pd.read_csv(SENS / "base" / "importance_per_fold.csv").groupby("feature")["perm_auc_decrease"].mean()
base_top5 = list(base_imp.sort_values(ascending=False).index[:5])

rows, imp_tab = [], {}
for tag, (lo, hi, buf) in tags.items():
    fdir = SENS / tag
    fm = pd.read_csv(fdir / "fold_metrics.csv")
    oof = pd.read_csv(fdir / "oof_predictions.csv")
    imp = pd.read_csv(fdir / "importance_per_fold.csv")
    log = pd.read_csv(fdir / "uz_call_log.csv")

    y, p = oof["label"].values.astype(int), oof["suit_oof"].values.astype(float)
    occ = oof["occ_id"].values.astype(int)
    rf_ids = (oof["rep"].astype(str) + "_" + oof["fold"].astype(str)).values
    lo_ci, hi_ci = st21.bootstrap_ci_clustered(y, p, occ, rf_ids, metric="auc_roc", seed=42)

    m_imp = imp.groupby("feature")["perm_auc_decrease"].mean()
    top5 = list(m_imp.sort_values(ascending=False).index[:5])
    rho = spearmanr(m_imp.reindex(base_imp.index), base_imp).statistic
    imp_tab[tag] = m_imp

    # paired difference vs baseline over the 15 partitions
    b = base.reset_index()
    mm = fm.merge(b[["rep", "fold", "auc_roc", "pr_auc", "brier"]], on=["rep", "fold"], suffixes=("", "_base"))
    d_pr = (mm["pr_auc"] - mm["pr_auc_base"]).values

    outer = log[log.phase == "outer"]
    rows.append({
        "variant": tag, "interval": f"[{lo:.2f}, {hi:.2f}]", "buffer_km": buf,
        "fold_auc_mean": fm.auc_roc.mean(), "fold_auc_sd": fm.auc_roc.std(),
        "fold_pr_auc_mean": fm.pr_auc.mean(), "fold_brier_mean": fm.brier.mean(),
        "fold_tss_mean": fm.tss.mean(),
        "pooled_auc": roc_auc_score(y, p), "pooled_auc_ci_lo": lo_ci, "pooled_auc_ci_hi": hi_ci,
        "pooled_pr_auc": average_precision_score(y, p), "pooled_brier": brier_score_loss(y, p),
        "pr_auc_diff_vs_base": d_pr.mean(), "pr_auc_p_vs_base": exact_signflip(d_pr) if tag != "base" else np.nan,
        "top5_importance": "; ".join(LABEL[f] for f in top5),
        "top5_overlap_with_base": len(set(top5) & set(base_top5)),
        "spearman_importance_vs_base": rho,
        "pa_ratio_1to10_share": (fm.pa_ratio == "1:10").mean(),
        "uz_calls_total": len(log),
        "sparse_fallback_pct_all": 100 * log.sparse_zone_fallback.mean(),
        "sparse_fallback_outer": int(outer.sparse_zone_fallback.sum()),
        "invalid_pilot_fallback_all": int(log.invalid_pilot_fallback.sum()),
        "empty_buffer_fallback_all": int(log.empty_buffer_fallback.sum()),
        "median_candidates_to_requested": float(np.nanmedian(log.n_candidates / log.n_requested)),
        "incomplete_cells_scored": int(np.nansum(log.n_incomplete_scored)),
    })

s7 = pd.DataFrame(rows)
order = ["base"] + [t for t in s7.variant if t != "base"]
s7 = s7.set_index("variant").loc[order].reset_index()
s7.to_csv(TAB / "TableS8_uz_sensitivity.csv", index=False)
pd.DataFrame(imp_tab).rename(index=LABEL).round(4).to_csv(TAB / "TableS8_importance_by_setting.csv")

# other pseudo-absence strategies, RF/M_SAP, archived results (reference rows)
ref = fm_arch[(fm_arch.algo == "RF") & (fm_arch.feat_set == "M_SAP")].groupby("pa_method").agg(
    fold_auc_mean=("auc_roc", "mean"), fold_auc_sd=("auc_roc", "std"),
    fold_pr_auc_mean=("pr_auc", "mean"), fold_brier_mean=("brier", "mean"), fold_tss_mean=("tss", "mean"))
ref.to_csv(TAB / "TableS8_reference_other_strategies.csv")

pd.set_option("display.width", 250)
cols = ["variant", "interval", "buffer_km", "fold_auc_mean", "fold_auc_sd", "fold_pr_auc_mean",
        "fold_brier_mean", "pooled_auc", "pooled_auc_ci_lo", "pooled_auc_ci_hi", "pooled_pr_auc",
        "pooled_brier", "pr_auc_diff_vs_base", "pr_auc_p_vs_base"]
print(s7[cols].round(3).to_string(index=False))
print(s7[["variant", "top5_overlap_with_base", "spearman_importance_vs_base", "pa_ratio_1to10_share",
          "uz_calls_total", "sparse_fallback_pct_all", "sparse_fallback_outer", "invalid_pilot_fallback_all",
          "empty_buffer_fallback_all", "median_candidates_to_requested", "incomplete_cells_scored"]].round(3).to_string(index=False))
print("\nTop-5 by variant:")
for _, r in s7.iterrows():
    print(f"  {r.variant:12s} {r.top5_importance}")
print("\nReference (archived) RF/M_SAP other strategies:\n", ref.round(3))
