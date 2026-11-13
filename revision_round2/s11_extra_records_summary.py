"""Compare the selected configuration fitted to the 38 archived records with 52 records
(38 + 14 cells omitted by the de-duplication/year-filter order). Table S10."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
TAB = HERE / "tables"
spec = importlib.util.spec_from_file_location("st21", ROOT / "blackchin_tilapia_analysis" / "scripts" / "21_pooled_metrics_stats_v2.py")
st21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st21)
LABEL = pd.read_csv(TAB / "TableS4_rank_stability.csv").set_index("feature")["label"].to_dict()

runs = {"38 archived records": HERE / "uz_sensitivity" / "base",
        "52 records (38 + 14 omitted cells)": HERE / "extra_records" / "plus14"}
rows, imps = [], {}
for name, d in runs.items():
    fm = pd.read_csv(d / "fold_metrics.csv")
    oof = pd.read_csv(d / "oof_predictions.csv")
    imp = pd.read_csv(d / "importance_per_fold.csv").groupby("feature")["perm_auc_decrease"].mean()
    imps[name] = imp
    y, p = oof.label.values.astype(int), oof.suit_oof.values
    rf = (oof.rep.astype(str) + "_" + oof.fold.astype(str)).values
    lo, hi = st21.bootstrap_ci_clustered(y, p, oof.occ_id.values.astype(int), rf, metric="auc_roc", seed=42)
    top5 = list(imp.sort_values(ascending=False).index[:5])
    rows.append({"dataset": name, "presences": int(oof[oof.label == 1].occ_id.nunique()),
                 "test_presences_per_partition": f"{fm.n_pres.min()}-{fm.n_pres.max()}",
                 "presences_dropped": int(fm.n_pres_dropped.sum()),
                 "fold_auc_mean": fm.auc_roc.mean(), "fold_auc_sd": fm.auc_roc.std(),
                 "fold_auc_range": f"{fm.auc_roc.min():.3f}-{fm.auc_roc.max():.3f}",
                 "fold_pr_auc_mean": fm.pr_auc.mean(), "fold_brier_mean": fm.brier.mean(),
                 "fold_tss_mean": fm.tss.mean(),
                 "pooled_auc": roc_auc_score(y, p), "pooled_auc_ci": f"{lo:.3f}-{hi:.3f}",
                 "pooled_pr_auc": average_precision_score(y, p), "pr_baseline": y.mean(),
                 "pooled_brier": brier_score_loss(y, p),
                 "top5": "; ".join(LABEL[f] for f in top5)})
s10 = pd.DataFrame(rows)
a, b = imps.values()
s10["top5_overlap_with_38"] = [5, len(set(a.sort_values(ascending=False).index[:5]) & set(b.sort_values(ascending=False).index[:5]))]
s10["spearman_importance_vs_38"] = [1.0, spearmanr(a, b.reindex(a.index)).statistic]
s10.to_csv(TAB / "TableS10_extra_records_sensitivity.csv", index=False)
pd.DataFrame(imps).rename(index=LABEL).round(4).sort_values(list(imps)[0], ascending=False).to_csv(
    TAB / "TableS10_importance_38_vs_52.csv")
pd.set_option("display.width", 250)
print(s10.T.to_string())
print(pd.DataFrame(imps).rename(index=LABEL).round(3).sort_values(list(imps)[0], ascending=False).to_string())
