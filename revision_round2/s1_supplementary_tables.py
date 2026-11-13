"""
Round-2 revision: supplementary tables computed from the existing outputs
(no model refitting).

S4  Rank stability of held-out individual permutation importance (15 outer RF models)
S5  Selection-independent check: leave-one-repetition-out (LORO) configuration selection
S6  Pseudo-absence strategy comparison: AUC, PR-AUC, Brier, TSS
     - matched-pair permutation + Holm (same function as 21_pooled_metrics_stats_v2.py)
     - exact sign-flip over the 15 outer partitions (dependence-robust)
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "blackchin_tilapia_analysis" / "output"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "st21", ROOT / "blackchin_tilapia_analysis" / "scripts" / "21_pooled_metrics_stats_v2.py")
st21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st21)

PA_LABEL = {"random": "Random", "spatial_constrained": "Spatially constrained",
            "two_step": "Two-step", "uncertainty_zone": "Uncertainty-zone"}
ALGO_LABEL = {"RF": "RF", "XGB": "XGB", "MXN": "EN-Hinge"}

# ───────────────────────────── S4 rank stability ─────────────────────────────
fi = pd.read_csv(OUT / "factor_explainability" / "feature_importance_per_fold.csv")
summ = pd.read_csv(OUT / "factor_explainability" / "feature_importance_summary.csv")
fi["rank"] = fi.groupby(["rep", "fold"])["permutation_auc_decrease"].rank(
    ascending=False, method="min")
g = fi.groupby(["feature", "label", "group"])
s4 = pd.DataFrame({
    "selection_frequency": g["selected"].mean(),
    "times_rank1": g["rank"].apply(lambda r: int((r == 1).sum())),
    "times_top5": g["rank"].apply(lambda r: int((r <= 5).sum())),
    "median_rank": g["rank"].median(),
}).reset_index()
s4 = s4.merge(summ[["feature", "permutation_auc_decrease_mean", "permutation_ci_lo",
                    "permutation_ci_hi"]], on="feature", how="left")
s4 = s4.sort_values(["times_top5", "permutation_auc_decrease_mean"], ascending=False)
s4.to_csv(TAB / "TableS4_rank_stability.csv", index=False)
print("S4\n", s4[["label", "selection_frequency", "times_rank1", "times_top5",
                  "median_rank", "permutation_auc_decrease_mean"]].round(3).to_string(index=False))
print("distinct rank-1 predictors:", int((s4.times_rank1 > 0).sum()))

# ───────────────────────────── S5 LORO selection ─────────────────────────────
fm = pd.read_csv(OUT / "experiment_fold_metrics.csv")
pm = pd.read_csv(OUT / "pooled_oof_metrics.csv")
oof = pd.read_csv(OUT / "experiment_oof_predictions.csv")
for d in (fm, pm, oof):
    d["cfg"] = d["algo"].map(ALGO_LABEL) + "/" + d["feat_set"] + "/" + d["pa_method"].map(PA_LABEL)


def pooled_auc(df):
    return roc_auc_score(df["label"].values, df["suit_oof"].values)


rows = []
reps = sorted(oof["rep"].unique())
for r in reps:
    train = oof[oof["rep"] != r]
    test = oof[oof["rep"] == r]
    # criterion used in the manuscript: pooled out-of-fold AUC
    crit = train.groupby("cfg").apply(pooled_auc, include_groups=False)
    best = crit.idxmax()
    held = pooled_auc(test[test["cfg"] == best])
    # alternative criterion: mean fold AUC
    crit2 = fm[fm["rep"] != r].groupby("cfg")["auc_roc"].mean()
    best2 = crit2.idxmax()
    held2 = fm[(fm["rep"] == r) & (fm["cfg"] == best2)]["auc_roc"].mean()
    rows.append({"held_out_repetition": r,
                 "selected_by_pooled_auc": best, "selection_pooled_auc": crit.max(),
                 "held_out_pooled_auc": held,
                 "selected_by_mean_fold_auc": best2, "selection_mean_fold_auc": crit2.max(),
                 "held_out_mean_fold_auc": held2})
s5 = pd.DataFrame(rows)
s5.to_csv(TAB / "TableS5_loro_selection.csv", index=False)
q = pm["auc_roc"].quantile([0.25, 0.5, 0.75])
top6 = pm.sort_values("auc_roc", ascending=False).head(6)
s5_summary = pd.DataFrame([
    {"quantity": "Selected configuration pooled AUC (post-selection)",
     "value": f"{pm.auc_roc.max():.3f}"},
    {"quantity": "Median pooled AUC across 48 configurations (IQR)",
     "value": f"{q[0.5]:.3f} ({q[0.25]:.3f}-{q[0.75]:.3f})"},
    {"quantity": "Range of the six highest pooled AUCs",
     "value": f"{top6.auc_roc.min():.3f}-{top6.auc_roc.max():.3f}"},
    {"quantity": "LORO held-out pooled AUC, selection by pooled AUC: mean (range)",
     "value": f"{s5.held_out_pooled_auc.mean():.3f} ({s5.held_out_pooled_auc.min():.3f}-{s5.held_out_pooled_auc.max():.3f})"},
    {"quantity": "LORO held-out mean fold AUC, selection by mean fold AUC: mean (range of repetition means)",
     "value": f"{s5.held_out_mean_fold_auc.mean():.3f} ({s5.held_out_mean_fold_auc.min():.3f}-{s5.held_out_mean_fold_auc.max():.3f})"},
])
s5_summary.to_csv(TAB / "TableS5_summary.csv", index=False)
print("\nS5\n", s5.round(3).to_string(index=False))
print(s5_summary.to_string(index=False))

# ───────────────────────────── S6 PA comparison ─────────────────────────────
fm["part"] = fm["rep"].astype(str) + "-" + fm["fold"].astype(str)
metrics = ["auc_roc", "pr_auc", "brier", "tss"]

means = fm.groupby("pa_method")[metrics].agg(["mean", "std"]).round(3)
means.to_csv(TAB / "TableS6_means_by_strategy.csv")
print("\nS6a\n", means)

# matched-pair permutation + Holm (original function, finer resolution)
pp = pd.concat([st21._paired_permutation(fm, "pa_method", m, n_perm=9999, seed=42)
                for m in metrics], ignore_index=True)
pp.to_csv(TAB / "TableS7_matched_pair_permutation.csv", index=False)
print("\nS6b\n", pp[["response", "level_a", "level_b", "diff_a_b", "n_pairs",
                     "p_paired_perm", "p_holm", "sig"]].to_string(index=False))


def exact_signflip(d):
    d = np.asarray(d)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    return float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (len(p) - rank)))
        adj[idx] = running
    return adj


rows = []
for m in metrics:
    w = fm.pivot_table(index=["part", "algo", "feat_set"], columns="pa_method", values=m)
    sub = []
    for other in ["random", "spatial_constrained", "two_step"]:
        diff = (w["uncertainty_zone"] - w[other]).groupby(level="part").mean()
        better = (diff < 0).sum() if m == "brier" else (diff > 0).sum()
        sub.append({"response": m, "comparison": f"Uncertainty-zone - {PA_LABEL[other]}",
                    "mean_difference": diff.mean(), "partitions_favouring_uz": int(better),
                    "n_partitions": len(diff), "p_exact_signflip": exact_signflip(diff.values)})
    adj = holm([s["p_exact_signflip"] for s in sub])
    for s, a in zip(sub, adj):
        s["p_holm"] = a
    rows.extend(sub)
s6c = pd.DataFrame(rows)
s6c.to_csv(TAB / "TableS7_partition_signflip.csv", index=False)
print("\nS6c\n", s6c.round(4).to_string(index=False))

g = fm.groupby(["algo", "feat_set", "pa_method"])[metrics].mean()
best_counts = {}
for m in metrics:
    u = g[m].unstack()
    best = u.idxmin(axis=1) if m == "brier" else u.idxmax(axis=1)
    best_counts[m] = best.value_counts().reindex(list(PA_LABEL), fill_value=0)
s6d = pd.DataFrame(best_counts)
s6d.index = [PA_LABEL[i] for i in s6d.index]
s6d.to_csv(TAB / "TableS6_best_counts.csv")
print("\nS6d (number of 12 algorithm x predictor-set combinations in which each strategy is best)\n", s6d)

s6e = pd.concat({m: fm.pivot_table(index="pa_ratio", columns="pa_method", values=m,
                                   aggfunc="mean") for m in ["pr_auc", "brier"]}, axis=1).round(3)
s6e["n_uz"] = fm[fm.pa_method == "uncertainty_zone"].groupby("pa_ratio").size()
s6e.to_csv(TAB / "TableS6_by_pa_ratio.csv")
print("\nS6e\n", s6e)

sel = fm[(fm.algo == "RF") & (fm.feat_set == "M_SAP")]
s6f = sel.groupby("pa_method")[metrics].agg(["mean", "std"]).round(3)
s6f.to_csv(TAB / "TableS6_rf_msap_by_strategy.csv")
print("\nS6f RF/M_SAP\n", s6f)
