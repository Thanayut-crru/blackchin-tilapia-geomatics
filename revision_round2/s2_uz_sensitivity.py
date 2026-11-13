"""
Round-2 revision: sensitivity of the uncertainty-zone (UZ) pseudo-absence rule.

Re-runs the selected configuration (RF / M_SAP / uncertainty_zone) through the
identical repeated nested spatial CV of 20_full_experiment_v2.py, changing only
the UZ candidate-score interval [lo, hi] and the presence-exclusion buffer.
All other code paths (folds, inner CV, VIF, HP search, evaluation backgrounds,
seeds) are imported unchanged from the original script.

Additionally records, for every UZ call, the candidate-set size and whether the
sparse-zone or invalid-pilot fallback was triggered, and computes held-out
permutation importance (30 permutations, same seeding as 27_factor_explainability.py).

Usage:
  python revision_round2/s2_uz_sensitivity.py --lo 0.2 --hi 0.6 --buffer 10 --tag base
Requires the WorldClim rasters reconstructed with prepare_worldclim.py.
"""
import argparse
import functools
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]          # release root
sys.path.insert(0, str(ROOT))
EXP_PATH = ROOT / "blackchin_tilapia_analysis" / "scripts" / "20_full_experiment_v2.py"

spec = importlib.util.spec_from_file_location("exp20", EXP_PATH)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

N_PERM = 30
STATE = {"phase": "outer", "model": None, "log": [], "rep": None, "fold": None}


def uz_variant(cell_lons, cell_lats, pres_lons, pres_lats, cell_X, n,
               buffer_km=10.0, fuzzy_lo=0.2, fuzzy_hi=0.6, seed=0):
    """Line-by-line copy of exp.sample_uncertainty_zone_pa with parameters and logging."""
    np.random.seed(seed)
    d = exp.hav_dist_to_nearest(cell_lons, cell_lats, pres_lons, pres_lats)
    outside = np.where(d >= buffer_km)[0]
    empty_buffer = len(outside) == 0
    if empty_buffer:
        outside = np.arange(len(cell_lons))
    pres_X = exp.sample_at_coords_direct(pres_lons, pres_lats, cell_lons, cell_lats, cell_X)
    pres_ok = np.isfinite(pres_X).all(axis=1)
    bg_idx = np.random.choice(outside, size=min(200, len(outside)), replace=False)
    bg_X = cell_X[bg_idx]
    bg_ok = np.isfinite(bg_X).all(axis=1)
    X_tr = np.vstack([pres_X[pres_ok], bg_X[bg_ok]])
    y_tr = np.concatenate([np.ones(pres_ok.sum()), np.zeros(bg_ok.sum())])
    entry = {"phase": STATE["phase"], "rep": STATE["rep"], "fold": STATE["fold"],
             "n_requested": int(n), "n_buffered_cells": int(len(outside)),
             "empty_buffer_fallback": bool(empty_buffer)}
    if y_tr.sum() == 0 or len(X_tr) < 5:
        entry.update({"invalid_pilot_fallback": True, "n_candidates": np.nan,
                      "n_incomplete_scored": np.nan, "sparse_zone_fallback": False})
        STATE["log"].append(entry)
        return exp.sample_random_pa(cell_lons, cell_lats, pres_lons, pres_lats, n,
                                    buffer_km=buffer_km, seed=seed)
    rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    out_X = cell_X[outside]
    ok = np.isfinite(out_X).all(axis=1)
    sp = np.full(len(outside), 0.4)
    if ok.sum() > 0:
        sp[ok] = rf.predict_proba(out_X[ok])[:, 1]
    fuzzy = outside[(sp >= fuzzy_lo) & (sp <= fuzzy_hi)]
    n_cand = len(fuzzy)
    sparse = n_cand < n // 2
    if sparse:
        fuzzy = outside
    chosen = np.random.choice(fuzzy, size=min(n, len(fuzzy)), replace=False)
    entry.update({"invalid_pilot_fallback": False, "n_candidates": int(n_cand),
                  "n_incomplete_scored": int((~ok).sum()),
                  "sparse_zone_fallback": bool(sparse)})
    STATE["log"].append(entry)
    return cell_lons[chosen], cell_lats[chosen]


_orig_inner_cv = exp.inner_cv
_orig_fit_rf = exp.fit_rf


def inner_cv_wrap(*args, **kwargs):
    STATE["phase"] = "inner"
    try:
        return _orig_inner_cv(*args, **kwargs)
    finally:
        STATE["phase"] = "outer"


def fit_rf_wrap(X_tr, y_tr, hp, seed):
    model = _orig_fit_rf(X_tr, y_tr, hp, seed)
    if STATE["phase"] == "outer":
        STATE["model"] = model
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--buffer", type=float, default=10.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--occ", default=None, help="alternative occurrence CSV (latitude, longitude, year)")
    ap.add_argument("--outroot", default="uz_sensitivity")
    args = ap.parse_args()

    exp.sample_uncertainty_zone_pa = functools.partial(
        uz_variant, buffer_km=args.buffer, fuzzy_lo=args.lo, fuzzy_hi=args.hi)
    exp.inner_cv = inner_cv_wrap
    exp.fit_rf = fit_rf_wrap

    out_dir = Path(__file__).resolve().parent / args.outroot / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "settings.json").write_text(json.dumps(vars(args), indent=2))

    occ = pd.read_csv(args.occ or exp.OCC_FINAL, encoding="utf-8-sig")
    occ = occ[occ["year"] <= 2024].copy()
    occ = occ.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    occ["occ_id"] = occ.index
    print(f"[{args.tag}] occurrences = {len(occ)}  interval=[{args.lo},{args.hi}] buffer={args.buffer} km", flush=True)

    arrays, transform, _ = exp.load_all_rasters()
    common_lo, common_la = exp.get_common_valid_cells(arrays, transform)
    feats = [f for f in exp.MODEL_VARIANTS["M_SAP"] if f in arrays]
    mv_arrays = {k: arrays[k] for k in feats}

    fold_rows, oof_rows, imp_rows = [], [], []
    t_start = time.time()
    for rep in range(exp.N_OUTER_REPS):
        rep_seed = exp.RANDOM_STATE + rep * 100
        occ_blocks, actual_k, centroids = exp.assign_spatial_folds_stable(
            occ, k=exp.N_OUTER_FOLDS, base_seed=rep_seed)
        cell_block = exp.assign_cells_to_blocks(common_lo, common_la, centroids)
        for fold in range(1, actual_k + 1):
            STATE["rep"], STATE["fold"] = rep + 1, fold
            te_mask = occ_blocks["spatial_fold"] == fold
            tr = occ_blocks[~te_mask]
            te = occ_blocks[te_mask]
            tr_cell = cell_block != (fold - 1)
            te_cell = cell_block == (fold - 1)
            test_lo, test_la = common_lo[te_cell], common_la[te_cell]
            n_eval = min(exp.N_EVAL_BG, len(test_lo))
            rng_ev = np.random.default_rng(rep_seed + fold * 97 + 1000)
            ev_idx = rng_ev.choice(len(test_lo), n_eval, replace=False)
            eval_lo, eval_la = test_lo[ev_idx], test_la[ev_idx]

            t0 = time.time()
            STATE["model"] = None
            met, oof = exp.run_outer_fold_v2(
                fold_id=fold, rep_seed=rep_seed,
                pres_tr_lo=tr["longitude"].values, pres_tr_la=tr["latitude"].values,
                pres_te_lo=te["longitude"].values, pres_te_la=te["latitude"].values,
                pres_te_occ_ids=te["occ_id"].values,
                train_cell_lo=common_lo[tr_cell], train_cell_la=common_la[tr_cell],
                test_cell_lo=test_lo, test_cell_la=test_la,
                eval_bg_lo=eval_lo, eval_bg_la=eval_la,
                feature_names=feats, arrays=mv_arrays, transform=transform,
                algo="RF", pa_method="uncertainty_zone")
            if met is None:
                print(f"  rep={rep+1} fold={fold} SKIP", flush=True)
                continue
            fold_rows.append({"tag": args.tag, "rep": rep + 1, "fold": fold, **met})
            for r in oof:
                r.update({"tag": args.tag, "rep": rep + 1, "fold": fold})
            oof_rows.extend(oof)

            # Held-out permutation importance (same seeding as 27_factor_explainability.py)
            model = STATE["model"]
            sel = met["selected"].split(",")
            sel_idx = [feats.index(s) for s in sel]
            x_p = exp.sample_at_coords(mv_arrays, feats, te["longitude"].values,
                                       te["latitude"].values, transform)[:, sel_idx]
            x_b = exp.sample_at_coords(mv_arrays, feats, eval_lo, eval_la, transform)[:, sel_idx]
            x_test = np.vstack([x_p, x_b])
            y_test = np.concatenate([np.ones(len(x_p)), np.zeros(len(x_b))])
            valid = np.isfinite(x_test).all(axis=1)
            x_test, y_test = x_test[valid], y_test[valid]
            base_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
            imp = {f: 0.0 for f in feats}
            for li, name in enumerate(sel):
                dec = []
                for perm in range(N_PERM):
                    xp = x_test.copy()
                    order = np.random.default_rng(
                        rep_seed + fold * 1000 + li * 100 + perm).permutation(len(xp))
                    xp[:, li] = xp[order, li]
                    dec.append(base_auc - roc_auc_score(y_test, model.predict_proba(xp)[:, 1]))
                imp[name] = float(np.mean(dec))
            for f in feats:
                imp_rows.append({"tag": args.tag, "rep": rep + 1, "fold": fold, "feature": f,
                                 "selected": f in sel, "perm_auc_decrease": imp[f],
                                 "baseline_auc": base_auc})
            print(f"  rep={rep+1} fold={fold} AUC={met['auc_roc']:.3f} PR={met['pr_auc']:.3f} "
                  f"Brier={met['brier']:.3f} PA={met['pa_ratio']} {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(out_dir / "importance_per_fold.csv", index=False)
    pd.DataFrame(STATE["log"]).to_csv(out_dir / "uz_call_log.csv", index=False)
    print(f"[{args.tag}] done in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
