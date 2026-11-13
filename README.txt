# Associations of Environmental, Waterway-Context, and Human-Activity Predictors with Blackchin Tilapia Records in Thailand

Version **1.2.0** is the Zenodo-ready computational companion to the
associated unpublished article. The manuscript itself is not included. This
archive contains the exact analysis code, the 38-row modelling occurrence table,
redistributable/share-alike model-ready inputs, archived out-of-fold predictions,
revision diagnostics, figures, and machine-readable checks.

The national surface is a hypothesis-generating screening product. It is not a
validated occurrence probability, occupancy estimate, invasion forecast, or
operational risk map.

## Fast verification

Python 3.11 or later is recommended.

```powershell
python verify_release.py
python pre_upload_check.py
```

The first command verifies the manuscript-facing numerical claims and all file
checksums. The second intentionally fails until the repository URL, data DOI,
and reserved Zenodo DOI are completed before publication.

## Reproduction levels

1. **Archived-result verification (minutes, no third-party download):** run
   `python verify_release.py`.
2. **Environmental input reconstruction:** run
   `python prepare_worldclim.py --download`, then
   `python verify_inputs.py`. WorldClim files are fetched directly because the
   provider does not permit redistribution.
3. **Full factorial experiment (long-running):** create the pinned environment,
   prepare WorldClim, then run
   `python run_reproduction.py --stage full`.
4. **Revision diagnostics:** see `REPRODUCIBILITY.txt`; national AOA and province
   regeneration additionally requires a user-obtained GADM boundary.

## Main reported configuration

- Algorithms: RF, XGB, and EN-Hinge (elastic-net hinge logistic regression; labelled `MXN` in code
  and output files)
- Predictor sets: M_S, M_SA, M_SP, and M_SAP
- Pseudo-absence methods: random, spatial-constrained, two-step, uncertainty-zone
- Outer design: 5 repetitions x 3 spatial folds
- Uncertainty-zone interval: 0.20 through 0.60, inclusive
- Selected result: RF/M_SAP/uncertainty-zone, conditional pooled AUC 0.852

Read `REPRODUCIBILITY.txt`, `THIRD_PARTY_DATA.txt`, and `PRE_UPLOAD_CHECKLIST.txt`
before depositing the ZIP. Third-party data remain governed by their source
terms; the repository-level MIT licence applies to original code only.

## New in version 1.2.0: peer-review analyses (`revision_round2/`)

| Table in the revised article | Output file(s) in `revision_round2/tables/` | Script |
|---|---|---|
| S3 record provenance and photographs | `TableS3_record_provenance.csv`, `TableS3_photographs.csv` | `s9_photo_provenance.py` |
| S4 rank stability of importance | `TableS4_rank_stability.csv` | `s1_supplementary_tables.py` |
| S5 leave-one-repetition-out selection | `TableS5_loro_selection.csv`, `TableS5_summary.csv` | `s1_supplementary_tables.py` |
| S6 pseudo-absence strategy means | `TableS6_*.csv` | `s1_supplementary_tables.py` |
| S7 paired tests of strategies | `TableS7_matched_pair_permutation.csv`, `TableS7_partition_signflip.csv` | `s1_supplementary_tables.py` |
| S8 uncertainty-zone sensitivity | `TableS8_*.csv`; runs in `uz_sensitivity/` | `s2_uz_sensitivity.py`, `s3_uz_sensitivity_summary.py` |
| S9 residual correlogram | `TableS9_moran_correlogram.csv` | `s4_moran_correlogram.py` |
| S10 omitted-cell sensitivity | `TableS10_*.csv`; run in `extra_records/plus14/` | `s10_extra_records.py`, `s2_uz_sensitivity.py`, `s11_extra_records_summary.py` |

Scripts that only read archived outputs run in seconds to minutes from the release root, for example
`python revision_round2/s1_supplementary_tables.py`. Re-running the model fits (`s2_uz_sensitivity.py`,
about 2 minutes per setting) requires the WorldClim rasters reconstructed with `prepare_worldclim.py`;
the archived runs are included so that the summaries can be checked without re-fitting.
