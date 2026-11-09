# Manuscript result provenance

| Manuscript item | Source data / analysis |
|---|---|
| Table 1, Panel A (six fixed-budget layouts) | `results_summaries/final_descriptive_by_attack.csv`; raw records: `results_raw/main_COCO500_results.csv`, `results_raw/main_DIV2K100_results.csv` |
| Table 1, Panel B (decoder and anchor-ratio ablations) | `results_summaries/vote_descriptive.csv`, `results_summaries/rho_descriptive.csv`; raw ratio records: `results_raw/rho_vote_ablation.csv` |
| Supporting Information Tables S1--S2 (per-attack paired tests) | `results_summaries/final_paired_by_attack.csv` |
| Supporting Information Table S3 (component removal) | `results_summaries/component_removal_attacks_500_condition_means.csv`; raw: `results_raw/component_removal_attacks_500.csv` |
| Supporting Information Table S4 (LaMa) | `results_summaries/lama_object_inpainting_382_summary.csv`; raw records and runner: `outputs/lama_reproducibility/full382_20261103/` (including `lama_object_inpainting_382.csv`, `clean_vs_attacked_382.csv`, `clean_matched_382.csv`, `final_statistics/paired_analysis_all_images.csv`, and analysis scripts) |
| Supporting Information Tables S5--S6 (grid and detector fallback) | `results_summaries/detection_overall_four_attack.csv`, `results_summaries/fallback_threshold_summary.csv`; raw sensitivity records in `results_raw/` |
| Supporting Information Table S7 (placement IoU) | `results_summaries/iou_sensitivity_manuscript_summary.csv`; raw: `results_raw/iou_sensitivity_results.csv` |
| Supporting Information Table S8 (runtime) | `results_summaries/runtime_cost_summary.json`; raw: `results_raw/runtime_cost_results.csv` |
| Supporting Information Table S9 (per-attack anchor ratio) | `results_summaries/rho_descriptive.csv` |
| Supporting Information Table S10 (clean-image quality) | `results_summaries/quality_summary.csv`; raw: `results_raw/quality_per_image.csv` |
| Placement fallback audit in Section S2 | `results_raw/placement_audit_500.csv` |
| Clean-image negative control in Section S11 | `results_summaries/clean_negative_control_summary.json`; raw: `results_raw/clean_negative_control_results.csv` |

The LaMa runner, full 382-image raw outputs, matched clean/attacked records, paired statistical outputs and environment export are included in this package. The LaMa checkpoint, COCO image/annotation files and WAM checkpoint remain upstream dependencies and must be obtained separately under their respective terms.
