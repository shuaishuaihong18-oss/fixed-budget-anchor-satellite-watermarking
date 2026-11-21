# Manuscript result provenance - Multimedia Systems full-length paper

This map follows the consolidated Regular Paper manuscript, which contains 18 numbered tables and 5 numbered figures in its main text. The original experimental measurements are unchanged. The additional attack-weight analysis is a descriptive reweighting of the existing per-attack means; it is not a new model run or an inferential test. The former supplementary tables and essential protocol definitions are integrated into the main manuscript. The archive does not include WAM/LaMa checkpoints or third-party source images.

| Main-text item | Reproducibility evidence |
|---|---|
| Table 1, allocation policies | `configs/experiment_config.yaml`; `watermark_anything/redundant_region.py`; area records in `results_summaries/final_descriptive_by_attack.csv`; placement audit in `results_raw/placement_audit_500.csv`. |
| Table 2, readout alternatives | `watermark_anything/redundant_region.py`; `results_summaries/vote_descriptive.csv`; `results_summaries/vote_paired_tests_holm.csv`; `results_raw/rho_vote_ablation.csv`. |
| Table 3, local-edit definitions | `scripts/run_fixed_budget_exp.py`; `configs/experiment_config.yaml`. The paper's Retain20 corresponds to implementation label `splicing20`, which uses same-source replacement, not cross-image semantic splicing. |
| Table 4, overall fixed-budget means | `results_summaries/final_descriptive_by_attack.csv`; per-image source records in `results_raw/main_COCO500_results.csv` and `results_raw/main_DIV2K100_results.csv`. Equal weighting applies to four edits only. |
| Table 5, aggregate paired statistical comparisons | `results_summaries/final_paired_overall.csv`; `scripts/analyze_final_stats.py`. |
| Tables 6-9, per-attack paired BitAcc and Recovery31 | `results_summaries/final_paired_by_attack.csv`; `scripts/analyze_final_stats.py`; matched per-image records above. |
| Table 10, anchor-ratio sensitivity | `results_summaries/rho_descriptive.csv`; `results_summaries/rho_paired_tests_holm.csv`; `results_raw/rho_vote_ablation.csv`. |
| Table 11, operational carrier removal | `results_summaries/component_removal_attacks_500_condition_means.csv`; `results_summaries/component_removal_attacks_500_paired_vs_hybrid.csv`; `results_raw/component_removal_attacks_500.csv`; `scripts/analyze_component_removal.py`. |
| Table 12, LaMa object-level inpainting | `results_summaries/lama_object_inpainting_382_summary.csv`; full clean/attacked paired analyses, runner and logs under `outputs/lama_reproducibility/full382_20261103/`. |
| Table 13, candidate-grid and threshold sensitivity | `results_summaries/detection_overall_four_attack.csv`; `results_raw/detection_param_sensitivity_results.csv`. |
| Table 14, global fallback threshold | `results_summaries/fallback_threshold_summary.csv`; `results_raw/fallback_threshold_results.csv`. |
| Table 15, placement IoU sensitivity | `results_summaries/iou_sensitivity_manuscript_summary.csv`; `results_raw/iou_sensitivity_results.csv`. |
| Table 16, original/watermarked visual quality | `results_summaries/quality_summary.csv`; `results_raw/quality_per_image.csv`; `scripts/quality_only.py`. |
| Table 17, runtime | `results_summaries/runtime_cost_summary.json`; `results_raw/runtime_cost_results.csv`; `scripts/runtime_cost_audit.py`. |
| Table 18, attack-weight sensitivity | `results_summaries/attack_weight_sensitivity.csv`, generated from `results_summaries/final_descriptive_by_attack.csv` by `scripts/analyze_attack_weight_sensitivity.py`. Erase40 receives weight p; the remaining three edits share (1-p)/3. The reported intervals identify where Hybrid has the highest descriptive weighted mean among the six fixed-budget policies; no inferential tests are attached. |
| Fig. 1, original illustration of support survival | The original composite is supplied within the manuscript source ZIP (`figures/Fig1.pdf` and `figures/Fig1.svg`). It is schematic and is not a detector-map measurement. |
| Fig. 2, per-attack Recovery31 trade-off | Values from `results_summaries/final_descriptive_by_attack.csv`; exact 24 plotted values and figure PDF are supplied in the manuscript source ZIP. |
| Fig. 3, COCO500 anchor-ratio trade-off | Values from `results_summaries/rho_descriptive.csv`; the manuscript source ZIP supplies `figures/Fig3_data.csv`, `figures/Fig3.pdf`, and the plotting script. |
| Fig. 4, anchor-removal effect | Values from `results_summaries/component_removal_attacks_500_condition_means.csv`; the manuscript source ZIP supplies `figures/Fig4_data.csv`, `figures/Fig4.pdf`, and the plotting script. |
| Fig. 5, attack-weight sensitivity | Recovery31 values are taken from `results_summaries/final_descriptive_by_attack.csv`; the manuscript source ZIP supplies the selected values in `figures/Fig5_data.csv` and the reproducible plotting script `figures/plot_fig5_attack_weight_sensitivity.py`. |
| Equal-budget anchor-free control | `results_summaries/anchor_equal_budget_overall_four_attack_tests.csv`; source records in `results_raw/anchor_free_equal_budget_results.csv`. |
| Placement fallback and clean-image negative control | `results_raw/placement_audit_500.csv`; `results_summaries/clean_negative_control_summary.json`; `results_raw/clean_negative_control_results.csv`. |

The frozen WAM weights, upstream WAM code, COCO and DIV2K images, and pretrained LaMa weights are not redistributed. The model/checkpoint identifiers and external dependencies are documented separately in `README.md` and `UPSTREAM_DEPENDENCIES.md`. The LaMa experiment is distinct from the four-edit aggregate and must not be included in that mean.
