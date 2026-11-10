# LaMa object-inpainting evidence (382 images)

This directory contains the archived full-run scripts, raw outputs, paired clean/attacked records, statistical analyses, logs, and metadata for the separate object-level LaMa inpainting stress test. The 382-image run contains 2,292 strategy-image rows (six strategies per image). The `smoke5/` directory retains the 30-row smoke-test record and overlap audit.

## Important external inputs

The archive does not redistribute COCO images/annotations, the official WAM repository or checkpoint, or the pretrained LaMa weights. Obtain these from their respective upstream sources and comply with their terms. The runners use portable environment variables rather than machine-specific paths. Set `WAM_ROOT` to the local WAM checkout, `COCO_ANN` to the COCO `instances_val2017.json` file, `COCO_MANIFEST` to the included manifest, and `COCO_IMAGE_DIR` to the local COCO image subset before execution. The archived CSVs and logs are the records used for the manuscript; they should not be overwritten when reproducing the run.

## Key files

- `full382_20261103/run_lama_object_inpainting_full382.py`: full 382-image runner.
- `full382_20261103/lama_object_inpainting_382.csv`: attacked-image records, 2,292 rows.
- `full382_20261103/clean_vs_attacked_382.csv`: image/strategy-matched clean and attacked records.
- `full382_20261103/clean_matched_382.csv`: matched clean control output.
- `full382_20261103/final_statistics/paired_analysis_all_images.csv`: paired per-image analysis.
- `full382_20261103/final_statistics/`: strategy summaries and pairwise comparisons.
- `full382_20261103/statistical_analysis/`: degradation and Friedman/post-hoc analyses.
- `full382_20261103/metadata.json`: experiment settings and source hashes.


## Smoke-test log interpretation

`smoke5/run.log` preserves the initial launch attempt, which stopped with `ModuleNotFoundError` because the WAM `notebooks` module was unavailable in that invocation. The initial log is retained as part of the execution record. `smoke5/run_retry.log` records the successful retry: all five images were processed and 30 strategy-image rows were written. The successful retry, rather than the initial failed launch, is the completed Smoke5 run.

## Reproduction path clarification

Set `COCO_MANIFEST` to `$PACKAGE_ROOT/manifests/COCO500_clean_manifest.csv`, where `PACKAGE_ROOT` is this reproducibility package. The full runner uses `TORCH_HOME/hub/checkpoints/big-lama.pt` for the SimpleLama checkpoint cache; it does not read a `LAMA_CHECKPOINT` environment variable. Keep the upstream WAM checkout separate and set `WAM_ROOT` to that checkout before running the scripts.
