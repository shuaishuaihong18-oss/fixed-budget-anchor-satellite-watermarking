# Reproducibility package — Fixed-Budget Anchor--Satellite Watermarking

**Status: prepared release candidate 0.1.0; not yet published.** This package combines the confirmed manuscript source archive, the main revision evidence, and the 382-image LaMa object-inpainting evidence bundle. It is not a public release until deposited at a stable repository/archive and assigned an immutable release identifier.

## Scope

The package contains the authors’ region-allocation and regional-readout implementation, evaluation and analysis scripts, raw records for the main comparison and reported ablations, summary tables, data manifests, execution logs, the LaMa object-inpainting runner and paired outputs, and the exported `wam310` environment specification. It is intended to be used with the official upstream Watermark Anything (WAM) repository.

## Included

- Main COCO500 and DIV2K100 evaluation scripts and per-image result CSVs.
- Paired bootstrap, Wilcoxon and Holm statistical analysis scripts and outputs.
- Decoder and anchor-ratio ablation scripts and raw records.
- Equal-budget anchor-free control and component-removal experiment, including all three two-satellite deletion pairs in the raw records.
- Candidate-grid, cell-threshold, detector fallback-mask and placement-IoU sensitivity scripts and records.
- Placement audit, clean-image negative control, runtime audit, and clean-image PSNR/SSIM/LPIPS records.
- LaMa full-run script, 382-image attacked records, matched clean/attacked records, per-image paired analysis, statistical summaries, metadata and logs under `outputs/lama_reproducibility/`.
- The 30-row LaMa smoke-test record and mask-overlap audit.
- `environment/wam310_environment.yml`, exported from the experiment environment; the machine-specific `prefix` line has been removed.
- A result-to-manuscript map and SHA-256 provenance for the source and evidence archives.

## Not included

- Official WAM source repository, pretrained checkpoint (`wam_mit.pth`) or `params.json`.
- COCO or DIV2K image files and the full COCO annotation archive. The relevant image manifests and selected LaMa annotations are included; obtain the original data from their official sources and comply with their terms.
- Pretrained LaMa weights. Obtain them from the upstream release.

## Recorded environment

The exported environment is supplied at `environment/wam310_environment.yml`. The experiment record identifies Python 3.10, PyTorch 2.3.1+cu121, and an NVIDIA GeForce RTX 4090. The WAM checkpoint SHA-256 recorded by the experiment is `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2`.

Create the environment with `conda env create -f environment/wam310_environment.yml`. The export was captured on the original Linux/CUDA setup; platform-specific package availability may require adjustment on a different operating system or GPU.

## Reproduction outline

1. Clone the official upstream WAM repository and follow its installation instructions. The upstream repository supplies the WAM implementation and inference utilities.
2. Overlay this package onto the WAM repository root.
3. Obtain the official checkpoint and configuration separately; verify the checkpoint SHA-256 above.
4. Place the COCO500 and DIV2K100 subsets in `data/eval/coco500_clean` and `data/eval/div2k100`, following the included manifests.
5. Run the main evaluation from the WAM repository root using the commands below.

```bash
PYTHONPATH=. python scripts/run_fixed_budget_exp.py --ckpt checkpoints/wam_mit.pth --json checkpoints/params.json --img_dir data/eval/coco500_clean --out_dir outputs/revision_202610/raw/coco500_clean_allattacks --total_area 0.20 --num_regions 4 --anchor_ratio 0.40 --limit 500 --attacks none erase40 randomcrop50 cornercrop50 splicing20 --strategies single single_random random texture wam_random_mask hybrid --seed 3
PYTHONPATH=. python scripts/run_fixed_budget_exp.py --ckpt checkpoints/wam_mit.pth --json checkpoints/params.json --img_dir data/eval/div2k100 --out_dir outputs/revision_202610/raw/paired100_allattacks --total_area 0.20 --num_regions 4 --anchor_ratio 0.40 --limit 100 --attacks none erase40 randomcrop50 cornercrop50 splicing20 --strategies single single_random random texture wam_random_mask hybrid --seed 3
```

The LaMa runner is under `outputs/lama_reproducibility/full382_20261103/`; the smoke-test artifacts are under `outputs/lama_reproducibility/smoke5/`. The scripts now use portable path settings. Set `WAM_ROOT` to the local WAM checkout, `COCO_ANN` to the COCO `instances_val2017.json` annotation file, `COCO_MANIFEST` to the included manifest, and `COCO_IMAGE_DIR` to the local COCO image subset. For example:

```bash
export WAM_ROOT=/path/to/watermark-anything
export COCO_ANN=/path/to/annotations/instances_val2017.json
export COCO_MANIFEST=$WAM_ROOT/outputs/revision_202610/manifests/coco500_clean_manifest.csv
export COCO_IMAGE_DIR=$WAM_ROOT/data/eval/coco500_clean
```

The LaMa runner expects the WAM checkpoint/configuration and pretrained LaMa weights to be installed separately. If needed, set `LAMA_CHECKPOINT` to the local `big-lama.pt` path. The archived result CSVs are the records used in the manuscript and should not be overwritten.

Targeted analysis scripts are in `scripts/`; their outputs and manuscript locations are mapped in `MANUSCRIPT_RESULTS_MAP.md`. The scripts in this package have been checked for machine-specific absolute paths. The main evaluation scripts still expect to be launched from a WAM checkout with the documented datasets, checkpoint, and configuration installed separately. Run the documented smoke tests before any full evaluation.

## Important protocol notes

- Main comparison: frozen WAM backbone, 32-bit payload, nominal area budget `B=0.20`, `K=4`, anchor ratio `rho=0.40`, maximum pairwise IoU `0.03`.
- Candidate regions are selected from the edited-image detector map, not from the embedding mask.
- `Retain20` is the paper-facing name for implementation label `splicing20`; it retains a watermarked square and restores the remainder from the same unwatermarked source image. It is not cross-image semantic splicing.
- The full-image WAM reference uses area 1.0 and is contextual only; it is not part of the fixed-budget ranking.
- The 382-image LaMa test is a separate object-level inpainting stress test, not a replacement for the four controlled local-edit protocols.

## License and third-party materials

The included MIT license covers only the authors' original code files. It does not relicense upstream WAM materials, model checkpoints, datasets, pretrained LaMa weights, or third-party packages. See `LICENSE_SCOPE.md` and `UPSTREAM_DEPENDENCIES.md`.

## Public-release checklist

Before resubmission: (1) deposit this package in a public repository/archive; (2) verify access without login and record an immutable release identifier; (3) check the README and paths on a fresh WAM checkout; (4) verify that upstream checkpoint, data and LaMa-weight instructions are clear; and (5) insert the real repository URL and release identifier into the manuscript and response letter. Do not insert a placeholder or an unverified URL.
