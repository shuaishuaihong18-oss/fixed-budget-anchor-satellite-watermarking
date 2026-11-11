# Reproducibility package — Fixed-Budget Anchor--Satellite Watermarking

This repository accompanies the manuscript *Fixed-Budget Anchor--Satellite Region Allocation for Robust Localized Image Watermarking*. It contains the authors' region-allocation and regional-readout implementation, evaluation scripts, configurations, and selected experiment records.

The implementation uses the official upstream Watermark Anything (WAM) repository. This repository adds the authors' `watermark_anything/redundant_region.py` module and the scripts used for evaluation and analysis; it does not replace or redistribute the upstream WAM source code or checkpoint.

The original manuscript-source archive and the complete supporting-evidence archives are not included. Their filenames, sizes, and SHA-256 digests are recorded in `protocols/source_archive_provenance.json`. The selected experiment records and supporting files described below are included.

## Scope

The package is intended to be used alongside the specified upstream WAM revision. Keep the upstream checkout separate from this repository; the reproduction instructions below show how to add the authors' regional module without replacing upstream files.

## Included

- Main COCO500 and DIV2K100 evaluation scripts and per-image result CSVs.
- Paired bootstrap, Wilcoxon and Holm statistical analysis scripts and outputs.
- Decoder and anchor-ratio ablation scripts and raw records.
- Equal-budget anchor-free control and component-removal experiment, including all three two-satellite deletion pairs in the raw records.
- Candidate-grid, cell-threshold, detector fallback-mask and placement-IoU sensitivity scripts and records.
- Placement audit, clean-image negative control, runtime audit, and clean-image PSNR/SSIM/LPIPS records.
- LaMa full-run script, 382-image attacked records, matched clean/attacked records, per-image paired analysis, statistical summaries, metadata and logs under `outputs/lama_reproducibility/`.
- The 30-row LaMa smoke-test record, mask-overlap audit, and both launch logs. `run.log` records the initial failed launch because the WAM `notebooks` module was unavailable in that invocation; `run_retry.log` records the successful retry, with all five images processed and 30 rows written.
- `environment/wam310_environment.yml`, exported from the experiment environment; the machine-specific `prefix` line has been removed.
- A result-to-manuscript map and SHA-256 provenance for the source and evidence archives.

## Not included

- Official WAM source repository, pretrained checkpoint (`wam_mit.pth`) or `params.json`.
- COCO and DIV2K image files and the full COCO annotation archive. Image manifests and selected metadata derived from COCO instance annotations are included; the source image files and complete annotation archive are not.
- Pretrained LaMa weights. Obtain them from the upstream release.

## Recorded environment

The exported environment is supplied at `environment/wam310_environment.yml`. The experiment record identifies Python 3.10, PyTorch 2.3.1+cu121, and an NVIDIA GeForce RTX 4090. The WAM checkpoint SHA-256 recorded by the experiment is `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2`.

Create the environment with `conda env create -f environment/wam310_environment.yml`. The pip subsection includes the official PyTorch CUDA 12.1 wheel index for the pinned `torch==2.3.1+cu121` and `torchvision==0.18.1+cu121` packages. The export was captured on the original Linux/CUDA setup; platform-specific package availability may require adjustment on a different operating system or GPU.



## Reproduction outline

Keep this package and the upstream WAM checkout in separate directories. Do not
overlay this package wholesale onto the upstream repository. The upstream checkout
provides the original WAM implementation; this package provides the author-developed
regional module, experiment scripts, configurations, manifests, and archived records.

### 1. Prepare the upstream checkout and author-developed module

```bash
export PACKAGE_ROOT=/path/to/fixed-budget-anchor-satellite-watermarking
export WAM_ROOT="$HOME/watermark-anything"

git clone https://github.com/facebookresearch/watermark-anything.git "$WAM_ROOT"
git -C "$WAM_ROOT" checkout 2c08af04d037d5667c02f6ddebbda9ff04581c3e

# Add only the author-developed module; do not replace upstream files.
cp "$PACKAGE_ROOT/watermark_anything/redundant_region.py" \
   "$WAM_ROOT/watermark_anything/"

export PYTHONPATH="$WAM_ROOT:$PACKAGE_ROOT"
```

The upstream commit and the 40-file fingerprint comparison are documented in
`UPSTREAM_DEPENDENCIES.md` and
`protocols/wam_upstream_fingerprint_audit.csv`.

### 2. Prepare the environment, checkpoint, and datasets

Create the recorded Linux/CUDA environment with:

```bash
conda env create -f "$PACKAGE_ROOT/environment/wam310_environment.yml"
conda activate wam310
```

Obtain the official WAM checkpoint and its `params.json` separately. Verify the
checkpoint against the SHA-256 recorded above. Place the COCO500 and DIV2K100
image subsets in `$WAM_ROOT/data/eval/coco500_clean` and
`$WAM_ROOT/data/eval/div2k100`, respectively, following
`$PACKAGE_ROOT/manifests/COCO500_clean_manifest.csv` and
`$PACKAGE_ROOT/manifests/DIV2K100_manifest.csv`.

### 3. Run the main fixed-budget evaluations

Run from the WAM checkout. The scripts remain in this package; the commands
write new outputs under the WAM checkout and do not overwrite the archived
manuscript records in this package.

```bash
cd "$WAM_ROOT"

PYTHONPATH="$WAM_ROOT:$PACKAGE_ROOT" python \
  "$PACKAGE_ROOT/scripts/run_fixed_budget_exp.py" \
  --ckpt "$WAM_ROOT/checkpoints/wam_mit.pth" \
  --json "$WAM_ROOT/checkpoints/params.json" \
  --img_dir "$WAM_ROOT/data/eval/coco500_clean" \
  --out_dir "$WAM_ROOT/outputs/revision_202610/raw/coco500_clean_allattacks" \
  --total_area 0.20 --num_regions 4 --anchor_ratio 0.40 \
  --limit 500 \
  --attacks none erase40 randomcrop50 cornercrop50 splicing20 \
  --strategies single single_random random texture wam_random_mask hybrid \
  --seed 3

PYTHONPATH="$WAM_ROOT:$PACKAGE_ROOT" python \
  "$PACKAGE_ROOT/scripts/run_fixed_budget_exp.py" \
  --ckpt "$WAM_ROOT/checkpoints/wam_mit.pth" \
  --json "$WAM_ROOT/checkpoints/params.json" \
  --img_dir "$WAM_ROOT/data/eval/div2k100" \
  --out_dir "$WAM_ROOT/outputs/revision_202610/raw/paired100_allattacks" \
  --total_area 0.20 --num_regions 4 --anchor_ratio 0.40 \
  --limit 100 \
  --attacks none erase40 randomcrop50 cornercrop50 splicing20 \
  --strategies single single_random random texture wam_random_mask hybrid \
  --seed 3
```

### 4. Run the separate LaMa object-inpainting evaluation

Set the external data and model paths explicitly. The manifest is supplied by
this package; the COCO images, COCO annotations, WAM checkpoint, and LaMa
weights must be obtained separately.

```bash
export WAM_ROOT
export PACKAGE_ROOT
export COCO_ANN=/path/to/annotations/instances_val2017.json
export COCO_MANIFEST="$PACKAGE_ROOT/manifests/COCO500_clean_manifest.csv"
export COCO_IMAGE_DIR="$WAM_ROOT/data/eval/coco500_clean"

# SimpleLama uses the PyTorch hub cache. Place big-lama.pt at:
# $TORCH_HOME/hub/checkpoints/big-lama.pt
export TORCH_HOME="$HOME/.cache/torch"
mkdir -p "$TORCH_HOME/hub/checkpoints"

PYTHONPATH="$WAM_ROOT:$PACKAGE_ROOT" python \
  "$PACKAGE_ROOT/outputs/lama_reproducibility/full382_20261103/run_lama_object_inpainting_full382.py"

PYTHONPATH="$WAM_ROOT:$PACKAGE_ROOT" python \
  "$PACKAGE_ROOT/outputs/lama_reproducibility/full382_20261103/run_clean_matched382.py"
```

The LaMa runner writes new run outputs under
`$WAM_ROOT/outputs/revision_202610/lama_object_inpainting_full382_20261103/`.
The archived CSVs, logs, and statistics under
`$PACKAGE_ROOT/outputs/lama_reproducibility/` are the manuscript records and
should not be overwritten.

## Important protocol notes

- Main comparison: frozen WAM backbone, 32-bit payload, nominal area budget `B=0.20`, `K=4`, anchor ratio `rho=0.40`, maximum pairwise IoU `0.03`.
- Candidate regions are selected from the edited-image detector map, not from the embedding mask.
- `Retain20` is the paper-facing name for implementation label `splicing20`; it retains a watermarked square and restores the remainder from the same unwatermarked source image. It is not cross-image semantic splicing.
- The full-image WAM reference uses area 1.0 and is contextual only; it is not part of the fixed-budget ranking.
- The 382-image LaMa test is a separate object-level inpainting stress test, not a replacement for the four controlled local-edit protocols.

## License and third-party materials

The MIT license applies only to the authors' original source-code files identified in `LICENSE_SCOPE.md`. It does not grant rights to upstream WAM materials, model checkpoints, datasets, pretrained LaMa weights, or third-party packages. COCO annotation-derived metadata is attributed separately under the applicable CC BY 4.0 terms. See `LICENSE_SCOPE.md` and `UPSTREAM_DEPENDENCIES.md`.

## Access and version

Set the GitHub repository to Public and confirm that its files can be opened in a browser session that is not signed in to GitHub before citing it as a publicly available resource. No Zenodo DOI has been assigned. For a fixed version, cite a tagged release or an exact commit rather than the moving `main` branch.
