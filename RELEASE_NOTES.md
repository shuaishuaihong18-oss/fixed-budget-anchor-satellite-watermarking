# Release candidate 0.1.0

This release candidate contains the authors' region-allocation and regional-readout
implementation, evaluation and statistical-analysis scripts, experiment configurations,
raw per-image result records, summaries, logs, and the 382-image LaMa inpainting
reproducibility records used for the manuscript revision.

## Public-release status

Prepared for public deposit, but not yet published. No GitHub repository URL or
Zenodo DOI is assigned at this stage.

## Changes made for portability and safe release

- Replaced machine-specific absolute paths in scripts with environment-variable
  configuration (`WAM_ROOT`, `COCO_ANN`, `COCO_MANIFEST`, `COCO_IMAGE_DIR`,
  and `LAMA_CHECKPOINT`).
- Converted COCO manifest source paths to relative dataset paths while retaining
  image identifiers and checksums.
- Sanitized machine-specific paths in logs.
- Clarified that the MIT license applies only to original author-written code.
- Confirmed that upstream checkpoints, dataset image files, COCO annotations,
  and pretrained LaMa weights are not redistributed.

## Before publication

Run the documented smoke tests on a clean WAM checkout, confirm all output paths,
review the package license scope, and then create the GitHub repository and Zenodo
archive. Insert the final repository URL and DOI only after they have been verified.


## Additional pre-release clarifications

- Clarified that the manuscript source archive and supporting evidence archives are documented by provenance hashes but are not included in this repository.
- Added attribution for metadata derived from COCO instance annotations and clarified the distinction between annotation licensing and image-level rights.
- Documented the official PyTorch CUDA 12.1 wheel index for the pinned PyTorch and torchvision versions.
- Explained the initial Smoke5 launch failure and retained the successful retry record.

## Additional upstream-source and reproduction clarifications

- Recorded the official WAM reference commit and a file-by-file fingerprint comparison for 40 selected upstream files.
- Added the 40-file fingerprint record under `protocols/wam_upstream_fingerprint_audit.csv`.
- Replaced wholesale overlay instructions with instructions that keep the upstream checkout and this package separate.
- Corrected the LaMa manifest and PyTorch hub-cache paths in the reproduction instructions.

## Additional upstream-source and reproduction clarifications

- Recorded the official WAM reference commit and a file-by-file fingerprint comparison for 40 selected upstream files.
- Added the 40-file fingerprint record under `protocols/wam_upstream_fingerprint_audit.csv`.
- Replaced wholesale overlay instructions with instructions that keep the upstream checkout and this package separate.
- Corrected the LaMa manifest and PyTorch hub-cache paths in the reproduction instructions.
