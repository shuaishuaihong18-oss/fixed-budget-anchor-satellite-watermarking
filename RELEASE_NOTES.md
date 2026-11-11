# Release notes — 0.1.0

This version collects the authors' region-allocation and regional-readout implementation, evaluation and statistical-analysis scripts, experiment configurations, per-image result records, summary tables, execution logs, and the 382-image LaMa object-inpainting records used in the manuscript revision.

## Repository status

The GitHub repository has been created and the release-candidate files are committed to the `main` branch. Set the repository visibility to Public and check access without signing in before citing it as a public resource. No Zenodo DOI has been assigned.

## Changes included in this version

- Replaced machine-specific paths in the experiment scripts with environment-variable settings, including `WAM_ROOT`, `COCO_ANN`, `COCO_MANIFEST`, and `COCO_IMAGE_DIR`.
- Converted COCO manifest source paths to relative dataset paths while retaining image identifiers and checksums.
- Removed machine-specific paths from the retained logs.
- Defined the scope of the MIT license and separated it from upstream code, checkpoints, datasets, and third-party packages.
- Recorded the source archives and environment files by filename, size, and SHA-256 digest.
- Added attribution for the selected metadata derived from COCO instance annotations.
- Specified the official PyTorch CUDA 12.1 wheel index for the pinned PyTorch and torchvision versions.
- Retained both LaMa Smoke5 launch logs and identified the successful retry.
- Recorded the WAM upstream reference commit and the comparison of 40 selected source files.
- Kept the upstream WAM checkout separate from the authors' module and experiment scripts in the reproduction instructions.
- Documented the `TORCH_HOME` cache location used to load `big-lama.pt`.

## Reproduction status

The repository includes the experiment records used for the manuscript. The packaging checks cover file integrity, Python syntax, paths, and selected upstream-source fingerprints. A clean-machine, end-to-end rerun of all experiments has not been performed as part of this release preparation. The archived records should not be described as results newly reproduced by this packaging work.

## External materials

The WAM source repository, WAM checkpoint, COCO and DIV2K image files, full COCO annotation archive, and pretrained LaMa weights are not redistributed here. Obtain them from their respective sources and follow the applicable license terms. The selected COCO annotation-derived metadata and the scope of the repository's MIT license are described in `LICENSE_SCOPE.md`.
