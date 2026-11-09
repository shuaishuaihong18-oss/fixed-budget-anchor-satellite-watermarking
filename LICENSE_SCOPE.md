# License scope and third-party materials

The MIT License in `LICENSE` applies only to the authors' original source-code files
in `scripts/`, `watermark_anything/redundant_region.py`, and the original Python
experiment/analysis scripts under `outputs/`.

The MIT License does **not** grant rights to:
- the upstream Watermark Anything (WAM) repository or its files;
- WAM checkpoints, including `wam_mit.pth`, or `params.json`;
- COCO, DIV2K, or COCO annotation/image files;
- pretrained LaMa model weights or third-party model packages;
- third-party libraries listed in the environment specification.

Those materials are not redistributed in this package. Obtain them from their
respective official sources and comply with the applicable upstream licenses and
terms. The CSV files, manifests, logs, and summary tables are research records
provided to document the reported experiments; they do not contain the underlying
COCO/DIV2K image files. The manifest paths are relative placeholders for local
dataset locations.
