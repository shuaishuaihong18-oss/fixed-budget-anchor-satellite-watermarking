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


## COCO annotation-derived metadata

The file `outputs/lama_reproducibility/full382_20261103/eligible_annotations.csv` includes selected COCO annotation identifiers and category identifiers, together with mask-area fractions calculated from COCO instance annotations. The source COCO annotations are attributed to the COCO Consortium and are licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

Attribution: COCO Consortium, Common Objects in Context (COCO), annotations. Source: https://cocodataset.org/. License: https://creativecommons.org/licenses/by/4.0/.

No COCO image files or complete COCO annotation archive are redistributed in this package. The COCO Consortium does not own the copyrights to the images; image use remains subject to the applicable image-level rights and terms.
