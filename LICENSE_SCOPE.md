# License scope and third-party materials

No single license is asserted for every file in this repository. The MIT license in `LICENSE` applies only to source code written by the authors for this project:

- the files in `scripts/`;
- `watermark_anything/redundant_region.py`; and
- the authors' original Python experiment and analysis scripts under `outputs/`.

The MIT license does **not** grant rights to:

- the upstream Watermark Anything (WAM) repository or its files;
- WAM checkpoints, including `wam_mit.pth`, or `params.json`;
- COCO or DIV2K images, or the complete COCO annotation archive;
- pretrained LaMa model weights; or
- third-party libraries listed in the environment specification.

Obtain these materials from their respective sources and follow the applicable licenses and terms. The CSV files, manifests, logs, and summary tables are research records included to document the reported experiments. They are not relicensed by the MIT notice in this repository. The manifests use relative paths for local dataset files.

## COCO annotation-derived metadata

`outputs/lama_reproducibility/full382_20261103/eligible_annotations.csv` contains selected COCO annotation identifiers and category identifiers, together with mask-area fractions calculated from COCO instance annotations. The source annotations are attributed to the COCO Consortium and are made available under the Creative Commons Attribution 4.0 (CC BY 4.0) terms.

Attribution: COCO Consortium, Common Objects in Context (COCO), annotations. Source: https://cocodataset.org/. License: https://creativecommons.org/licenses/by/4.0/.

No COCO image files or complete COCO annotation archive are redistributed here. The COCO Consortium does not own the copyrights to the images; use of each image remains subject to its applicable image-level rights and terms.
