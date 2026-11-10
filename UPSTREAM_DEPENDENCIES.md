# Upstream dependencies

The experiments use the official Meta/FacebookResearch Watermark Anything repository: https://github.com/facebookresearch/watermark-anything (archived upstream repository). Clone that repository separately and follow its original installation instructions. The official repository supplies the WAM network implementation, `notebooks/inference_utils.py`, transforms, and checkpoint configuration.

This package supplies the authors' additional `watermark_anything/redundant_region.py` module and evaluation scripts. It does not redistribute WAM source files, the official checkpoint, or datasets. The experiments used the MIT-licensed `wam_mit.pth` checkpoint identified by the hash recorded in the README; obtain it from the official release and verify its hash locally.


## COCO dataset and annotations

The evaluation uses selected COCO images and instance annotations. The package does not redistribute the COCO image files or the complete annotation archive. The selected annotation-derived metadata is documented in the evidence files and attributed to the COCO Consortium. COCO states that its annotations and website are licensed under CC BY 4.0: https://cocodataset.org/dataset/termsofuse.htm. The image files remain subject to their respective image-level rights and terms.


## Verified WAM upstream snapshot

The official upstream source reference used for the selected-file comparison is:

- Repository: https://github.com/facebookresearch/watermark-anything
- Commit: `2c08af04d037d5667c02f6ddebbda9ff04581c3e`
- Commit date: 2025-06-20
- Commit message: `Update inference_utils.py`

The Git blob SHA-1 identifiers of 40 selected files in the retained experiment
environment match the corresponding blobs at this commit. The file-by-file
comparison is recorded in
`protocols/wam_upstream_fingerprint_audit.csv`.

This is a selected-file fingerprint comparison, not a claim that every file in
the experiment directory is identical to upstream. The original experiment
directory did not retain `.git` metadata, so its complete historical checkout
cannot be established from that directory.

For a reproducible upstream checkout:

```bash
git clone https://github.com/facebookresearch/watermark-anything.git
git -C watermark-anything checkout 2c08af04d037d5667c02f6ddebbda9ff04581c3e
```

Do not overlay this package wholesale onto the upstream repository. Preserve
upstream files and add only the author-developed
`watermark_anything/redundant_region.py` module as described in the main README.
