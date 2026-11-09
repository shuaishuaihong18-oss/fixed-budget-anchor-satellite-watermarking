# Upstream dependencies

The experiments use the official Meta/FacebookResearch Watermark Anything repository: https://github.com/facebookresearch/watermark-anything (archived upstream repository). Clone that repository separately and follow its original installation instructions. The official repository supplies the WAM network implementation, `notebooks/inference_utils.py`, transforms, and checkpoint configuration.

This package supplies the authors' additional `watermark_anything/redundant_region.py` module and evaluation scripts. It does not redistribute WAM source files, the official checkpoint, or datasets. The experiments used the MIT-licensed `wam_mit.pth` checkpoint identified by the hash recorded in the README; obtain it from the official release and verify its hash locally.
