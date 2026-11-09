import random
import os
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from watermark_anything.data.transforms import default_transform
from watermark_anything.redundant_region import select_regions

ROOT = Path(os.environ.get("WAM_ROOT", Path(__file__).resolve().parents[3])).resolve()
OUT = ROOT / "outputs/revision_202610/lama_object_inpainting_smoke5_20261103"

SEED = 3
BUDGET = 0.20
K = 4
RHO = 0.40

STRATEGIES = [
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
    "hybrid",
]

ANN = Path(os.environ.get("COCO_ANN", str(ROOT / "data/annotations/instances_val2017.json"))).resolve()
MANIFEST = Path(os.environ.get("COCO_MANIFEST", str(ROOT / "outputs/revision_202610/manifests/coco500_clean_manifest.csv"))).resolve()
IMG_DIR = Path(os.environ.get("COCO_IMAGE_DIR", str(ROOT / "data/eval/coco500_clean"))).resolve()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda")

manifest = pd.read_csv(MANIFEST)
coco = COCO(str(ANN))

categories = {
    x["id"]: x["name"]
    for x in coco.loadCats(coco.getCatIds())
}

eligible = []

for row in manifest.itertuples():
    image_id = int(row.source_id)

    info = coco.loadImgs([image_id])[0]
    total = info["height"] * info["width"]

    anns = coco.loadAnns(
        coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    )

    best = None

    for ann in anns:
        m = coco.annToMask(ann)
        frac = float(m.sum()) / total

        if 0.03 <= frac <= 0.30:
            if best is None or frac > best["mask_frac"]:
                best = {
                    "image": row.dataset_filename,
                    "source_id": image_id,
                    "ann_id": ann["id"],
                    "category": categories[ann["category_id"]],
                    "mask_frac": frac,
                    "mask": m,
                }

    if best is not None:
        eligible.append(best)

eligible.sort(key=lambda x: x["image"])

# Use exactly the five images in the Smoke5 experiment.
eligible = eligible[:5]

results = []

for idx, item in enumerate(eligible):

    img_pil = Image.open(
        IMG_DIR / item["image"]
    ).convert("RGB")

    img = default_transform(img_pil).unsqueeze(0).to(device)

    object_mask = torch.tensor(
        item["mask"],
        dtype=torch.float32,
        device=device
    )[None, None]

    for strategy in STRATEGIES:

        nreg = 1 if strategy.startswith("single") else K

        masks, boxes = select_regions(
            img,
            strategy=strategy,
            total_area=BUDGET,
            num_regions=nreg,
            seed=SEED + idx,
            anchor_ratio=RHO,
        )

        union = masks.sum(dim=0, keepdim=True).clamp(0, 1)

        intersection = (object_mask * union).sum().item()
        object_area = object_mask.sum().item()
        watermark_area = union.sum().item()

        results.append({
            "image": item["image"],
            "category": item["category"],
            "strategy": strategy,
            "object_mask_area": object_area / (img.shape[-2] * img.shape[-1]),
            "watermark_area": watermark_area / (img.shape[-2] * img.shape[-1]),
            "intersection_area": intersection / (img.shape[-2] * img.shape[-1]),
            "overlap_of_watermark": intersection / max(watermark_area, 1e-8),
            "overlap_of_object": intersection / max(object_area, 1e-8),
        })

df = pd.DataFrame(results)

df.to_csv(
    OUT / "smoke5_mask_overlap_audit.csv",
    index=False
)

print("\n========== SELECTED OBJECTS ==========")

for x in eligible:
    print(
        x["image"],
        "|", x["category"],
        "| mask area =", round(x["mask_frac"], 4)
    )

print("\n========== OVERLAP BY STRATEGY ==========")

print(
    df.groupby("strategy")[
        [
            "overlap_of_watermark",
            "overlap_of_object",
            "intersection_area"
        ]
    ].mean().round(4).to_string()
)

print("\n========== PER-IMAGE OVERLAP ==========")

print(
    df.pivot(
        index="image",
        columns="strategy",
        values="overlap_of_watermark"
    ).round(3).to_string()
)

print("\nCSV:", OUT / "smoke5_mask_overlap_audit.csv")
print("PASS")
