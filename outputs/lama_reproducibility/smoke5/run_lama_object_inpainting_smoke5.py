import csv
import json
import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from simple_lama_inpainting import SimpleLama

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform
from watermark_anything.redundant_region import (
    select_regions,
    detect_confidence_voting,
    bit_accuracy,
    recovery_31_32,
)
from watermark_anything.data.metrics import msg_predict_inference

ROOT = Path(os.environ.get("WAM_ROOT", Path(__file__).resolve().parents[3])).resolve()
OUT = ROOT / "outputs/revision_202610/lama_object_inpainting_smoke5_20261103"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 3
LIMIT = 5
BUDGET = 0.20
K = 4
ANCHOR_RATIO = 0.40

MASK_AREA_MIN = 0.03
MASK_AREA_MAX = 0.30

STRATEGIES = [
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
    "hybrid",
]

ATTACK_NAME = "lama_object_inpaint"

ANN = Path(os.environ.get("COCO_ANN", str(ROOT / "data/annotations/instances_val2017.json"))).resolve()
MANIFEST = Path(os.environ.get("COCO_MANIFEST", str(ROOT / "outputs/revision_202610/manifests/coco500_clean_manifest.csv"))).resolve()
IMG_DIR = Path(os.environ.get("COCO_IMAGE_DIR", str(ROOT / "data/eval/coco500_clean"))).resolve()

def original_wam_decode(wam, img_pt, gt_msg):
    preds = wam.detect(img_pt)["preds"]
    mask_preds = torch.sigmoid(preds[:, 0:1])
    bit_preds = preds[:, 1:]
    return msg_predict_inference(bit_preds, mask_preds).to(gt_msg.device).int()

def pil_to_tensor(img, device):
    return default_transform(img.convert("RGB")).unsqueeze(0).to(device)

def tensor_to_pil01(img_t):
    x = img_t.detach().cpu()[0]
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)

def find_eligible_images():
    df = pd.read_csv(MANIFEST)
    coco = COCO(str(ANN))
    rows = []

    for row in df.itertuples():
        image_id = int(row.source_id)
        img_info = coco.loadImgs([image_id])[0]
        H, W = img_info["height"], img_info["width"]
        area_total = H * W

        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        best = None
        for ann in anns:
            try:
                m = coco.annToMask(ann)
            except Exception:
                continue
            frac = float(m.sum()) / float(area_total)
            if MASK_AREA_MIN <= frac <= MASK_AREA_MAX:
                if best is None or frac > best["mask_frac"]:
                    best = {
                        "dataset_filename": row.dataset_filename,
                        "source_id": image_id,
                        "mask_frac": frac,
                        "ann_id": ann["id"],
                        "category_id": ann["category_id"],
                    }

        if best is not None:
            rows.append(best)

    rows = sorted(rows, key=lambda x: x["dataset_filename"])
    return rows[:LIMIT]

def crop_back_to_original(im, size_wh):
    w, h = size_wh
    return im.crop((0, 0, w, h))

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    eligible = find_eligible_images()
    assert len(eligible) == LIMIT, f"Expected {LIMIT} eligible images, got {len(eligible)}"
    print("Eligible images:", len(eligible))

    coco = COCO(str(ANN))
    lama = SimpleLama()

    wam = load_model_from_checkpoint(
        str(ROOT / "checkpoints/params.json"),
        str(ROOT / "checkpoints/wam_mit.pth"),
    ).to(device).eval()

    main_rows = pd.read_csv(ROOT / "outputs/revision_202610/raw/coco500_clean_allattacks/results.csv")
    messages = dict(zip(main_rows["image"], main_rows["message"]))

    out_csv = OUT / "smoke5_results.csv"
    if out_csv.exists():
        raise FileExistsError(out_csv)

    fields = [
        "image", "source_id", "strategy", "attack", "mask_frac",
        "actual_area", "regions_detected",
        "orig_bitacc", "vote_bitacc",
        "orig_recovery31", "vote_recovery31",
    ]

    vis_dir = OUT / "visuals"
    vis_dir.mkdir(exist_ok=True)

    rows_written = 0

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with torch.no_grad():
            for idx, item in enumerate(eligible):
                image_name = item["dataset_filename"]
                image_path = IMG_DIR / image_name
                assert image_path.exists(), image_path

                img_pil = Image.open(image_path).convert("RGB")
                orig_size = img_pil.size
                img = pil_to_tensor(img_pil, device)

                msg_string = messages[image_name]
                msg = torch.tensor(
                    [[int(b) for b in msg_string]],
                    dtype=torch.float32,
                    device=device
                )

                ann = coco.loadAnns([item["ann_id"]])[0]
                mask_np = coco.annToMask(ann).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_np).convert("L")

                for strategy in STRATEGIES:
                    nreg = 1 if strategy.startswith("single") else K

                    masks, boxes = select_regions(
                        img,
                        strategy=strategy,
                        total_area=BUDGET,
                        num_regions=nreg,
                        seed=SEED + idx,
                        anchor_ratio=ANCHOR_RATIO
                    )

                    union_mask = masks.sum(dim=0, keepdim=True).clamp(0, 1)

                    full_w = wam.embed(img, msg)["imgs_w"]
                    img_w = full_w * union_mask + img * (1.0 - union_mask)

                    img_w_pil = tensor_to_pil01(img_w)

                    attacked_pil = lama(img_w_pil, mask_pil)
                    attacked_pil = crop_back_to_original(attacked_pil, orig_size)

                    attacked = pil_to_tensor(attacked_pil, device)

                    pred_orig = original_wam_decode(wam, attacked, msg)
                    det_vote = detect_confidence_voting(
                        wam,
                        attacked,
                        vote_mode="proposed"
                    )
                    pred_vote = det_vote["pred_msg"]

                    row = {
                        "image": image_name,
                        "source_id": item["source_id"],
                        "strategy": strategy,
                        "attack": ATTACK_NAME,
                        "mask_frac": round(item["mask_frac"], 6),
                        "actual_area": float(union_mask.mean().item()),
                        "regions_detected": int(det_vote["region_count"].item()),
                        "orig_bitacc": bit_accuracy(pred_orig, msg),
                        "vote_bitacc": bit_accuracy(pred_vote, msg),
                        "orig_recovery31": recovery_31_32(pred_orig, msg),
                        "vote_recovery31": recovery_31_32(pred_vote, msg),
                    }
                    writer.writerow(row)
                    rows_written += 1

                    if idx < 2:
                        stem = image_name.replace(".jpg", "")
                        img_pil.save(vis_dir / f"{stem}_{strategy}_orig.png")
                        img_w_pil.save(vis_dir / f"{stem}_{strategy}_wm.png")
                        mask_pil.save(vis_dir / f"{stem}_{strategy}_mask.png")
                        attacked_pil.save(vis_dir / f"{stem}_{strategy}_attacked.png")

                print(f"Processed {idx+1}/{LIMIT}")

    print("=" * 70)
    print("SMOKE TEST COMPLETE")
    print("Rows written:", rows_written)
    print("Output:", out_csv)
    print("Visuals:", vis_dir)

if __name__ == "__main__":
    main()
