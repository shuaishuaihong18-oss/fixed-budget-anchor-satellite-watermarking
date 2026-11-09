import csv
import json
import hashlib
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
OUT = ROOT / "outputs/revision_202610/lama_object_inpainting_full382_20261103"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 3
LIMIT = 382
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

    out_csv = OUT / "lama_object_inpainting_382.partial.csv"
    final_csv = OUT / "lama_object_inpainting_382.csv"
    metadata_path = OUT / "metadata.json"
    eligible_path = OUT / "eligible_annotations.csv"

    for path in [out_csv, final_csv, metadata_path, eligible_path]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing evidence: {path}"
            )

    fields = [
        "image", "source_id", "eligible_index", "geometry_seed",
        "ann_id", "category_id", "message",
        "strategy", "attack", "mask_frac",
        "actual_area", "intersection_area",
        "overlap_of_watermark", "overlap_of_object",
        "inside_change_mae", "outside_change_mae",
        "regions_detected",
        "orig_bitacc", "vote_bitacc",
        "orig_recovery31", "vote_recovery31",
    ]

    vis_dir = OUT / "visuals"
    vis_dir.mkdir(exist_ok=True)

    smoke_path = ROOT / (
        "outputs/revision_202610/"
        "lama_object_inpainting_smoke5_20261103/"
        "smoke5_results.csv"
    )
    smoke_ref = pd.read_csv(smoke_path)
    assert len(smoke_ref) == 30
    smoke_ref = smoke_ref.set_index(["image", "strategy"])

    pd.DataFrame(eligible).to_csv(
        eligible_path, index=False
    )

    rows_written = 0
    reference_checks = 0

    with out_csv.open("x", newline="") as f:
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
                assert mask_pil.size == orig_size

                object_mask = torch.as_tensor(
                    mask_np > 0,
                    device=device,
                    dtype=img.dtype
                ).unsqueeze(0).unsqueeze(0)

                # Same complete watermark for all allocation methods.
                full_w = wam.embed(img, msg)["imgs_w"]

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

                    img_w = full_w * union_mask + img * (1.0 - union_mask)

                    img_w_pil = tensor_to_pil01(img_w)
                    assert img_w_pil.size == orig_size

                    attacked_padded = lama(img_w_pil, mask_pil)
                    expected_size = tuple(
                        ((v + 7) // 8) * 8 for v in orig_size
                    )
                    assert attacked_padded.size == expected_size, (
                        image_name, strategy,
                        attacked_padded.size, expected_size
                    )

                    attacked_pil = crop_back_to_original(
                        attacked_padded, orig_size
                    )
                    assert attacked_pil.size == orig_size

                    attacked = pil_to_tensor(
                        attacked_pil, device
                    )

                    # Diagnostic only: do not use overlap to
                    # select images or change attack masks.
                    intersection = float(
                        (union_mask * object_mask).sum().item()
                    )
                    watermark_pixels = float(
                        union_mask.sum().item()
                    )
                    object_pixels = float(
                        object_mask.sum().item()
                    )
                    image_pixels = float(
                        img.shape[-2] * img.shape[-1]
                    )

                    image_before = np.asarray(
                        img_w_pil, dtype=np.float32
                    )
                    image_after = np.asarray(
                        attacked_pil, dtype=np.float32
                    )
                    pixel_change = np.abs(
                        image_after - image_before
                    ).mean(axis=2) / 255.0
                    object_bool = mask_np > 0

                    inside_mae = float(
                        pixel_change[object_bool].mean()
                    )
                    outside_mae = float(
                        pixel_change[~object_bool].mean()
                    )

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
                        "eligible_index": idx,
                        "geometry_seed": SEED + idx,
                        "ann_id": item["ann_id"],
                        "category_id": item["category_id"],
                        "message": msg_string,
                        "strategy": strategy,
                        "attack": ATTACK_NAME,
                        "mask_frac": round(item["mask_frac"], 6),
                        "actual_area": float(union_mask.mean().item()),
                        "intersection_area": intersection / image_pixels,
                        "overlap_of_watermark": intersection / max(watermark_pixels, 1e-8),
                        "overlap_of_object": intersection / max(object_pixels, 1e-8),
                        "inside_change_mae": inside_mae,
                        "outside_change_mae": outside_mae,
                        "regions_detected": int(det_vote["region_count"].item()),
                        "orig_bitacc": bit_accuracy(pred_orig, msg),
                        "vote_bitacc": bit_accuracy(pred_vote, msg),
                        "orig_recovery31": recovery_31_32(pred_orig, msg),
                        "vote_recovery31": recovery_31_32(pred_vote, msg),
                    }
                    # Reproduce the successful Smoke5 results
                    # before proceeding to the remaining images.
                    if idx < 5:
                        reference = smoke_ref.loc[
                            (image_name, strategy)
                        ]
                        for key in [
                            "mask_frac",
                            "actual_area",
                            "orig_bitacc",
                            "vote_bitacc",
                            "orig_recovery31",
                            "vote_recovery31",
                        ]:
                            diff = abs(
                                float(row[key])
                                - float(reference[key])
                            )
                            if diff > 1e-6:
                                raise RuntimeError(
                                    f"Smoke5 mismatch: "
                                    f"{image_name}, {strategy}, "
                                    f"{key}, diff={diff}"
                                )
                        reference_checks += 1

                    writer.writerow(row)
                    rows_written += 1

                    if idx < 2:
                        stem = image_name.replace(".jpg", "")
                        img_pil.save(vis_dir / f"{stem}_{strategy}_orig.png")
                        img_w_pil.save(vis_dir / f"{stem}_{strategy}_wm.png")
                        mask_pil.save(vis_dir / f"{stem}_{strategy}_mask.png")
                        attacked_pil.save(vis_dir / f"{stem}_{strategy}_attacked.png")

                if (idx + 1) % 10 == 0 or idx + 1 == LIMIT:
                    f.flush()
                    print(
                        f"Processed {idx+1}/{LIMIT}; "
                        f"rows={rows_written}",
                        flush=True
                    )

    result = pd.read_csv(out_csv)

    expected_rows = LIMIT * len(STRATEGIES)

    assert rows_written == expected_rows
    assert len(result) == expected_rows
    assert result["image"].nunique() == LIMIT
    assert result["strategy"].nunique() == len(STRATEGIES)
    assert reference_checks == 30

    assert not result.duplicated(
        ["image", "strategy"]
    ).any()

    assert result.groupby("image")["strategy"].nunique().eq(
        len(STRATEGIES)
    ).all()

    assert result.groupby("image")["ann_id"].nunique().eq(1).all()
    assert result.groupby("image")["message"].nunique().eq(1).all()

    metrics = [
        "actual_area",
        "orig_bitacc", "vote_bitacc",
        "orig_recovery31", "vote_recovery31",
        "overlap_of_watermark", "overlap_of_object",
        "inside_change_mae", "outside_change_mae"
    ]
    assert not result[metrics].isna().any().any()

    for key in [
        "orig_bitacc", "vote_bitacc",
        "orig_recovery31", "vote_recovery31",
        "overlap_of_watermark", "overlap_of_object",
    ]:
        assert result[key].between(0, 1).all(), key

    assert result["actual_area"].between(0.15, 0.25).all()

    # Publish final CSV only after integrity checks pass.
    out_csv.rename(final_csv)

    def filehash(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(
                lambda: fh.read(1024 * 1024), b""
            ):
                h.update(chunk)
        return h.hexdigest()

    model_path = ROOT / "checkpoints/wam_mit.pth"
    torch_home = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch"))).expanduser().resolve()
    lama_path = torch_home / "hub/checkpoints/big-lama.pt"

    metadata = {
        "experiment": "LaMa semantic object inpainting",
        "n_images": LIMIT,
        "n_strategies": len(STRATEGIES),
        "n_records": len(result),
        "seed": SEED,
        "geometry_seed_rule": "3 + eligible_image_index",
        "payload_bits": 32,
        "budget": BUDGET,
        "K": K,
        "anchor_ratio": ANCHOR_RATIO,
        "mask_area_range": [
            MASK_AREA_MIN, MASK_AREA_MAX
        ],
        "object_selection": (
            "largest eligible COCO instance per image"
        ),
        "object_selection_uses_watermark_results": False,
        "detector": "original plus proposed voting",
        "smoke5_reproduced_rows": reference_checks,
        "WAM_sha256": filehash(model_path),
        "LaMa_sha256": filehash(lama_path),
        "script_sha256": filehash(Path(__file__)),
        "result_sha256": filehash(final_csv),
    }

    with metadata_path.open("x") as f:
        json.dump(
            metadata, f, indent=2,
            ensure_ascii=False
        )

    print("=" * 70)
    print("FULL LAMA EXPERIMENT COMPLETE")
    print("Images:", LIMIT)
    print("Rows:", len(result), "/", expected_rows)
    print("Smoke5 reproduced:", reference_checks, "/30")
    print("Missing primary metrics: 0")
    print("Duplicate image-strategy keys: 0")
    print("Output:", final_csv)
    print("Metadata:", metadata_path)
    print("PASS")

if __name__ == "__main__":
    main()
