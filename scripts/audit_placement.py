import csv
import json
import random
from pathlib import Path

import pandas as pd
import torch
import sys

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from watermark_anything import redundant_region as rr

ROOT = Path("outputs/revision_202610")
SRC = ROOT / (
    "iou_sensitivity_preflight_20261028/"
    "iou_geometry_sensitivity_preflight.csv"
)
OUT = ROOT / (
    "placement_fallback_audit_20261029/"
    "placement_audit_500.csv"
)

B = 0.20
K = 4
RHO = 0.40
IOU = 0.03
SEED = 3
MAX_TRIALS = 1000

if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite: {OUT}")

df = pd.read_csv(SRC, dtype={"image": str})
ref = df[
    (df["max_iou"] - IOU).abs() < 1e-9
].copy()

assert len(ref) == 500
assert ref["image_index"].nunique() == 500

records = []

for r in ref.sort_values("image_index").itertuples(index=False):
    idx = int(r.image_index)
    H, W = int(r.height), int(r.width)
    seed = SEED + idx

    img = torch.zeros((1, 3, H, W), dtype=torch.float32)

    # Actual existing implementation.
    masks, actual_boxes = rr.select_hybrid_regions(
        img,
        total_area=B,
        num_regions=K,
        anchor_ratio=RHO,
        max_iou=IOU,
        seed=seed,
        max_trials=MAX_TRIALS,
    )

    actual = [
        [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
        for b in actual_boxes
    ]

    # Independently replay the original random placement loop
    # to measure rejection and fallback statistics.
    rng = random.Random(seed)

    anchor_side = rr._area_to_side(H, W, B * RHO)
    sat_side = rr._area_to_side(
        H, W, B * (1.0 - RHO) / (K - 1)
    )

    ax1 = max(0, (W - anchor_side) // 2)
    ay1 = max(0, (H - anchor_side) // 2)

    selected = [
        rr.RegionBox(
            ax1, ay1,
            ax1 + anchor_side,
            ay1 + anchor_side,
            1.0,
        )
    ]

    trials = 0
    rejected = 0

    while len(selected) < K and trials < MAX_TRIALS:
        trials += 1

        x1 = rng.randint(0, max(0, W - sat_side))
        y1 = rng.randint(0, max(0, H - sat_side))

        candidate = rr.RegionBox(
            x1, y1,
            x1 + sat_side,
            y1 + sat_side,
            0.0,
        )

        if all(
            rr._iou(candidate, previous) <= IOU
            for previous in selected
        ):
            selected.append(candidate)
        else:
            rejected += 1

    random_placed = len(selected) - 1
    fallback_triggered = len(selected) < K
    fallback_added = 0

    if fallback_triggered:
        for y1 in range(0, H - sat_side + 1, sat_side):
            for x1 in range(0, W - sat_side + 1, sat_side):
                candidate = rr.RegionBox(
                    x1, y1,
                    x1 + sat_side,
                    y1 + sat_side,
                    0.0,
                )

                if all(
                    rr._iou(candidate, previous) <= IOU
                    for previous in selected
                ):
                    selected.append(candidate)
                    fallback_added += 1

                    if len(selected) >= K:
                        break

            if len(selected) >= K:
                break

    replay = [
        [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
        for b in selected
    ]

    reference = json.loads(r.boxes)
    reference = [
        [
            int(b["x1"]), int(b["y1"]),
            int(b["x2"]), int(b["y2"]),
        ]
        for b in reference
    ]

    if actual != replay:
        raise AssertionError(
            f"Replay mismatch: image {idx}"
        )

    if actual != reference:
        raise AssertionError(
            f"IoU experiment geometry mismatch: image {idx}"
        )

    if len(actual) != K:
        raise AssertionError(
            f"Incomplete layout: image {idx}"
        )

    union_area = float(
        masks.sum(dim=0).clamp(0, 1).mean().item()
    )

    records.append({
        "image_index": idx,
        "image": r.image,
        "height": H,
        "width": W,
        "seed": seed,
        "random_trials": trials,
        "random_rejections": rejected,
        "random_satellites_placed": random_placed,
        "fallback_triggered": int(fallback_triggered),
        "fallback_satellites_added": fallback_added,
        "final_region_count": len(actual),
        "actual_union_area": union_area,
        "geometry_match": 1,
        "boxes": json.dumps(actual),
    })

result = pd.DataFrame(records)
assert len(result) == 500
assert result["geometry_match"].eq(1).all()
assert result["final_region_count"].eq(4).all()

result.to_csv(OUT, index=False)

print("=" * 65)
print("PLACEMENT FALLBACK AUDIT")
print("=" * 65)
print("Images:", len(result))
print("Geometry matches:", int(result["geometry_match"].sum()))
print("Complete layouts:", int(result["final_region_count"].eq(4).sum()))

print("\n========== RANDOM PLACEMENT ==========")
print("Mean random trials:", round(result["random_trials"].mean(), 4))
print("Median random trials:", result["random_trials"].median())
print("Maximum random trials:", result["random_trials"].max())
print("Total rejected candidates:", int(result["random_rejections"].sum()))
print("Images with rejected candidates:",
      int(result["random_rejections"].gt(0).sum()))

print("\n========== FALLBACK ==========")
print("Fallback-triggered images:",
      int(result["fallback_triggered"].sum()))
print("Fallback satellites added:",
      int(result["fallback_satellites_added"].sum()))
print("Incomplete layouts:",
      int(result["final_region_count"].ne(4).sum()))

print("\n========== AREA ==========")
print("Mean actual area:", round(result["actual_union_area"].mean(), 8))

print("\nSaved:", OUT)
print("AUDIT COMPLETE")
