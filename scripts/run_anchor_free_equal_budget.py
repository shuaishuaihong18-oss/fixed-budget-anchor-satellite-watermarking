import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
from PIL import Image

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform
from watermark_anything.redundant_region import (
    RegionBox,
    boxes_to_masks,
    select_regions,
    detect_confidence_voting,
    bit_accuracy,
    recovery_31_32,
    _area_to_side,
)
from scripts.run_fixed_budget_exp import (
    apply_attack,
    original_wam_decode,
    iter_images,
    psnr01,
    ssim01,
)

SEED = 3
LIMIT = 500
BUDGET = 0.20
K = 4
ANCHOR_RATIO = 0.40

ATTACKS = [
    "none",
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]

ROOT = REPO / "outputs/revision_202610"
MAIN_CSV = ROOT / "raw/coco500_clean_allattacks/results.csv"
OUT_DIR = ROOT / "anchor_equal_budget_20261028"
OUT_CSV = OUT_DIR / "anchor_free_equal_budget_results.csv"


def box_record(box):
    return {
        "x1": int(box.x1),
        "y1": int(box.y1),
        "x2": int(box.x2),
        "y2": int(box.y2),
    }


def intersection_area(a, b):
    w = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    h = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return w * h


def center_reference_box(img_pt):
    """Reconstruct the exact center-anchor geometry used by Hybrid."""
    _, _, H, W = img_pt.shape
    side = _area_to_side(H, W, BUDGET * ANCHOR_RATIO)
    x1 = max(0, (W - side) // 2)
    y1 = max(0, (H - side) // 2)
    return RegionBox(x1, y1, x1 + side, y1 + side, 1.0)


def select_anchor_free_regions(
    img_pt,
    seed,
    num_regions=3,
    max_trials=50000,
):
    """
    Equal-budget control:
      - 3 enlarged satellite regions;
      - each targets BUDGET / 3 of the image area;
      - no region may intersect the center-anchor reference box;
      - satellite regions must not overlap one another.
    """
    _, _, H, W = img_pt.shape
    center = center_reference_box(img_pt)
    side = _area_to_side(H, W, BUDGET / num_regions)

    if side > min(H, W):
        raise ValueError(f"Satellite side {side} exceeds image dimensions")

    rng = random.Random(seed)
    selected = []

    def valid(box):
        if intersection_area(box, center) != 0:
            return False
        return all(
            intersection_area(box, other) == 0
            for other in selected
        )

    # Deterministic random placement.
    trials = 0
    while len(selected) < num_regions and trials < max_trials:
        trials += 1
        x1 = rng.randint(0, W - side)
        y1 = rng.randint(0, H - side)
        box = RegionBox(x1, y1, x1 + side, y1 + side, 0.0)

        if valid(box):
            selected.append(box)

    # Deterministic grid fallback.
    if len(selected) < num_regions:
        step = max(1, side // 8)

        for y1 in range(0, H - side + 1, step):
            for x1 in range(0, W - side + 1, step):
                box = RegionBox(
                    x1, y1, x1 + side, y1 + side, 0.0
                )
                if valid(box):
                    selected.append(box)

                    if len(selected) == num_regions:
                        break

            if len(selected) == num_regions:
                break

    if len(selected) != num_regions:
        raise RuntimeError(
            f"Could not place {num_regions} non-overlapping "
            f"satellites outside the center region; "
            f"placed={len(selected)}"
        )

    masks = boxes_to_masks(
        selected, H, W, img_pt.device, img_pt.dtype
    )

    return masks, selected, center


def load_main_reference():
    if not MAIN_CSV.exists():
        raise FileNotFoundError(MAIN_CSV)

    messages = {}
    rows = {}
    attacks_by_image = {}

    with MAIN_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] != "hybrid":
                continue

            image = r["image"]
            attack = r["attack"]
            key = (image, attack)

            if key in rows:
                raise ValueError(f"Duplicate main result: {key}")

            msg = r["message"]
            if len(msg) != 32 or set(msg) - {"0", "1"}:
                raise ValueError(f"Invalid message for {image}: {msg}")

            if image in messages and messages[image] != msg:
                raise ValueError(f"Inconsistent messages for {image}")

            messages[image] = msg
            rows[key] = r
            attacks_by_image.setdefault(image, set()).add(attack)

    expected = LIMIT * len(ATTACKS)

    if len(rows) != expected:
        raise ValueError(f"Expected {expected} main rows, got {len(rows)}")

    if len(messages) != LIMIT:
        raise ValueError(
            f"Expected {LIMIT} images, got {len(messages)}"
        )

    for image, attacks in attacks_by_image.items():
        if attacks != set(ATTACKS):
            raise ValueError(
                f"Incomplete attacks for {image}: {sorted(attacks)}"
            )

    return messages, rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_CSV.exists():
        raise FileExistsError(
            f"Output already exists: {OUT_CSV}. "
            "Inspect or archive it before rerunning."
        )

    messages, main_rows = load_main_reference()

    paths = iter_images(REPO / "data/eval/coco500_clean", LIMIT)
    names = [p.name for p in paths]

    if len(paths) != LIMIT or len(set(names)) != LIMIT:
        raise ValueError("Image list is incomplete or contains duplicates")

    if set(names) != set(messages):
        raise ValueError("Image manifest differs from the main experiment")

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading frozen WAM model...")
    wam = load_model_from_checkpoint(
        str(REPO / "checkpoints/params.json"),
        str(REPO / "checkpoints/wam_mit.pth"),
    ).to(device).eval()

    fields = [
        "image",
        "message",
        "method",
        "attack",
        "configured_budget",
        "anchor_ratio",
        "actual_area",
        "anchor_area",
        "satellite_union_area",
        "anchor_satellite_overlap_area",
        "center_reference_overlap_area",
        "anchor_box",
        "center_reference_box",
        "satellite_boxes",
        "psnr",
        "ssim",
        "regions_detected",
        "orig_bitacc",
        "vote_bitacc",
        "orig_recovery31",
        "vote_recovery31",
    ]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with torch.no_grad():
            for idx, path in enumerate(paths):
                with Image.open(path) as im:
                    img = default_transform(
                        im.convert("RGB")
                    ).unsqueeze(0).to(device)

                msg_string = messages[path.name]
                msg = torch.tensor(
                    [[int(bit) for bit in msg_string]],
                    dtype=torch.float32,
                    device=device,
                )

                # Embed the same message once for both conditions.
                full_w = wam.embed(img, msg)["imgs_w"]

                # Condition A: original Hybrid allocation.
                masks, boxes = select_regions(
                    img,
                    strategy="hybrid",
                    total_area=BUDGET,
                    num_regions=K,
                    seed=SEED + idx,
                    anchor_ratio=ANCHOR_RATIO,
                )

                if len(boxes) != K or masks.shape[0] != K:
                    raise ValueError(
                        f"Unexpected Hybrid region count for {path.name}"
                    )

                anchor_mask = masks[0:1]
                satellite_union = (
                    masks[1:].sum(dim=0, keepdim=True).clamp(0, 1)
                )
                hybrid_union = (
                    masks.sum(dim=0, keepdim=True).clamp(0, 1)
                )

                center_box = center_reference_box(img)

                if box_record(boxes[0]) != box_record(center_box):
                    raise ValueError(
                        f"Hybrid center box mismatch: {path.name}"
                    )

                overlap_area = float(
                    (anchor_mask * satellite_union).mean().item()
                )

                hybrid_img = (
                    full_w * hybrid_union
                    + img * (1.0 - hybrid_union)
                )

                # Condition B: 3 enlarged satellites, no center watermark.
                free_masks, free_boxes, free_center = (
                    select_anchor_free_regions(
                        img,
                        seed=SEED + idx,
                        num_regions=3,
                    )
                )

                free_union = (
                    free_masks.sum(dim=0, keepdim=True).clamp(0, 1)
                )

                center_mask = boxes_to_masks(
                    [free_center],
                    img.shape[-2],
                    img.shape[-1],
                    img.device,
                    img.dtype,
                )

                center_overlap = float(
                    (free_union * center_mask).mean().item()
                )

                if center_overlap != 0.0:
                    raise ValueError(
                        f"Anchor-free regions intersect center: {path.name}"
                    )

                for i, box in enumerate(free_boxes):
                    for other in free_boxes[i + 1:]:
                        if intersection_area(box, other) != 0:
                            raise ValueError(
                                f"Overlapping satellites: {path.name}"
                            )

                free_img = (
                    full_w * free_union
                    + img * (1.0 - free_union)
                )

                hybrid_area = float(hybrid_union.mean().item())
                free_area = float(free_union.mean().item())

                common_center_box = json.dumps(
                    box_record(center_box), separators=(",", ":")
                )

                conditions = [
                    {
                        "method": "hybrid_control",
                        "watermarked": hybrid_img,
                        "actual_area": hybrid_area,
                        "anchor_area": float(anchor_mask.mean().item()),
                        "satellite_union_area": float(
                            satellite_union.mean().item()
                        ),
                        "anchor_satellite_overlap_area": overlap_area,
                        "center_reference_overlap_area": overlap_area,
                        "anchor_box": common_center_box,
                        "center_reference_box": common_center_box,
                        "satellite_boxes": json.dumps(
                            [box_record(b) for b in boxes[1:]],
                            separators=(",", ":"),
                        ),
                    },
                    {
                        "method": "anchor_free_equal_budget_3sat",
                        "watermarked": free_img,
                        "actual_area": free_area,
                        "anchor_area": 0.0,
                        "satellite_union_area": free_area,
                        "anchor_satellite_overlap_area": "",
                        "center_reference_overlap_area": center_overlap,
                        "anchor_box": "",
                        "center_reference_box": json.dumps(
                            box_record(free_center),
                            separators=(",", ":"),
                        ),
                        "satellite_boxes": json.dumps(
                            [box_record(b) for b in free_boxes],
                            separators=(",", ":"),
                        ),
                    },
                ]

                for condition in conditions:
                    watermarked = condition["watermarked"]
                    quality_psnr = psnr01(watermarked, img)
                    quality_ssim = ssim01(watermarked, img)

                    for attack in ATTACKS:
                        attacked = apply_attack(
                            watermarked,
                            img,
                            attack,
                            seed=SEED + idx,
                        )

                        pred_orig = original_wam_decode(
                            wam, attacked, msg
                        )

                        det = detect_confidence_voting(
                            wam,
                            attacked,
                            vote_mode="proposed",
                        )
                        pred_vote = det["pred_msg"].to(device)

                        row = {
                            "image": path.name,
                            "message": msg_string,
                            "method": condition["method"],
                            "attack": attack,
                            "configured_budget": BUDGET,
                            "anchor_ratio": ANCHOR_RATIO,
                            "actual_area": condition["actual_area"],
                            "anchor_area": condition["anchor_area"],
                            "satellite_union_area": condition[
                                "satellite_union_area"
                            ],
                            "anchor_satellite_overlap_area": condition[
                                "anchor_satellite_overlap_area"
                            ],
                            "center_reference_overlap_area": condition[
                                "center_reference_overlap_area"
                            ],
                            "anchor_box": condition["anchor_box"],
                            "center_reference_box": condition[
                                "center_reference_box"
                            ],
                            "satellite_boxes": condition["satellite_boxes"],
                            "psnr": quality_psnr,
                            "ssim": quality_ssim,
                            "regions_detected": int(
                                det["region_count"].item()
                            ),
                            "orig_bitacc": bit_accuracy(pred_orig, msg),
                            "vote_bitacc": bit_accuracy(pred_vote, msg),
                            "orig_recovery31": recovery_31_32(
                                pred_orig, msg
                            ),
                            "vote_recovery31": recovery_31_32(
                                pred_vote, msg
                            ),
                        }
                        writer.writerow(row)

                f.flush()

                if (idx + 1) % 25 == 0:
                    print(f"Processed {idx + 1}/{LIMIT} images")

    # Final integrity and paired-control audits.
    with OUT_CSV.open(newline="") as f:
        output_rows = list(csv.DictReader(f))

    expected = LIMIT * 2 * len(ATTACKS)
    if len(output_rows) != expected:
        raise ValueError(
            f"Expected {expected} rows, found {len(output_rows)}"
        )

    methods = {
        "hybrid_control",
        "anchor_free_equal_budget_3sat",
    }

    for method in methods:
        subset = [r for r in output_rows if r["method"] == method]
        if len(subset) != LIMIT * len(ATTACKS):
            raise ValueError(f"Incomplete rows for {method}")

        keys = {(r["image"], r["attack"]) for r in subset}
        expected_keys = {
            (image, attack)
            for image in names
            for attack in ATTACKS
        }

        if keys != expected_keys:
            raise ValueError(f"Incomplete image-attack pairs for {method}")

    control_rows = {
        (r["image"], r["attack"]): r
        for r in output_rows
        if r["method"] == "hybrid_control"
    }

    metric_cols = [
        "actual_area",
        "orig_bitacc",
        "vote_bitacc",
        "orig_recovery31",
        "vote_recovery31",
    ]

    max_diffs = {c: 0.0 for c in metric_cols}

    for key, new_row in control_rows.items():
        ref = main_rows[key]

        for col in metric_cols:
            diff = abs(float(new_row[col]) - float(ref[col]))
            max_diffs[col] = max(max_diffs[col], diff)

    if any(v > 1e-5 for v in max_diffs.values()):
        raise RuntimeError(
            f"Hybrid control failed reproducibility audit: {max_diffs}"
        )

    # One clean row per image is sufficient for the area summary.
    clean_rows = {
        r["method"]: [
            float(x["actual_area"])
            for x in output_rows
            if x["method"] == r["method"] and x["attack"] == "none"
        ]
        for r in output_rows
        if r["attack"] == "none"
    }

    # Deduplicate the dictionary keys without changing the values.
    area_means = {
        method: sum(
            float(r["actual_area"])
            for r in output_rows
            if r["method"] == method and r["attack"] == "none"
        ) / LIMIT
        for method in methods
    }

    max_center_overlap = max(
        float(r["center_reference_overlap_area"])
        for r in output_rows
        if r["method"] == "anchor_free_equal_budget_3sat"
    )

    area_gap = abs(
        area_means["hybrid_control"]
        - area_means["anchor_free_equal_budget_3sat"]
    )

    print("\n========== EQUAL-BUDGET ABLATION COMPLETE ==========")
    print("Images:", LIMIT)
    print("Rows:", len(output_rows))
    print("Expected rows:", expected)
    print("Output:", OUT_CSV)

    print("\n========== MEAN ACTUAL AREA ==========")
    for method, mean_area in sorted(area_means.items()):
        print(f"{method}: {mean_area:.6f} ({mean_area * 100:.3f}%)")

    print("Absolute mean area gap:", f"{area_gap:.6f}")
    print("Maximum center-reference overlap:", max_center_overlap)

    print("\n========== HYBRID CONTROL AUDIT ==========")
    for col, diff in max_diffs.items():
        print(f"{col}: max_abs_diff={diff:.10f}")

    if area_gap > 0.005:
        raise RuntimeError(
            "Actual areas differ by more than 0.5 percentage points. "
            "Do not interpret the equal-budget comparison yet."
        )

    if max_center_overlap != 0.0:
        raise RuntimeError("Center-reference exclusion audit failed")

    print("PASS: Hybrid control matches the existing main results.")
    print("PASS: Satellite regions avoid the center reference box.")
    print("PASS: Satellite regions are mutually non-overlapping.")
    print("PASS: Mean actual area gap is within 0.5 percentage points.")
    print("No existing main experiment files were modified.")


if __name__ == "__main__":
    main()
