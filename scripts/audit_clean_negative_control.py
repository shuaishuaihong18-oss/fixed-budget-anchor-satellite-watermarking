import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

import watermark_anything.redundant_region as rr

SOURCE = (
    REPO / "outputs/revision_202610/"
    "iou_sensitivity_full_20261028/run_iou_sensitivity.py"
)

OUT_DIR = REPO / "outputs/revision_202610/false_positive_audit_20261029"
OUT_CSV = OUT_DIR / "clean_negative_control_results.csv"
OUT_JSON = OUT_DIR / "clean_negative_control_summary.json"

if OUT_CSV.exists() or OUT_JSON.exists():
    raise FileExistsError(
        "Output already exists. Preserve the existing evidence; "
        "do not overwrite it."
    )

spec = importlib.util.spec_from_file_location(
    "iou_sensitivity_negative_control", SOURCE
)
sens = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sens
spec.loader.exec_module(sens)

base = sens.base
messages, main_rows = base.load_main_reference()

image_paths = list(
    sens.iter_images(REPO / "data/eval/coco500_clean", sens.LIMIT)
)

if len(image_paths) != 500:
    raise ValueError(f"Expected 500 images, got {len(image_paths)}")

if len({p.name for p in image_paths}) != 500:
    raise ValueError("Duplicate image filenames")

if {p.name for p in image_paths} != set(messages):
    raise ValueError("Image/message mapping mismatch")

for p in image_paths:
    if len(messages[p.name]) != 32 or set(messages[p.name]) - {"0", "1"}:
        raise ValueError(f"Invalid 32-bit message: {p.name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(sens.SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(sens.SEED)

checkpoint = REPO / "checkpoints/wam_mit.pth"
params = REPO / "checkpoints/params.json"

print("=" * 72)
print("CLEAN-IMAGE NEGATIVE-CONTROL AUDIT")
print("=" * 72)
print("Images:", len(image_paths))
print("Device:", device)
print("Checkpoint:", checkpoint)
print("Detector: existing confidence-weighted voting")
print("Grid=8, min_cell_score=0.08, mask_threshold=0.5")
print("min_pixels=16, vote_mode=proposed")
print("No embedding, training, or image attacks in this experiment.")

wam = sens.load_model_from_checkpoint(
    str(params), str(checkpoint)
).to(device).eval()

fields = [
    "image",
    "assigned_message",
    "predicted_message",
    "bit_errors",
    "bitacc_vs_assigned_message",
    "recovery31_vs_assigned_message",
    "exact_match_vs_assigned_message",
    "region_count",
    "mean_mask_prob",
    "mask_fraction_ge_0_5",
    "mean_bit_margin",
]

rows = []
partial = OUT_CSV.with_suffix(".partial.csv")

if partial.exists():
    raise FileExistsError(
        f"Partial output already exists: {partial}"
    )

with partial.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    with torch.no_grad():
        for i, path in enumerate(image_paths):
            with Image.open(path) as im:
                img = sens.default_transform(
                    im.convert("RGB")
                ).unsqueeze(0).to(device)

            target_string = messages[path.name]
            target = torch.tensor(
                [[int(b) for b in target_string]],
                dtype=torch.float32,
                device=device,
            )

            # Run the unchanged detector on the original, unwatermarked image.
            det = rr.detect_confidence_voting(
                wam,
                img,
                mask_threshold=0.5,
                grid=8,
                min_cell_score=0.08,
                min_pixels=16,
                vote_mode="proposed",
            )

            pred = det["pred_msg"].int()
            target_int = target.int()
            errors = int(
                (pred != target_int).sum().item()
            )
            bitacc = float(
                (pred == target_int).float().mean().item()
            )

            mask_prob = det["mask_prob"].float()
            fused_score = det["fused_score"].float()

            row = {
                "image": path.name,
                "assigned_message": target_string,
                "predicted_message": "".join(
                    str(int(x))
                    for x in pred.reshape(-1).tolist()
                ),
                "bit_errors": errors,
                "bitacc_vs_assigned_message": bitacc,
                "recovery31_vs_assigned_message": int(errors <= 1),
                "exact_match_vs_assigned_message": int(errors == 0),
                "region_count": int(
                    det["region_count"].reshape(-1)[0].item()
                ),
                "mean_mask_prob": float(mask_prob.mean().item()),
                "mask_fraction_ge_0_5": float(
                    (mask_prob >= 0.5).float().mean().item()
                ),
                "mean_bit_margin": float(
                    ((fused_score - 0.5).abs() * 2).mean().item()
                ),
            }

            writer.writerow(row)
            rows.append(row)

            if (i + 1) % 25 == 0:
                f.flush()
                print(f"Processed {i + 1}/500 images")

if len(rows) != 500:
    raise RuntimeError(f"Expected 500 rows, got {len(rows)}")

# Promote the completed CSV only after all 500 rows have been written.
partial.rename(OUT_CSV)

def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]

    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        / denom
    )
    return [max(0.0, center - half), min(1.0, center + half)]

n_exact = sum(r["exact_match_vs_assigned_message"] for r in rows)
n_near = sum(r["recovery31_vs_assigned_message"] for r in rows)
n_zero = sum(r["region_count"] == 0 for r in rows)

summary = {
    "experiment": "clean_image_negative_control",
    "n_images": len(rows),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest(),
    "settings": {
        "grid": 8,
        "min_cell_score": 0.08,
        "mask_threshold": 0.5,
        "min_pixels": 16,
        "vote_mode": "proposed",
    },
    "exact_match_count": n_exact,
    "exact_match_rate": n_exact / len(rows),
    "exact_match_wilson_ci95": wilson_ci(n_exact, len(rows)),
    "recovery31_count": n_near,
    "recovery31_rate": n_near / len(rows),
    "recovery31_wilson_ci95": wilson_ci(n_near, len(rows)),
    "mean_bitacc_vs_assigned_message": sum(
        r["bitacc_vs_assigned_message"] for r in rows
    ) / len(rows),
    "mean_region_count": sum(
        r["region_count"] for r in rows
    ) / len(rows),
    "zero_region_count": n_zero,
    "mean_mask_probability": sum(
        r["mean_mask_prob"] for r in rows
    ) / len(rows),
    "mean_mask_fraction_ge_0_5": sum(
        r["mask_fraction_ge_0_5"] for r in rows
    ) / len(rows),
    "interpretation": (
        "Negative-control agreement with the per-image assigned message. "
        "Recovery31 means at most one bit error. These are not a generic "
        "system-level false-positive rate because the detector has no "
        "independent watermark-acceptance decision."
    ),
    "raw_results": str(OUT_CSV),
}

OUT_JSON.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)

print("\n========== NEGATIVE-CONTROL SUMMARY ==========")
print("Images:", len(rows))
print(f"Exact matches: {n_exact}/500 = {n_exact / 500:.6f}")
print(f"Recovery31 matches: {n_near}/500 = {n_near / 500:.6f}")
print(
    "Mean BitAcc against assigned message:",
    f"{summary['mean_bitacc_vs_assigned_message']:.6f}",
)
print("Mean accepted candidate regions:", f"{summary['mean_region_count']:.3f}")
print("Images with zero accepted regions:", n_zero)
print("Mean mask probability:", f"{summary['mean_mask_probability']:.6f}")
print("CSV:", OUT_CSV)
print("Summary:", OUT_JSON)
print("\nAUDIT COMPLETE")
