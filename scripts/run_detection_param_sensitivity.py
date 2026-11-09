import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Reuse the validated data loading, geometry and model interfaces.
SOURCE = (
    REPO / "outputs/revision_202610/"
    "iou_sensitivity_full_20261028/run_iou_sensitivity.py"
)

spec = importlib.util.spec_from_file_location(
    "iou_sensitivity_reuse", SOURCE
)
sens = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sens
spec.loader.exec_module(sens)

base = sens.base
OUT_DIR = Path(__file__).resolve().parent
SEED = sens.SEED
LIMIT = sens.LIMIT
ATTACKS = list(sens.ATTACKS)
BASELINE = "grid8_score008"
GEOMETRY_IOU = 0.03

CONFIGS = [
    ("grid4_score008", 4, 0.08),
    ("grid8_score008", 8, 0.08),
    ("grid16_score008", 16, 0.08),
    ("grid8_score004", 8, 0.04),
    ("grid8_score012", 8, 0.12),
]

FIELDS = [
    "image_index", "image", "message", "attack", "attack_seed",
    "config", "grid", "min_cell_score", "mask_threshold",
    "vote_mode", "region_count",
    "vote_bitacc", "vote_recovery31",
]


def scalar_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got {value.shape}")
        return float(value.detach().cpu().item())
    return float(value)


def scalar_int(value):
    return int(round(scalar_float(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--output",
        default=str(OUT_DIR / "detection_param_sensitivity_results.csv"),
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= LIMIT:
        raise ValueError(f"--limit must be between 1 and {LIMIT}")

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. "
            "Choose a new filename; do not overwrite evidence."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DETECTION PARAMETER SENSITIVITY")
    print("=" * 72)

    messages, main_rows = base.load_main_reference()

    all_paths = list(
        sens.iter_images(REPO / "data/eval/coco500_clean", LIMIT)
    )
    if len(all_paths) != LIMIT:
        raise ValueError(f"Expected {LIMIT} image paths, got {len(all_paths)}")
    if {p.name for p in all_paths} != set(messages):
        raise ValueError("Image paths do not match the Hybrid reference")

    geometry = sens.load_geometry(all_paths)
    paths = all_paths[:args.limit]

    expected = len(paths) * len(ATTACKS) * len(CONFIGS)
    print("Images:", len(paths))
    print("Attacks:", ATTACKS)
    print("Configurations:", CONFIGS)
    print("Expected result rows:", expected)

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Loading frozen WAM model...")

    wam = sens.load_model_from_checkpoint(
        str(REPO / "checkpoints/params.json"),
        str(REPO / "checkpoints/wam_mit.pth"),
    ).to(device).eval()

    written = 0
    baseline_max_diff = {
        "vote_bitacc": 0.0,
        "vote_recovery31": 0.0,
    }
    baseline_checked = 0

    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        with torch.no_grad():
            for idx, path in enumerate(paths):
                from PIL import Image
                from watermark_anything.data.transforms import default_transform

                with Image.open(path) as im:
                    img = default_transform(
                        im.convert("RGB")
                    ).unsqueeze(0).to(device)

                H, W = img.shape[-2:]
                g = geometry[(idx, GEOMETRY_IOU)]

                if (H, W) != (int(g["height"]), int(g["width"])):
                    raise ValueError(
                        f"Geometry dimension mismatch: {path.name}"
                    )

                message = messages[path.name]
                msg = torch.tensor(
                    [[int(bit) for bit in message]],
                    dtype=torch.float32,
                    device=device,
                )

                # Generate one full watermarked image per source image.
                full_w = wam.embed(img, msg)["imgs_w"]

                # Use the exact validated IoU=0.03 Hybrid geometry.
                mask = torch.zeros(
                    (1, 1, H, W), device=device, dtype=img.dtype
                )
                for raw_box in json.loads(g["boxes"]):
                    box = base.parse_box(raw_box)
                    mask[:, :, box.y1:box.y2, box.x1:box.x2] = 1.0

                watermarked = full_w * mask + img * (1.0 - mask)
                attack_seed = SEED + idx

                for attack in ATTACKS:
                    # Apply each attack only once. Every detector
                    # configuration receives this same attacked tensor.
                    attacked = sens.apply_attack(
                        watermarked, img, attack, seed=attack_seed
                    )

                    for config, grid, min_score in CONFIGS:
                        det = sens.detect_confidence_voting(
                            wam,
                            attacked,
                            grid=grid,
                            min_cell_score=min_score,
                            vote_mode="proposed",
                        )

                        pred = det["pred_msg"].to(device)
                        vote_bitacc = scalar_float(
                            sens.bit_accuracy(pred, msg)
                        )
                        vote_recovery31 = scalar_float(
                            sens.recovery_31_32(pred, msg)
                        )

                        if "region_count" not in det:
                            raise KeyError(
                                "Detector did not return region_count; "
                                f"available keys: {list(det.keys())}"
                            )

                        regions = scalar_int(det["region_count"])

                        writer.writerow({
                            "image_index": idx,
                            "image": path.name,
                            "message": message,
                            "attack": attack,
                            "attack_seed": attack_seed,
                            "config": config,
                            "grid": grid,
                            "min_cell_score": min_score,
                            "mask_threshold": 0.5,
                            "vote_mode": "proposed",
                            "region_count": regions,
                            "vote_bitacc": vote_bitacc,
                            "vote_recovery31": vote_recovery31,
                        })
                        written += 1

                        # Verify the default detector against the
                        # original per-image Hybrid reference.
                        if config == BASELINE:
                            ref = main_rows[(path.name, attack)]
                            for metric, value in [
                                ("vote_bitacc", vote_bitacc),
                                ("vote_recovery31", vote_recovery31),
                            ]:
                                diff = abs(value - float(ref[metric]))
                                baseline_max_diff[metric] = max(
                                    baseline_max_diff[metric], diff
                                )
                            baseline_checked += 1

                f.flush()

                if (idx + 1) % 10 == 0 or idx + 1 == len(paths):
                    print(
                        f"Processed {idx + 1}/{len(paths)} images; "
                        f"rows={written}",
                        flush=True,
                    )

    if written != expected:
        raise RuntimeError(
            f"Row count mismatch: expected {expected}, got {written}"
        )

    print("\n========== INTEGRITY CHECK ==========")
    print("Rows written:", written)
    print("Expected rows:", expected)
    print("Default baseline comparisons:", baseline_checked)
    print("Maximum baseline differences:", baseline_max_diff)

    if max(baseline_max_diff.values()) <= 1e-4:
        print("PASS: Default detector reproduces the Hybrid reference.")
    else:
        print(
            "WARNING: Default detector differs from the Hybrid reference. "
            "Investigate before interpreting the sensitivity results."
        )

    print("Output:", output)
    print("No existing experiment files were modified.")


if __name__ == "__main__":
    main()
