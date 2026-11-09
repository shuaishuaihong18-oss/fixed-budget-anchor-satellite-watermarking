import csv
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

SENS_PATH = (
    REPO / "outputs/revision_202610/"
    "iou_sensitivity_full_20261028/run_iou_sensitivity.py"
)
spec = importlib.util.spec_from_file_location("sens_runtime", SENS_PATH)
sens = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sens
spec.loader.exec_module(sens)

base = sens.base

from scripts.run_fixed_budget_exp import apply_attack
from watermark_anything import redundant_region as rr

OUT = Path(__file__).resolve().parent
RESULT = OUT / "runtime_cost_results.csv"
SUMMARY = OUT / "runtime_cost_summary_by_attack.csv"
REPORT = OUT / "runtime_cost_summary.json"
META = OUT / "runtime_cost_metadata.json"

for p in (RESULT, SUMMARY, REPORT, META):
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite: {p}")

SEED = sens.SEED
LIMIT = sens.LIMIT
ATTACKS = list(sens.ATTACKS)
IOU = 0.03

BUDGET = 0.20
K = 4
ANCHOR_RATIO = 0.40
MAX_TRIALS = 1000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
if DEVICE.type != "cuda":
    raise RuntimeError("CUDA is unavailable; refusing a CPU timing run.")

def sync():
    torch.cuda.synchronize(DEVICE)

def timed(fn):
    sync()
    t0 = time.perf_counter()
    result = fn()
    sync()
    return result, (time.perf_counter() - t0) * 1000.0

def coords(box):
    return tuple(int(getattr(box, k))
                 for k in ("x1", "y1", "x2", "y2"))

def normalize_union_mask(mask, k, h, w, dtype):
    mask = mask.to(device=DEVICE, dtype=dtype)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[0] == k and k > 1:
            mask = mask.sum(dim=0, keepdim=True).clamp(0, 1)
            mask = mask.unsqueeze(0)
        elif mask.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
    elif mask.ndim == 4:
        if mask.shape[0] == k and k > 1:
            mask = mask.sum(dim=0, keepdim=True).clamp(0, 1)
    else:
        raise ValueError(f"Unexpected mask dimensions: {mask.ndim}")

    if tuple(mask.shape) != (1, 1, h, w):
        raise ValueError(f"Unexpected union-mask shape: {tuple(mask.shape)}")
    return mask.clamp(0, 1)

def measure_detector(wam, fn):
    """Wall time plus CUDA-event time spent inside wam.detect()."""
    original_detect = wam.detect
    had_instance_override = "detect" in wam.__dict__
    old_instance_value = wam.__dict__.get("detect")

    events = []
    cpu_detect_ms = []

    def wrapped_detect(*args, **kwargs):
        if DEVICE.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_detect(*args, **kwargs)
            end.record()
            events.append((start, end))
            return result

        t0 = time.perf_counter()
        result = original_detect(*args, **kwargs)
        cpu_detect_ms.append((time.perf_counter() - t0) * 1000)
        return result

    wam.detect = wrapped_detect
    try:
        output, wall_ms = timed(fn)
    finally:
        if had_instance_override:
            wam.detect = old_instance_value
        else:
            delattr(wam, "detect")

    sync()
    if not events and not cpu_detect_ms:
        raise RuntimeError(
            "The detector call did not invoke wam.detect(); "
            "cannot isolate its forward-pass time."
        )

    detect_ms = (
        sum(start.elapsed_time(end) for start, end in events)
        if DEVICE.type == "cuda"
        else sum(cpu_detect_ms)
    )

    # Approximation: wall time minus GPU time inside wam.detect().
    # This is not a pure Python-only measurement.
    non_forward_ms = max(0.0, wall_ms - detect_ms)

    return output, wall_ms, detect_ms, non_forward_ms

def bit_metrics(pred, msg):
    return {
        "bitacc": float(rr.bit_accuracy(pred, msg)),
        "recovery31": float(rr.recovery_31_32(pred, msg)),
    }

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("=" * 72)
print("RUNTIME COST AUDIT")
print("=" * 72)
print("Device:", torch.cuda.get_device_name(DEVICE))
print("Images:", LIMIT)
print("Attacks:", ATTACKS)
print("Geometry IoU:", IOU)

messages, main_rows = base.load_main_reference()
paths = list(
    sens.iter_images(REPO / "data/eval/coco500_clean", LIMIT)
)
if len(paths) != LIMIT:
    raise ValueError(f"Expected {LIMIT} images, got {len(paths)}")
if {p.name for p in paths} != set(messages):
    raise ValueError("Image/message mapping differs from main experiment")

geometry = sens.load_geometry(paths)

torch.manual_seed(SEED)
random.seed(SEED)

wam = sens.load_model_from_checkpoint(
    str(REPO / "checkpoints/params.json"),
    str(REPO / "checkpoints/wam_mit.pth"),
).to(DEVICE).eval()

# Warm-up only; these calls are excluded from all reported timings.
with Image.open(paths[0]) as im:
    warm_img = sens.default_transform(
        im.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

warm_msg = torch.tensor(
    [[int(b) for b in messages[paths[0].name]]],
    dtype=torch.float32, device=DEVICE,
)

with torch.no_grad():
    _ = wam.embed(warm_img, warm_msg)
    _ = rr.select_hybrid_regions(
        warm_img, total_area=BUDGET, num_regions=K,
        anchor_ratio=ANCHOR_RATIO, max_iou=IOU,
        seed=SEED, max_trials=MAX_TRIALS,
    )
    _ = base.original_wam_decode(wam, warm_img, warm_msg)
    _ = base.detect_confidence_voting(
        wam, warm_img, vote_mode="proposed"
    )
sync()

rows = []
geometry_matches = 0
baseline_mismatches = 0
max_baseline_diff = 0.0

with torch.no_grad():
    for idx, path in enumerate(paths):
        with Image.open(path) as im:
            img = sens.default_transform(
                im.convert("RGB")
            ).unsqueeze(0).to(DEVICE)

        h, w = img.shape[-2:]
        msg = torch.tensor(
            [[int(b) for b in messages[path.name]]],
            dtype=torch.float32, device=DEVICE,
        )

        g = geometry[(idx, IOU)]
        expected_boxes = [
            base.parse_box(b) for b in json.loads(g["boxes"])
        ]

        (selection, selection_ms) = timed(
            lambda: rr.select_hybrid_regions(
                img, total_area=BUDGET, num_regions=K,
                anchor_ratio=ANCHOR_RATIO, max_iou=IOU,
                seed=SEED + idx, max_trials=MAX_TRIALS,
            )
        )
        mask, selected_boxes = selection

        if [coords(b) for b in selected_boxes] != [
            coords(b) for b in expected_boxes
        ]:
            raise ValueError(
                f"Geometry mismatch at image {idx}: {path.name}"
            )
        geometry_matches += 1

        union_mask = normalize_union_mask(
            mask, len(selected_boxes), h, w, img.dtype
        )
        area = float(union_mask.mean().item())
        if abs(area - float(g["actual_area"])) > 1e-6:
            raise ValueError(f"Area mismatch: {path.name}")

        full_w, embed_ms = timed(
            lambda: wam.embed(img, msg)["imgs_w"]
        )

        watermarked, compose_ms = timed(
            lambda: full_w * union_mask + img * (1.0 - union_mask)
        )

        # Verify the measured composition matches the existing main runner.
        reference_condition = base.make_condition(
            "hybrid", "center", expected_boxes,
            img, full_w, base.center_reference_box_hw(h, w),
        )
        compose_diff = float(
            (watermarked - reference_condition["watermarked"])
            .abs().max().item()
        )
        if compose_diff > 1e-6:
            raise ValueError(
                f"Composition mismatch at {path.name}: {compose_diff}"
            )

        pipeline_ms = selection_ms + embed_ms + compose_ms

        for attack in ATTACKS:
            attacked = apply_attack(
                watermarked, img, attack, seed=SEED + idx
            )

            pred_orig, orig_wall, orig_gpu, orig_extra = measure_detector(
                wam,
                lambda: base.original_wam_decode(wam, attacked, msg),
            )
            det, vote_wall, vote_gpu, vote_extra = measure_detector(
                wam,
                lambda: base.detect_confidence_voting(
                    wam, attacked, vote_mode="proposed"
                ),
            )
            pred_vote = det["pred_msg"].to(DEVICE)

            om = bit_metrics(pred_orig, msg)
            vm = bit_metrics(pred_vote, msg)

            computed = {
                "orig_bitacc": om["bitacc"],
                "orig_recovery31": om["recovery31"],
                "vote_bitacc": vm["bitacc"],
                "vote_recovery31": vm["recovery31"],
            }
            ref = main_rows[(path.name, attack)]

            diffs = {
                key: abs(value - float(ref[key]))
                for key, value in computed.items()
            }
            row_max_diff = max(diffs.values())
            max_baseline_diff = max(max_baseline_diff, row_max_diff)
            if row_max_diff > 1e-6:
                baseline_mismatches += 1

            rows.append({
                "image_index": idx,
                "image": path.name,
                "attack": attack,
                "attack_seed": SEED + idx,
                "actual_area": area,
                "region_selection_ms": selection_ms,
                "full_wam_embed_ms": embed_ms,
                "mask_composition_ms": compose_ms,
                "embed_pipeline_ms": pipeline_ms,
                "original_decode_total_ms": orig_wall,
                "original_detect_gpu_ms": orig_gpu,
                "original_wall_minus_detect_gpu_ms": orig_extra,
                "voting_decode_total_ms": vote_wall,
                "voting_detect_gpu_ms": vote_gpu,
                "voting_wall_minus_detect_gpu_ms": vote_extra,
                "voting_minus_original_ms": vote_wall - orig_wall,
                **computed,
                "max_metric_diff_vs_main": row_max_diff,
            })

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{LIMIT} images")

df = pd.DataFrame(rows)
expected_rows = LIMIT * len(ATTACKS)
if len(df) != expected_rows:
    raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
if df.duplicated(["image", "attack"]).any():
    raise ValueError("Duplicate image/attack rows")

df.to_csv(RESULT, index=False)

timing_cols = [
    "region_selection_ms",
    "full_wam_embed_ms",
    "mask_composition_ms",
    "embed_pipeline_ms",
    "original_decode_total_ms",
    "original_detect_gpu_ms",
    "original_wall_minus_detect_gpu_ms",
    "voting_decode_total_ms",
    "voting_detect_gpu_ms",
    "voting_wall_minus_detect_gpu_ms",
    "voting_minus_original_ms",
]

def stats(series):
    a = np.asarray(series, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
    }

summary_rows = []
for attack, group in df.groupby("attack", sort=False):
    r = {"attack": attack, "n_images": int(group["image"].nunique())}
    for col in timing_cols:
        for stat_name, value in stats(group[col]).items():
            r[f"{col}_{stat_name}"] = value
    summary_rows.append(r)

pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False)

report = {
    "images": LIMIT,
    "attacks": ATTACKS,
    "result_rows": len(df),
    "geometry_matches": geometry_matches,
    "baseline_metric_mismatches": baseline_mismatches,
    "max_metric_diff_vs_main": max_baseline_diff,
    "timings_ms_over_all_image_attack_pairs": {
        col: stats(df[col]) for col in timing_cols
    },
    "gpu": torch.cuda.get_device_name(DEVICE),
    "torch_version": torch.__version__,
    "iou_threshold": IOU,
    "seed": SEED,
    "interpretation_note": (
        "CUDA event time measures work inside wam.detect(). "
        "Wall time minus this GPU time is an approximate residual, "
        "not a pure Python-only measurement. Timing includes attack "
        "inference but excludes image loading and model initialization."
    ),
}

with open(REPORT, "x") as f:
    json.dump(report, f, indent=2)

source_files = {
    "timing_script": Path(__file__),
    "iou_runner": SENS_PATH,
    "anchor_runner": Path(base.__file__),
    "region_implementation": Path(rr.__file__),
    "attack_implementation": Path(apply_attack.__code__.co_filename),
    "model_params": REPO / "checkpoints/params.json",
    "model_checkpoint": REPO / "checkpoints/wam_mit.pth",
}
metadata = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(DEVICE),
    "source_sha256": {
        name: sha256(path)
        for name, path in source_files.items()
    },
}
with open(META, "x") as f:
    json.dump(metadata, f, indent=2)

print("\n========== FINAL INTEGRITY ==========")
print("Rows:", len(df), "/", expected_rows)
print("Geometry matches:", geometry_matches, "/", LIMIT)
print("Baseline metric mismatches:", baseline_mismatches)
print("Maximum baseline metric difference:", max_baseline_diff)
print("Raw results:", RESULT)
print("Summary:", SUMMARY)
print("Metadata:", META)

if geometry_matches != LIMIT:
    raise RuntimeError("Geometry integrity check failed.")
if baseline_mismatches:
    print("WARNING: Metrics differ from the original main results.")
else:
    print("PASS: All measured detector metrics reproduce the main results.")
