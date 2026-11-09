#!/usr/bin/env python3
"""Matched attacked component-removal audit; run from repository root."""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wilcoxon

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
import watermark_anything.redundant_region as rr
from scripts.run_fixed_budget_exp import apply_attack, original_wam_decode

SOURCE = REPO / "outputs/revision_202610/iou_sensitivity_full_20261028/run_iou_sensitivity.py"
REFERENCE = REPO / "outputs/revision_202610/iou_sensitivity_full_20261028/iou_sensitivity_results.csv"
IMAGE_DIR = REPO / "data/eval/coco500_clean"
CHECKPOINT = REPO / "checkpoints/wam_mit.pth"
PARAMS = REPO / "checkpoints/params.json"
IOU = 0.03
ATTACKS = ("erase40", "randomcrop50", "cornercrop50", "splicing20")
METRICS = ("orig_bitacc", "vote_bitacc", "orig_recovery31", "vote_recovery31")

# Region 0 is the anchor; regions 1, 2, 3 are satellites.
CONDITIONS = [
    ("hybrid_all", (0, 1, 2, 3)),
    ("anchor_removed", (1, 2, 3)),
]
for s in (1, 2, 3):
    CONDITIONS.append((f"remove_S{s}", tuple(i for i in range(4) if i != s)))
for pair in combinations((1, 2, 3), 2):
    CONDITIONS.append(("remove_S" + "_S".join(map(str, pair)), tuple(i for i in range(4) if i not in pair)))
CONDITIONS.extend([("anchor_only", (0,)), ("all_regions_removed", ())])


def load_sensitivity():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    spec = importlib.util.spec_from_file_location("component_attack_sensitivity", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="mergesort")
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty(len(p), dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        raise ValueError("--limit must be between 1 and 500")
    if args.bootstrap < 1000:
        raise ValueError("--bootstrap must be at least 1000")
    for p in (REFERENCE, IMAGE_DIR, CHECKPOINT, PARAMS):
        if not p.exists():
            raise FileNotFoundError(p)

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.csv")
    means_path = out.with_name(out.stem + "_condition_means.csv")
    paired_path = out.with_name(out.stem + "_paired_vs_hybrid.csv")
    meta_path = out.with_name(out.stem + "_summary.json")
    for p in (out, partial, means_path, paired_path, meta_path):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite existing evidence: {p}")

    sens = load_sensitivity()
    base = sens.base
    seed = int(sens.SEED)
    if not set(ATTACKS).issubset(set(base.ATTACKS)):
        raise ValueError(f"Attack mismatch: expected {ATTACKS}, source has {base.ATTACKS}")
    if len(CONDITIONS) != 10:
        raise RuntimeError(f"Expected 10 conditions, got {len(CONDITIONS)}")

    messages, _ = base.load_main_reference()
    all_paths = list(sens.iter_images(IMAGE_DIR, 500))
    if len(all_paths) != 500 or len({p.name for p in all_paths}) != 500:
        raise ValueError("Expected 500 unique source images")
    if {p.name for p in all_paths} != set(messages):
        raise ValueError("Image/message mapping mismatch")
    paths = all_paths[:args.limit]
    geometry = sens.load_geometry(all_paths)

    ref = pd.read_csv(REFERENCE)
    required = {"image", "attack", "max_iou", "actual_area", *METRICS}
    if not required.issubset(ref.columns):
        raise ValueError(f"Reference CSV missing columns: {sorted(required-set(ref.columns))}")
    ref = ref[np.isclose(ref.max_iou.astype(float), IOU) & ref.attack.isin(ATTACKS) & ref.image.isin({p.name for p in paths})].copy()
    if ref.duplicated(["image", "attack"]).any():
        raise ValueError("Duplicate image/attack keys in reference CSV")
    ref = ref.set_index(["image", "attack"]).sort_index()
    expected_ref = args.limit * len(ATTACKS)
    if len(ref) != expected_ref:
        raise ValueError(f"Expected {expected_ref} reference rows, got {len(ref)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_rows = args.limit * len(CONDITIONS) * len(ATTACKS)

    print("=" * 72)
    print("MATCHED ATTACKED COMPONENT-REMOVAL AUDIT")
    print("Images:", args.limit, " Conditions:", len(CONDITIONS), " Attacks:", list(ATTACKS))
    print("Expected rows:", expected_rows, " IoU:", IOU, " Device:", device)
    print("Same attack seed per image across all conditions; frozen WAM; no retraining.")
    print("=" * 72)

    wam = sens.load_model_from_checkpoint(str(PARAMS), str(CHECKPOINT)).to(device).eval()
    fields = [
        "image", "image_index", "message", "condition", "retained_regions", "removed_regions",
        "attack", "attack_seed", "max_iou", "geometry_seed", "max_observed_pair_iou",
        "configured_budget", "anchor_ratio", "actual_area", *METRICS,
        "orig_bit_errors", "vote_bit_errors", "region_count",
    ]
    written = 0
    with partial.open("x", newline="") as f:
        writer = __import__("csv").DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for idx, path in enumerate(paths):
                with Image.open(path) as pil:
                    img = sens.default_transform(pil.convert("RGB")).unsqueeze(0).to(device)
                msg_string = messages[path.name]
                msg = torch.tensor([[int(b) for b in msg_string]], dtype=torch.float32, device=device)
                msg_int = msg.int()
                g = geometry[(idx, IOU)]
                boxes = [base.parse_box(b) for b in json.loads(g["boxes"])]
                if len(boxes) != 4:
                    raise ValueError(f"{path.name}: expected four regions, got {len(boxes)}")
                h, w = img.shape[-2:]
                masks = rr.boxes_to_masks(boxes, h, w, img.device, img.dtype)
                # One full-image embedding per image, as in the original IoU experiment.
                full_w = wam.embed(img, msg)["imgs_w"]
                attack_seed = seed + idx

                for condition, retained in CONDITIONS:
                    if retained:
                        union = masks[list(retained)].sum(dim=0, keepdim=True).clamp(0, 1)
                    else:
                        union = torch.zeros_like(img[:, :1])
                    condition_image = full_w * union + img * (1.0 - union)
                    area = float(union.mean().item())
                    removed = sorted(set(range(4)) - set(retained))

                    for attack in ATTACKS:
                        attacked = apply_attack(condition_image, img, attack, seed=attack_seed)
                        pred_orig = original_wam_decode(wam, attacked, msg).int()
                        det = rr.detect_confidence_voting(
                            wam, attacked, mask_threshold=0.5, grid=8,
                            min_cell_score=0.08, min_pixels=16, vote_mode="proposed",
                        )
                        pred_vote = det["pred_msg"].to(device).int()
                        row = {
                            "image": path.name, "image_index": idx, "message": msg_string,
                            "condition": condition, "retained_regions": ",".join(map(str, retained)),
                            "removed_regions": ",".join(map(str, removed)), "attack": attack,
                            "attack_seed": attack_seed, "max_iou": IOU,
                            "geometry_seed": int(g["seed"]),
                            "max_observed_pair_iou": float(g["max_observed_pair_iou"]),
                            "configured_budget": float(base.BUDGET),
                            "anchor_ratio": float(base.ANCHOR_RATIO), "actual_area": area,
                            "orig_bitacc": rr.bit_accuracy(pred_orig, msg),
                            "vote_bitacc": rr.bit_accuracy(pred_vote, msg),
                            "orig_recovery31": rr.recovery_31_32(pred_orig, msg),
                            "vote_recovery31": rr.recovery_31_32(pred_vote, msg),
                            "orig_bit_errors": int((pred_orig != msg_int).sum().item()),
                            "vote_bit_errors": int((pred_vote != msg_int).sum().item()),
                            "region_count": int(det["region_count"].reshape(-1)[0].item()),
                        }
                        writer.writerow(row)
                        written += 1
                if (idx + 1) % 10 == 0:
                    f.flush()
                    print(f"Processed {idx+1}/{args.limit}; rows={written}/{expected_rows}")

    if written != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, wrote {written}; partial retained: {partial}")
    df = pd.read_csv(partial, keep_default_na=False)
    if len(df) != expected_rows or df.isna().any().any():
        raise RuntimeError("CSV row count or missing-value integrity check failed")
    if df.duplicated(["image", "condition", "attack"]).any():
        raise RuntimeError("Duplicate image/condition/attack rows")
    if df.image.nunique() != args.limit or df.condition.nunique() != 10 or df.attack.nunique() != 4:
        raise RuntimeError("Image/condition/attack coverage mismatch")

    # Exact paired reproduction of the original Hybrid baseline, for every attack.
    hybrid = df[df.condition == "hybrid_all"].set_index(["image", "attack"]).sort_index()
    common = hybrid.index.intersection(ref.index)
    if len(common) != expected_ref:
        raise RuntimeError(f"Hybrid/reference matched rows {len(common)} != {expected_ref}")
    baseline_diffs = {}
    for metric in (*METRICS, "actual_area"):
        diff = (hybrid.loc[common, metric].astype(float) - ref.loc[common, metric].astype(float)).abs()
        baseline_diffs[metric] = float(diff.max())
        if baseline_diffs[metric] > 1e-6:
            raise RuntimeError(f"Hybrid baseline mismatch {metric}: {baseline_diffs[metric]}; partial retained: {partial}")
    print("PASS: Hybrid reproduces IoU=0.03 reference under all four attacks.")
    print("Maximum absolute baseline differences:", baseline_diffs)

    means = df.groupby(["attack", "condition"], sort=False).agg(
        n_images=("image", "nunique"), actual_area=("actual_area", "mean"),
        orig_bitacc=("orig_bitacc", "mean"), vote_bitacc=("vote_bitacc", "mean"),
        orig_recovery31=("orig_recovery31", "mean"), vote_recovery31=("vote_recovery31", "mean"),
        mean_region_count=("region_count", "mean"),
    ).reset_index()

    # Paired image-level comparisons, stratified by attack; one global Holm family (144 tests).
    wide = df.pivot(index=["image", "attack"], columns="condition", values=list(METRICS))
    rng = np.random.default_rng(20261030)
    comparisons = []
    for attack in ATTACKS:
        sub = wide.loc[wide.index.get_level_values("attack") == attack]
        for condition, _ in CONDITIONS:
            if condition == "hybrid_all":
                continue
            for metric in METRICS:
                candidate = sub[(metric, condition)].to_numpy(dtype=float)
                baseline = sub[(metric, "hybrid_all")].to_numpy(dtype=float)
                delta = candidate - baseline
                p_raw = 1.0 if np.allclose(delta, 0) else float(wilcoxon(delta, zero_method="pratt").pvalue)
                boot_idx = rng.integers(0, len(delta), size=(args.bootstrap, len(delta)))
                boot_mean = delta[boot_idx].mean(axis=1)
                comparisons.append({
                    "attack": attack, "condition": condition, "metric": metric,
                    "n_images": len(delta), "mean_diff_candidate_minus_hybrid": float(delta.mean()),
                    "ci95_low": float(np.quantile(boot_mean, 0.025)),
                    "ci95_high": float(np.quantile(boot_mean, 0.975)), "p_raw": p_raw,
                })
    adjusted = holm_adjust([x["p_raw"] for x in comparisons])
    for item, p_adj in zip(comparisons, adjusted):
        item["p_holm_global144"] = float(p_adj)
    paired = pd.DataFrame(comparisons)

    # Promote evidence only after row coverage and Hybrid reproduction pass.
    partial.rename(out)
    means.to_csv(means_path, index=False)
    paired.to_csv(paired_path, index=False)
    meta = {
        "experiment": "matched_attacked_anchor_satellite_component_removal",
        "n_images": args.limit, "rows": len(df),
        "conditions": [name for name, _ in CONDITIONS], "attacks": list(ATTACKS),
        "iou": IOU, "seed_rule": "SEED + image_index, shared across conditions per image",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "baseline_reproduction": "PASS", "baseline_max_abs_differences": baseline_diffs,
        "paired_comparisons": len(comparisons),
        "significant_after_global_holm144": int((paired.p_holm_global144 < 0.05).sum()),
        "bootstrap_replicates": args.bootstrap,
        "raw_results": str(out), "condition_means": str(means_path),
        "paired_statistics": str(paired_path),
        "interpretation_note": "Component removal also reduces retained watermark area; these are operational component-removal ablations, not area-controlled pure position effects.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print("\n========== FINAL INTEGRITY ==========")
    print(f"Rows: {len(df)} / {expected_rows}; images={df.image.nunique()}; conditions={df.condition.nunique()}; attacks={df.attack.nunique()}")
    print(f"Paired comparisons: {len(comparisons)}; significant after global Holm: {meta['significant_after_global_holm144']}")
    print("Raw:", out)
    print("Condition means:", means_path)
    print("Paired statistics:", paired_path)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
