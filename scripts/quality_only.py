import csv
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform
from watermark_anything.redundant_region import select_regions
from scripts.run_fixed_budget_exp import iter_images, psnr01, ssim01

SEED = 3
BUDGET = 0.20
K = 4
ANCHOR_RATIO = 0.40

STRATEGIES = [
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
    "hybrid",
]
METHODS = ["full_wam_reference"] + STRATEGIES
METRICS = ["psnr", "ssim", "lpips"]

DATASETS = {
    "COCO500": ("data/eval/coco500_clean", 500),
    "DIV2K100": ("data/eval/div2k100", 100),
}

OUT = Path("outputs/revision_202610/quality_only_20261027")
OUT.mkdir(parents=True, exist_ok=False)

# LPIPS is optional: PSNR and SSIM remain available if its
# pretrained weights cannot be loaded.
lpips_model = None
try:
    import lpips
    lpips_model = lpips.LPIPS(net="alex", verbose=False)
    lpips_model.eval()
    print("LPIPS AlexNet initialized.")
except Exception as e:
    print("WARNING: LPIPS unavailable:", repr(e))
    print("PSNR and SSIM will still be calculated.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and lpips_model is not None:
    lpips_model = lpips_model.to(device)

quality_csv = OUT / "quality_per_image.csv"
fields = [
    "dataset", "image", "message", "method",
    "actual_area", "allocated_regions",
    "psnr", "ssim", "lpips",
]

with quality_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for dataset, (img_dir, expected_n) in DATASETS.items():
        # Reset before loading the model, matching the original
        # experiment's per-dataset RNG initialization.
        random.seed(SEED)
        torch.manual_seed(SEED)

        print(f"\n========== {dataset} ==========")
        print("Loading frozen WAM model...")

        wam = load_model_from_checkpoint(
            "checkpoints/params.json",
            "checkpoints/wam_mit.pth",
        ).to(device).eval()

        paths = iter_images(Path(img_dir), expected_n)
        if len(paths) != expected_n:
            raise RuntimeError(
                f"{dataset}: expected {expected_n} images, "
                f"found {len(paths)}"
            )

        print("Images:", len(paths))
        print("Output:", OUT)

        with torch.no_grad():
            for idx, path in enumerate(paths):
                img = default_transform(
                    Image.open(path).convert("RGB")
                ).unsqueeze(0).to(device)

                # Same message generation and embedding sequence
                # as the original experiment.
                msg = torch.randint(0, 2, (1, 32), device=device).float()
                msg_string = "".join(
                    str(int(x)) for x in msg[0].tolist()
                )

                full_w = wam.embed(img, msg)["imgs_w"]

                methods = {
                    "full_wam_reference": (full_w, 1.0, 0)
                }

                for strategy in STRATEGIES:
                    nreg = 1 if strategy.startswith("single") else K

                    masks, boxes = select_regions(
                        img,
                        strategy=strategy,
                        total_area=BUDGET,
                        num_regions=nreg,
                        seed=SEED + idx,
                        anchor_ratio=ANCHOR_RATIO,
                    )

                    union_mask = masks.sum(
                        dim=0, keepdim=True
                    ).clamp(0, 1)

                    local_w = (
                        full_w * union_mask
                        + img * (1.0 - union_mask)
                    )

                    methods[strategy] = (
                        local_w,
                        float(union_mask.mean().item()),
                        len(boxes),
                    )

                for method, (watermarked, area, nregions) in methods.items():
                    p = psnr01(watermarked, img)
                    s = ssim01(watermarked, img)

                    lp = float("nan")
                    if lpips_model is not None:
                        lp = float(
                            lpips_model(img, watermarked).mean().item()
                        )

                    writer.writerow({
                        "dataset": dataset,
                        "image": path.name,
                        "message": msg_string,
                        "method": method,
                        "actual_area": f"{area:.8f}",
                        "allocated_regions": nregions,
                        "psnr": f"{p:.8f}",
                        "ssim": f"{s:.8f}",
                        "lpips": "" if not np.isfinite(lp) else f"{lp:.8f}",
                    })

                # Preserve progress if a long run is interrupted.
                f.flush()

                if (idx + 1) % 25 == 0:
                    print(f"Processed {idx + 1}/{len(paths)}")

        del wam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("\nPer-image metrics saved:", quality_csv)

# ---------- Summary and paired bootstrap CIs ----------
with quality_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))

rng = np.random.default_rng(20261027)
N_BOOT = 10000

def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")

    estimates = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[i] = sample.mean()

    return tuple(np.quantile(estimates, [0.025, 0.975]))

summary_path = OUT / "quality_summary.csv"
summary_fields = [
    "dataset", "method", "metric", "n",
    "mean", "ci95_low", "ci95_high",
]

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()

    for dataset in DATASETS:
        for method in METHODS:
            subset = [
                r for r in rows
                if r["dataset"] == dataset and r["method"] == method
            ]

            for metric in METRICS:
                values = np.array([
                    float(r[metric]) if r[metric].strip() else np.nan
                    for r in subset
                ])
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    continue

                lo, hi = bootstrap_ci(values)
                writer.writerow({
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": f"{values.mean():.8f}",
                    "ci95_low": f"{lo:.8f}",
                    "ci95_high": f"{hi:.8f}",
                })

paired_path = OUT / "quality_paired_comparisons.csv"
paired_fields = [
    "dataset", "metric", "comparison", "n",
    "mean_diff_hybrid_minus_baseline",
    "ci95_low", "ci95_high",
]

BASELINES = [
    "single", "single_random", "random",
    "texture", "wam_random_mask",
]

with paired_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=paired_fields)
    writer.writeheader()

    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset]
        by_method = {
            method: {
                r["image"]: r for r in subset if r["method"] == method
            }
            for method in METHODS
        }

        for metric in METRICS:
            for baseline in BASELINES:
                shared = sorted(
                    set(by_method["hybrid"]) & set(by_method[baseline])
                )

                diffs = []
                for image in shared:
                    a = by_method["hybrid"][image][metric]
                    b = by_method[baseline][image][metric]
                    if not a.strip() or not b.strip():
                        continue

                    av, bv = float(a), float(b)
                    if np.isfinite(av) and np.isfinite(bv):
                        diffs.append(av - bv)

                if not diffs:
                    continue

                lo, hi = bootstrap_ci(diffs)
                writer.writerow({
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": f"hybrid_vs_{baseline}",
                    "n": len(diffs),
                    "mean_diff_hybrid_minus_baseline":
                        f"{np.mean(diffs):.8f}",
                    "ci95_low": f"{lo:.8f}",
                    "ci95_high": f"{hi:.8f}",
                })

print("Summary:", summary_path)
print("Paired comparisons:", paired_path)
print("QUALITY EVALUATION COMPLETE")
