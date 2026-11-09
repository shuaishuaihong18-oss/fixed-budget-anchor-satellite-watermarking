import csv
import random
from pathlib import Path

import torch
from PIL import Image

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform
from watermark_anything.redundant_region import (
    select_regions,
    detect_confidence_voting,
    bit_accuracy,
    recovery_31_32,
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

RHOS = [0.2, 0.4, 0.6, 0.8]
ATTACKS = [
    "none",
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]
STRATEGIES = [
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
    "hybrid",
]

MAIN_CSV = Path(
    "outputs/revision_202610/raw/"
    "coco500_clean_allattacks/results.csv"
)
OUT_DIR = Path(
    "outputs/revision_202610/ablation_clean500"
)
OUT_CSV = OUT_DIR / "rho_vote_ablation.csv"

if not MAIN_CSV.exists():
    raise SystemExit(f"Missing main results: {MAIN_CSV}")

if OUT_CSV.exists():
    raise SystemExit(
        f"Output already exists: {OUT_CSV}\n"
        "Inspect it first; do not overwrite."
    )

# Reuse the exact messages saved by the CLEAN main experiment.
messages = {}

with MAIN_CSV.open(newline="") as f:
    for row in csv.DictReader(f):
        if (
            row["method"] == "hybrid"
            and row["attack"] == "none"
        ):
            image = row["image"]
            message = row["message"]

            if len(message) != 32 or set(message) - {"0", "1"}:
                raise SystemExit(
                    f"Invalid 32-bit message for {image}"
                )

            if image in messages and messages[image] != message:
                raise SystemExit(
                    f"Inconsistent saved messages: {image}"
                )

            messages[image] = message

if len(messages) != 500:
    raise SystemExit(
        f"Expected 500 saved messages; got {len(messages)}"
    )

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading frozen WAM model...")
wam = load_model_from_checkpoint(
    "checkpoints/params.json",
    "checkpoints/wam_mit.pth",
).to(device).eval()

paths = iter_images(
    Path("data/eval/coco500_clean"),
    LIMIT,
)

if len(paths) != 500:
    raise SystemExit(
        f"Expected 500 CLEAN images; got {len(paths)}"
    )

path_names = [p.name for p in paths]

if len(set(path_names)) != 500:
    raise SystemExit("Duplicate filenames in CLEAN dataset")

missing_messages = sorted(set(path_names) - set(messages))
extra_messages = sorted(set(messages) - set(path_names))

if missing_messages or extra_messages:
    raise SystemExit(
        f"Message/image mismatch.\n"
        f"Missing messages: {missing_messages[:10]}\n"
        f"Extra messages: {extra_messages[:10]}"
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)

fields = [
    "image",
    "message",
    "rho",
    "vote_mode",
    "attack",
    "actual_area",
    "allocated_regions",
    "regions_detected",
    "psnr",
    "ssim",
    "orig_bitacc",
    "vote_bitacc",
    "orig_recovery31",
    "vote_recovery31",
]

rows_written = 0

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    with torch.no_grad():
        for idx, path in enumerate(paths):
            img = default_transform(
                Image.open(path).convert("RGB")
            ).unsqueeze(0).to(device)

            message = messages[path.name]
            msg = torch.tensor(
                [int(b) for b in message],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            # Generate the same full-image watermark used
            # as the basis of the main fixed-budget experiment.
            full_w = wam.embed(img, msg)["imgs_w"]

            # Reproduce the main experiment's strategy order.
            # This also provides the original rho=0.4 hybrid.
            main_hybrid = None

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

                if strategy == "hybrid":
                    main_hybrid = (
                        local_w,
                        float(union_mask.mean().item()),
                        len(boxes),
                    )

            if main_hybrid is None:
                raise SystemExit("Failed to create main hybrid")

            # Rho ablation: proposed voting at all four rho values.
            # Voting ablation: all four modes at rho=0.4.
            for rho in RHOS:
                if abs(rho - ANCHOR_RATIO) < 1e-9:
                    watermarked, area, nregions = main_hybrid
                else:
                    masks, boxes = select_regions(
                        img,
                        strategy="hybrid",
                        total_area=BUDGET,
                        num_regions=K,
                        seed=SEED + idx,
                        anchor_ratio=rho,
                    )

                    union_mask = masks.sum(
                        dim=0, keepdim=True
                    ).clamp(0, 1)

                    watermarked = (
                        full_w * union_mask
                        + img * (1.0 - union_mask)
                    )
                    area = float(union_mask.mean().item())
                    nregions = len(boxes)

                q_psnr = psnr01(watermarked, img)
                q_ssim = ssim01(watermarked, img)

                modes = (
                    ["orig", "uniform", "mask", "proposed"]
                    if abs(rho - ANCHOR_RATIO) < 1e-9
                    else ["proposed"]
                )

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

                    for mode in modes:
                        if mode == "orig":
                            pred_vote = pred_orig
                            detected_regions = 0
                        else:
                            det = detect_confidence_voting(
                                wam,
                                attacked,
                                vote_mode=mode,
                            )
                            pred_vote = det["pred_msg"].to(device)
                            detected_regions = int(
                                det["region_count"].item()
                            )

                        writer.writerow({
                            "image": path.name,
                            "message": message,
                            "rho": f"{rho:.1f}",
                            "vote_mode": mode,
                            "attack": attack,
                            "actual_area": f"{area:.6f}",
                            "allocated_regions": nregions,
                            "regions_detected": detected_regions,
                            "psnr": f"{q_psnr:.6f}",
                            "ssim": f"{q_ssim:.6f}",
                            "orig_bitacc": (
                                f"{bit_accuracy(pred_orig, msg):.6f}"
                            ),
                            "vote_bitacc": (
                                f"{bit_accuracy(pred_vote, msg):.6f}"
                            ),
                            "orig_recovery31": (
                                f"{recovery_31_32(pred_orig, msg):.6f}"
                            ),
                            "vote_recovery31": (
                                f"{recovery_31_32(pred_vote, msg):.6f}"
                            ),
                        })
                        rows_written += 1

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/500 images")

expected = 500 * len(ATTACKS) * (len(RHOS) + 3)

print("\n========== CLEAN500 ABLATION COMPLETE ==========")
print("Images:", len(paths))
print("Rows:", rows_written)
print("Expected rows:", expected)
print("Output:", OUT_CSV)

if rows_written != expected:
    raise SystemExit("FAIL: unexpected number of rows")

print("PASS: row count verified")
