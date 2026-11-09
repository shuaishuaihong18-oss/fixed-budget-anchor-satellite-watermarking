"""Run fixed-budget redundant localized watermarking experiments.

Run from repository root:
PYTHONPATH=. python scripts/run_fixed_budget_exp.py \
  --ckpt checkpoints/wam_mit.pth --json checkpoints/params.json \
  --img_dir data/eval/coco500 --out_dir outputs/coco_B20_K4 \
  --total_area 0.20 --num_regions 4
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.utils import save_image

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.metrics import msg_predict_inference
from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img
from watermark_anything.redundant_region import (
    bit_accuracy,
    detect_confidence_voting,
    embed_with_region_strategy,
    recovery_31_32,
    select_regions,
)

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


def tensor_to_pil(img_pt: torch.Tensor) -> Image.Image:
    img = unnormalize_img(img_pt.detach().cpu()).clamp(0, 1)[0]
    return transforms.ToPILImage()(img)


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    return default_transform(img.convert("RGB")).unsqueeze(0).to(device)


def _rect_mask(H: int, W: int, ratio: float, seed: int, device, dtype) -> torch.Tensor:
    rng = random.Random(seed)
    side = int((H * W * ratio) ** 0.5)
    side = max(8, min(side, H, W))
    x1 = rng.randint(0, max(0, W - side))
    y1 = rng.randint(0, max(0, H - side))
    m = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    m[:, :, y1:y1 + side, x1:x1 + side] = 1.0
    return m


def apply_attack(img_w: torch.Tensor, img_orig: torch.Tensor, attack: str, seed: int) -> torch.Tensor:
    """Cheap local/global attacks. Diffusion inpainting can be added later as a separate script."""
    device = img_w.device
    if attack == "none":
        return img_w

    pil = tensor_to_pil(img_w)

    if attack.startswith("jpeg"):
        q = int(attack.replace("jpeg", ""))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return pil_to_tensor(Image.open(buf), device)

    if attack.startswith("resize"):
        scale = float(attack.replace("resize", "")) / 100.0 if attack.replace("resize", "").isdigit() else 0.5
        w, h = pil.size
        small = pil.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
        return pil_to_tensor(small.resize((w, h), Image.BILINEAR), device)

    if attack == "blur":
        return pil_to_tensor(pil.filter(ImageFilter.GaussianBlur(radius=1.5)), device)

    if attack.startswith("randomcrop"):
        # randomcrop50 means keep a random 50% side-length crop, then resize back.
        keep = float(attack.replace("randomcrop", "")) / 100.0
        keep = max(0.1, min(1.0, keep))
        w, h = pil.size
        cw, ch = int(w * keep), int(h * keep)
        rng = random.Random(seed + 31)
        x1 = rng.randint(0, max(0, w - cw))
        y1 = rng.randint(0, max(0, h - ch))
        crop = pil.crop((x1, y1, x1 + cw, y1 + ch)).resize((w, h), Image.BILINEAR)
        return pil_to_tensor(crop, device)

    if attack.startswith("cornercrop"):
        # cornercrop50 means keep the upper-left 50% side-length crop, then resize back.
        keep = float(attack.replace("cornercrop", "")) / 100.0
        keep = max(0.1, min(1.0, keep))
        w, h = pil.size
        cw, ch = int(w * keep), int(h * keep)
        crop = pil.crop((0, 0, cw, ch)).resize((w, h), Image.BILINEAR)
        return pil_to_tensor(crop, device)

    if attack.startswith("crop"):
        # crop50 means keep a centered 50% side-length crop, then resize back.
        keep = float(attack.replace("crop", "")) / 100.0
        keep = max(0.1, min(1.0, keep))
        w, h = pil.size
        cw, ch = int(w * keep), int(h * keep)
        x1, y1 = (w - cw) // 2, (h - ch) // 2
        crop = pil.crop((x1, y1, x1 + cw, y1 + ch)).resize((w, h), Image.BILINEAR)
        return pil_to_tensor(crop, device)

    if attack.startswith("erase"):
        # erase20/erase40: replace a random local area with the original unwatermarked pixels.
        ratio = float(attack.replace("erase", "")) / 100.0
        _, _, H, W = img_w.shape
        m = _rect_mask(H, W, ratio, seed=seed + 17, device=device, dtype=img_w.dtype)
        return img_w * (1.0 - m) + img_orig * m

    if attack.startswith("splicing"):
        # splicing20: only keep a random 20% region from the watermarked image; background is original.
        ratio = float(attack.replace("splicing", "")) / 100.0
        _, _, H, W = img_w.shape
        m = _rect_mask(H, W, ratio, seed=seed + 23, device=device, dtype=img_w.dtype)
        return img_w * m + img_orig * (1.0 - m)

    raise ValueError(f"Unknown attack: {attack}")


@torch.no_grad()
def original_wam_decode(wam, img_pt: torch.Tensor, gt_msg: torch.Tensor) -> torch.Tensor:
    preds = wam.detect(img_pt)["preds"]
    mask_preds = torch.sigmoid(preds[:, 0:1])
    bit_preds = preds[:, 1:]
    return msg_predict_inference(bit_preds, mask_preds).to(gt_msg.device).int()


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    a01 = unnormalize_img(a.detach()).clamp(0, 1)
    b01 = unnormalize_img(b.detach()).clamp(0, 1)
    mse = F.mse_loss(a01, b01).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim01(a: torch.Tensor, b: torch.Tensor) -> float:
    if skimage_ssim is None:
        return float("nan")
    import numpy as np
    a_np = unnormalize_img(a.detach()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    b_np = unnormalize_img(b.detach()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()









    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/wam_mit.pth")
    parser.add_argument("--json", default="checkpoints/params.json")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--total_area", type=float, default=0.20)
    parser.add_argument("--num_regions", type=int, default=4)
    parser.add_argument("--anchor_ratio", type=float, default=0.40, help="Anchor area ratio for hybrid allocation.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--attacks", nargs="+", default=["none", "jpeg50", "jpeg30", "resize50", "blur", "crop50", "randomcrop50", "randomcrop33", "cornercrop50", "erase20", "erase40", "splicing20", "splicing40"])
    parser.add_argument("--strategies", nargs="+", default=["single", "single_random", "random", "texture", "wam_random_mask", "hybrid"])
    parser.add_argument(
        "--vote_mode",
        type=str,
        default="proposed",
        choices=["orig", "uniform", "mask", "proposed"],
        help="Message decoding mode: orig uses the original WAM decoder; uniform/mask/proposed use region voting."
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wam = load_model_from_checkpoint(args.json, args.ckpt).to(device).eval()
    image_paths = iter_images(Path(args.img_dir), args.limit)

    csv_path = out_dir / "results.csv"
    fieldnames = [
        "image", "message", "method", "attack", "actual_area",
        "allocated_regions", "candidate_cells_used", "orig_bitacc",
        "vote_bitacc", "orig_recovery31", "vote_recovery31",
        "psnr", "ssim",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for idx, img_path in enumerate(image_paths):
                img = Image.open(img_path).convert("RGB")
                img_pt = default_transform(img).unsqueeze(0).to(device)
                msg = torch.randint(0, 2, (1, 32), device=device).float()
                msg_string = "".join(str(int(x)) for x in msg[0].tolist())

                # Embed once per image so every allocation uses the identical full-image watermark.
                full_w = wam.embed(img_pt, msg)["imgs_w"]
                methods = [("full_wam_reference", full_w, 1.0, 0)]

                for strategy in args.strategies:
                    nreg = 1 if strategy.startswith("single") else args.num_regions
                    masks, boxes = select_regions(
                        img_pt,
                        strategy=strategy,
                        total_area=args.total_area,
                        num_regions=nreg,
                        seed=args.seed + idx,
                        anchor_ratio=args.anchor_ratio,
                    )
                    union_mask = masks.sum(dim=0, keepdim=True).clamp(0, 1)
                    img_w = full_w * union_mask + img_pt * (1.0 - union_mask)
                    methods.append((strategy, img_w, float(union_mask.mean().item()), len(boxes)))

                for method, watermarked, actual_area, nregions in methods:
                    q_psnr = psnr01(watermarked, img_pt)
                    q_ssim = ssim01(watermarked, img_pt)
                    for attack in args.attacks:
                        attacked = apply_attack(watermarked, img_pt, attack, seed=args.seed + idx)
                        pred_orig = original_wam_decode(wam, attacked, msg)
                        if args.vote_mode == "orig":
                            pred_vote = pred_orig
                            det_vote = {"region_count": torch.tensor([0], device=device)}
                        else:
                            det_vote = detect_confidence_voting(wam, attacked, vote_mode=args.vote_mode)
                            pred_vote = det_vote["pred_msg"].to(device)

                        row = {
                            "image": img_path.name,
                            "message": msg_string,
                            "method": method,
                            "attack": attack,
                            "actual_area": f"{actual_area:.4f}",
                            "allocated_regions": nregions,
                            "candidate_cells_used": int(det_vote["region_count"].item()),
                            "orig_bitacc": f"{bit_accuracy(pred_orig, msg):.4f}",
                            "vote_bitacc": f"{bit_accuracy(pred_vote, msg):.4f}",
                            "orig_recovery31": f"{recovery_31_32(pred_orig, msg):.4f}",
                            "vote_recovery31": f"{recovery_31_32(pred_vote, msg):.4f}",
                            "psnr": f"{q_psnr:.4f}",
                            "ssim": f"{q_ssim:.4f}",
                        }
                        writer.writerow(row)

                if (idx + 1) % 20 == 0:
                    print(f"Processed {idx + 1}/{len(image_paths)} images")

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
