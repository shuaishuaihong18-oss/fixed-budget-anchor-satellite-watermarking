"""
Fixed-budget redundant localized watermarking add-on for WAM.

This module does NOT modify WAM's network or checkpoint. It only changes:
1) where the local watermark regions are placed under a fixed total area budget;
2) how multiple surviving local regions are fused during detection.

Paper-friendly comparison:
- single: one centered local watermarked region with total area B;
- single_random: one random local region with total area B;
- random: K random local regions, each B/K;
- texture: K texture-guided local regions, each B/K;
- hybrid: one compact anchor region plus distributed redundant satellite regions;
- wam_random_mask: WAM-style random block mask baseline under the same total area budget;
- voting: mask-aware confidence voting over surviving regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import random
import torch
import torch.nn.functional as F

from watermark_anything.data.transforms import unnormalize_img


@dataclass
class RegionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def _to_gray01(img_pt: torch.Tensor) -> torch.Tensor:
    rgb = unnormalize_img(img_pt.detach()).clamp(0, 1)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    return gray


def texture_energy_map(img_pt: torch.Tensor) -> torch.Tensor:
    """Sobel-gradient texture map, returned as [B,1,H,W] in [0,1]."""
    gray = _to_gray01(img_pt)
    device, dtype = gray.device, gray.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    energy = torch.sqrt(gx * gx + gy * gy + 1e-8)
    flat = energy.flatten(1)
    mn = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (energy - mn) / (mx - mn + 1e-8)


def _iou(a: RegionBox, b: RegionBox) -> float:
    xx1, yy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xx2, yy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    denom = a.area + b.area - inter + 1e-8
    return float(inter / denom)


def boxes_to_masks(boxes: List[RegionBox], H: int, W: int, device, dtype) -> torch.Tensor:
    masks = torch.zeros((len(boxes), 1, H, W), device=device, dtype=dtype)
    for i, b in enumerate(boxes):
        masks[i, :, b.y1:b.y2, b.x1:b.x2] = 1.0
    return masks


def _area_to_side(H: int, W: int, area_ratio: float) -> int:
    """Square side length for a target area ratio."""
    if not (0 < area_ratio <= 1):
        raise ValueError("area_ratio must be in (0, 1].")
    side = int((H * W * area_ratio) ** 0.5)
    return max(8, min(side, H, W))


def _region_side(H: int, W: int, total_area: float, num_regions: int) -> int:
    if not (0 < total_area <= 1):
        raise ValueError("total_area must be in (0, 1].")
    if num_regions < 1:
        raise ValueError("num_regions must be >= 1.")
    return _area_to_side(H, W, total_area / num_regions)


def select_texture_regions(
    img_pt: torch.Tensor,
    total_area: float = 0.20,
    num_regions: int = 4,
    stride_ratio: float = 0.5,
    max_iou: float = 0.03,
) -> Tuple[torch.Tensor, List[RegionBox]]:
    """Texture-guided non-overlapping square regions under a fixed total area budget."""
    if img_pt.ndim != 4 or img_pt.shape[0] != 1:
        raise ValueError("Expected img_pt shape [1,3,H,W].")
    _, _, H, W = img_pt.shape
    side = _region_side(H, W, total_area, num_regions)
    stride = max(4, int(side * stride_ratio))
    energy = texture_energy_map(img_pt)[0, 0]

    candidates: List[RegionBox] = []
    for y1 in range(0, H - side + 1, stride):
        for x1 in range(0, W - side + 1, stride):
            y2, x2 = y1 + side, x1 + side
            tex = float(energy[y1:y2, x1:x2].mean().item())
            cy, cx = (y1 + y2) / 2.0 / H, (x1 + x2) / 2.0 / W
            center_prior = 1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / 0.707)
            candidates.append(RegionBox(x1, y1, x2, y2, tex + 0.03 * center_prior))

    candidates.sort(key=lambda b: b.score, reverse=True)
    selected: List[RegionBox] = []
    for box in candidates:
        if all(_iou(box, prev) <= max_iou for prev in selected):
            selected.append(box)
            if len(selected) >= num_regions:
                break

    # Fallback if the image is small or the non-overlap constraint is too strict.
    if len(selected) < num_regions:
        for y1 in range(0, H - side + 1, side):
            for x1 in range(0, W - side + 1, side):
                box = RegionBox(x1, y1, x1 + side, y1 + side, 0.0)
                if all(_iou(box, prev) <= max_iou for prev in selected):
                    selected.append(box)
                    if len(selected) >= num_regions:
                        break
            if len(selected) >= num_regions:
                break

    return boxes_to_masks(selected, H, W, img_pt.device, img_pt.dtype), selected


def select_random_regions(
    img_pt: torch.Tensor,
    total_area: float = 0.20,
    num_regions: int = 4,
    max_iou: float = 0.03,
    seed: int = 0,
    max_trials: int = 1000,
) -> Tuple[torch.Tensor, List[RegionBox]]:
    """Random non-overlapping square regions under the same fixed total area budget."""
    if img_pt.ndim != 4 or img_pt.shape[0] != 1:
        raise ValueError("Expected img_pt shape [1,3,H,W].")
    _, _, H, W = img_pt.shape
    side = _region_side(H, W, total_area, num_regions)
    rng = random.Random(seed)

    selected: List[RegionBox] = []
    trials = 0
    while len(selected) < num_regions and trials < max_trials:
        trials += 1
        x1 = rng.randint(0, max(0, W - side))
        y1 = rng.randint(0, max(0, H - side))
        box = RegionBox(x1, y1, x1 + side, y1 + side, 0.0)
        if all(_iou(box, prev) <= max_iou for prev in selected):
            selected.append(box)

    if len(selected) < num_regions:
        # deterministic grid fallback
        for y1 in range(0, H - side + 1, side):
            for x1 in range(0, W - side + 1, side):
                box = RegionBox(x1, y1, x1 + side, y1 + side, 0.0)
                if all(_iou(box, prev) <= max_iou for prev in selected):
                    selected.append(box)
                    if len(selected) >= num_regions:
                        break
            if len(selected) >= num_regions:
                break

    return boxes_to_masks(selected, H, W, img_pt.device, img_pt.dtype), selected


def select_single_region(
    img_pt: torch.Tensor,
    total_area: float = 0.20,
    mode: str = "center",
    seed: int = 0,
) -> Tuple[torch.Tensor, List[RegionBox]]:
    """One region with the same total area budget. This is the fair single-region baseline."""
    _, _, H, W = img_pt.shape
    side = _region_side(H, W, total_area, 1)
    mode = mode.lower()
    if mode == "texture":
        return select_texture_regions(img_pt, total_area=total_area, num_regions=1)
    if mode == "random":
        rng = random.Random(seed)
        x1 = rng.randint(0, max(0, W - side))
        y1 = rng.randint(0, max(0, H - side))
    else:
        x1 = max(0, (W - side) // 2)
        y1 = max(0, (H - side) // 2)
    box = RegionBox(x1, y1, x1 + side, y1 + side, 0.0)
    return boxes_to_masks([box], H, W, img_pt.device, img_pt.dtype), [box]


def select_hybrid_regions(
    img_pt: torch.Tensor,
    total_area: float = 0.20,
    num_regions: int = 4,
    anchor_ratio: float = 0.40,
    max_iou: float = 0.03,
    seed: int = 0,
    max_trials: int = 1000,
) -> Tuple[torch.Tensor, List[RegionBox]]:
    """Anchor + satellites hybrid region allocation.

    The fixed total area budget B is split into:
    - one centered anchor region using anchor_ratio * B;
    - num_regions-1 distributed random satellite regions using the remaining budget.

    Motivation: the anchor preserves robustness under center-preserving crops/splicing, while
    the distributed satellites preserve the repeated message under random local erasing/editing.
    """
    if img_pt.ndim != 4 or img_pt.shape[0] != 1:
        raise ValueError("Expected img_pt shape [1,3,H,W].")
    if num_regions < 2:
        return select_single_region(img_pt, total_area=total_area, mode="center", seed=seed)
    if not (0 < anchor_ratio < 1):
        raise ValueError("anchor_ratio must be in (0, 1).")

    _, _, H, W = img_pt.shape
    rng = random.Random(seed)

    anchor_area = total_area * anchor_ratio
    satellite_area_each = total_area * (1.0 - anchor_ratio) / (num_regions - 1)

    anchor_side = _area_to_side(H, W, anchor_area)
    ax1 = max(0, (W - anchor_side) // 2)
    ay1 = max(0, (H - anchor_side) // 2)
    selected: List[RegionBox] = [RegionBox(ax1, ay1, ax1 + anchor_side, ay1 + anchor_side, 1.0)]

    sat_side = _area_to_side(H, W, satellite_area_each)
    trials = 0
    while len(selected) < num_regions and trials < max_trials:
        trials += 1
        x1 = rng.randint(0, max(0, W - sat_side))
        y1 = rng.randint(0, max(0, H - sat_side))
        box = RegionBox(x1, y1, x1 + sat_side, y1 + sat_side, 0.0)
        if all(_iou(box, prev) <= max_iou for prev in selected):
            selected.append(box)

    # Deterministic grid fallback if random placement cannot find enough satellites.
    if len(selected) < num_regions:
        for y1 in range(0, H - sat_side + 1, sat_side):
            for x1 in range(0, W - sat_side + 1, sat_side):
                box = RegionBox(x1, y1, x1 + sat_side, y1 + sat_side, 0.0)
                if all(_iou(box, prev) <= max_iou for prev in selected):
                    selected.append(box)
                    if len(selected) >= num_regions:
                        break
            if len(selected) >= num_regions:
                break

    return boxes_to_masks(selected, H, W, img_pt.device, img_pt.dtype), selected



def select_wam_random_mask_regions(
    img_pt: torch.Tensor,
    total_area: float = 0.20,
    grid: int = 8,
    seed: int = 0,
) -> Tuple[torch.Tensor, List[RegionBox]]:
    """WAM-style random mask baseline under a fixed total area budget.

    This baseline is meant to represent a generic random local mask setting rather than
    our designed equal-size random regions or anchor-satellite layout. The image is
    divided into a coarse grid and a random subset of cells is selected so that the
    union mask area is approximately equal to total_area.

    Difference from existing baselines:
    - single_random: one random square with area B;
    - random: K equal-size random square regions, each about B/K;
    - wam_random_mask: a stochastic block-wise random mask with many small cells and
      no anchor, no texture guidance, and no equal-region design.

    The resulting mask is still constrained by the same total area budget B, so it is
    a fair allocation baseline for the proposed fixed-budget study.
    """
    if img_pt.ndim != 4 or img_pt.shape[0] != 1:
        raise ValueError("Expected img_pt shape [1,3,H,W].")
    if not (0 < total_area <= 1):
        raise ValueError("total_area must be in (0, 1].")
    _, _, H, W = img_pt.shape
    grid = max(2, int(grid))
    rng = random.Random(seed)

    # Build coarse cells. The last row/column absorbs any remainder.
    cells: List[RegionBox] = []
    for gy in range(grid):
        y1 = int(round(gy * H / grid))
        y2 = int(round((gy + 1) * H / grid))
        for gx in range(grid):
            x1 = int(round(gx * W / grid))
            x2 = int(round((gx + 1) * W / grid))
            if x2 > x1 and y2 > y1:
                cells.append(RegionBox(x1, y1, x2, y2, 0.0))

    rng.shuffle(cells)
    target_area = total_area * H * W
    selected: List[RegionBox] = []
    area = 0
    for box in cells:
        if len(selected) == 0 or area < target_area:
            selected.append(box)
            area += box.area
        else:
            break

    return boxes_to_masks(selected, H, W, img_pt.device, img_pt.dtype), selected


def select_regions(
    img_pt: torch.Tensor,
    strategy: str,
    total_area: float,
    num_regions: int,
    seed: int = 0,
    anchor_ratio: float = 0.40,
) -> Tuple[torch.Tensor, List[RegionBox]]:
    strategy = strategy.lower()
    if strategy in {"single", "single_center"}:
        return select_single_region(img_pt, total_area=total_area, mode="center", seed=seed)
    if strategy == "single_random":
        return select_single_region(img_pt, total_area=total_area, mode="random", seed=seed)
    if strategy == "single_texture":
        return select_single_region(img_pt, total_area=total_area, mode="texture", seed=seed)
    if strategy == "random":
        return select_random_regions(img_pt, total_area=total_area, num_regions=num_regions, seed=seed)
    if strategy == "texture":
        return select_texture_regions(img_pt, total_area=total_area, num_regions=num_regions)
    if strategy == "hybrid":
        return select_hybrid_regions(img_pt, total_area=total_area, num_regions=num_regions, seed=seed, anchor_ratio=anchor_ratio)
    if strategy in {"wam_random_mask", "wam_default", "random_mask", "wam_randmask"}:
        return select_wam_random_mask_regions(img_pt, total_area=total_area, seed=seed)
    raise ValueError(f"Unknown region strategy: {strategy}")


@torch.no_grad()
def embed_with_region_strategy(
    wam,
    img_pt: torch.Tensor,
    msg: torch.Tensor,
    strategy: str = "texture",
    total_area: float = 0.20,
    num_regions: int = 4,
    seed: int = 0,
    anchor_ratio: float = 0.40,
) -> Dict[str, object]:
    if msg.ndim == 1:
        msg = msg.unsqueeze(0)
    msg = msg.float().to(img_pt.device)
    masks, boxes = select_regions(img_pt, strategy=strategy, total_area=total_area, num_regions=num_regions, seed=seed, anchor_ratio=anchor_ratio)
    union_mask = masks.sum(dim=0, keepdim=True).clamp(0, 1)
    base = wam.embed(img_pt, msg)
    img_w = base["imgs_w"] * union_mask + img_pt * (1.0 - union_mask)
    return {
        "imgs_w": img_w,
        "msgs": msg,
        "masks": masks,
        "union_mask": union_mask,
        "boxes": boxes,
        "actual_area": float(union_mask.mean().item()),
    }


def _grid_candidate_masks(mask_prob: torch.Tensor, grid: int = 8, min_score: float = 0.08) -> List[torch.Tensor]:
    _, _, H, W = mask_prob.shape
    candidates: List[torch.Tensor] = []
    cell_h, cell_w = max(1, H // grid), max(1, W // grid)
    for gy in range(grid):
        for gx in range(grid):
            y1, y2 = gy * cell_h, H if gy == grid - 1 else min(H, (gy + 1) * cell_h)
            x1, x2 = gx * cell_w, W if gx == grid - 1 else min(W, (gx + 1) * cell_w)
            score = float(mask_prob[:, :, y1:y2, x1:x2].mean().item())
            if score >= min_score:
                m = torch.zeros_like(mask_prob)
                m[:, :, y1:y2, x1:x2] = 1.0
                candidates.append(m)
    return candidates


@torch.no_grad()
def detect_confidence_voting(
    wam,
    img_pt: torch.Tensor,
    mask_threshold: float = 0.5,
    grid: int = 8,
    min_cell_score: float = 0.08,
    min_pixels: int = 16,
    vote_mode: str = "proposed",) -> Dict[str, torch.Tensor]:
    preds = wam.detect(img_pt)["preds"]
    mask_prob = torch.sigmoid(preds[:, 0:1])
    bit_prob = torch.sigmoid(preds[:, 1:])

    candidate_masks = _grid_candidate_masks(mask_prob, grid=grid, min_score=min_cell_score)
    if len(candidate_masks) == 0:
        candidate_masks = [(mask_prob > mask_threshold).float()]

    local_scores: List[torch.Tensor] = []
    local_weights: List[torch.Tensor] = []
    for cmask in candidate_masks:
        w = cmask * mask_prob
        if int((w > 0).sum().item()) < min_pixels or float(w.sum().item()) <= 1e-8:
            continue
        bit_score = (bit_prob * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-8)
        margin = (bit_score - 0.5).abs().mean(dim=1, keepdim=True) * 2.0
        area_weight = w.mean(dim=(1, 2, 3)).view(1, 1)
        local_scores.append(bit_score)
        if vote_mode == "uniform":
            weight = torch.ones_like(margin)
        elif vote_mode == "mask":
            weight = area_weight
        elif vote_mode == "proposed":
            weight = margin * area_weight
        else:
            raise ValueError(f"Unknown vote_mode: {vote_mode}")
        local_weights.append(weight + 1e-8)

    if len(local_scores) == 0:
        w = mask_prob
        fused_score = (bit_prob * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-8)
        region_count = torch.tensor([0], device=img_pt.device)
    else:
        scores = torch.cat(local_scores, dim=0)
        weights = torch.cat(local_weights, dim=0)
        fused_score = (scores * weights).sum(dim=0, keepdim=True) / weights.sum()
        region_count = torch.tensor([scores.shape[0]], device=img_pt.device)

    pred_msg = (fused_score > 0.5).int()
    return {
        "pred_msg": pred_msg,
        "fused_score": fused_score,
        "mask_prob": mask_prob,
        "bit_prob": bit_prob,
        "region_count": region_count,
    }


def bit_accuracy(pred_msg: torch.Tensor, gt_msg: torch.Tensor) -> float:
    if gt_msg.ndim == 1:
        gt_msg = gt_msg.unsqueeze(0)
    gt_msg = gt_msg.to(pred_msg.device).int()
    return float((pred_msg.int() == gt_msg.int()).float().mean().item())








def recovery_31_32(pred_msg: torch.Tensor, gt_msg: torch.Tensor) -> float:
    """Return the fraction of images with no more than one bit error."""
    if gt_msg.ndim == 1:
        gt_msg = gt_msg.unsqueeze(0)
    gt_msg = gt_msg.to(pred_msg.device).int()
    errors = (pred_msg.int() != gt_msg.int()).sum(dim=1)
    return float((errors <= 1).float().mean().item())
