import json
import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform
from watermark_anything.data.metrics import msg_predict_inference

from watermark_anything.redundant_region import (
    select_regions,
    detect_confidence_voting,
    bit_accuracy,
    recovery_31_32,
)

ROOT = Path(os.environ.get("WAM_ROOT", Path(__file__).resolve().parents[3])).resolve()
D = ROOT / "outputs/revision_202610/lama_object_inpainting_full382_20261103"

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

ANN = Path(os.environ.get("COCO_ANN", str(ROOT / "data/annotations/instances_val2017.json"))).resolve()
IMG_DIR = Path(os.environ.get("COCO_IMAGE_DIR", str(ROOT / "data/eval/coco500_clean"))).resolve()

ELIGIBLE_PATH = D / "eligible_annotations.csv"
ATTACK_PATH = D / "lama_object_inpainting_382.csv"

CLEAN_PATH = D / "clean_matched_382.csv"
PAIRED_PATH = D / "clean_vs_attacked_382.csv"
PARTIAL_PATH = D / "clean_matched_382.partial.csv"

assert ELIGIBLE_PATH.exists()
assert ATTACK_PATH.exists()

for p in [CLEAN_PATH, PAIRED_PATH, PARTIAL_PATH]:
    if p.exists():
        raise FileExistsError(p)


def pil_to_tensor(img, device):
    return default_transform(
        img.convert("RGB")
    ).unsqueeze(0).to(device)


def tensor_to_pil01(img_t):
    x = img_t.detach().cpu()[0]
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = (
        x.permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(x)


def original_wam_decode(wam, img_pt, gt_msg):
    preds = wam.detect(img_pt)["preds"]
    mask_preds = torch.sigmoid(preds[:, 0:1])
    bit_preds = preds[:, 1:]
    return msg_predict_inference(
        bit_preds, mask_preds
    ).to(gt_msg.device).int()


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    eligible = pd.read_csv(
        ELIGIBLE_PATH
    ).sort_values(
        "dataset_filename"
    ).reset_index(drop=True)

    assert len(eligible) == 382

    attacked = pd.read_csv(
        ATTACK_PATH,
        dtype={"image": str}
    )

    assert len(attacked) == 2292

    assert not attacked.duplicated(
        ["image", "strategy"]
    ).any()

    attack_ref = attacked.set_index(
        ["image", "strategy"]
    )

    # Recover exactly the messages used in the main experiment.
    main_path = ROOT / (
        "outputs/revision_202610/raw/"
        "coco500_clean_allattacks/results.csv"
    )

    main_df = pd.read_csv(
        main_path,
        dtype={"image": str, "message": str}
    )

    assert (
        main_df.groupby("image")["message"].nunique() == 1
    ).all()

    messages = dict(
        zip(
            main_df.drop_duplicates("image")["image"],
            main_df.drop_duplicates("image")["message"]
        )
    )

    coco = COCO(str(ANN))

    wam = load_model_from_checkpoint(
        str(ROOT / "checkpoints/params.json"),
        str(ROOT / "checkpoints/wam_mit.pth"),
    ).to(device).eval()

    rows = []

    with torch.no_grad():

        for idx, item in enumerate(
            eligible.to_dict("records")
        ):

            image_name = item["dataset_filename"]
            image_path = IMG_DIR / image_name

            assert image_path.exists()

            img_pil = Image.open(
                image_path
            ).convert("RGB")

            img = pil_to_tensor(img_pil, device)

            msg_string = messages[image_name]

            msg = torch.tensor(
                [[int(b) for b in msg_string]],
                dtype=torch.float32,
                device=device
            )

            ann = coco.loadAnns(
                [int(item["ann_id"])]
            )[0]

            mask_np = (
                coco.annToMask(ann).astype(np.uint8)
            )

            object_mask = torch.as_tensor(
                mask_np > 0,
                dtype=img.dtype,
                device=device
            ).unsqueeze(0).unsqueeze(0)

            # Same complete watermark for all six strategies.
            full_w = wam.embed(
                img, msg
            )["imgs_w"]

            for strategy in STRATEGIES:

                nreg = (
                    1 if strategy.startswith("single")
                    else K
                )

                masks, boxes = select_regions(
                    img,
                    strategy=strategy,
                    total_area=BUDGET,
                    num_regions=nreg,
                    seed=SEED + idx,
                    anchor_ratio=ANCHOR_RATIO
                )

                union_mask = masks.sum(
                    dim=0, keepdim=True
                ).clamp(0, 1)

                img_w = (
                    full_w * union_mask
                    + img * (1.0 - union_mask)
                )

                # Quantize exactly as in the LaMa experiment.
                img_w_pil = tensor_to_pil01(img_w)

                clean_input = pil_to_tensor(
                    img_w_pil, device
                )

                # Verify exact same watermark geometry.
                intersection = float(
                    (union_mask * object_mask).sum().item()
                )

                watermark_pixels = float(
                    union_mask.sum().item()
                )

                object_pixels = float(
                    object_mask.sum().item()
                )

                area = float(union_mask.mean().item())

                overlap_w = (
                    intersection /
                    max(watermark_pixels, 1e-8)
                )

                overlap_o = (
                    intersection /
                    max(object_pixels, 1e-8)
                )

                ref = attack_ref.loc[
                    (image_name, strategy)
                ]

                for key, value in [
                    ("actual_area", area),
                    ("overlap_of_watermark", overlap_w),
                    ("overlap_of_object", overlap_o),
                ]:

                    diff = abs(
                        float(ref[key]) - value
                    )

                    if diff > 1e-6:
                        raise RuntimeError(
                            f"Geometry mismatch: "
                            f"{image_name}, {strategy}, "
                            f"{key}, diff={diff}"
                        )

                # Original detector.
                pred_orig = original_wam_decode(
                    wam, clean_input, msg
                )

                # Same proposed voting detector.
                det_vote = detect_confidence_voting(
                    wam,
                    clean_input,
                    vote_mode="proposed"
                )

                pred_vote = det_vote["pred_msg"]

                rows.append({
                    "image": image_name,
                    "source_id": int(item["source_id"]),
                    "eligible_index": idx,
                    "geometry_seed": SEED + idx,
                    "strategy": strategy,

                    "clean_orig_bitacc":
                        bit_accuracy(pred_orig, msg),

                    "clean_vote_bitacc":
                        bit_accuracy(pred_vote, msg),

                    "clean_orig_recovery31":
                        recovery_31_32(pred_orig, msg),

                    "clean_vote_recovery31":
                        recovery_31_32(pred_vote, msg),

                    "clean_regions_detected":
                        int(det_vote["region_count"].item()),
                })

            if (idx + 1) % 20 == 0 or idx + 1 == 382:
                print(
                    f"Processed {idx+1}/382; "
                    f"rows={len(rows)}",
                    flush=True
                )

    clean = pd.DataFrame(rows)

    assert len(clean) == 2292

    assert not clean.duplicated(
        ["image", "strategy"]
    ).any()

    assert not clean.isna().any().any()

    assert clean.groupby("image")["strategy"].nunique().eq(
        6
    ).all()

    clean.to_csv(PARTIAL_PATH, index=False)

    merged = attacked.merge(
        clean,
        on=["image", "source_id", "strategy"],
        how="inner",
        validate="one_to_one"
    )

    assert len(merged) == 2292

    # Positive delta = improved after attack;
    # negative delta = performance degradation.
    for metric in ["bitacc", "recovery31"]:

        for detector in ["orig", "vote"]:

            merged[
                f"delta_{detector}_{metric}"
            ] = (
                merged[f"{detector}_{metric}"]
                - merged[f"clean_{detector}_{metric}"]
            )

    assert not merged.isna().any().any()

    CLEAN_PATH.write_bytes(
        PARTIAL_PATH.read_bytes()
    )

    PAIRED_PATH.write_text(
        merged.to_csv(index=False)
    )

    PARTIAL_PATH.unlink()

    metadata = {
        "experiment": "matched_clean_control_for_lama",
        "images": 382,
        "rows": len(clean),
        "strategies": STRATEGIES,
        "budget": BUDGET,
        "seed": SEED,
        "geometry_match_tolerance": 1e-6,
        "geometry_verified": True,
        "WAM_sha256": file_sha256(
            ROOT / "checkpoints/wam_mit.pth"
        ),
    }

    (D / "clean_control_metadata.json").write_text(
        json.dumps(
            metadata, indent=2, ensure_ascii=False
        )
    )

    print("=" * 70)
    print("MATCHED CLEAN CONTROL COMPLETE")
    print("Images:", clean.image.nunique())
    print("Rows:", len(clean))
    print("Geometry matches: 2292/2292")
    print("Paired attack-control rows:", len(merged))
    print("Clean results:", CLEAN_PATH)
    print("Paired results:", PAIRED_PATH)
    print("PASS")


if __name__ == "__main__":
    main()
