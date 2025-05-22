"""
Evaluation / Inference Script – conceptual skeleton

Purpose
-------
Run trained CAD-Recode model on validation or test split to produce metrics and
(optional) qualitative outputs without gradient updates.

Top-level flow
--------------
main(cfg):
    • Load config & resolve checkpoint path.
    • Instantiate dataset (split=val/test).
    • Build DataLoader (no shuffle).
    • Load CADRecodeModel weights (strict=True).
    • For each batch:
        – Generate predicted CADQuery code (beam search or greedy).
        – Compute metrics: Chamfer distance, IoU (exact or Monte-Carlo fallback).
        – Accumulate stats.
    • Print / log mean metrics.
    • Optionally save per-sample comparison visuals (if cfg.save_visuals=True).

CLI arguments
-------------
• --config CONFIG.YAML             # path to config (same as train)
• --checkpoint PATH                # model .pth file
• --split {val,test}
• --max_length INT                 # max generated tokens
• --num_candidates INT             # beam size or n-best
• --save_visuals                   # flag to write visuals under results/.

Implementation notes
--------------------
• Reuse utils.sample_points_on_shape & chamfer_distance from package.
• Wrap model in torch.no_grad() context, set model.eval().
• If GPU(s) available, use .to('cuda') – multi-GPU inference not critical.
• Qualitative outputs saved like train/epoch visuals but under separate folder (e.g., results/eval_<date>/).
"""

from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
from pathlib import Path
from typing import Tuple


import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from cad_recode.data.dataset import CadRecodeDataset, _execute_cadquery
from cad_recode.models.model import CADRecodeModel
from cad_recode.utils import sample_points_on_shape, chamfer_distance
from cad_recode.utils.common import setup_logger, compare_point_clouds


# --------------------------------------------------------------------------- #
#                                IoU helper                                   #
# --------------------------------------------------------------------------- #
def _approx_iou(a_shape, b_shape, n_samples=25000) -> float:
    if a_shape is None or b_shape is None:
        return 0.0
    bb_a = a_shape.BoundingBox()
    bb_b = b_shape.BoundingBox()
    xmin, xmax = min(bb_a.xmin, bb_b.xmin), max(bb_a.xmax, bb_b.xmax)
    ymin, ymax = min(bb_a.ymin, bb_b.ymin), max(bb_a.ymax, bb_b.ymax)
    zmin, zmax = min(bb_a.zmin, bb_b.zmin), max(bb_a.zmax, bb_b.zmax)
    if xmax - xmin < 1e-6 or ymax - ymin < 1e-6 or zmax - zmin < 1e-6:
        return 0.0
    pts = np.random.rand(n_samples, 3)
    pts[:, 0] = pts[:, 0] * (xmax - xmin) + xmin
    pts[:, 1] = pts[:, 1] * (ymax - ymin) + ymin
    pts[:, 2] = pts[:, 2] * (zmax - zmin) + zmin
    def inside(shp, p):
        return shp.distToShape(tuple(p))[0] < 1e-9
    inside_a = np.fromiter((inside(a_shape, p) for p in pts), dtype=bool, count=n_samples)
    inside_b = np.fromiter((inside(b_shape, p) for p in pts), dtype=bool, count=n_samples)
    inter = (inside_a & inside_b).sum()
    union = (inside_a | inside_b).sum()
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
#                          main evaluation routine                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(cfg):
    logger = setup_logger(None)  # stdout only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    ds = CadRecodeDataset(cfg.data.path, split=cfg.split,
                          n_points=cfg.data.n_points,
                          noise_std=0.0, noise_prob=0.0,
                          preload=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Model
    model = CADRecodeModel(cfg.model.name,
                           freeze_decoder=True,   # inference – memory lower
                           pos_enc=True).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    chamfers, ious, invalid = [], [], 0
    out_dir = None
    if getattr(cfg, "save_visuals", False):
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg.get("output_dir", "results")) / f"eval_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (pts_tensor, code_gt) in enumerate(tqdm(loader)):
        pts_tensor = pts_tensor.to(device)
        embeds, mask = model.prepare_prefix(pts_tensor)
        gen = model.decoder.generate(inputs_embeds=embeds,
                                     attention_mask=mask,
                                     max_length=cfg.max_length,
                                     num_beams=cfg.num_candidates,
                                     eos_token_id=model.end_id)
        code_pred = model.tokenizer.decode(gen[0], skip_special_tokens=False)

        shp_pred = _execute_cadquery(code_pred)
        shp_gt   = _execute_cadquery(code_gt[0])
        if shp_pred is None or shp_gt is None:
            invalid += 1
            continue

        pts_pred = sample_points_on_shape(shp_pred, 256)
        pts_gt   = sample_points_on_shape(shp_gt,   256)
        cd = chamfer_distance(pts_pred, pts_gt)
        chamfers.append(cd)
        iou = _approx_iou(shp_pred, shp_gt)
        ious.append(iou)

        # Qualitative dump?
        if out_dir is not None and idx < 20:   # cap saved samples
            sample_dir = out_dir / f"sample_{idx:04d}"
            sample_dir.mkdir(exist_ok=True)
            (sample_dir / "codes.txt").write_text(
                f"[GT]\n{code_gt[0]}\n\n[PRED]\n{code_pred}"
            )
            compare_point_clouds(pts_gt, pts_pred,
                                 sample_dir / "compare.png",
                                 title=f"CD {cd:.4f}, IoU {iou:.3f}")

    mean_cd  = float(torch.tensor(chamfers).mean()) if chamfers else None
    mean_iou = float(torch.tensor(ious).mean())     if ious else None
    invalid_ratio = invalid / len(ds)

    logger.info(f"Finished ⇢ Chamfer={mean_cd:.4f}, IoU={mean_iou:.3f}, "
                f"Invalid={invalid_ratio*100:.2f}%")

    if out_dir:
        (out_dir / "metrics.json").write_text(
            json.dumps({"mean_cd": mean_cd,
                        "mean_iou": mean_iou,
                        "invalid_ratio": invalid_ratio},
                       indent=2)
        )


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #
def _parse_cli():
    ap = argparse.ArgumentParser(description="Evaluate CAD-Recode model.")
    ap.add_argument("--config",  required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_candidates", type=int, default=1)
    ap.add_argument("--save_visuals", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_cli()
    cfg = OmegaConf.load(args.config)
    cfg.merge_with(vars(args))          # merge CLI overrides
    evaluate(cfg)
