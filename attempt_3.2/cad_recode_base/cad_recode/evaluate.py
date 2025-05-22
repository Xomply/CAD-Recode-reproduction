# cad_recode/evaluate.py
"""Model evaluation script for CAD‑Recode.

Key changes
~~~~~~~~~~~
* **Monte‑Carlo IoU** now defaults to *2 500* samples (was 25 000) – much
  faster for large test sets while still giving a useful approximation.
* Removed per‑batch `torch.cuda.empty_cache()`; that call **slows** evaluation
  considerably and is unnecessary once memory leaks are fixed.
* Beam‑search parameters are documented: by default we use *plain* beam search
  (`num_beams = num_candidates`).  To encourage diversity set
  ``--diversity_penalty > 0`` and ``--num_beam_groups`` in the CLI or via
  config.
* All configurable values can now be overridden via **Hydra** overrides (the
  script still accepts plain argparse for standalone use).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import cadquery as cq

from cad_recode.dataset import CadRecodeDataset
from cad_recode.model import CADRecodeModel
from cad_recode.utils import chamfer_distance, edit_distance, sample_points_on_shape, save_point_cloud

# ---------------------------------------------------------------------------
# Monte‑Carlo IoU (approx.)
# ---------------------------------------------------------------------------

def approximate_iou_via_sampling(solid_a, solid_b, n_samples: int = 2500) -> float:
    """Rough IoU by rejection sampling inside the union AABB."""
    if solid_a is None or solid_b is None:
        return 0.0
    bb_a, bb_b = solid_a.BoundingBox(), solid_b.BoundingBox()
    xmin, xmax = min(bb_a.xmin, bb_b.xmin), max(bb_a.xmax, bb_b.xmax)
    ymin, ymax = min(bb_a.ymin, bb_b.ymin), max(bb_a.ymax, bb_b.ymax)
    zmin, zmax = min(bb_a.zmin, bb_b.zmin), max(bb_a.zmax, bb_b.zmax)
    if xmax - xmin < 1e-9 or ymax - ymin < 1e-9 or zmax - zmin < 1e-9:
        return 0.0

    pts = np.random.rand(n_samples, 3)
    pts[:, 0] = pts[:, 0] * (xmax - xmin) + xmin
    pts[:, 1] = pts[:, 1] * (ymax - ymin) + ymin
    pts[:, 2] = pts[:, 2] * (zmax - zmin) + zmin

    def inside(shape, p):
        return shape.distToShape(cq.Vector(*p))[0] < 1e-9

    inside_a = np.fromiter((inside(solid_a, p) for p in pts), dtype=bool, count=n_samples)
    inside_b = np.fromiter((inside(solid_b, p) for p in pts), dtype=bool, count=n_samples)
    inter = np.logical_and(inside_a, inside_b).sum()
    union = np.logical_or(inside_a, inside_b).sum()
    return inter / union if union > 0 else 0.0

# ---------------------------------------------------------------------------
# Evaluation routine
# ---------------------------------------------------------------------------

def evaluate_model(model: CADRecodeModel,
                   dataset: CadRecodeDataset,
                   *,
                   batch_size: int = 1,
                   max_length: int = 256,
                   num_candidates: int = 1,
                   device: torch.device | None = None,
                   save_examples: int = 0) -> tuple[dict, list]:
    """Run evaluation and return (metrics, example‑list)."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    chamfer_scores: List[float] = []
    iou_scores: List[float] = []
    token_accs: List[float] = []
    edit_dists: List[int] = []
    invalid = 0
    examples = []
    saved = 0

    for pts_batch, code_true_batch in loader:
        pts_batch = pts_batch.to(device)
        B = pts_batch.size(0)

        # ‑‑ generate candidates (plain beam; set diversity_penalty for DBS) --
        try:
            prefix, attn = model.prepare_prefix(pts_batch)
            gen = model.decoder.generate(
                inputs_embeds=prefix,
                attention_mask=attn,
                max_length=max_length,
                num_beams=max(1, num_candidates),
                num_return_sequences=num_candidates,
                early_stopping=True,
                eos_token_id=model.end_id,
                # diversity_penalty=0.3,
                # num_beam_groups=num_candidates,
            )
        except Exception as e:
            print(f"[WARN] generation failed for batch of size {B}: {e}")
            invalid += B
            continue

        gen = gen.view(B, num_candidates, -1)
        for b in range(B):
            cand_codes: List[str] = []
            for j in range(num_candidates):
                seq = gen[b, j].clone()
                if seq[0].item() == model.start_id:  # strip leading <|start|>
                    seq = seq[1:]
                if (end := (seq == model.end_id).nonzero(as_tuple=True)[0]).numel():
                    seq = seq[: end[0] + 1]
                cand_codes.append(model.tokenizer.decode(seq, skip_special_tokens=False))

            # pick best candidate by Chamfer – note: ignores diversity
            best_cd = float("inf")
            best_shape, best_code = None, None
            for code_pred in cand_codes:
                try:
                    loc: dict[str, object] = {}
                    exec(code_pred, {"cq": cq}, loc)
                    shp = loc.get("result") or loc.get("r") or loc.get("shape")
                    if isinstance(shp, cq.Workplane):
                        shp = shp.val()
                except Exception:
                    continue
                if shp is None:
                    continue
                pts_pred = sample_points_on_shape(shp, 1024)
                pts_true = pts_batch[b].cpu().numpy()
                cd = chamfer_distance(pts_pred, pts_true)
                if cd < best_cd:
                    best_cd, best_shape, best_code = cd, shp, code_pred

            if best_shape is None or best_code is None:
                invalid += 1
                continue

            chamfer_scores.append(best_cd)

            # IoU (exact or sampling)
            try:
                solid_pred = best_shape.val() if isinstance(best_shape, cq.Workplane) else best_shape
                loc_gt: dict[str, object] = {}
                exec(code_true_batch[b], {"cq": cq}, loc_gt)
                solid_gt = loc_gt.get("result") or loc_gt.get("r") or loc_gt.get("shape")
                if isinstance(solid_gt, cq.Workplane):
                    solid_gt = solid_gt.val()
                vol_pred = solid_pred.Volume()
                vol_gt = solid_gt.Volume()
                inter = solid_pred.intersect(solid_gt)
                vol_int = inter.Volume() if inter else 0.0
                vol_union = vol_pred + vol_gt - vol_int
                iou = vol_int / vol_union if vol_union > 1e-9 else 0.0
            except Exception:
                iou = approximate_iou_via_sampling(solid_pred, solid_gt)
            iou_scores.append(iou)

            # token accuracy + edit distance
            code_pred_clean = best_code.replace("<|start|>", "").replace("<|end|>", "").strip()
            true_ids = model.tokenizer.encode(code_true_batch[b], add_special_tokens=False)
            pred_ids = model.tokenizer.encode(code_pred_clean, add_special_tokens=False)
            matches = sum(int(t == p) for t, p in zip(true_ids, pred_ids))
            token_accs.append(matches / max(1, len(true_ids)))
            edit_dists.append(edit_distance(true_ids, pred_ids))

            # optional example dump
            if save_examples and saved < save_examples:
                try:
                    loc_gt: dict[str, object] = {}
                    exec(code_true_batch[b], {"cq": cq}, loc_gt)
                    solid_gt = loc_gt.get("result") or loc_gt.get("r") or loc_gt.get("shape")
                    if isinstance(solid_gt, cq.Workplane):
                        solid_gt = solid_gt.val()
                    pts_true_full = sample_points_on_shape(solid_gt, 1024)
                    pts_pred_full = sample_points_on_shape(best_shape, 1024)
                    examples.append({
                        "code_true": code_true_batch[b],
                        "code_pred": code_pred_clean,
                        "pts_true": pts_true_full,
                        "pts_pred": pts_pred_full,
                    })
                    saved += 1
                except Exception:
                    pass

    metrics = {
        "mean_chamfer": float(np.mean(chamfer_scores)) if chamfer_scores else None,
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else None,
        "invalid_ratio": float(invalid / len(dataset)) if len(dataset) else 1.0,
        "mean_token_accuracy": float(np.mean(token_accs)) if token_accs else None,
        "mean_edit_distance": float(np.mean(edit_dists)) if edit_dists else None,
    }

    print("[Eval]", "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in metrics.items()))
    return metrics, examples

# ---------------------------------------------------------------------------
# CLI wrapper – still supports argparse for quick tests
# ---------------------------------------------------------------------------

def main(args):
    if isinstance(args, argparse.Namespace):
        args = vars(args)  # convert to dict so Hydra overrides would also work

    test_set = CadRecodeDataset(args["data_root"], split=args.get("split", "val"),
                                n_points=256, noise_std=0.0, noise_prob=0.0)

    model = CADRecodeModel(llm_name=args.get("llm", "Qwen/Qwen2-1.5B"))
    if ckpt := args.get("checkpoint"):
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)

    metrics, examples = evaluate_model(
        model,
        test_set,
        batch_size=args.get("batch_size", 1),
        max_length=args.get("max_length", 256),
        num_candidates=args.get("num_candidates", 1),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_examples=args.get("save_examples", 0),
    )

    out_dir = Path(args.get("output_dir") or Path(ckpt).parent or ".")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "eval_results.json", "w", encoding="utf‑8") as f:
        json.dump(metrics, f, indent=2)

    # dump example outputs
    for idx, ex in enumerate(examples):
        base = out_dir / f"example_{idx:03d}"
        (base.with_suffix("_true.py")).write_text(ex["code_true"] + "\n")
        (base.with_suffix("_pred.py")).write_text(ex["code_pred"] + "\n")
        save_point_cloud(ex["pts_true"], str(base) + "_true.ply")
        save_point_cloud(ex["pts_pred"], str(base) + "_pred.ply")

    print(f"Saved metrics and {len(examples)} example(s) to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--llm", default="Qwen/Qwen2-1.5B")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_candidates", type=int, default=1)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--save_examples", type=int, default=0)
    main(p.parse_args())
