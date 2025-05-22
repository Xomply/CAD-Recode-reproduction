# cad_recode/evaluate.py
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

import cadquery as cq                              # NEW: the missing import

# local imports – adjust the relative path if your package layout is different
from cad_recode.dataset import CadRecodeDataset
from cad_recode.model   import CADRecodeModel
from cad_recode.utils   import sample_points_on_shape, chamfer_distance

# --------------------------------------------------------------------------- #
#  Monte-Carlo IoU fallback                                                   #
# --------------------------------------------------------------------------- #
def approximate_iou_via_sampling(solid_a, solid_b, n_samples=25000):
    """
    Estimate IoU by Monte-Carlo sampling inside the union’s axis-aligned
    bounding box when exact OCC Boolean ops fail.
    """
    if solid_a is None or solid_b is None:
        return 0.0
    # Axis-aligned bounding boxes
    bb_a = solid_a.BoundingBox()
    bb_b = solid_b.BoundingBox()
    # Union BB
    xmin = min(bb_a.xmin, bb_b.xmin)
    xmax = max(bb_a.xmax, bb_b.xmax)
    ymin = min(bb_a.ymin, bb_b.ymin)
    ymax = max(bb_a.ymax, bb_b.ymax)
    zmin = min(bb_a.zmin, bb_b.zmin)
    zmax = max(bb_a.zmax, bb_b.zmax)
    if xmax - xmin < 1e-9 or ymax - ymin < 1e-9 or zmax - zmin < 1e-9:
        return 0.0
    # Uniformly sample points in the BB
    pts = np.random.rand(n_samples, 3)
    pts[:, 0] = pts[:, 0] * (xmax - xmin) + xmin
    pts[:, 1] = pts[:, 1] * (ymax - ymin) + ymin
    pts[:, 2] = pts[:, 2] * (zmax - zmin) + zmin
    # OCC has a “isInside” style check via the ShapeAnalysis package,
    # but CadQuery exposes `.distToShape`; we use a quick workaround:
    def inside(shape, p):
        # Positive distance means outside, -ve = inside/on
        return shape.distToShape(cq.Vector(*p))[0] < 1e-9
    inside_a = np.fromiter((inside(solid_a, p) for p in pts), dtype=bool, count=n_samples)
    inside_b = np.fromiter((inside(solid_b, p) for p in pts), dtype=bool, count=n_samples)
    inter = np.logical_and(inside_a, inside_b).sum()
    union = np.logical_or(inside_a, inside_b).sum()
    return inter / union if union > 0 else 0.0

# --------------------------------------------------------------------------- #
#  Evaluation routine                                                         #
# --------------------------------------------------------------------------- #
def evaluate_model(model, dataset, *,
                   batch_size       = 1,
                   max_length       = 256,
                   num_candidates   = 5,
                   device           = None):
    """
    Evaluate `model` on `dataset`. Returns (mean_CD, mean_IoU, invalid_ratio).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    chamfer_scores, iou_scores = [], []
    invalid = 0

    for pts_batch, code_true_batch in loader:
        pts_batch = pts_batch.to(device)          # (B,256,3)
        B = pts_batch.size(0)

        # ------- generate candidate codes ---------------------------------- #
        prefix_emb = model.projector(pts_batch)           # (B, embed_dim)

        start_ids  = torch.full((B, 1), model.start_id, dtype=torch.long, device=device)
        start_emb  = model.decoder.get_input_embeddings()(start_ids)
        init_emb   = torch.cat([prefix_emb, start_emb], dim=1)
        attn_mask  = torch.ones((B, 2), dtype=torch.long, device=device)

        try:
            gen = model.decoder.generate(
                inputs_embeds     = init_emb,
                attention_mask    = attn_mask,
                max_length        = max_length,
                num_beams         = num_candidates,
                num_return_sequences = num_candidates,
                early_stopping    = True,
                eos_token_id      = model.end_id,
            )
        except Exception as e:
            print(f"[WARN] Generation failed, skipping {B} samples: {e}")
            invalid += B
            continue

        # gen shape = (B*num_candidates, T)
        gen = gen.view(B, num_candidates, -1)

        for b in range(B):
            cand_codes = []
            for j in range(num_candidates):
                seq = gen[b, j]
                # remove first token (<|start|>) if present
                if seq[0].item() == model.start_id: seq = seq[1:]
                # trim at end-token
                end_idx = (seq == model.end_id).nonzero(as_tuple=True)[0]
                if len(end_idx): seq = seq[:end_idx[0]+1]
                cand_codes.append(model.tokenizer.decode(seq, skip_special_tokens=False))

            best_cd = float("inf")
            best_shape = None
            for code_pred in cand_codes:
                try:
                    loc = {}
                    exec(code_pred, {"cq": cq}, loc)
                    shp = loc.get("result") or loc.get("r") or loc.get("shape")
                    if isinstance(shp, cq.Workplane): shp = shp.val()
                except Exception:
                    continue
                if shp is None:
                    continue
                pts_pred = sample_points_on_shape(shp, 1024)
                pts_true = pts_batch[b].cpu().numpy()          # (256,3)
                cd = chamfer_distance(pts_pred, pts_true)
                if cd < best_cd:
                    best_cd, best_shape = cd, shp

            if best_shape is None:
                invalid += 1
                continue

            chamfer_scores.append(best_cd)

            # ------------------ IoU (exact or fallback) ------------------- #
            try:
                solid_pred = best_shape.val() if isinstance(best_shape, cq.Workplane) else best_shape
                loc_gt = {}
                exec(code_true_batch[b], {"cq": cq}, loc_gt)
                solid_gt = loc_gt.get("result") or loc_gt.get("r") or loc_gt.get("shape")
                if isinstance(solid_gt, cq.Workplane): solid_gt = solid_gt.val()

                vol_pred = solid_pred.Volume()
                vol_gt   = solid_gt.Volume()
                inter    = solid_pred.intersect(solid_gt)
                vol_int  = inter.Volume() if inter else 0.0
                vol_union = vol_pred + vol_gt - vol_int
                iou = vol_int / vol_union if vol_union > 1e-9 else 0.0
            except Exception:
                iou = approximate_iou_via_sampling(solid_pred, solid_gt)
            iou_scores.append(iou)

    # ---------------- aggregate ------------------------------------------- #
    mean_cd  = float(np.mean(chamfer_scores)) if chamfer_scores else None
    mean_iou = float(np.mean(iou_scores))     if iou_scores     else None
    invalid_ratio = invalid / len(dataset)

    print(f"[Eval] Chamfer  (mean) : {mean_cd}")
    print(f"[Eval] IoU      (mean) : {mean_iou}")
    print(f"[Eval] Invalid%        : {invalid_ratio*100:.2f}")

    return mean_cd, mean_iou, invalid_ratio

# --------------------------------------------------------------------------- #
#  CLI wrapper                                                                #
# --------------------------------------------------------------------------- #
def main(cfg):
    # Load dataset
    test_set = CadRecodeDataset(cfg.data_root, split=cfg.split,
                                n_points=256, noise_std=0.0, noise_prob=0.0)

    # Load model checkpoint
    model = CADRecodeModel(llm_name=cfg.llm)
    if cfg.checkpoint:
        model.load_state_dict(torch.load(cfg.checkpoint, map_location="cpu"))

    evaluate_model(model,
                   dataset        = test_set,
                   batch_size     = 1,
                   max_length     = cfg.max_length,
                   num_candidates = cfg.num_candidates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True, help="Path to cad-recode dataset directory")
    parser.add_argument("--checkpoint",  required=True, help="Path to trained model .pt")
    parser.add_argument("--split",       default="val", choices=["train","val","test"])
    parser.add_argument("--llm",         default="Qwen/Qwen2-0.5B")
    parser.add_argument("--max_length",  type=int, default=256)
    parser.add_argument("--num_candidates", type=int, default=5)
    args = parser.parse_args()
    main(args)
