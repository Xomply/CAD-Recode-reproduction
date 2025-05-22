"""
Training Entry-Point – conceptual skeleton

Responsibilities
----------------
1. Parse configuration (OmegaConf / Hydra) and initialise logging.
2. Initialise distributed environment if launched via torchrun (DDP).
3. Build Dataset + DataLoader (with DistributedSampler).
4. Instantiate CADRecodeModel and wrap in DDP if world_size > 1.
5. Set up optimizer (AdamW) and LR scheduler (warmup + cosine).
6. Run training loop with checkpointing, validation, and optional W&B logging.
7. Implement resume logic (load checkpoint, restore epoch/step, optimizer, scheduler).
8. Save qualitative validation outputs (point-cloud comparison) per epoch.

High-Level Flow
---------------
main(cfg):
    setup_distributed()
    log_cfg(cfg)
    dataset = CadRecodeDataset(...)
    train_loader, val_loader = build_dataloaders(dataset)

    model = CADRecodeModel(...)
    if distributed: model = DDP(model, device_ids=[local_rank])
    optimizer, scheduler = build_optim(model, cfg)

    start_epoch, global_step = maybe_resume(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, cfg.training.max_epochs):
        train_one_epoch()
        validate()
        save_checkpoint(epoch, is_best)
        save_visuals(epoch)           # ground-truth vs predicted
        if convergence_achieved: break

    write_summary()

Key helper functions
--------------------
• setup_distributed() – sets rank, world_size, local_device.
• build_dataloaders() – handles DistributedSampler.
• train_one_epoch() – inner loop, gradient accumulation.
• validate() – compute val loss, IoU/CD metrics.
• save_checkpoint(epoch, is_best) – keep `latest.pth` + `best.pth` only.
• save_visuals(epoch) – generate & save example images / code diff to results/.
• maybe_resume(path, ...) – load state if found.
• write_summary() – final run-wide metadata.

CLI usage
---------
• Standard *python -m cad_recode.train* with Hydra overrides, or via `torchrun` for multi-GPU.
• Accept config override `training.resume_path` for manual resume as well.

Note: All heavy imports (torch, hf transformers) inside `main` to keep module load light.
"""

from __future__ import annotations
import os, sys, time, json, random, hydra
from pathlib import Path
from contextlib import nullcontext
from typing import Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import numpy as np

from cad_recode.common import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    get_rank,
    world_size,
    is_main_process,
    barrier,
    compare_point_clouds,
)
from cad_recode.dataset import CadRecodeDataset
from cad_recode.model import CADRecodeModel
from cad_recode.utils import chamfer_distance   # existing util


# --------------------------------------------------------------------------- #
#                           distributed helpers                               #
# --------------------------------------------------------------------------- #
def _init_dist() -> Tuple[torch.device, int]:
    """
    Initialise distributed environment if run under torchrun / srun.

    Returns
    -------
    device       : torch.device for this rank
    local_rank   : int
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, local_rank


# --------------------------------------------------------------------------- #
#                              data loaders                                   #
# --------------------------------------------------------------------------- #
def _build_loaders(cfg, rank: int):
    train_set = CadRecodeDataset(cfg.data.path, split="train",
                                 n_points=cfg.data.n_points,
                                 noise_std=cfg.data.noise_std,
                                 noise_prob=cfg.data.noise_prob,
                                 preload=True)
    val_set   = CadRecodeDataset(cfg.data.path, split="val",
                                 n_points=cfg.data.n_points,
                                 noise_std=0.0, noise_prob=0.0,
                                 preload=True)

    if world_size() > 1:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_set,   shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(train_set,
                              batch_size=cfg.training.batch_size,
                              sampler=train_sampler,
                              shuffle=train_sampler is None,
                              num_workers=cfg.training.num_workers,
                              pin_memory=True)

    val_loader   = DataLoader(val_set,
                              batch_size=cfg.training.batch_size,
                              sampler=val_sampler,
                              shuffle=False,
                              num_workers=cfg.training.num_workers // 2,
                              pin_memory=True)
    return train_loader, val_loader, train_sampler


# --------------------------------------------------------------------------- #
#                               validate                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _validate(model: CADRecodeModel,
              loader: DataLoader,
              device: torch.device) -> float:
    model.eval()
    losses, n = 0.0, 0
    for pts, code in loader:
        pts = pts.to(device, non_blocking=True)
        code_end = [s + "<|end|>" for s in code]
        out = model(pts, code=code_end, labels=code_end)
        losses += out.loss.item() * pts.size(0)
        n += pts.size(0)
    model.train()
    return losses / max(n, 1)


# --------------------------------------------------------------------------- #
#                             qualitative dump                                #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _save_visuals(model: CADRecodeModel,
                  epoch_dir: Path,
                  val_loader: DataLoader,
                  device: torch.device,
                  max_samples: int = 3):
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = 0
    for pts, code_gt in val_loader:
        pts = pts.to(device)
        embeds, mask = model.prepare_prefix(pts)
        out = model.decoder.generate(inputs_embeds=embeds,
                                     attention_mask=mask,
                                     max_new_tokens=256,
                                     num_beams=3,
                                     eos_token_id=model.end_id)
        pred_code = model.tokenizer.decode(out[0], skip_special_tokens=False)

        # Execute predicted code ➜ point cloud
        from cad_recode.dataset import _execute_cadquery  # reuse helper
        shape_pred = _execute_cadquery(pred_code)
        shape_gt   = _execute_cadquery(code_gt[0])
        if shape_pred is None or shape_gt is None:
            continue
        pts_pred = torch.from_numpy(
            sample_points_on_shape(shape_pred, 256)
        )
        pts_gt = torch.from_numpy(
            sample_points_on_shape(shape_gt, 256)
        )
        cd = chamfer_distance(pts_pred.numpy(), pts_gt.numpy())

        # Write codes
        (epoch_dir / f"sample_{count:02d}_codes.txt").write_text(
            f"Chamfer: {cd:.4f}\n\n[GT]\n{code_gt[0]}\n\n[PRED]\n{pred_code}"
        )
        # Plot comparison
        compare_point_clouds(pts_gt.numpy(), pts_pred.numpy(),
                             epoch_dir / f"sample_{count:02d}.png",
                             title=f"Chamfer {cd:.4f}")

        count += 1
        if count >= max_samples:
            break
    model.train()


# --------------------------------------------------------------------------- #
#                              main training loop                             #
# --------------------------------------------------------------------------- #
def train(cfg):
    device, local_rank = _init_dist()
    logger = setup_logger(cfg.training.output_dir, rank=get_rank())

    if is_main_process():
        logger.info("⚙️  Resolved config:\n" + OmegaConf.to_yaml(cfg))
        (Path(cfg.training.output_dir) / "run_config.yaml").write_text(
            OmegaConf.to_yaml(cfg)
        )

    # Seed
    random.seed(42 + get_rank())
    np.random.seed(42 + get_rank())
    torch.manual_seed(42 + get_rank())

    # Data
    train_loader, val_loader, train_sampler = _build_loaders(cfg, get_rank())

    # Model
    model = CADRecodeModel(cfg.model.name,
                           freeze_decoder=cfg.model.freeze_decoder,
                           pos_enc=True)
    model.to(device)
    if world_size() > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optim + sched
    optim = torch.optim.AdamW(model.parameters(),
                              lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)
    warm_sched = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-6, total_iters=cfg.training.warmup_steps
    )
    cos_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(cfg.training.max_steps - cfg.training.warmup_steps, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim, [warm_sched, cos_sched], milestones=[cfg.training.warmup_steps]
    )

    # Resume?
    start_epoch = 0
    global_step = 0
    ckpt_path = Path(cfg.training.output_dir) / "latest_checkpoint.pth"
    if getattr(cfg.training, "resume_path", None) and Path(cfg.training.resume_path).exists():
        ckpt_path = Path(cfg.training.resume_path)
    if ckpt_path.exists():
        start_epoch, global_step = load_checkpoint(
            ckpt_path, model, optim, scheduler
        )
        logger.info(f"🔄 Resumed from {ckpt_path} (epoch {start_epoch}, step {global_step})")
    barrier()

    # Training loop
    best_val = float("inf")
    accum = max(cfg.training.get("accumulation_steps", 1), 1)
    grad_ctx = model.no_sync if (world_size() > 1 and accum > 1) else nullcontext

    for epoch in range(start_epoch, cfg.training.max_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, disable=not is_main_process(),
                    desc=f"Epoch {epoch}", leave=False)

        for b_idx, (pts, codes) in enumerate(pbar):
            pts = pts.to(device, non_blocking=True)
            codes_end = [c + "<|end|>" for c in codes]

            outputs = model(pts, code=codes_end, labels=codes_end)
            loss = outputs.loss / accum

            with grad_ctx():
                loss.backward()

            if (b_idx + 1) % accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if is_main_process() and global_step % cfg.logging.log_interval == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(f"[{global_step}] loss={loss.item()*accum:.4f} lr={lr_now:.2e}")
                    pbar.set_postfix(loss=loss.item()*accum, lr=f"{lr_now:.1e}")

                # Validation
                if global_step % cfg.training.val_interval == 0:
                    val_loss = _validate(model, val_loader, device)
                    if is_main_process():
                        logger.info(f"💡 Step {global_step} | val_loss={val_loss:.4f}")

                        # Check best model
                        is_best = val_loss < best_val
                        if is_best:
                            best_val = val_loss

                        # Checkpoint
                        state = {"model": model.state_dict(),
                                 "optimizer": optim.state_dict(),
                                 "scheduler": scheduler.state_dict(),
                                 "epoch": epoch,
                                 "step": global_step,
                                 "best_val": best_val}
                        save_checkpoint(state,
                                        Path(cfg.training.output_dir) / "latest_checkpoint.pth")
                        if is_best:
                            save_checkpoint(state,
                                            Path(cfg.training.output_dir) / "best_model.pth")

        # End epoch – qualitative dump & checkpoint
        if is_main_process():
            epoch_dir = Path(cfg.training.output_dir) / f"epoch_{epoch:02d}"
            _save_visuals(model.module if isinstance(model, DDP) else model,
                          epoch_dir, val_loader, device)

    barrier()
    if is_main_process():
        # Final summary
        summary = {"best_val": best_val,
                   "epochs_run": cfg.training.max_epochs,
                   "total_steps": global_step}
        (Path(cfg.training.output_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info("🏁 Training finished")


# --------------------------------------------------------------------------- #
#                      Hydra / CLI entry-point                                #
# --------------------------------------------------------------------------- #
def _parse_and_run():
    @hydra.main(version_base=None, config_path="../../config", config_name="config")
    def _inner(cfg):
        Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
        train(cfg)
    _inner()  # pylint: disable=no-value-for-parameter

if __name__ == "__main__":
    # If launched via torchrun, Hydra’s path may be off – handle gracefully
    in_jupyter = "ipykernel" in sys.modules
    if in_jupyter:
        cfg = OmegaConf.load("config.yaml")
        train(cfg)
    else:
        _parse_and_run()
