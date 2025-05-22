# cad_recode/train.py
"""Top‑level training entry point for CAD‑Recode.

All runtime parameters are read from **config.yaml** via Hydra.  No value is
hard‑coded in the script – edit *config.yaml* to change data paths, model
hyper‑parameters, training schedule, logging, etc.

Style guidelines
----------------
* The **main training loop** is intentionally compact and calls helper
  functions so you can see the flow at a glance.
* All side‑effects (checkpoints, logs, wandb runs, …) live under
  `output/run_*` which Hydra keeps independent of source files.
* Every RNG (Python `random`, NumPy, PyTorch CPU/GPU) is seeded from
  `cfg.training.seed` to guarantee reproducibility.
* Optional **Weights & Biases** tracking is enabled by
  `cfg.logging.use_wandb`.  If `false`, the code falls back to plain console
  + file logging only.
"""
from __future__ import annotations

import gc
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:  # Hydra not installed – hard error
    raise RuntimeError("Hydra is required:  pip install hydra-core") from e

# ‑‑ local imports -----------------------------------------------------------
from cad_recode.dataset import CadRecodeDataset
from cad_recode.model import CADRecodeModel
from cad_recode.utils import chamfer_distance, sample_points_on_shape, save_point_cloud

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def init_distributed() -> SimpleNamespace:
    """Initialise (optional) DDP from standard `torchrun` env variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        local_rank = rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return SimpleNamespace(distributed=distributed, world_size=world_size,
                           local_rank=local_rank, rank=rank, device=device)


def seed_everything(seed: int) -> None:
    """Seed *all* relevant random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def init_logger(log_file: Path) -> logging.Logger:
    """Console + file logger without duplicate handlers (Jupyter‑safe)."""
    logger = logging.getLogger("cad_recode.train")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):  # reset if re‑running in notebook
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                             "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Dataset + dataloader helpers
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DictConfig,
                       env: SimpleNamespace) -> tuple[DataLoader, DataLoader]:
    """Return train / val dataloaders obeying distributed sampling."""
    ds_train = CadRecodeDataset(cfg.data.path, "train",
                                n_points=cfg.data.n_points,
                                noise_std=cfg.data.noise_std,
                                noise_prob=cfg.data.noise_prob)
    ds_val = CadRecodeDataset(cfg.data.path, "val",
                              n_points=cfg.data.n_points,
                              noise_std=0.0, noise_prob=0.0)

    if cfg.training.mode == "smoke":  # tiny subset for pipeline testing
        ds_train.files = ds_train.files[:8]
        ds_val.files = ds_val.files[:4]

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if env.distributed:
        train_sampler = DistributedSampler(ds_train, shuffle=True)
        val_sampler = DistributedSampler(ds_val, shuffle=False)

    dl_train = DataLoader(ds_train,
                          batch_size=cfg.training.batch_size,
                          sampler=train_sampler,
                          shuffle=train_sampler is None,
                          num_workers=cfg.training.num_workers,
                          pin_memory=True)

    dl_val = DataLoader(ds_val,
                        batch_size=cfg.evaluation.batch_size,
                        sampler=val_sampler,
                        shuffle=False,
                        num_workers=max(cfg.training.num_workers // 2, 0),
                        pin_memory=True)
    return dl_train, dl_val


# ---------------------------------------------------------------------------
# Validation step (no‑grad)
# ---------------------------------------------------------------------------

def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for pts, code in loader:
            pts = pts.to(device, non_blocking=True)
            code_io = ["<|start|>" + s + "<|end|>" for s in code]
            out = model(pts, code=code_io, labels=code_io)
            loss_sum += out.loss.item() * pts.size(0)
            n += pts.size(0)
    model.train()
    return loss_sum / max(1, n)


# ---------------------------------------------------------------------------
# Qualitative epoch snapshot
# ---------------------------------------------------------------------------

def save_epoch_snapshot(model: nn.Module, sample: tuple[torch.Tensor, str],
                        out_dir: Path, device: torch.device,
                        max_tokens: int = 256) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pts, gt_code = sample
    pts = pts.unsqueeze(0).to(device)

    with torch.no_grad():
        prefix, attn = model.prepare_prefix(pts)
        gen_tok = model.decoder.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            max_new_tokens=max_tokens,
            num_beams=3,
            do_sample=False,
            eos_token_id=model.end_id,
        )[0]
        pred_code = model.tokenizer.decode(gen_tok, skip_special_tokens=False)

    # always write code strings – even if shapes are missing
    (out_dir / "ground_truth.py").write_text(gt_code + "\n")
    (out_dir / "predicted.py").write_text(pred_code + "\n")

    # try to export PLYs – silently skip if shapes not present
    for tag, code in (("ground_truth", gt_code), ("predicted", pred_code)):
        try:
            loc: dict[str, object] = {}
            exec(code, {"cq": __import__("cadquery").cq}, loc)
            solid = loc.get("result") or loc.get("r") or loc.get("shape")
            if isinstance(solid, __import__("cadquery").cq.Workplane):
                solid = solid.val()
            if solid is None:
                continue
            pc = sample_points_on_shape(solid, 1024)
            save_point_cloud(pc, out_dir / f"{tag}.ply")
        except Exception:
            continue  # soft‑fail – snapshot remains useful


# ---------------------------------------------------------------------------
# Main training entry – Hydra manages CLI / overrides
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):  # noqa: C901 – complex but readable
    env = init_distributed()
    seed_everything(cfg.training.seed)

    # --- output directories ------------------------------------------------
    proj_root = Path(__file__).resolve().parent.parent
    runs_root = proj_root / "output"
    runs_root.mkdir(exist_ok=True)
    next_idx = max([int(p.name.split("_")[1]) for p in runs_root.glob("run_*")
                    if p.name.split("_")[1].isdigit()] or [0]) + 1
    run_dir = runs_root / f"run_{next_idx:03d}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir()
    ckpt_dir.mkdir()

    logger = init_logger(run_dir / "log.txt")
    logger.info("New run directory → %s", run_dir)
    logger.info("Config (flattened):\n%s", OmegaConf.to_yaml(cfg))

    # --- optional Weights & Biases ----------------------------------------
    if cfg.logging.use_wandb:
        import wandb  # local import so project works without wandb installed
        wandb.init(project=cfg.logging.project_name, name=run_dir.name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # --- dataloaders -------------------------------------------------------
    dl_train, dl_val = build_dataloaders(cfg, env)
    first_val_sample: Optional[tuple[torch.Tensor, str]] = dl_val.dataset[0] if len(dl_val.dataset) else None

    # --- model / optimiser / scheduler ------------------------------------
    model = CADRecodeModel(llm_name=cfg.model.name,
                           freeze_decoder=cfg.model.freeze_decoder,
                           pos_enc=cfg.model.pos_enc)
    model.to(env.device)
    if env.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[env.local_rank])

    optim = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    total_steps = cfg.training.max_steps if cfg.training.max_steps > 0 else None
    if total_steps is None:
        total_steps = cfg.training.max_epochs * len(dl_train)
    warmup_steps = cfg.training.warmup_steps
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-3, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps - warmup_steps, 1))
        ],
        milestones=[warmup_steps]
    )

    # --- training loop -----------------------------------------------------
    step, best_val = 0, float("inf")
    accum_steps = max(1, cfg.training.accumulation_steps)
    grad_ctx = model.no_sync if (accum_steps > 1 and hasattr(model, "no_sync")) else torch.enable_grad

    epoch = 0
    while step < total_steps and epoch < cfg.training.max_epochs:
        epoch += 1
        if env.distributed:
            dl_train.sampler.set_epoch(epoch)  # type: ignore[arg-type]

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}", disable=env.rank != 0)
        for i, (pts, code_str) in enumerate(pbar):
            pts = pts.to(env.device, non_blocking=True)
            code_io = ["<|start|>" + s + "<|end|>" for s in code_str]
            outputs = model(pts, code=code_io, labels=code_io)
            loss = outputs.loss / accum_steps

            with grad_ctx():
                loss.backward()
            if (i + 1) % accum_steps == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                if env.rank == 0 and step % cfg.logging.log_interval == 0:
                    lr_now = sched.get_last_lr()[0]
                    logger.info("step %d/%d  loss=%.4f  lr=%e", step, total_steps, loss.item() * accum_steps, lr_now)
                    if cfg.logging.use_wandb:
                        wandb.log({"loss": loss.item() * accum_steps, "lr": lr_now, "epoch": epoch, "step": step})

                if step % cfg.training.val_interval == 0 or step >= total_steps:
                    if env.rank == 0:
                        val_loss = validate(model, dl_val, env.device)
                        logger.info("val_loss=%.4f (step %d)", val_loss, step)
                        if cfg.logging.use_wandb:
                            wandb.log({"val_loss": val_loss, "epoch": epoch, "step": step})
                        best_val = min(best_val, val_loss)
                    if env.distributed:
                        dist.barrier()

            if step >= total_steps:
                break

        # --- end‑of‑epoch snapshot + ckpt ---------------------------------
        if env.rank == 0:
            val_loss = validate(model, dl_val, env.device)
            best_val = min(best_val, val_loss)
            snap_dir = ckpt_dir / f"epoch_{epoch:03d}"
            if first_val_sample is not None:
                save_epoch_snapshot(model, first_val_sample, snap_dir, env.device)

            sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            ckpt_file = ckpt_dir / f"ckpt_epoch_{epoch:03d}.pt"
            torch.save({"model": sd, "optim": optim.state_dict(), "sched": sched.state_dict(),
                        "step": step, "val_loss": val_loss}, ckpt_file)
            logger.info("Checkpoint saved → %s", ckpt_file.name)

        if env.distributed:
            dist.barrier()

    # --------------------------------------------------------------------
    # Final evaluation (optional) – delegate to `evaluate.py`
    # --------------------------------------------------------------------
    if cfg.training.run_final_test and env.rank == 0:
        logger.info("Running final evaluation on the test split …")
        from cad_recode.evaluate import main as evaluate_main
        eval_args = SimpleNamespace(
            data_root=cfg.data.path,
            checkpoint=str(ckpt_file),  # last epoch ckpt
            split="test",
            llm=cfg.model.name,
            max_length=cfg.evaluation.max_length,
            num_candidates=cfg.evaluation.num_candidates,
            output_dir=str(run_dir),
            save_examples=cfg.evaluation.save_examples,
        )

        # free GPU before spawning a fresh model
        model.to("cpu")
        del model, optim, sched
        gc.collect()
        torch.cuda.empty_cache()

        evaluate_main(eval_args)

    # --------------------------------------------------------------------
    if cfg.logging.use_wandb and env.rank == 0:
        wandb.finish()

    if env.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
