# cad_recode/train.py
"""
Unified training entry‑point for CAD‑Recode.

• Works with Hydra when launched from a normal Python process.
• Falls back to a direct OmegaConf load when executed inside Jupyter/IPython
  so you avoid the notorious "--f=kernel‑xxxx.json" argparse error.
"""

import os, sys, time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm                    # nice progress bars
import wandb

# local imports
from cad_recode.dataset import CadRecodeDataset
from cad_recode.model   import CADRecodeModel



# --------------------------------------------------------------------------- #
#                           core training routine                              #
# --------------------------------------------------------------------------- #
def train(cfg):
    """
    cfg : OmegaConf DictConfig  (or any object with the same attribute names)
    """
    # ---------------- dataset & dataloader ----------------------------------
    nw_train = getattr(cfg.training, "num_workers", 0)
    nw_val   = max(nw_train // 2, 0)
    train_ds = CadRecodeDataset(
        cfg.data.path, split="train",
        n_points   = cfg.data.n_points,
        noise_std  = cfg.data.noise_std,
        noise_prob = cfg.data.noise_prob,
    )
    val_ds   = CadRecodeDataset(
        cfg.data.path, split="val",
        n_points   = cfg.data.n_points,
        noise_std  = 0.0,
        noise_prob = 0.0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.training.batch_size,
        shuffle     = True,
        num_workers = nw_train,
        pin_memory  = True,
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size  = cfg.training.batch_size,
        shuffle     = False,
        num_workers = nw_val,
        pin_memory  = True,
    )

    # ---------------- model, optim, sched ----------------------------------
    model = CADRecodeModel(
        llm_name      = cfg.model.name,
        freeze_decoder= cfg.model.freeze_decoder,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA device = ", device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr          = cfg.training.lr,
        weight_decay= cfg.training.weight_decay,
    )

    total_steps  = cfg.training.max_iterations
    warmup_steps = cfg.training.warmup_steps
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.000001, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=0.0
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    # ---------------- logging (W&B) ----------------------------------------
    if cfg.logging.use_wandb:
        wandb.init(project=cfg.logging.project_name, config=dict(cfg))
        wandb.watch(model, log="parameters", log_freq=500)

    log_every = cfg.logging.log_interval
    next_log  = log_every if log_every else float("inf")

    # ---------------- training loop ----------------------------------------
    model.train()
    step           = 0
    accum_steps    = max(getattr(cfg.training, "accumulation_steps", 1), 1)
    grad_ctx       = model.no_sync if accum_steps > 1 else nullcontext

    while step < total_steps:
        for epoch in range(cfg.training.max_epochs):
            pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
            for batch_idx, (points, code_str) in enumerate(pbar):
                points = points.to(device, non_blocking=True)

                # Append <|end|> token for training labels
                code_end = [s + "<|end|>" for s in code_str]
                outputs  = model(points, code=code_end, labels=code_end)
                loss     = outputs.loss / accum_steps

                with grad_ctx():
                    loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1

                    lr_now = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=loss.item()*accum_steps, lr=f"{lr_now:.2e}")

                    if cfg.logging.use_wandb:
                        wandb.log({"train_loss": loss.item()*accum_steps,
                                   "lr": lr_now}, step=step)

                    # ---- validation --------------------------------------
                    if step % cfg.training.val_interval == 0 or step == total_steps:
                        val_loss = validate(model, val_loader, device)
                        if cfg.logging.use_wandb:
                            wandb.log({"val_loss": val_loss}, step=step)
                        print(f"[{time.strftime('%H:%M:%S')}] "
                              f"step {step}/{total_steps}  val_loss={val_loss:.4f}")
                        
                    # ---- debug logging (interval-crossing) --------
                    if step >= next_log:
                        print(f"[{step}] loss={loss.item() * accum_steps:.4f}")

                        # ---- DEBUGGING OUTPUT ----
                        gt_code = code_str[0]  # first item in the batch (str)
                        with torch.no_grad():
                            input_points = points[:1]  # (1, N, 3)
                            prefix = model.projector(input_points)        # (1, 256, E)
                            start  = torch.full((1, 1), model.start_id, dtype=torch.long, device=device)
                            start_emb = model.decoder.get_input_embeddings()(start)  # (1,1,E)
                            init_emb  = torch.cat([prefix, start_emb], dim=1)        # (1,257,E)
                            attn_mask = torch.ones((1, 257), dtype=torch.long, device=device)
                            pred_tokens = model.decoder.generate(
                                inputs_embeds=init_emb,
                                attention_mask=attn_mask,
                                max_new_tokens=256,
                                num_beams=3,
                                do_sample=False,
                                eos_token_id=model.end_id
                            )
                            pred_code = model.tokenizer.decode(pred_tokens[0], skip_special_tokens=False)

                        print("=" * 40)
                        print(f"[DEBUG] Step {step}")
                        print("[GT CODE]\n", gt_code)
                        print("[PREDICTED CODE]\n", pred_code)
                        print("=" * 40)
                        print(f"[DEBUG] GT tokens: {len(model.tokenizer.encode(gt_code))}, Pred tokens: {len(pred_tokens[0])}")
                        # ---- END DEBUGGING ----

                        # Advance logging interval
                        while next_log <= step:
                            next_log += log_every

                    if step >= total_steps:
                        break

    # ---------------- save --------------------------------------------------
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.output_dir, "cad_recode_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    if cfg.logging.use_wandb:
        wandb.save(ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


# --------------------------------------------------------------------------- #
#                        simple validation helper                             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss_sum, n = 0.0, 0
    for pts, code in loader:
        pts = pts.to(device)
        code_end = [s + "<|end|>" for s in code]
        out = model(pts, code=code_end, labels=code_end)
        loss_sum += out.loss.item() * pts.size(0)
        n += pts.size(0)
    model.train()
    return loss_sum / max(n, 1)


# --------------------------------------------------------------------------- #
#                   entry‑point that handles all cases                         #
# --------------------------------------------------------------------------- #
def run():
    """
    • If called from IPython/Jupyter -> load YAML manually, no Hydra.
    • Else (terminal)               -> use Hydra to parse overrides.
    """
    in_notebook = "ipykernel" in sys.argv[0] or "spyder" in sys.argv[0]
    try:
        import hydra
        from omegaconf import OmegaConf, DictConfig
    except ImportError:  # Hydra not installed – fall back to plain config
        hydra = None
        from omegaconf import OmegaConf

    if in_notebook or hydra is None:
        # ------------- notebook / fallback mode ---------------------------
        if not os.path.exists("D:/ML/CAD-Recode reproduction/attempt_1/config/config.yaml"):
            raise FileNotFoundError(
                "Could not find config/config.yaml – create one first."
            )
        cfg = OmegaConf.load("D:/ML/CAD-Recode reproduction/attempt_1/config/config.yaml")
        # Safe defaults for Windows/Jupyter
        cfg.training.num_workers = 0
        print("⚙️  Using OmegaConf config\n", OmegaConf.to_yaml(cfg))
        train(cfg)

    else:
        # ------------- CLI (Hydra) mode -----------------------------------
        @hydra.main(version_base=None, config_path="../config",
                    config_name="config")
        def _main(cfg: DictConfig):
            print("⚙️  Resolved config\n", cfg)
            train(cfg)
        _main()  # pylint: disable=no-value-for-parameter


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run()
