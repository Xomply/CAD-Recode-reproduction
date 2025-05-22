# cad_recode/model.py
"""CAD-Recode: point-cloud encoder + Causal LM decoder.

Major fixes vs. original implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Config-driven positional encoding** – set via ``pos_enc`` flag.
* **Special-token initialisation** – `<|start|>` / `<|end|>` embeddings are
  initialised to the **mean** of the pretrained vocabulary to avoid training
  instability.
* **Token-sequence consistency** –​ training now *prepends* `<|start|>` to
  every input and *appends* `<|end|>` to every label, mirroring generation.
* **Freeze decoder** optional via config.

The class is self-contained; no external globals are required.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["CADRecodeModel", "PointCloudProjector"]

# ---------------------------------------------------------------------------
#  Point-cloud → token projector
# ---------------------------------------------------------------------------


class PointCloudProjector(nn.Module):
    """MLP that converts *(B, N, 3)* points → *(B, N, E)* token embeddings.

    If ``pos_enc`` is *True* we concatenate **Fourier features** (sin/cos of
    4 frequency bands) to the raw XYZ prior to the MLP.
    """

    def __init__(self, output_dim: int, *, pos_enc: bool = False, n_freq: int = 4) -> None:  # noqa: D401
        super().__init__()
        self.pos_enc = pos_enc
        self.n_freq = n_freq

        in_dim = 3 * (1 + n_freq * 2) if pos_enc else 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:  # (B, N, 3)
        if self.pos_enc:
            B, N, _ = pts.shape
            freqs = 2 ** torch.arange(self.n_freq, device=pts.device, dtype=pts.dtype)  # (F,)
            pts_freq = (pts.unsqueeze(-1) * freqs).view(B, N, -1)  # (B, N, 3*F)
            pts_pe = torch.cat([pts_freq.sin(), pts_freq.cos()], dim=-1)  # (B, N, 3*F*2)
            x = torch.cat([pts, pts_pe], dim=-1)
        else:
            x = pts
        return self.mlp(x)


# ---------------------------------------------------------------------------
#  CAD-Recode main model
# ---------------------------------------------------------------------------


class CADRecodeModel(nn.Module):
    """Combine a point-cloud projector with a HuggingFace Causal LM."""

    def __init__(
        self,
        *,
        llm_name: str = "Qwen/Qwen2-1.5B",
        freeze_decoder: bool = False,
        pos_enc: bool = False,
    ) -> None:
        super().__init__()

        # 1) Decoder LM ------------------------------------------------------
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(llm_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not load decoder model '{llm_name}': {exc}") from exc

        embed_dim = self.decoder.get_input_embeddings().embedding_dim

        # 2) Point-token projector -----------------------------------------
        self.projector = PointCloudProjector(embed_dim, pos_enc=pos_enc)

        # 3) Tokeniser + special tokens ------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|start|>", "<|end|>"]})
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start|>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|end|>")

        # Initialise special tokens to mean of original embeddings ----------
        self._init_special_token_embeddings()

        # 4) Optionally freeze decoder weights -----------------------------
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    # ---------------------------------------------------------------------
    #  Special-token helpers
    # ---------------------------------------------------------------------

    def _init_special_token_embeddings(self) -> None:
        """Set `<|start|>`/`<|end|>` rows to mean of pretrained embeddings."""
        weight = self.decoder.get_input_embeddings().weight.data
        # assume the two new tokens are *last* in vocab
        mean_emb = weight[:-2].mean(dim=0, keepdim=True)
        weight[self.start_id] = mean_emb
        weight[self.end_id] = mean_emb

    # ---------------------------------------------------------------------
    #  Forward (training)
    # ---------------------------------------------------------------------

    def forward(
        self,
        points: torch.Tensor,  # (B, N, 3)
        code: Optional[Union[List[str], torch.Tensor]] = None,
        labels: Optional[Union[List[str], torch.Tensor]] = None,
    ) -> torch.nn.modules.module.Module:  # return HF CausalLMOutput
        device = points.device
        B, N_pts, _ = points.shape

        # 1) encode points → token embeddings
        pt_tokens = self.projector(points)  # (B, N_pts, E)

        # 2) prepare textual part
        if code is not None and not torch.is_tensor(code):
            # prepend <|start|>
            code_proc = ["<|start|>" + s for s in code]
            tok = self.tokenizer(code_proc, return_tensors="pt", padding=True)
            input_ids = tok["input_ids"].to(device)
        else:
            input_ids = code or torch.full((B, 1), self.start_id, dtype=torch.long, device=device)

        txt_embeds = self.decoder.get_input_embeddings()(input_ids)

        # 3) concatenate projector prefix + text token embeddings
        inputs_embeds = torch.cat([pt_tokens, txt_embeds], dim=1)  # (B, N+T, E)
        attn_mask = torch.ones((B, inputs_embeds.size(1)), dtype=torch.long, device=device)

        # 4) labels – ignore point tokens & include <|end|>
        new_labels = None
        if labels is not None:
            if not torch.is_tensor(labels):
                labels_proc = [s + "<|end|>" for s in labels]
                tok_lbl = self.tokenizer(labels_proc, return_tensors="pt", padding=True)
                lbl_ids = tok_lbl["input_ids"].to(device)
            else:
                lbl_ids = labels.to(device)
            ignore = torch.full((B, N_pts), -100, dtype=torch.long, device=device)
            new_labels = torch.cat([ignore, lbl_ids], dim=1)

        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=new_labels)

    # ---------------------------------------------------------------------
    #  Prefix for generation
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def prepare_prefix(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return *(inputs_embeds, attention_mask)* for *generate()* call."""
        pt_emb = self.projector(points)  # (B, N, E)
        B, N, _ = pt_emb.shape
        start_ids = torch.full((B, 1), self.start_id, dtype=torch.long, device=points.device)
        start_emb = self.decoder.get_input_embeddings()(start_ids)
        inputs = torch.cat([pt_emb, start_emb], dim=1)  # (B, N+1, E)
        attn = torch.ones((B, N + 1), dtype=torch.long, device=points.device)
        return inputs, attn
