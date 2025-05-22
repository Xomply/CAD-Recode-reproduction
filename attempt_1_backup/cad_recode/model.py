# cad_recode/model.py – *revised* (per‑point token version)
"""CAD‑Recode model with a **sequence** of point‑query tokens (one per
input point) instead of a single pooled vector.

Public API is identical:  ``CADRecodeModel(points, code, labels)`` returns
HF `CausalLMOutput`.  Only the dimensionality of the *prefix* part changes:
(B, N_pts, E) rather than (B, 1, E).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
#  Point‑cloud → sequence of learnable *query tokens*
# ---------------------------------------------------------------------------
class PointCloudProjector(nn.Module):
    """Encode a 3‑D point cloud into *N* token embeddings.

    Each point is mapped independently by an MLP followed by **no pooling**;
    the result has shape ``(B, N_pts, E)``.  Optionally, Fourier
    positional–encodings (`pos_enc=True`) can be appended to the raw XYZ.
    """

    def __init__(self, output_dim: int, pos_enc: bool = False):
        super().__init__()
        self.pos_enc = pos_enc

        in_dim = 3 * (1 + 4) if pos_enc else 3  # 4 pairs of sin/cos for each

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),  # per‑point embedding
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points – (B, N_pts, 3) in **normalised** space."""
        if self.pos_enc:
            # Fourier features (fixed wavelength 1, 2, 4, 8)
            B, N, _ = points.shape
            freqs = 2 ** torch.arange(4, device=points.device).float()  # (4,)
            pts_freq = (points.unsqueeze(-1) * freqs).view(B, N, -1)  # (B,N,3*4)
            pts_pe = torch.cat([pts_freq.sin(), pts_freq.cos()], dim=-1)  # (B,N,3*8)
            x = torch.cat([points, pts_pe], dim=-1)
        else:
            x = points  # (B, N_pts, 3)

        tok = self.mlp(x)  # (B, N_pts, E)
        return tok


# ---------------------------------------------------------------------------
#  CAD‑Recode main wrapper
# ---------------------------------------------------------------------------
class CADRecodeModel(nn.Module):
    """Combine point‑token *prefix* with any causal‑LM decoder (e.g. Qwen2)."""

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2-1.5B",
        freeze_decoder: bool = False,
        pos_enc: bool = False,
    ) -> None:
        super().__init__()

        # 1) Decoder first – so we know the embedding size
        self.decoder = AutoModelForCausalLM.from_pretrained(llm_name)
        embed_dim = self.decoder.get_input_embeddings().embedding_dim

        # 2) Point‑cloud projector returns (B, N_pts, embed_dim)
        self.projector = PointCloudProjector(embed_dim, pos_enc=pos_enc)

        # 3) Tokeniser + special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|start|>", "<|end|>"]
        })
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start|>")
        self.end_id   = self.tokenizer.convert_tokens_to_ids("<|end|>")

        # 4) Optionally freeze decoder (memory‑constrained GPUs)
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    #                        forward / loss
    # ------------------------------------------------------------------
    def forward(
        self,
        points: torch.Tensor,          # (B, N_pts, 3)
        code:   list[str] | torch.Tensor | None = None,
        labels: list[str] | torch.Tensor | None = None,
    ):
        device = points.device
        B, N_pts, _ = points.shape

        # -- 1) point‑cloud → *query token* embeddings -------------------
        pt_tokens = self.projector(points)                      # (B, N_pts, E)

        # -- 2) process textual code input ------------------------------
        if code is not None and not torch.is_tensor(code):
            tokenised = self.tokenizer(list(code), return_tensors="pt", padding=True)
            input_ids = tokenised["input_ids"].to(device)
        else:
            input_ids = code  # may be None (during generation)

        if input_ids is None:
            input_ids = torch.full((B, 1), self.start_id, dtype=torch.long, device=device)

        txt_embeds = self.decoder.get_input_embeddings()(input_ids)   # (B,T,E)

        # -- 3) concatenate:  [pt_tokens] + [code tokens]  ---------------
        inputs_embeds = torch.cat([pt_tokens, txt_embeds], dim=1)      # (B,N_pts+T,E)
        seq_len = inputs_embeds.size(1)
        attention_mask = torch.ones((B, seq_len), dtype=torch.long, device=device)

        # -- 4) labels & loss -------------------------------------------
        new_labels = None
        if labels is not None:
            if not torch.is_tensor(labels):
                lbl_tok = self.tokenizer(list(labels), return_tensors="pt", padding=True)
                lbl_ids = lbl_tok["input_ids"]
            else:
                lbl_ids = labels
            lbl_ids = lbl_ids.to(device)
            pad = torch.full((B, N_pts), -100, dtype=torch.long, device=device)
            new_labels = torch.cat([pad, lbl_ids], dim=1)              # (B,N_pts+T)

        return self.decoder(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            labels         = new_labels,
        )

    # ------------------------------------------------------------------
    #  Convenience helper for *generation* with point tokens             
    # ------------------------------------------------------------------
    def prepare_prefix(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return *(inputs_embeds, attention_mask)* for `model.generate`.

        Adds a textual <|start|> token after the point‑tokens so downstream
        code can do a plain `.generate()` without touching internals.
        """
        with torch.no_grad():
            pt_tok  = self.projector(points)                             # (B,N,E)
            B, N, E = pt_tok.shape
            # textual <|start|>
            start_ids   = torch.full((B, 1), self.start_id, dtype=torch.long, device=points.device)
            start_embed = self.decoder.get_input_embeddings()(start_ids)  # (B,1,E)
            inp_embeds  = torch.cat([pt_tok, start_embed], dim=1)         # (B,N+1,E)
            attn_mask   = torch.ones((B, N+1), dtype=torch.long, device=points.device)
        return inp_embeds, attn_mask
