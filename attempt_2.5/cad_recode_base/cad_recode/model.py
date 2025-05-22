"""
CADRecodeModel – conceptual skeleton

Architecture Overview
----------------------
• PointCloudProjector → per-point token embeddings (256×3 → N×E)
• Causal LM decoder (Qwen-1.5B) as text generator.
• Special tokens: <|start|>, <|end|>.
• Forward path returns `transformers.CausalLMOutput` (loss if labels given).

Class layout
------------
class PointCloudProjector(nn.Module):
    - __init__(output_dim: int, pos_enc: bool = False)
    - forward(points: Tensor[B,N,3]) → Tensor[B,N,E]

class CADRecodeModel(nn.Module):
    - __init__(llm_name: str, freeze_decoder: bool, pos_enc: bool)
        * load tokenizer + HF model, add special tokens.
        * instantiate projector.
    - forward(points, code=None, labels=None)
        * compose embeddings: [point_tokens] + [text tokens].
    - prepare_prefix(points) → (inputs_embeds, attention_mask)
        * convenience for model.generate().

Distributed considerations
-------------------------
• Model will be wrapped by torch.nn.parallel.DistributedDataParallel in train.py – this class itself is framework-agnostic.
• `.to(device)` must move both decoder and projector.

Checkpoint interface
--------------------
• State dict saving / loading handled externally by `train.py` (keep interface simple).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
#                         Point cloud → token projector                       #
# --------------------------------------------------------------------------- #
class PointCloudProjector(nn.Module):
    """Encode each 3-D point independently to an embedding."""

    def __init__(self, output_dim: int, pos_enc: bool = False):
        super().__init__()
        self.pos_enc = pos_enc
        in_dim = 3 * (1 + 4) if pos_enc else 3  # Fourier 4 frequencies × sin+cos
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pts : (B, N, 3)  – normalised point clouds

        Returns
        -------
        tok : (B, N, E)  – per-point embeddings
        """
        if self.pos_enc:
            # Fourier features – fixed λ = 1,2,4,8
            freqs = torch.pow(2.0, torch.arange(4, device=pts.device))
            pts_exp = pts.unsqueeze(-1) * freqs        # (B,N,3,4)
            pe = torch.cat([pts_exp.sin(), pts_exp.cos()], dim=-1)  # (B,N,3,8)
            pe = pe.flatten(-2)                        # (B,N,24)
            x = torch.cat([pts, pe], dim=-1)
        else:
            x = pts
        return self.mlp(x)


# --------------------------------------------------------------------------- #
#                          Main CAD-Recode wrapper                            #
# --------------------------------------------------------------------------- #
class CADRecodeModel(nn.Module):
    """Combine point-token prefix with a HuggingFace causal decoder."""

    def __init__(self,
                 llm_name: str = "Qwen/Qwen2-1.5B",
                 freeze_decoder: bool = False,
                 pos_enc: bool = False):
        super().__init__()
        # 1) Load decoder first – we need embedding size
        self.decoder = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        embed_dim = self.decoder.get_input_embeddings().embedding_dim

        # 2) Instantiate projector
        self.projector = PointCloudProjector(embed_dim, pos_enc=pos_enc)

        # 3) Tokeniser & special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|start|>", "<|end|>"]}
        )
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start|>")
        self.end_id   = self.tokenizer.convert_tokens_to_ids("<|end|>")

        # 4) Optional freeze
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    #                               forward                              #
    # ------------------------------------------------------------------ #
    def forward(self,
                points: torch.Tensor,
                code:   list[str] | torch.Tensor | None = None,
                labels: list[str] | torch.Tensor | None = None):
        """
        Parameters
        ----------
        points : (B, N_pts, 3) torch.float32
        code   : list[str] *or* token IDs (input sequence)
        labels : if provided, used for LM loss.

        Returns
        -------
        transformers.CausalLMOutput
        """
        device = points.device
        B, N_pts, _ = points.shape

        # -- prefix from point cloud ------------------------------------
        pt_tok = self.projector(points)                # (B, N_pts, E)

        # -- text input -------------------------------------------------
        if code is not None:
            if torch.is_tensor(code):
                input_ids = code.to(device)
            else:  # list[str]
                enc = self.tokenizer(code, return_tensors="pt",
                                      padding=True, truncation=True)
                input_ids = enc["input_ids"].to(device)
        else:
            input_ids = torch.full((B, 1), self.start_id,
                                   dtype=torch.long, device=device)

        txt_emb = self.decoder.get_input_embeddings()(input_ids)

        # concat
        inputs_embeds = torch.cat([pt_tok, txt_emb], dim=1)  # (B, N_pts+T, E)
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                               device=device)

        # -- labels -----------------------------------------------------
        new_labels = None
        if labels is not None:
            if torch.is_tensor(labels):
                lbl_ids = labels.to(device)
            else:
                enc = self.tokenizer(labels, return_tensors="pt",
                                     padding=True, truncation=True)
                lbl_ids = enc["input_ids"].to(device)
            pad = torch.full((B, N_pts), -100, dtype=torch.long, device=device)
            new_labels = torch.cat([pad, lbl_ids], dim=1)

        return self.decoder(inputs_embeds=inputs_embeds,
                            attention_mask=attn_mask,
                            labels=new_labels)

    # ------------------------------------------------------------------ #
    #               helper for `.generate()` convenience                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prepare_prefix(self, points: torch.Tensor):
        """
        Return inputs_embeds + attn_mask ready for HF `.generate()`.

        Adds a textual <|start|> token immediately after point tokens.
        """
        pt_tok = self.projector(points)  # (B,N,E)
        B, N, E = pt_tok.shape
        start_ids = torch.full((B, 1), self.start_id,
                               dtype=torch.long, device=points.device)
        start_emb = self.decoder.get_input_embeddings()(start_ids)
        embeds = torch.cat([pt_tok, start_emb], dim=1)
        mask = torch.ones((B, N + 1), dtype=torch.long, device=points.device)
        return embeds, mask
