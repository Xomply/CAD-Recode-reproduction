# cad_recode/model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class PointCloudProjector(nn.Module):
    """Encodes a 3D point cloud into a fixed-width vector that matches the LLM."""
    def __init__(self, output_dim: int, pos_enc: bool = False):
        super(PointCloudProjector, self).__init__()
        self.pos_enc = pos_enc
        # If using Fourier positional encoding, define here (not shown for brevity)
        input_dim = 3  # using raw (x,y,z)
        # Simple MLP to expand point features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # output feature per point
        )
        # Note: output_dim should match LLM embedding size (1536 for Qwen2-1.5B).
    
    def forward(self, points):
        """
        points: Tensor of shape (B, N, 3) -- batch of point clouds.
        Returns: Tensor of shape (B, output_dim) encoding each point cloud.
        """
        x = points  # (B, N, 3)
        # (Optional) apply positional encoding to x here if pos_enc=True.
        feat = self.mlp(x)          # (B, N, output_dim)
        return feat

class CADRecodeModel(nn.Module):
    """Combined model: uses a projector to encode points and an LLM to decode code."""
    def __init__(self, llm_name="Qwen/Qwen2-0.5B", freeze_decoder=False):
        super(CADRecodeModel, self).__init__()
        # 1) load decoder first so we can read its embedding size
        self.decoder   = AutoModelForCausalLM.from_pretrained(llm_name)
        self.embed_dim = self.decoder.get_input_embeddings().embedding_dim

        # 2) now build the projector with the **correct** output size
        self.projector = PointCloudProjector(output_dim=self.embed_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        # (Optionally freeze the decoder if we only want to train projector)
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        # Add special start/end tokens to tokenizer, if not present
        special_tokens = {"additional_special_tokens": ["<|start|>", "<|end|>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        # Save the IDs for convenience
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start|>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
    
    def forward(self, points, code=None, labels=None):
        """
        points: Tensor (B, N, 3)
        code: either a token ID tensor of shape (B, T) or a list of code strings.
        labels: (B, T) tensor of token IDs for training (including end token), optional.
        """
        B = points.size(0)
        device = points.device
        # Project points to get prefix embedding
        prefix_emb = self.projector(points)              # (B, embed_dim)
        # If code is provided as strings, tokenize to get input_ids
        if code is not None and not torch.is_tensor(code):
            # code is a list of strings
            tokenized = self.tokenizer(list(code), return_tensors='pt', padding=True)
            input_ids = tokenized["input_ids"].to(device)
        else:
            input_ids = code  # assume already token IDs tensor on device
        if input_ids is None:
            # If no code provided (e.g. during generation only prefix), we create a dummy start token
            input_ids = torch.full((B, 1), self.start_id, dtype=torch.long, device=device)
        # Get token embeddings for the input code tokens
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B, T, embed_dim)
        # safety check while debugging
        assert prefix_emb.size(-1) == inputs_embeds.size(-1)
        print("prefix_emb shape:", prefix_emb.shape)
        print("inputs_embeds shape:", inputs_embeds.shape)
        assert prefix_emb.dim() == 3 and prefix_emb.size(1) == 256, \
        f"Expected (B,256,E), got {prefix_emb.shape}"

        inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)   # (B,1+T,embed_dim)

        # Create attention mask (1 for all real tokens including prefix)
        seq_length = inputs_embeds.size(1)
        attention_mask = torch.ones((B, seq_length), dtype=torch.long, device=device)
        # If labels are provided for training, we need to align them with the decoder outputs.
        # The decoder will output a sequence of length equal to input_embeds length. We want it to predict the code tokens.
        new_labels = None
        if labels is not None:
            if not torch.is_tensor(labels):
                # tokenize labels if given as strings
                labels_tok = self.tokenizer(list(labels), return_tensors='pt', padding=True)
                labels_tensor = labels_tok["input_ids"]
            else:
                labels_tensor = labels
            labels_tensor = labels_tensor.to(device)
            # Prepend a dummy label for the prefix token so that loss for prefix is ignored
            prefix_len = prefix_emb.size(1)  # = 256
            pad = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
            new_labels = torch.cat([pad, labels_tensor], dim=1)
        # Forward pass through the decoder LLM
        outputs = self.decoder(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            labels=new_labels
        )
        # outputs is a CausalLMOutput. If labels were provided, outputs.loss is the NLL loss.

        assert prefix_emb.dim() == 3 and prefix_emb.size(1) == 256, \
        f"Projector must output (B,256,E) but got {prefix_emb.shape}"
        return outputs  # we will use outputs.loss during training
