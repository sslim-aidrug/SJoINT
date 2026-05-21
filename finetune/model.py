"""Downstream model: pretrained SJoINT backbone + MLP prediction head."""
import importlib.util
import os

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from dataset import build_cross_attn_mask

_PRETRAIN_MODEL_PY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "pretrain", "model.py")


def _load_sjoint_class(model_py=None):
    """Import SJoINTModel from pretrain/model.py without polluting sys.path."""
    path = model_py or _PRETRAIN_MODEL_PY
    spec = importlib.util.spec_from_file_location("pretrain_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SJoINTModel


def _detect_proj_dim(state_dict):
    """Auto-detect proj_dim from checkpoint weights."""
    for key in ("mol_proj.2.weight", "_orig_mod.mol_proj.2.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    return 32  # default


def _detect_arch(state_dict):
    """Auto-detect hidden_dim, mol_num_layers, jt_num_layers from ckpt state_dict."""
    # hidden_dim from final_norm_mol.weight shape (LayerNorm: (H,))
    hidden = None
    for key in ("final_norm_mol.weight", "_orig_mod.final_norm_mol.weight"):
        if key in state_dict:
            hidden = int(state_dict[key].shape[0]); break
    # Count conv layers via *.convs.{i}.lin_l.weight (GATv2)
    mol_n = sum(1 for k in state_dict if k.startswith(("mol_encoder.convs.", "_orig_mod.mol_encoder.convs."))
                and k.endswith(".lin_l.weight"))
    jt_n = sum(1 for k in state_dict if k.startswith(("jt_encoder.convs.", "_orig_mod.jt_encoder.convs."))
               and k.endswith(".lin_l.weight"))
    return (hidden or 32, mol_n or 3, jt_n or 2)


def load_pretrained_backbone(pretrain_ckpt, device, jt_vocab_size=1753, num_ca_blocks=3):
    """Load pretrained SJoINTModel from checkpoint."""
    SJoINTModel = _load_sjoint_class()

    state = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    proj_dim = _detect_proj_dim(cleaned)
    hidden_dim, mol_n, jt_n = _detect_arch(cleaned)
    backbone = SJoINTModel(
        jt_vocab_size=jt_vocab_size, num_ca_blocks=num_ca_blocks,
        proj_dim=proj_dim, hidden_dim=hidden_dim,
        mol_num_layers=mol_n, jt_num_layers=jt_n).to(device)
    backbone.load_state_dict(cleaned, strict=False)
    return backbone


def create_random_backbone(jt_vocab_size=1753, num_ca_blocks=3, device="cpu"):
    """Create a randomly initialised SJoINTModel (no pretrain)."""
    SJoINTModel = _load_sjoint_class()
    return SJoINTModel(jt_vocab_size=jt_vocab_size, num_ca_blocks=num_ca_blocks).to(device)


class ResidualBlock(nn.Module):
    """Linear -> norm -> activation -> dropout, with optional residual connection."""

    def __init__(self, in_dim, out_dim, dropout, activation="relu", norm="layer"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if norm == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()
        self.act = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }.get(activation, nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.use_residual = (in_dim == out_dim)

    def forward(self, x):
        out = self.drop(self.act(self.norm(self.linear(x))))
        return out + x if self.use_residual else out


class DownstreamModel(nn.Module):
    """Pretrained SJoINT backbone + MLP head for property prediction.

    Args:
        pretrained_model: SJoINTModel instance (pretrained or random).
        num_tasks: Number of output targets.
        num_head_layers: Depth of the MLP prediction head.
        head_hidden: Hidden dimension of the MLP head.
        head_dropout: Dropout rate in the MLP head.
        head_activation: Activation function ('relu', 'gelu', 'silu').
        head_norm: Normalisation type ('layer', 'batch', 'none').
        pool_mode: 'both' (concat JT + MOL), 'jt_only', or 'mol_only'.
    """

    def __init__(self, pretrained_model, num_tasks, *,
                 num_head_layers=2, head_hidden=32, head_dropout=0.1,
                 head_activation="relu", head_norm="layer", pool_mode="both"):
        super().__init__()
        self.hidden_dim = pretrained_model.hidden_dim
        self.mol_encoder = pretrained_model.mol_encoder
        self.jt_encoder = pretrained_model.jt_encoder
        self.blocks = pretrained_model.blocks
        self.final_norm_jt = pretrained_model.final_norm_jt
        self.final_norm_mol = pretrained_model.final_norm_mol

        self.pool_mode = pool_mode
        if pool_mode == "both":
            input_dim = self.hidden_dim * 2
        else:
            input_dim = self.hidden_dim
        self.pred_head = self._build_head(
            input_dim, num_tasks, num_head_layers, head_hidden,
            head_dropout, head_activation, head_norm)

    @staticmethod
    def _build_head(input_dim, num_tasks, num_layers, hidden, dropout, activation, norm):
        if num_layers == 1:
            return nn.Linear(input_dim, num_tasks)

        layers = []
        prev = input_dim
        dim = hidden
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(prev, dim, dropout, activation, norm))
            prev = dim
            dim = max(dim // 2, 8)
        layers.append(nn.Linear(prev, num_tasks))
        return nn.Sequential(*layers)

    @staticmethod
    def _pool(h_padded, pad_mask):
        """Mean pooling: (B, N, D) -> (B, D)."""
        m = pad_mask.unsqueeze(-1)
        return (h_padded * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, jt_batch, mol_batch,
                jt2mol_b, jt2mol_r, jt2mol_c,
                mol2jt_b, mol2jt_r, mol2jt_c):
        with torch.amp.autocast("cuda", enabled=False):
            jt_h = self.jt_encoder(
                jt_batch.vocab_ids, jt_batch.features, jt_batch.edge_index)
            mol_h = self.mol_encoder(
                mol_batch.atom_ids, mol_batch.features,
                mol_batch.edge_index, mol_batch.edge_attr)

        jt_h, jt_pm = to_dense_batch(jt_h, jt_batch.batch)
        mol_h, mol_pm = to_dense_batch(mol_h, mol_batch.batch)
        B, max_jt = jt_h.size(0), jt_h.size(1)
        max_mol = mol_h.size(1)

        jt_mask = build_cross_attn_mask(
            jt2mol_b, jt2mol_r, jt2mol_c, B, max_jt, max_mol, mol_pm)
        mol_mask = build_cross_attn_mask(
            mol2jt_b, mol2jt_r, mol2jt_c, B, max_mol, max_jt, jt_pm)

        for block in self.blocks:
            jt_h, mol_h = block(jt_h, mol_h, jt_mask, mol_mask)
            jt_h = jt_h.masked_fill(~jt_pm.unsqueeze(-1), 0.0)
            mol_h = mol_h.masked_fill(~mol_pm.unsqueeze(-1), 0.0)

        jt_h = self.final_norm_jt(jt_h)
        mol_h = self.final_norm_mol(mol_h)

        if self.pool_mode == "jt_only":
            rep = self._pool(jt_h, jt_pm)
        elif self.pool_mode == "mol_only":
            rep = self._pool(mol_h, mol_pm)
        else:
            rep = torch.cat(
                [self._pool(jt_h, jt_pm),
                 self._pool(mol_h, mol_pm)], dim=-1)
        return self.pred_head(rep)


def freeze_backbone(model):
    """Disable gradients for all parameters except the prediction head."""
    for name, param in model.named_parameters():
        if not name.startswith("pred_head"):
            param.requires_grad = False


def unfreeze_backbone(model):
    """Enable gradients for all parameters."""
    for param in model.parameters():
        param.requires_grad = True
