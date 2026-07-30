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


# --- Backbone architecture auto-detection from a checkpoint state_dict ---
# The finetuner reconstructs the exact backbone from the ckpt alone, so any
# valid pretrain checkpoint loads without passing its architecture by hand.

def _detect_proj_dim(state_dict):
    for key in ("mol_proj.2.weight", "_orig_mod.mol_proj.2.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    return 32


def _detect_arch(state_dict):
    """Auto-detect hidden_dim, mol_num_layers, jt_num_layers from ckpt state_dict."""
    hidden = None
    for key in ("final_norm_mol.weight", "_orig_mod.final_norm_mol.weight"):
        if key in state_dict:
            hidden = int(state_dict[key].shape[0]); break
    # GATv2 conv layers use convs.{i}.lin_l.weight
    mol_n_gat = sum(1 for k in state_dict if k.startswith(("mol_encoder.convs.", "_orig_mod.mol_encoder.convs."))
                    and k.endswith(".lin_l.weight"))
    jt_n_gat = sum(1 for k in state_dict if k.startswith(("jt_encoder.convs.", "_orig_mod.jt_encoder.convs."))
                   and k.endswith(".lin_l.weight"))
    # GIN/GINE conv layers use convs.{i}.nn.* keys instead of lin_l
    mol_n_gin = len({k.split(".")[2] for k in state_dict
                     if k.startswith(("mol_encoder.convs.", "_orig_mod.mol_encoder.convs.")) and ".nn." in k})
    jt_n_gin = len({k.split(".")[2] for k in state_dict
                    if k.startswith(("jt_encoder.convs.", "_orig_mod.jt_encoder.convs.")) and ".nn." in k})
    mol_n = mol_n_gat or mol_n_gin or 3
    jt_n = jt_n_gat or jt_n_gin or 3
    return (hidden or 128, mol_n, jt_n)


def _detect_encoder_gnn_type(state_dict, encoder_prefix):
    """Detect GNN type ('gine'/'gatv2') for a specific encoder from ckpt keys."""
    nn_prefix = f"{encoder_prefix}.convs.0.nn."
    gat_prefix = f"{encoder_prefix}.convs.0.lin_l"
    for k in state_dict:
        if nn_prefix in k:
            return "gine"
        if gat_prefix in k:
            return "gatv2"
    return "gatv2"


def _detect_jt_use_edge_attr(state_dict):
    for key in ("jt_encoder.convs.0.lin_edge.weight", "_orig_mod.jt_encoder.convs.0.lin_edge.weight"):
        if key in state_dict:
            return True
    return False


def _detect_jt_use_jk(state_dict):
    for key in ("jt_encoder.jk_proj.weight", "_orig_mod.jt_encoder.jk_proj.weight"):
        if key in state_dict:
            return True
    return False


def _detect_jt_feature_dim(state_dict):
    for key in ("jt_encoder.content_proj.weight", "_orig_mod.jt_encoder.content_proj.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return 61


def _detect_num_ca_blocks(state_dict):
    """Count cross-attention blocks from state_dict keys."""
    idxs = set()
    for k in state_dict:
        kk = k.replace("_orig_mod.", "")
        if kk.startswith("blocks."):
            idxs.add(int(kk.split(".")[1]))
    return (max(idxs) + 1) if idxs else 0


def load_pretrained_backbone(pretrain_ckpt, device, jt_vocab_size=1753, num_ca_blocks=3):
    """Load a pretrained SJoINTModel from a checkpoint, reconstructing its
    architecture from the state_dict."""
    SJoINTModel = _load_sjoint_class()

    state = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    proj_dim = _detect_proj_dim(cleaned)
    hidden_dim, mol_n, jt_n = _detect_arch(cleaned)
    mol_gnn_type = _detect_encoder_gnn_type(cleaned, "mol_encoder")
    jt_gnn_type = _detect_encoder_gnn_type(cleaned, "jt_encoder")
    jt_use_edge_attr = _detect_jt_use_edge_attr(cleaned)
    jt_use_jk = _detect_jt_use_jk(cleaned)
    jt_feature_dim = _detect_jt_feature_dim(cleaned)
    num_ca_blocks = _detect_num_ca_blocks(cleaned) or num_ca_blocks
    for k in ("jt_encoder.jt_embedding.weight", "_orig_mod.jt_encoder.jt_embedding.weight"):
        if k in cleaned:
            jt_vocab_size = int(cleaned[k].shape[0])
            break

    backbone = SJoINTModel(
        jt_vocab_size=jt_vocab_size, jt_feature_dim=jt_feature_dim,
        num_ca_blocks=num_ca_blocks, proj_dim=proj_dim, hidden_dim=hidden_dim,
        mol_num_layers=mol_n, jt_num_layers=jt_n,
        mol_gnn_type=mol_gnn_type, jt_gnn_type=jt_gnn_type,
        jt_use_edge_attr=jt_use_edge_attr, jt_use_jk=jt_use_jk).to(device)
    ret = backbone.load_state_dict(cleaned, strict=False)
    if ret.missing_keys or ret.unexpected_keys:
        problems = []
        if ret.missing_keys:
            problems.append(f"missing={ret.missing_keys}")
        if ret.unexpected_keys:
            problems.append(f"unexpected={ret.unexpected_keys}")
        raise RuntimeError(
            "Pretrained backbone did not load cleanly: " + "; ".join(problems))
    return backbone


def create_random_backbone(jt_vocab_size=1753, num_ca_blocks=3, device="cpu",
                           jt_feature_dim=61):
    """Create a randomly initialised SJoINTModel (no-pretrain baseline)."""
    SJoINTModel = _load_sjoint_class()
    return SJoINTModel(jt_vocab_size=jt_vocab_size, num_ca_blocks=num_ca_blocks,
                       jt_feature_dim=jt_feature_dim).to(device)


class ResidualBlock(nn.Module):
    """Linear → norm → activation → dropout, with a residual when dims match."""

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
        pool_mode: 'both' (concat JT ⊕ MOL), 'jt_only', or 'mol_only'.
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
        input_dim = self.hidden_dim * 2 if pool_mode == "both" else self.hidden_dim
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
        """Masked mean pooling: (B, N, D) → (B, D)."""
        m = pad_mask.unsqueeze(-1)
        return (h_padded * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, jt_batch, mol_batch,
                jt2mol_b, jt2mol_r, jt2mol_c,
                mol2jt_b, mol2jt_r, mol2jt_c):
        # Encode both views in fp32 for GATv2 stability.
        with torch.amp.autocast("cuda", enabled=False):
            jt_h = self.jt_encoder(
                jt_batch.vocab_ids, jt_batch.features, jt_batch.edge_index,
                getattr(jt_batch, "edge_attr", None))          # (N_jt, H)
            mol_h = self.mol_encoder(
                mol_batch.atom_ids, mol_batch.features,
                mol_batch.edge_index, mol_batch.edge_attr)      # (N_mol, H)

        jt_h, jt_pm = to_dense_batch(jt_h, jt_batch.batch)      # (B, max_jt, H)
        mol_h, mol_pm = to_dense_batch(mol_h, mol_batch.batch)  # (B, max_mol, H)
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
        else:  # "both": concat JT ⊕ MOL
            rep = torch.cat([self._pool(jt_h, jt_pm), self._pool(mol_h, mol_pm)], dim=-1)
        return self.pred_head(rep)                              # (B, num_tasks)


# The backbone submodules are copied from SJoINTModel in DownstreamModel.__init__.
# Everything NOT under these prefixes (the prediction head) is randomly
# initialised downstream and must be trained during Stage 1 warm-up at the full
# head LR — not frozen at backbone_lr.
BACKBONE_PREFIXES = (
    "mol_encoder", "jt_encoder", "blocks", "final_norm_jt", "final_norm_mol",
)


def is_backbone_param(name):
    """True iff the parameter belongs to the pretrained backbone (vs. new head)."""
    return name.startswith(BACKBONE_PREFIXES)


def freeze_backbone(model):
    """Freeze the pretrained backbone; keep the new head trainable (Stage 1)."""
    for name, param in model.named_parameters():
        param.requires_grad = not is_backbone_param(name)


def unfreeze_backbone(model):
    """Enable gradients for all parameters (Stage 2)."""
    for param in model.parameters():
        param.requires_grad = True
