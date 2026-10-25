"""SJoINT pre-training model: molecular-graph and JT encoders fused by cross-attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, GINConv
from torch_geometric.utils import to_dense_batch


def _build_cross_attn_mask(b_idx, r_idx, c_idx, B, max_q, max_k, pad_mask_k):
    """Additive (B, 1, max_q, max_k) mask: 0 for member pairs, -inf elsewhere."""
    device = pad_mask_k.device
    mask = torch.full((B, 1, max_q, max_k), float("-inf"), device=device)
    if b_idx.numel() > 0:
        valid = (c_idx < max_k) & (r_idx < max_q)
        if valid.any():
            mask[b_idx[valid], 0, r_idx[valid], c_idx[valid]] = 0.0
    mask = mask.masked_fill(~pad_mask_k[:, None, None, :], float("-inf"))
    all_inf = mask.isinf().all(dim=-1, keepdim=True)
    mask = mask.masked_fill(all_inf, 0.0)
    return mask


class MolEncoder(nn.Module):
    """Molecular-graph encoder (GNN + Jumping Knowledge)."""

    def __init__(self, atom_embed_dim=16, feature_dim=51, edge_in=12,
                 hidden_dim=64, num_layers=3, num_heads=8, dropout=0.1, use_jk=True,
                 gnn_type="gatv2"):
        super().__init__()
        self.use_jk = use_jk
        self.gnn_type = gnn_type

        self.atom_type_embed = nn.Embedding(118, atom_embed_dim)
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_embed_dim + feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU())

        self.bond_embed = nn.Linear(edge_in, hidden_dim)

        assert hidden_dim % num_heads == 0
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gatv2":
                self.convs.append(GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    edge_dim=hidden_dim, concat=True))
            elif gnn_type == "gine":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))
                self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)

    def forward(self, atom_ids, features, edge_index, edge_attr):
        x = torch.cat([self.atom_type_embed(atom_ids), features.float()], dim=-1)
        x = self.atom_embed(x)
        edge_emb = self.bond_embed(edge_attr.float())

        if self.use_jk:
            layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = self.dropout(F.relu(bn(conv(x, edge_index, edge_emb))))
            x = x + x_res
            if self.use_jk:
                layer_outputs.append(x)

        if self.use_jk:
            return self.jk_proj(torch.cat(layer_outputs, dim=-1))
        return x


class JTEncoder(nn.Module):
    """JT encoder: vocabulary embedding + projected content features, then GNN + Jumping Knowledge."""

    def __init__(self, vocab_size=1750, feature_dim=61, hidden_dim=64,
                 num_layers=2, num_heads=8, dropout=0.1, use_jk=False,
                 gnn_type="gatv2", use_edge_attr=False, edge_feature_dim=55):
        super().__init__()
        self.use_jk = use_jk
        self.gnn_type = gnn_type
        self.feature_dim = feature_dim
        self.use_edge_attr = use_edge_attr

        self.jt_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.content_proj = nn.Linear(feature_dim, hidden_dim)
        nn.init.normal_(self.content_proj.weight, std=0.01)
        nn.init.zeros_(self.content_proj.bias)
        self.node_bn = nn.BatchNorm1d(hidden_dim)
        self.node_act = nn.ReLU()

        assert hidden_dim % num_heads == 0
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gatv2":
                self.convs.append(GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True,
                    edge_dim=edge_feature_dim if use_edge_attr else None))
            elif gnn_type == "gine":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))
                if use_edge_attr:
                    self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_feature_dim))
                else:
                    self.convs.append(GINConv(mlp, train_eps=True))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)

    def forward(self, vocab_ids, features, edge_index, edge_attr=None):
        clamped = vocab_ids.clamp(max=self.jt_embedding.num_embeddings - 1)
        h_id = self.jt_embedding(clamped)
        valid_mask = (vocab_ids >= 2).to(h_id.dtype).unsqueeze(-1)  # pad=0 / unk=1 -> no vocab embedding
        h_id = h_id * valid_mask
        h_content = self.content_proj(features.float())
        x = h_id + h_content

        x = self.node_act(self.node_bn(x))

        if self.use_jk:
            layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            if self.use_edge_attr:
                x = self.dropout(F.relu(bn(conv(x, edge_index, edge_attr))))
            else:
                x = self.dropout(F.relu(bn(conv(x, edge_index))))
            x = x + x_res
            if self.use_jk:
                layer_outputs.append(x)

        if self.use_jk:
            return self.jk_proj(torch.cat(layer_outputs, dim=-1))
        return x


class CABlock(nn.Module):
    """Bidirectional JT <-> molecule cross-attention block (pre-LayerNorm, FFN, residuals)."""

    def __init__(self, embed_dim, num_heads=8, ca_dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.q_jt = nn.Linear(embed_dim, embed_dim)
        self.k_mol = nn.Linear(embed_dim, embed_dim)
        self.v_mol = nn.Linear(embed_dim, embed_dim)
        self.out_jt = nn.Linear(embed_dim, embed_dim)

        self.q_mol = nn.Linear(embed_dim, embed_dim)
        self.k_jt = nn.Linear(embed_dim, embed_dim)
        self.v_jt = nn.Linear(embed_dim, embed_dim)
        self.out_mol = nn.Linear(embed_dim, embed_dim)

        self.norm1_jt = nn.LayerNorm(embed_dim)
        self.norm1_mol = nn.LayerNorm(embed_dim)
        self.norm2_jt = nn.LayerNorm(embed_dim)
        self.norm2_mol = nn.LayerNorm(embed_dim)

        ffn_dim = embed_dim * ffn_ratio
        self.ffn_jt = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(ca_dropout), nn.Linear(ffn_dim, embed_dim))
        self.ffn_mol = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(ca_dropout), nn.Linear(ffn_dim, embed_dim))

        self.resid_dropout = nn.Dropout(ca_dropout)
        self.attn_dropout_p = ca_dropout

    def _cross_attn(self, q, k, v, attn_mask, out_proj):
        B, N_q, _ = q.shape
        N_k = k.size(1)
        H, d = self.num_heads, self.head_dim
        q = q.view(B, N_q, H, d).transpose(1, 2)
        k = k.view(B, N_k, H, d).transpose(1, 2)
        v = v.view(B, N_k, H, d).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0)
        return out_proj(out.transpose(1, 2).contiguous().view(B, N_q, -1))

    def forward(self, jt_h, mol_h, jt_mask, mol_mask):
        jt_n = self.norm1_jt(jt_h)
        mol_n = self.norm1_mol(mol_h)

        jt_h = jt_h + self.resid_dropout(self._cross_attn(
            self.q_jt(jt_n), self.k_mol(mol_n), self.v_mol(mol_n), jt_mask, self.out_jt))
        mol_h = mol_h + self.resid_dropout(self._cross_attn(
            self.q_mol(mol_n), self.k_jt(jt_n), self.v_jt(jt_n), mol_mask, self.out_mol))

        jt_h = jt_h + self.resid_dropout(self.ffn_jt(self.norm2_jt(jt_h)))
        mol_h = mol_h + self.resid_dropout(self.ffn_mol(self.norm2_mol(mol_h)))
        return jt_h, mol_h


class SJoINTModel(nn.Module):
    """Dual-view backbone with NT-Xent projection heads."""

    def __init__(self, jt_vocab_size=1750, jt_feature_dim=61,
                 num_ca_blocks=3, proj_dim=32,
                 hidden_dim=128, mol_num_layers=3, jt_num_layers=3,
                 num_heads=8, dropout=0.0, ca_dropout=0.1, ffn_ratio=4,
                 mol_gnn_type="gine", jt_gnn_type="gatv2",
                 jt_use_edge_attr=True, jt_use_jk=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mol_gnn_type = mol_gnn_type
        self.jt_gnn_type = jt_gnn_type
        self.jt_feature_dim = jt_feature_dim
        h = self.hidden_dim

        self.mol_encoder = MolEncoder(
            atom_embed_dim=16, feature_dim=51, hidden_dim=h,
            num_layers=mol_num_layers, num_heads=num_heads, dropout=dropout, use_jk=True,
            gnn_type=self.mol_gnn_type)
        self.jt_encoder = JTEncoder(
            vocab_size=jt_vocab_size, feature_dim=jt_feature_dim, hidden_dim=h,
            num_layers=jt_num_layers, num_heads=num_heads, dropout=dropout, use_jk=jt_use_jk,
            gnn_type=self.jt_gnn_type, use_edge_attr=jt_use_edge_attr)

        self.blocks = nn.ModuleList([
            CABlock(h, num_heads, ca_dropout=ca_dropout, ffn_ratio=ffn_ratio)
            for _ in range(num_ca_blocks)])

        self.final_norm_jt = nn.LayerNorm(h)
        self.final_norm_mol = nn.LayerNorm(h)

        self.jt_proj = self._build_proj_head(h, h, proj_dim)
        self.mol_proj = self._build_proj_head(h, h, proj_dim)

    @staticmethod
    def _build_proj_head(in_dim, hidden, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))

    @staticmethod
    def _pool(h_padded, pad_mask):
        m = pad_mask.unsqueeze(-1)
        return (h_padded * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, jt_batch, mol_batch,
                jt2mol_b, jt2mol_r, jt2mol_c,
                mol2jt_b, mol2jt_r, mol2jt_c):
        with torch.amp.autocast('cuda', enabled=False):  # fp32 for GATv2 stability
            jt_h = self.jt_encoder(
                jt_batch.vocab_ids, jt_batch.features, jt_batch.edge_index,
                getattr(jt_batch, "edge_attr", None))
            mol_h = self.mol_encoder(
                mol_batch.atom_ids, mol_batch.features,
                mol_batch.edge_index, mol_batch.edge_attr)

        jt_h, jt_pm = to_dense_batch(jt_h, jt_batch.batch)
        mol_h, mol_pm = to_dense_batch(mol_h, mol_batch.batch)
        B, max_jt, max_mol = jt_h.size(0), jt_h.size(1), mol_h.size(1)

        jt_mask = _build_cross_attn_mask(
            jt2mol_b, jt2mol_r, jt2mol_c, B, max_jt, max_mol, mol_pm)
        mol_mask = _build_cross_attn_mask(
            mol2jt_b, mol2jt_r, mol2jt_c, B, max_mol, max_jt, jt_pm)

        for block in self.blocks:
            jt_h, mol_h = block(jt_h, mol_h, jt_mask, mol_mask)
            jt_h = jt_h.masked_fill(~jt_pm.unsqueeze(-1), 0.0)
            mol_h = mol_h.masked_fill(~mol_pm.unsqueeze(-1), 0.0)

        jt_h = self.final_norm_jt(jt_h)
        mol_h = self.final_norm_mol(mol_h)

        jt_out = self.jt_proj(self._pool(jt_h, jt_pm))
        mol_out = self.mol_proj(self._pool(mol_h, mol_pm))
        return jt_out, mol_out
