"""SJoINT pretrain model.

Architecture (shipped backbone; encoder GNN type is configurable):
    MolEncoder (GINE + JK)  ──┐
                               ├─► Cross-Attention Blocks (×3) ──► LayerNorm ──► MeanPool ──► Projection (NT-Xent)
    JTEncoder  (GATv2 + JK) ──┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, GINConv
from torch_geometric.utils import to_dense_batch


def _build_cross_attn_mask(b_idx, r_idx, c_idx, B, max_q, max_k, pad_mask_k):
    """Build dense additive attention mask from sparse (batch, row, col) indices.

    Args:
        b_idx, r_idx, c_idx: (num_edges,) flat index tensors for valid query-key pairs
        B: batch size
        max_q, max_k: padded sequence lengths for query / key
        pad_mask_k: (B, max_k) bool mask for valid key positions

    Returns:
        mask: (B, 1, max_q, max_k) additive mask (0 for valid, -inf for invalid)
    """
    device = pad_mask_k.device
    mask = torch.full((B, 1, max_q, max_k), float("-inf"), device=device)
    if b_idx.numel() > 0:
        valid = (c_idx < max_k) & (r_idx < max_q)
        if valid.any():
            mask[b_idx[valid], 0, r_idx[valid], c_idx[valid]] = 0.0
    mask = mask.masked_fill(~pad_mask_k[:, None, None, :], float("-inf"))
    # Prevent NaN softmax on all-inf rows
    all_inf = mask.isinf().all(dim=-1, keepdim=True)
    mask = mask.masked_fill(all_inf, 0.0)
    return mask


class MolEncoder(nn.Module):
    """Molecular graph encoder (GNN + optional Jump Knowledge).

    Input:  atom_ids (N,), features (N, 51), edge_index (2, E), edge_attr (E, 12)
    Output: node embeddings (N, hidden_dim)
    """

    def __init__(self, atom_embed_dim=16, feature_dim=51, edge_in=12,
                 hidden_dim=64, num_layers=3, num_heads=8, dropout=0.1, use_jk=True,
                 gnn_type="gatv2"):
        super().__init__()
        self.use_jk = use_jk
        self.gnn_type = gnn_type

        # Atom embedding: type_embed (N, 16) + features (N, 51) → (N, hidden_dim)
        self.atom_type_embed = nn.Embedding(118, atom_embed_dim)
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_embed_dim + feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU())

        # Bond embedding: (E, 12) → (E, hidden_dim)
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
                # GINEConv: edge feature 지원 GIN. MLP(2L) is the standard GINE choice.
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))
                self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Jump Knowledge: concat all layer outputs → project back
        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)

    def forward(self, atom_ids, features, edge_index, edge_attr):
        # (N,) + (N, 51) → (N, 16+51) → (N, H)
        x = torch.cat([self.atom_type_embed(atom_ids), features.float()], dim=-1)
        x = self.atom_embed(x)
        edge_emb = self.bond_embed(edge_attr.float())  # (E, 12) → (E, H)

        if self.use_jk:
            layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = self.dropout(F.relu(bn(conv(x, edge_index, edge_emb))))  # (N, H)
            x = x + x_res  # residual
            if self.use_jk:
                layer_outputs.append(x)

        if self.use_jk:
            # (N, H*num_layers) → (N, H)
            return self.jk_proj(torch.cat(layer_outputs, dim=-1))
        return x


class JTEncoder(nn.Module):
    """Junction tree encoder (GNN over substructure nodes).

    Node feature composition:
      h_init = vocab_emb(id) * 1[id>=2] + content_proj(sub_feat)

    sub_feat = jt_features_10d (substructure-level descriptors)
             ⊕ atom_feature_mean_51d          (= 61-D, shipped backbone)

    For id<2 (pad/unk), vocab_emb contributes 0, content fully drives h.

    Input:  vocab_ids (N,), features (N, feature_dim), edge_index (2, E)
    Output: node embeddings (N, hidden_dim)
    """

    def __init__(self, vocab_size=1753, feature_dim=61, hidden_dim=64,
                 num_layers=2, num_heads=8, dropout=0.1, use_jk=False,
                 gnn_type="gatv2", use_edge_attr=False, edge_feature_dim=55):
        super().__init__()
        self.use_jk = use_jk
        self.gnn_type = gnn_type
        self.feature_dim = feature_dim
        self.use_edge_attr = use_edge_attr

        # Dual injection: substructure vocab embedding + projected content features.
        self.jt_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.content_proj = nn.Linear(feature_dim, hidden_dim)
        # Small init for content_proj preserves the vocab_emb prior at start.
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
                # JT edges have no edge_attr → use plain GINConv (no edge_dim)
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
        h_id = self.jt_embedding(clamped)                            # (N, H)
        valid_mask = (vocab_ids >= 2).to(h_id.dtype).unsqueeze(-1)
        h_id = h_id * valid_mask                                     # pad/unk → 0
        h_content = self.content_proj(features.float())             # (N, H)
        x = h_id + h_content                                        # (N, H)

        x = self.node_act(self.node_bn(x))  # (N, H)

        if self.use_jk:
            layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            if self.use_edge_attr:
                x = self.dropout(F.relu(bn(conv(x, edge_index, edge_attr))))  # (N, H)
            else:
                x = self.dropout(F.relu(bn(conv(x, edge_index))))  # (N, H)
            x = x + x_res
            if self.use_jk:
                layer_outputs.append(x)

        if self.use_jk:
            return self.jk_proj(torch.cat(layer_outputs, dim=-1))
        return x  # (N, H)


class CABlock(nn.Module):
    """Bidirectional cross-attention block with pre-LayerNorm and FFN.

    JT ←→ MOL cross-attention with residual connections.

    Input:  jt_h (B, N_jt, D), mol_h (B, N_mol, D)
    Output: jt_h (B, N_jt, D), mol_h (B, N_mol, D)
    """

    def __init__(self, embed_dim, num_heads=8, ca_dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # JT queries MOL
        self.q_jt = nn.Linear(embed_dim, embed_dim)
        self.k_mol = nn.Linear(embed_dim, embed_dim)
        self.v_mol = nn.Linear(embed_dim, embed_dim)
        self.out_jt = nn.Linear(embed_dim, embed_dim)

        # MOL queries JT
        self.q_mol = nn.Linear(embed_dim, embed_dim)
        self.k_jt = nn.Linear(embed_dim, embed_dim)
        self.v_jt = nn.Linear(embed_dim, embed_dim)
        self.out_mol = nn.Linear(embed_dim, embed_dim)

        # Pre-norm (LayerNorm)
        self.norm1_jt = nn.LayerNorm(embed_dim)
        self.norm1_mol = nn.LayerNorm(embed_dim)
        self.norm2_jt = nn.LayerNorm(embed_dim)
        self.norm2_mol = nn.LayerNorm(embed_dim)

        # FFN
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
        """Multi-head cross-attention.

        Args:
            q: (B, N_q, D), k: (B, N_k, D), v: (B, N_k, D)
            attn_mask: (B, 1, N_q, N_k) additive mask
        Returns:
            (B, N_q, D) attended output
        """
        B, N_q, _ = q.shape
        N_k = k.size(1)
        H, d = self.num_heads, self.head_dim
        q = q.view(B, N_q, H, d).transpose(1, 2)  # (B, H, N_q, d)
        k = k.view(B, N_k, H, d).transpose(1, 2)  # (B, H, N_k, d)
        v = v.view(B, N_k, H, d).transpose(1, 2)  # (B, H, N_k, d)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0)
        return out_proj(out.transpose(1, 2).contiguous().view(B, N_q, -1))  # (B, N_q, D)

    def forward(self, jt_h, mol_h, jt_mask, mol_mask):
        jt_n = self.norm1_jt(jt_h)    # pre-norm
        mol_n = self.norm1_mol(mol_h)

        # Bidirectional cross-attention
        jt_h = jt_h + self.resid_dropout(self._cross_attn(
            self.q_jt(jt_n), self.k_mol(mol_n), self.v_mol(mol_n), jt_mask, self.out_jt))
        mol_h = mol_h + self.resid_dropout(self._cross_attn(
            self.q_mol(mol_n), self.k_jt(jt_n), self.v_jt(jt_n), mol_mask, self.out_mol))

        # FFN with pre-norm and residual
        jt_h = jt_h + self.resid_dropout(self.ffn_jt(self.norm2_jt(jt_h)))
        mol_h = mol_h + self.resid_dropout(self.ffn_mol(self.norm2_mol(mol_h)))
        return jt_h, mol_h


class SJoINTModel(nn.Module):
    """SJoINT contrastive pretrain model.

    Forward flow:
        1. Encode: MolEncoder(mol_graph) → (N_mol, H), JTEncoder(jt_graph) → (N_jt, H)
        2. to_dense_batch → (B, max_N, H) padded tensors
        3. Cross-Attention Blocks: bidirectional JT ↔ MOL attention
        4. Final LayerNorm
        5. Mean Pooling: (B, max_N, H) → (B, H)
        6. Projection Head: (B, H) → (B, proj_dim) for NT-Xent contrastive loss
    """

    def __init__(self, jt_vocab_size=1753, jt_feature_dim=61,
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

        # Graph encoders
        self.mol_encoder = MolEncoder(
            atom_embed_dim=16, feature_dim=51, hidden_dim=h,
            num_layers=mol_num_layers, num_heads=num_heads, dropout=dropout, use_jk=True,
            gnn_type=self.mol_gnn_type)
        self.jt_encoder = JTEncoder(
            vocab_size=jt_vocab_size, feature_dim=jt_feature_dim, hidden_dim=h,
            num_layers=jt_num_layers, num_heads=num_heads, dropout=dropout, use_jk=jt_use_jk,
            gnn_type=self.jt_gnn_type, use_edge_attr=jt_use_edge_attr)

        # Cross-attention blocks
        self.blocks = nn.ModuleList([
            CABlock(h, num_heads, ca_dropout=ca_dropout, ffn_ratio=ffn_ratio)
            for _ in range(num_ca_blocks)])

        # Final normalization (before pooling)
        self.final_norm_jt = nn.LayerNorm(h)
        self.final_norm_mol = nn.LayerNorm(h)

        # Projection heads for NT-Xent
        self.jt_proj = self._build_proj_head(h, h, proj_dim)
        self.mol_proj = self._build_proj_head(h, h, proj_dim)

    @staticmethod
    def _build_proj_head(in_dim, hidden, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))

    @staticmethod
    def _pool(h_padded, pad_mask):
        """Mean pooling: (B, N, D) → (B, D)."""
        m = pad_mask.unsqueeze(-1)  # (B, N, 1)
        return (h_padded * m).sum(1) / m.sum(1).clamp(min=1)  # (B, D)

    def forward(self, jt_batch, mol_batch,
                jt2mol_b, jt2mol_r, jt2mol_c,
                mol2jt_b, mol2jt_r, mol2jt_c):
        # 1. Encode graphs (fp32 for GATv2 stability)
        with torch.amp.autocast('cuda', enabled=False):
            jt_h = self.jt_encoder(
                jt_batch.vocab_ids, jt_batch.features, jt_batch.edge_index,
                getattr(jt_batch, "edge_attr", None))  # (N_jt_total, H)
            mol_h = self.mol_encoder(
                mol_batch.atom_ids, mol_batch.features,
                mol_batch.edge_index, mol_batch.edge_attr)  # (N_mol_total, H)

        # 2. Sparse → dense padded batch
        jt_h, jt_pm = to_dense_batch(jt_h, jt_batch.batch)    # (B, max_jt, H), (B, max_jt)
        mol_h, mol_pm = to_dense_batch(mol_h, mol_batch.batch)  # (B, max_mol, H), (B, max_mol)
        B, max_jt, max_mol = jt_h.size(0), jt_h.size(1), mol_h.size(1)

        # 3. Build cross-attention masks
        jt_mask = _build_cross_attn_mask(
            jt2mol_b, jt2mol_r, jt2mol_c, B, max_jt, max_mol, mol_pm)  # (B, 1, max_jt, max_mol)
        mol_mask = _build_cross_attn_mask(
            mol2jt_b, mol2jt_r, mol2jt_c, B, max_mol, max_jt, jt_pm)    # (B, 1, max_mol, max_jt)

        # 4. Cross-attention blocks
        for block in self.blocks:
            jt_h, mol_h = block(jt_h, mol_h, jt_mask, mol_mask)
            jt_h = jt_h.masked_fill(~jt_pm.unsqueeze(-1), 0.0)    # zero out padding
            mol_h = mol_h.masked_fill(~mol_pm.unsqueeze(-1), 0.0)

        # 5. Final norm
        jt_h = self.final_norm_jt(jt_h)    # (B, max_jt, H)
        mol_h = self.final_norm_mol(mol_h)  # (B, max_mol, H)

        # 6. Mean pooling → projection
        jt_out = self.jt_proj(self._pool(jt_h, jt_pm))      # (B, H) → (B, proj_dim)
        mol_out = self.mol_proj(self._pool(mol_h, mol_pm))  # (B, H) → (B, proj_dim)
        return jt_out, mol_out
