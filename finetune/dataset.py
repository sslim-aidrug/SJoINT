"""Fine-tuning dataset, collate, cross-attention masks, and label normalisation."""
import logging

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


def _flatten_mapping(mapping):
    """Convert a list-of-lists mapping into two flat 1D long tensors (rows, cols)."""
    rows, cols = [], []
    for i, targets in enumerate(mapping):
        if not targets:
            continue
        rows.extend([i] * len(targets))
        cols.extend(targets)
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
    )


class FinetuneDataset(Dataset):
    """Loads a preprocessed .pt file (list of dicts) into memory.

    Each sample contains:
      - jt_graph, mol_graph (PyG Data)
      - jt2mol_rows, jt2mol_cols, mol2jt_rows, mol2jt_cols (flat long tensors,
        precomputed once so collate_fn can skip the Python loop)
      - labels (num_tasks,)
    """

    def __init__(self, pt_path):
        super().__init__()
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, list):
            raw = [raw]

        self.data = []
        for item in raw:
            jt_graph = Data(
                vocab_ids=item["jt_vocab_ids"],          # (N_jt,)
                features=item["jt_features"],            # (N_jt, 61)
                edge_index=item["jt_edge_index"],        # (2, E_jt)
                edge_attr=item.get("jt_edge_attr"),      # (E_jt, 55) or None
                num_nodes=item["jt_vocab_ids"].size(0),
            )
            mol_graph = Data(
                atom_ids=item["mol_atom_ids"],           # (N_atom,)
                features=item["mol_x"],                  # (N_atom, 51)
                edge_index=item["mol_edge_index"],       # (2, E_mol)
                edge_attr=item["mol_edge_attr"],         # (E_mol, 12)
                num_nodes=item["mol_atom_ids"].size(0),
            )
            j2m_r, j2m_c = _flatten_mapping(item.get("jt2mol_map", []))
            m2j_r, m2j_c = _flatten_mapping(item.get("mol2jt_map", []))
            self.data.append({
                "jt_graph": jt_graph,
                "mol_graph": mol_graph,
                "jt2mol_rows": j2m_r,
                "jt2mol_cols": j2m_c,
                "mol2jt_rows": m2j_r,
                "mol2jt_cols": m2j_c,
                "labels": item["labels"],
            })
        logger.info(f"[Data] Loaded {len(self.data):,} samples from {pt_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _stack_mapping(batch, key_r, key_c):
    """Concatenate per-sample (row, col) tensors with a batch index tensor."""
    bs, rs, cs = [], [], []
    for b, d in enumerate(batch):
        r, c = d[key_r], d[key_c]
        n = r.size(0)
        if n == 0:
            continue
        bs.append(torch.full((n,), b, dtype=torch.long))
        rs.append(r)
        cs.append(c)
    if not bs:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty
    return torch.cat(bs), torch.cat(rs), torch.cat(cs)


def collate_fn(batch):
    """Collate a list of samples into a single batch dict.

    The cross-attention mask data is delivered as flat (b, r, c) index tensors;
    the model builds the dense mask on-device from these in O(N) operations,
    avoiding a per-batch Python loop.
    """
    batch = [d for d in batch if d is not None]
    if not batch:
        return None

    j2m_b, j2m_r, j2m_c = _stack_mapping(batch, "jt2mol_rows", "jt2mol_cols")
    m2j_b, m2j_r, m2j_c = _stack_mapping(batch, "mol2jt_rows", "mol2jt_cols")

    return {
        "jt_batch": Batch.from_data_list([d["jt_graph"] for d in batch]),
        "mol_batch": Batch.from_data_list([d["mol_graph"] for d in batch]),
        "jt2mol_b": j2m_b, "jt2mol_r": j2m_r, "jt2mol_c": j2m_c,
        "mol2jt_b": m2j_b, "mol2jt_r": m2j_r, "mol2jt_c": m2j_c,
        "labels": torch.stack([d["labels"] for d in batch]),   # (B, num_tasks)
    }


def build_cross_attn_mask(b_idx, r_idx, c_idx, B, max_q, max_k, pad_mask_k):
    """Build a (B, 1, max_q, max_k) additive attention mask from flat indices.

    Entries listed in (b_idx, r_idx, c_idx) become 0; everything else -inf.
    Rows that end up entirely -inf (padded keys only) are reset to 0 to
    avoid NaN in softmax.
    """
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


def compute_label_stats(dataset):
    """Mean / std of labels over the train split (per task), ignoring NaN."""
    labels = torch.stack([s["labels"] for s in dataset.data])  # (N, num_tasks)
    nan_mask = torch.isnan(labels)
    safe = labels.masked_fill(nan_mask, 0.0)
    cnt = (~nan_mask).sum(dim=0).clamp(min=1).float()
    mean = safe.sum(dim=0) / cnt
    var = ((safe - mean) ** 2 * (~nan_mask).float()).sum(dim=0) / cnt
    std = var.sqrt().clamp(min=1e-6)
    return mean, std


def normalize_labels(dataset, mean, std):
    """In-place z-score normalization of labels for a FinetuneDataset."""
    for s in dataset.data:
        s["labels"] = (s["labels"] - mean) / std
