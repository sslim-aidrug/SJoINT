"""Pretraining utilities: dataset, collate, and NT-Xent loss."""
import os
import glob
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


def _flatten_mapping(mapping):
    """Sparse atom<->substructure correspondence: list-of-lists -> flat (rows, cols).

    mapping[i] = list of target-node indices that source-node i maps to.
    Returns rows, cols: both (E,) long tensors of valid (source, target) pairs.
    """
    rows, cols = [], []
    for i, targets in enumerate(mapping):
        if not targets:
            continue
        rows.extend([i] * len(targets))
        cols.extend(targets)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


class GraphPairDataset(Dataset):
    """Loads pre-tensorised .pt chunks into memory. Each sample is a dict with the
    two views (jt_graph, mol_graph as PyG Data) and the flat atom<->substructure
    correspondence indices (jt2mol / mol2jt rows & cols) used to mask cross-attention.
    """

    def __init__(self, data_path, split_indices=None):
        super().__init__()
        all_chunks = sorted(glob.glob(os.path.join(data_path, "*.pt")))
        if not all_chunks:
            raise FileNotFoundError(f"No .pt files in {data_path}")
        if split_indices is not None:
            all_chunks = [all_chunks[i] for i in split_indices]

        self.data = []
        for fp in all_chunks:
            chunk = torch.load(fp, map_location="cpu", weights_only=False)
            if not isinstance(chunk, list):
                chunk = [chunk]
            for item in chunk:
                jt_graph = Data(
                    vocab_ids=item["jt_vocab_ids"],
                    features=item["jt_features"],
                    edge_index=item["jt_edge_index"],
                    edge_attr=item.get("jt_edge_attr"),
                    num_nodes=item["jt_vocab_ids"].size(0))
                mol_graph = Data(
                    atom_ids=item["mol_atom_ids"],
                    features=item["mol_x"],
                    edge_index=item["mol_edge_index"],
                    edge_attr=item.get("mol_edge_attr"),
                    num_nodes=item["mol_atom_ids"].size(0))
                j2m_r, j2m_c = _flatten_mapping(item.get("jt2mol_map", []))
                m2j_r, m2j_c = _flatten_mapping(item.get("mol2jt_map", []))
                self.data.append({
                    "jt_graph": jt_graph, "mol_graph": mol_graph,
                    "jt2mol_rows": j2m_r, "jt2mol_cols": j2m_c,
                    "mol2jt_rows": m2j_r, "mol2jt_cols": m2j_c,
                })
        logger.info(f"Loaded {len(self.data):,} samples from {len(all_chunks)} chunks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _stack_mapping(batch, key_r, key_c):
    """Concatenate per-sample correspondence indices into batch-flat tensors,
    adding a batch-index tensor. Returns (b_idx, r_idx, c_idx), each (sum_E,) long,
    where b_idx marks which sample in the batch each (row, col) pair belongs to.
    """
    bs, rs, cs = [], [], []
    for b, d in enumerate(batch):
        r, c = d[key_r], d[key_c]
        if r.size(0) == 0:
            continue
        bs.append(torch.full((r.size(0),), b, dtype=torch.long))
        rs.append(r)
        cs.append(c)
    if not bs:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty
    return torch.cat(bs), torch.cat(rs), torch.cat(cs)


def custom_collate(batch):
    """Collate a list of dual-view samples into one batch.

    Returns a dict with two batched PyG graphs and the flat correspondence indices;
    the model builds the dense (B, 1, N_q, N_k) cross-attention masks from these.
    """
    batch = [d for d in batch if d is not None]
    if not batch:
        return None
    # (b, r, c) flat indices for JT->MOL and MOL->JT attention masks
    j2m_b, j2m_r, j2m_c = _stack_mapping(batch, "jt2mol_rows", "jt2mol_cols")
    m2j_b, m2j_r, m2j_c = _stack_mapping(batch, "mol2jt_rows", "mol2jt_cols")
    return {
        "jt_batch": Batch.from_data_list([d["jt_graph"] for d in batch]),   # PyG Batch
        "mol_batch": Batch.from_data_list([d["mol_graph"] for d in batch]),  # PyG Batch
        "jt2mol_b": j2m_b, "jt2mol_r": j2m_r, "jt2mol_c": j2m_c,  # each (E_jt2mol,)
        "mol2jt_b": m2j_b, "mol2jt_r": m2j_r, "mol2jt_c": m2j_c,  # each (E_mol2jt,)
    }


def nt_xent_loss(z_jt, z_mol, temperature=0.07):
    """Symmetric NT-Xent (cross-view InfoNCE) loss with L2 normalisation.

    z_jt, z_mol: (B, proj_dim) projected embeddings of the JT / atom views; row i
    of each is the same molecule, so the diagonal of the similarity matrix is the
    positive pair and off-diagonal entries (other molecules in the batch) are negatives.
    """
    z_jt = F.normalize(z_jt, dim=-1)                          # (B, D)
    z_mol = F.normalize(z_mol, dim=-1)                        # (B, D)
    logits = torch.matmul(z_jt, z_mol.T) / temperature        # (B, B) sim matrix, diag = positives
    labels = torch.arange(z_jt.size(0), device=z_jt.device)   # (B,) positive index per row
    # average of both directions (JT->MOL and MOL->JT)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
