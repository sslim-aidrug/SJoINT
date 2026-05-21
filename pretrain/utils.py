"""Pretraining utilities: dataset, collate, and SimCLR loss."""
import os
import glob
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


def _flatten_mapping(mapping):
    rows, cols = [], []
    for i, targets in enumerate(mapping):
        if not targets:
            continue
        rows.extend([i] * len(targets))
        cols.extend(targets)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


class GraphPairDataset(Dataset):
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
    }


def simclr_loss(z_jt, z_mol, temperature=0.07):
    """Symmetric NT-Xent loss with L2 normalisation."""
    z_jt = F.normalize(z_jt, dim=-1)
    z_mol = F.normalize(z_mol, dim=-1)
    logits = torch.matmul(z_jt, z_mol.T) / temperature
    labels = torch.arange(z_jt.size(0), device=z_jt.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
