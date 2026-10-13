"""I/O utilities for preprocessing.

    Vocab loading → SMILES featurisation → tensor conversion → chunk saving
"""
from __future__ import annotations

import json
import os

import torch

from preprocess.core.features import (
    JT_FEATURE_DIM,
    JT_FEATURE_DIM_EXTENDED,
    build_jt_edge_index,
    compute_jt_features,
    compute_sub_atom_pool,
    get_atom_features,
    get_bond_features,
    jt_node_to_smiles,
    precompute_mol_properties,
    tree_decomp,
    build_jt_mol_mappings,
    MOL_FEATURE_DIM,
    BOND_FEATURE_DIM,
    build_jt_edge_attr,
)

import random
import numpy as np
from rdkit import Chem

_VOCAB = None
_VOCAB_FEATURES = None


def load_vocab(vocab_path: str) -> int:
    """Load JT vocabulary JSON into global cache. Returns vocab size."""
    global _VOCAB, _VOCAB_FEATURES
    with open(vocab_path) as f:
        data = json.load(f)
    _VOCAB = data["vocab"]
    _VOCAB_FEATURES = data["features"]
    return len(_VOCAB)


def get_vocab():
    if _VOCAB is None:
        raise RuntimeError("Vocab not loaded. Call load_vocab() first.")
    return _VOCAB, _VOCAB_FEATURES


def process_smiles(smiles: str) -> dict | None:
    """Convert a SMILES string to a raw feature dict (numpy/list)."""
    try:
        rd_mol = Chem.MolFromSmiles(smiles)
        if rd_mol is None:
            return None
        canonical = Chem.MolToSmiles(rd_mol, canonical=True, isomericSmiles=True)
        num_atoms = rd_mol.GetNumAtoms()

        jt_nodes, jt_edges = tree_decomp(rd_mol)
        jt2mol_map, mol2jt_map = build_jt_mol_mappings(jt_nodes, num_atoms)

        precompute_mol_properties(rd_mol)
        ring_info = rd_mol.GetRingInfo()
        atom_ids, atom_features = [], []
        for atom in rd_mol.GetAtoms():
            aid, feat = get_atom_features(atom, ring_info)
            atom_ids.append(aid)
            atom_features.append(feat)

        atom_ids = np.array(atom_ids, dtype=np.int64)
        atom_features = (np.stack(atom_features) if atom_features
                         else np.zeros((0, MOL_FEATURE_DIM), dtype=np.float32))

        vocab, vocab_features = get_vocab()
        jt_vocab_ids = []
        jt_feats = []
        jt_feats_extended = []
        for node_atoms in jt_nodes:
            smi = jt_node_to_smiles(rd_mol, node_atoms)
            vocab_id = vocab.get(smi, 1)   # 1 = <UNK> for out-of-vocab substructures
            jt_vocab_ids.append(vocab_id)
            base_feat = (vocab_features[smi] if smi in vocab_features
                         else compute_jt_features(rd_mol, node_atoms))
            jt_feats.append(base_feat)
            jt_feats_extended.append(list(base_feat) + compute_sub_atom_pool(atom_features, node_atoms))

        edge_src, edge_dst, edge_feats = [], [], []
        for bond in rd_mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = get_bond_features(bond)
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_feats.extend([bf, bf])

        mol_edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        mol_edge_attr = (np.stack(edge_feats) if edge_feats
                         else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32))

        return {
            "smiles": canonical,
            "jt_vocab_ids": jt_vocab_ids,
            "jt_features_10d": jt_feats,
            "jt_features_extended": jt_feats_extended,
            "jt_edges": jt_edges,
            "mol_atom_ids": atom_ids,
            "mol_features": atom_features,
            "mol_edge_index": mol_edge_index,
            "mol_edge_features": mol_edge_attr,
            "jt2mol_map": jt2mol_map,
            "mol2jt_map": mol2jt_map,
        }
    except Exception:
        return None


def to_sample(cpu_data: dict, labels: list[float] | None = None) -> dict:
    """Convert raw feature dict to a tensor sample for .pt saving."""
    jt_feat_ext = torch.tensor(cpu_data["jt_features_extended"], dtype=torch.float32)
    if jt_feat_ext.dim() == 1:
        jt_feat_ext = jt_feat_ext.unsqueeze(0)
    if jt_feat_ext.numel() == 0:
        jt_feat_ext = torch.zeros((0, JT_FEATURE_DIM_EXTENDED), dtype=torch.float32)
    sample = {
        "smiles": cpu_data["smiles"],
        "jt_vocab_ids": torch.tensor(cpu_data["jt_vocab_ids"], dtype=torch.long),
        "jt_features": jt_feat_ext,
        "jt_edge_index": build_jt_edge_index(cpu_data["jt_edges"]),
        "jt_edge_attr": build_jt_edge_attr(
            cpu_data["jt2mol_map"], cpu_data["jt_edges"],
            cpu_data["mol_atom_ids"], cpu_data["mol_features"]),
        "mol_atom_ids": torch.from_numpy(cpu_data["mol_atom_ids"]).long(),
        "mol_x": torch.from_numpy(cpu_data["mol_features"]).float(),
        "mol_edge_index": torch.from_numpy(cpu_data["mol_edge_index"]).long(),
        "mol_edge_attr": torch.from_numpy(cpu_data["mol_edge_features"]).float(),
        "jt2mol_map": cpu_data["jt2mol_map"],
        "mol2jt_map": cpu_data["mol2jt_map"],
    }
    if labels is not None:
        sample["labels"] = torch.tensor(labels, dtype=torch.float32)
    return sample


def save_chunk(data_list: list[dict], out_path: str, chunk_num: int):
    """Save a list of tensor samples as a numbered .pt chunk."""
    base, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".pt"
    chunk_path = f"{base}_chunk_{chunk_num}{ext}"
    os.makedirs(os.path.dirname(chunk_path) or ".", exist_ok=True)
    torch.save(data_list, chunk_path, pickle_protocol=4)
