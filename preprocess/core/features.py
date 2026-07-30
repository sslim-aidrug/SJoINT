"""Molecular featurisation and Junction-Tree decomposition.

    SMILES → atom features (51-D) + bond features (12-D)
           → JT decomposition + fallback features (10-D)
           → jt2mol / mol2jt membership maps
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

RDLogger.DisableLog("rdApp.*")

MOL_FEATURE_DIM = 51
BOND_FEATURE_DIM = 12
JT_FEATURE_DIM = 10
# Extended JT node feature: 10-D fallback descriptors + mean of the member atoms'
# 51-D features. This is the composition used by the released backbone (61-D).
JT_FEATURE_DIM_EXTENDED = JT_FEATURE_DIM + MOL_FEATURE_DIM
JT_EDGE_FEATURE_DIM = MOL_FEATURE_DIM + 4

DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRALITIES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
    Chem.rdchem.ChiralType.CHI_ALLENE,
]
RING_SIZES = [3, 4, 5, 6, 7, 8]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
PAULING_EN = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 11: 0.93, 12: 1.31,
    13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82,
    20: 1.00, 35: 2.96, 53: 2.66,
}

_PERIODIC_TABLE = Chem.GetPeriodicTable()
HDONOR_SMARTS = Chem.MolFromSmarts('[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),$([n;H1;+0])]')
HACCEPTOR_SMARTS = Chem.MolFromSmarts(
    '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),'
    '$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]')

HALOGEN_ATOMS = {9, 17, 35, 53}
MST_MAX_WEIGHT = 100


def precompute_mol_properties(mol):
    """Precompute Gasteiger charges, Crippen, TPSA, and H-bond properties."""
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp('_GasteigerCharge', 0.0)

    contribs = Crippen._GetAtomContribs(mol)
    for i, (logp, mr) in enumerate(contribs):
        mol.GetAtomWithIdx(i).SetDoubleProp('_CrippenLogP', logp)
        mol.GetAtomWithIdx(i).SetDoubleProp('_CrippenMR', mr)

    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
    for i, val in enumerate(tpsa_contribs):
        mol.GetAtomWithIdx(i).SetDoubleProp('_TPSAContrib', val)

    donor_matches = set()
    acceptor_matches = set()
    for match in mol.GetSubstructMatches(HDONOR_SMARTS):
        donor_matches.update(match)
    for match in mol.GetSubstructMatches(HACCEPTOR_SMARTS):
        acceptor_matches.update(match)
    for atom in mol.GetAtoms():
        atom.SetDoubleProp('_IsDonor', 1.0 if atom.GetIdx() in donor_matches else 0.0)
        atom.SetDoubleProp('_IsAcceptor', 1.0 if atom.GetIdx() in acceptor_matches else 0.0)


def get_atom_features(atom, ring_info=None) -> tuple[int, np.ndarray]:
    """Return (atomic_number, 51-D feature vector) for a single atom."""
    feats = np.zeros(MOL_FEATURE_DIM, dtype=np.float32)
    offset = 0
    atom_id = atom.GetAtomicNum()

    deg = min(atom.GetDegree(), 5)
    feats[offset + deg] = 1.0
    offset += 6

    feats[offset] = atom.GetMass() / 100.0
    offset += 1
    feats[offset] = 1.0 if atom.GetIsAromatic() else 0.0
    offset += 1

    hybrid = atom.GetHybridization()
    if hybrid in HYBRIDIZATIONS:
        feats[offset + HYBRIDIZATIONS.index(hybrid)] = 1.0
    offset += 5

    nh = min(atom.GetTotalNumHs(), 4)
    feats[offset + nh] = 1.0
    offset += 5

    chiral = atom.GetChiralTag()
    if chiral in CHIRALITIES:
        feats[offset + CHIRALITIES.index(chiral)] = 1.0
    offset += 5

    fc = max(-2, min(2, atom.GetFormalCharge()))
    try:
        feats[offset + FORMAL_CHARGES.index(fc)] = 1.0
    except ValueError:
        feats[offset + len(FORMAL_CHARGES) - 1] = 1.0
    offset += 5

    feats[offset] = 1.0 if atom.IsInRing() else 0.0
    offset += 1

    if ring_info is None:
        ring_info = atom.GetOwningMol().GetRingInfo()
    atom_idx = atom.GetIdx()
    for ri, rs in enumerate(RING_SIZES):
        if ring_info.IsAtomInRingOfSize(atom_idx, rs):
            feats[offset + ri] = 1.0
    offset += 6

    iv = min(atom.GetImplicitValence(), 5)
    feats[offset + iv] = 1.0
    offset += 6

    gasteiger = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
    if not np.isfinite(gasteiger):
        gasteiger = 0.0
    feats[offset] = gasteiger
    feats[offset + 1] = atom.GetDoubleProp('_CrippenLogP') if atom.HasProp('_CrippenLogP') else 0.0
    feats[offset + 2] = atom.GetDoubleProp('_CrippenMR') if atom.HasProp('_CrippenMR') else 0.0
    feats[offset + 3] = atom.GetDoubleProp('_TPSAContrib') if atom.HasProp('_TPSAContrib') else 0.0
    feats[offset + 4] = atom.GetDoubleProp('_IsDonor') if atom.HasProp('_IsDonor') else 0.0
    feats[offset + 5] = atom.GetDoubleProp('_IsAcceptor') if atom.HasProp('_IsAcceptor') else 0.0
    feats[offset + 6] = PAULING_EN.get(atom_id, 0.0) / 4.0
    feats[offset + 7] = _PERIODIC_TABLE.GetNOuterElecs(atom_id) / 8.0
    feats[offset + 8] = _PERIODIC_TABLE.GetRcovalent(atom_id) / 2.0
    feats[offset + 9] = _PERIODIC_TABLE.GetRvdw(atom_id) / 3.0

    return atom_id, feats


def get_bond_features(bond) -> np.ndarray:
    """Return 12-D feature vector for a single bond."""
    feats = np.zeros(BOND_FEATURE_DIM, dtype=np.float32)
    bt = bond.GetBondType()
    if bt in BOND_TYPES:
        feats[BOND_TYPES.index(bt)] = 1.0
    feats[4] = 1.0 if bond.IsInRing() else 0.0
    feats[5] = 1.0 if bond.GetIsConjugated() else 0.0
    stereo = bond.GetStereo()
    if stereo in STEREO_TYPES:
        feats[6 + STEREO_TYPES.index(stereo)] = 1.0
    return feats


def tree_decomp(mol):
    """Decompose molecule into a Junction Tree.

    Returns:
        jt_nodes: list of atom-index lists per JT node
        jt_edges: list of (src, dst) edge tuples
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    jt_nodes: list[list[int]] = []
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            jt_nodes.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    jt_nodes.extend([list(x) for x in Chem.GetSymmSSSR(mol)])
    if not jt_nodes:
        return [[0]], []

    nei_list = [[] for _ in range(n_atoms)]
    for i, node in enumerate(jt_nodes):
        for atom in node:
            nei_list[atom].append(i)

    node_sets = [set(n) if len(n) > 2 else None for n in jt_nodes]
    for i in range(len(jt_nodes)):
        if node_sets[i] is None or not jt_nodes[i]:
            continue
        for atom in jt_nodes[i]:
            for j in nei_list[atom]:
                if i >= j or node_sets[j] is None or not jt_nodes[j]:
                    continue
                if len(node_sets[i] & node_sets[j]) > 2:
                    node_sets[i] |= node_sets[j]
                    jt_nodes[i] = list(node_sets[i])
                    jt_nodes[j] = []
                    node_sets[j] = None

    jt_nodes = [n for n in jt_nodes if n]
    if not jt_nodes:
        return [[0]], []

    nei_list = [[] for _ in range(n_atoms)]
    for i, node in enumerate(jt_nodes):
        for atom in node:
            nei_list[atom].append(i)

    node_fsets = [frozenset(n) for n in jt_nodes]
    node_lens = [len(n) for n in jt_nodes]
    edges = {}

    for atom in range(n_atoms):
        cnei = nei_list[atom]
        if len(cnei) <= 1:
            continue
        bonds = [c for c in cnei if node_lens[c] == 2]
        rings = [c for c in cnei if node_lens[c] > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            new_idx = len(jt_nodes)
            jt_nodes.append([atom])
            node_fsets.append(frozenset([atom]))
            node_lens.append(1)
            for c1 in cnei:
                edges[(c1, new_idx)] = 1
        elif len(rings) > 2:
            new_idx = len(jt_nodes)
            jt_nodes.append([atom])
            node_fsets.append(frozenset([atom]))
            node_lens.append(1)
            for c1 in cnei:
                edges[(c1, new_idx)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = len(node_fsets[c1] & node_fsets[c2])
                    key = (c1, c2)
                    if key not in edges or edges[key] < inter:
                        edges[key] = inter

    if not edges:
        return jt_nodes, []

    n = len(jt_nodes)
    row, col, data = [], [], []
    for (u, v), w in edges.items():
        row.append(u)
        col.append(v)
        data.append(MST_MAX_WEIGHT - w)
    graph = coo_matrix((data, (row, col)), shape=(n, n))
    tree = minimum_spanning_tree(graph.tocsr())
    tree_row, tree_col = tree.nonzero()
    return jt_nodes, list(zip(tree_row.tolist(), tree_col.tolist()))


def jt_node_to_smiles(mol, atoms: list[int]) -> str:
    """Convert a JT node (atom indices) to canonical SMILES."""
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, canonical=True)
        return smiles if smiles else "C"
    except Exception:
        return ".".join(mol.GetAtomWithIdx(a).GetSymbol() for a in atoms)


def build_jt_mol_mappings(jt_nodes: list[list[int]], num_atoms: int):
    """Build bidirectional jt2mol / mol2jt membership maps."""
    jt2mol = [sorted(n) for n in jt_nodes]
    mol2jt = [[] for _ in range(num_atoms)]
    for jt_idx, atoms in enumerate(jt2mol):
        for a in atoms:
            mol2jt[a].append(jt_idx)
    return jt2mol, mol2jt


def build_jt_edge_index(jt_edges) -> torch.Tensor:
    """Convert edge list to undirected edge_index tensor (2, 2E)."""
    if not jt_edges:
        return torch.empty((2, 0), dtype=torch.long)
    ei = torch.tensor(jt_edges, dtype=torch.long).t().contiguous()
    return torch.cat([ei, ei.flip(0)], dim=1)


def build_jt_edge_attr(jt_nodes, jt_edges, atom_ids: np.ndarray, atom_features: np.ndarray) -> torch.Tensor:
    """Build undirected JT edge features from shared-atom chemistry.

    Feature layout:
      - mean shared-atom feature vector (51D)
      - shared atom count / 4
      - any aromatic shared atom
      - any hetero shared atom
      - any shared atom in ring
    """
    if not jt_edges:
        return torch.zeros((0, JT_EDGE_FEATURE_DIM), dtype=torch.float32)

    feats = []
    node_sets = [set(n) for n in jt_nodes]
    for u, v in jt_edges:
        shared = sorted(node_sets[u] & node_sets[v])
        if shared:
            shared_arr = atom_features[shared]
            mean_feat = shared_arr.mean(axis=0).astype(np.float32)
            count_norm = min(len(shared), 4) / 4.0
            aromatic_any = float(shared_arr[:, 7].max() > 0)   # atom aromatic flag
            hetero_any = float(np.any((atom_ids[shared] != 6) & (atom_ids[shared] != 1)))
            ring_any = float(shared_arr[:, 28].max() > 0)      # atom ring flag
        else:
            mean_feat = np.zeros((MOL_FEATURE_DIM,), dtype=np.float32)
            count_norm = aromatic_any = hetero_any = ring_any = 0.0
        feat = np.concatenate([
            mean_feat,
            np.array([count_norm, aromatic_any, hetero_any, ring_any], dtype=np.float32),
        ])
        feats.append(feat)

    arr = np.stack(feats, axis=0)
    arr = np.concatenate([arr, arr], axis=0)
    return torch.from_numpy(arr).float()


def compute_sub_atom_pool(atom_features: np.ndarray, atoms: list[int]) -> list[float]:
    """Mean over atom_features for the atoms in this substructure.

    Args:
        atom_features: (N_atoms, MOL_FEATURE_DIM) precomputed atom features
        atoms: atom indices in this JT node

    Returns:
        51-D list = mean of the member atoms' features, giving each JT node its
        molecule-context-aware atom chemistry.
    """
    if not atoms or atom_features.shape[0] == 0:
        return [0.0] * MOL_FEATURE_DIM
    sub = atom_features[atoms]
    return sub.mean(axis=0).astype(float).tolist()


def compute_jt_features(mol, atoms: list[int]) -> list[float]:
    """Compute 10-D fallback features for an out-of-vocabulary JT node."""
    n = len(atoms)
    atom_set = set(atoms)
    n_bonds = 0
    has_aromatic = False
    for a_idx in atoms:
        atom = mol.GetAtomWithIdx(a_idx)
        if atom.GetIsAromatic():
            has_aromatic = True
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(a_idx)
            if other in atom_set and a_idx < other:
                n_bonds += 1

    is_ring = 1.0 if n > 2 else 0.0
    ring_size = n / 8.0 if n > 2 else 0.0

    elem = {"C": 0, "N": 0, "O": 0, "S": 0, "hal": 0}
    for a_idx in atoms:
        anum = mol.GetAtomWithIdx(a_idx).GetAtomicNum()
        if anum == 6: elem["C"] += 1
        elif anum == 7: elem["N"] += 1
        elif anum == 8: elem["O"] += 1
        elif anum == 16: elem["S"] += 1
        elif anum in HALOGEN_ATOMS: elem["hal"] += 1

    denom = max(n, 1)
    return [
        n / 10.0, n_bonds / 10.0, is_ring, ring_size,
        1.0 if has_aromatic else 0.0,
        elem["C"] / denom, elem["N"] / denom, elem["O"] / denom,
        elem["S"] / denom, elem["hal"] / denom,
    ]
