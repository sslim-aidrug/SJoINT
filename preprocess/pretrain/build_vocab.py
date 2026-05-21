"""Build JT substructure vocabulary from a SMILES corpus.

Output: JSON with vocab_id -> SMILES mapping (0=<PAD>, 1=<UNK>, 2+=substructures)
        and 10-D fallback feature vectors per entry.

Usage:
    python -m preprocess.pretrain.build_vocab \\
        --input data/zinc250k/smiles.txt --output data/zinc250k/vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
from collections import Counter

from rdkit import Chem
from tqdm import tqdm

from preprocess.core.features import (
    compute_jt_features,
    jt_node_to_smiles,
    tree_decomp,
)


def extract_jt(smiles: str) -> list[tuple[str, list[float]]] | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        jt_nodes, _ = tree_decomp(Chem.RWMol(mol))
        return [(jt_node_to_smiles(mol, n), compute_jt_features(mol, n))
                for n in jt_nodes]
    except Exception:
        return None


def build_vocab(input_txt: str, num_workers: int = 60, min_count: int = 1):
    print(f"Loading SMILES from {input_txt}...")
    with open(input_txt) as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    total = len(all_smiles)
    print(f"  Loaded {total:,} SMILES")

    counter = Counter()
    feat_sums, feat_counts = {}, {}

    print(f"\nExtracting JT substructures with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_jt, all_smiles, chunksize=1024),
            total=total, desc="JT decomp"
        ):
            if result is None:
                continue
            for smi, feat in result:
                counter[smi] += 1
                if smi not in feat_sums:
                    feat_sums[smi] = [0.0] * 10
                    feat_counts[smi] = 0
                for i in range(10):
                    feat_sums[smi][i] += feat[i]
                feat_counts[smi] += 1

    filtered = [(smi, cnt) for smi, cnt in counter.most_common() if cnt >= min_count]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    features = {"<PAD>": [0.0] * 10, "<UNK>": [0.0] * 10}

    for smi, _ in filtered:
        vocab[smi] = len(vocab)
        fc = feat_counts[smi]
        features[smi] = [feat_sums[smi][i] / fc for i in range(10)]

    print(f"\n{'='*60}")
    print(f"  Total molecules:      {total:,}")
    print(f"  Unique substructures: {len(filtered):,}")
    print(f"  Vocab size:           {len(vocab):,} (incl. <PAD>, <UNK>)")
    print(f"{'='*60}")

    return {"vocab": vocab, "features": features,
            "stats": {"total_molecules": total, "vocab_size": len(vocab)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--min_count", type=int, default=1)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    result = build_vocab(args.input, args.workers, args.min_count)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
