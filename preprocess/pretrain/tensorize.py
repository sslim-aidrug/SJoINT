"""Pretraining tensoriser: raw SMILES (one per line) -> .pt chunks.

Usage:
    python -m preprocess.pretrain.tensorize \\
        --input data/zinc250k/smiles.txt --output data/zinc250k/data \\
        --vocab data/zinc250k/vocab.json
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from preprocess.core.features import MOL_FEATURE_DIM
from preprocess.core.utils import load_vocab, process_smiles, to_sample, save_chunk


def build_dataset(smiles_txt, out_path, vocab_path, num_workers, chunk_size):
    vocab_size = load_vocab(vocab_path)

    print("=" * 60)
    print("  SJoINT Pretrain Preprocess")
    print(f"  JT vocab : {vocab_size:,}  |  Mol: atom_id + {MOL_FEATURE_DIM}D")
    print(f"  Workers  : {num_workers}  |  Chunk: {chunk_size:,}")
    print("=" * 60)

    with open(smiles_txt) as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    total = len(all_smiles)
    print(f"\nLoaded {total:,} SMILES from {smiles_txt}")

    save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="saver")
    save_futures = []
    final_buf = []
    chunk_num = 0
    saved = 0
    failed = 0

    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_smiles, all_smiles, chunksize=512),
            total=total, desc="Processing", smoothing=0.1,
        )
        for result in pbar:
            if result is None:
                failed += 1
                continue
            final_buf.append(to_sample(result))
            saved += 1
            if len(final_buf) >= chunk_size:
                chunk_to_save = final_buf[:chunk_size]
                final_buf[:] = final_buf[chunk_size:]
                save_futures.append(save_executor.submit(
                    save_chunk, list(chunk_to_save), out_path, chunk_num))
                chunk_num += 1
                pbar.set_postfix(ok=saved, fail=failed, chunks=chunk_num)

    if final_buf:
        save_futures.append(save_executor.submit(
            save_chunk, list(final_buf), out_path, chunk_num))
        chunk_num += 1

    for fut in save_futures:
        fut.result()
    save_executor.shutdown(wait=True)

    print(f"\n{'='*60}")
    print(f"  Total: {total:,}  |  OK: {saved:,}  |  Fail: {failed:,}  |  Chunks: {chunk_num}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="SMILES txt file")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--vocab", required=True, help="JT vocab JSON")
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Use fork so workers inherit loaded vocab
    mp.set_start_method("fork", force=True)

    build_dataset(args.input, args.output, args.vocab, args.workers, args.chunk_size)


if __name__ == "__main__":
    main()
