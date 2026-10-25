# Preprocessing

Raw SMILES / CSV → `.pt` tensors for pre-training and fine-tuning.

```
preprocess/
├── run.sh                  entry point (pretrain | moleculenet | moleculeace)
├── core/                   featurizer (atom 51-D, bond 12-D, JT node 61-D), vocab/tensor I/O, dataset configs
├── pretrain/               JT vocabulary builder + ZINC250K tensorizer
├── moleculenet/            train/val/test split (random / scaffold) + tensorizer
└── moleculeace/            tensorizer for the fixed MoleculeACE splits (+ cliff_mask)
```

Vocabulary ids index the backbone's substructure embeddings, so tensors must be built with the
vocabulary the backbone was pre-trained with: the released backbone uses `data/vocab/vocab_zinc.json`.
A new backbone needs a new vocabulary (`build_vocab`), re-tensorized data, and `pretrain/train.py --vocab_path`.

## Usage

```bash
bash preprocess/run.sh pretrain                     # ZINC250K → data/processed/zinc250k/*.pt
bash preprocess/run.sh moleculenet                  # random split (seed 42) → data/processed/moleculenet/random_split
MODE=scaffold bash preprocess/run.sh moleculenet    # scaffold split → .../scaffold_split
bash preprocess/run.sh moleculeace                  # fixed splits → data/processed/moleculeace/<task>/
```

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `random` | `random` / `scaffold` / `both` (moleculenet) |
| `SEEDS` | `42` | split seed(s) (moleculenet) |
| `VOCAB_PATH` | `data/vocab/vocab_zinc.json` | JT vocabulary; built if missing |
| `WORKERS` | 60 / 30 | parallel workers (pretrain / moleculeace) |
| `CSV_DIR`, `OUT_DIR`, `CHUNK_SIZE`, `PYTHON` | see `run.sh` | paths, chunk size, interpreter |

Each step can also be run directly with `python -m preprocess.<pkg>.<module> --help`.

## Output layout

```
data/processed/
├── zinc250k/data_chunk_*.pt
├── moleculenet/{random_split,scaffold_split}/<dataset>/seed<n>/{train,val,test}.pt (+ .csv, metadata.json)
└── moleculeace/<task>/{train,test}.pt + cliff_mask.pt + metadata.json
```
