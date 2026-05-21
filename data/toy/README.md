# Toy data

Tiny public datasets shipped for smoke-testing the pipeline without external
downloads. Not sufficient to reproduce paper numbers.

| File | Source | Size |
|---|---|---:|
| `zinc_toy.smi` | First 500 lines of `jtnn_zinc250k.txt` (Jin et al., 2018; https://github.com/wengong-jin/icml18-jtnn) | 500 SMILES |
| `FreeSolv.csv` | MoleculeNet FreeSolv (Mobley & Guthrie, 2014; https://moleculenet.org) | 642 molecules |

## Quick smoke test

```bash
# 1) Preprocess toy ZINC + FreeSolv
python -m preprocess.pretrain.build_vocab \
    --input data/toy/zinc_toy.smi \
    --output data/toy/vocab.json --workers 4
python -m preprocess.pretrain.tensorize \
    --input data/toy/zinc_toy.smi \
    --output data/toy/data \
    --vocab data/toy/vocab.json --workers 4 --chunk_size 250
mkdir -p data/toy/MoleculeNet/raw && cp data/toy/FreeSolv.csv data/toy/MoleculeNet/raw/
python -m preprocess.moleculenet.split \
    --mode random --csv_dir data/toy/MoleculeNet/raw \
    --out_dir data/toy/MoleculeNet/split/random --seeds 1
python -m preprocess.moleculenet.tensorize \
    --data_dir data/toy/MoleculeNet/split/random \
    --vocab data/toy/vocab.json --seeds 1

# 2) Pre-train (2 epochs)
python pretrain/train.py \
    --data_path data/toy \
    --checkpoint_dir data/toy/ckpt \
    --batch_size 64 --max_epochs 2 --warmup_epochs 1 \
    --num_workers 2 --gpu_id 0

# 3) Fine-tune FreeSolv (toy vocab is 202, not the default 1753)
python finetune/train.py \
    --dataset FreeSolv --seed 1 \
    --pretrain_ckpt data/toy/ckpt/best_model.ckpt \
    --data_base data/toy/MoleculeNet/split/random \
    --jt_vocab_size 202 \
    --stage1_epochs 5 --max_epochs 15 --early_stop_patience 5 \
    --learning_rate 0.01 --weight_decay 5e-5 --batch_size 32 \
    --num_head_layers 2 --head_hidden 32 --head_dropout 0.2 \
    --gpu_id 0
```

End-to-end runs in ~1–2 minutes on a single GPU. The fine-tune RMSE on the
toy split is not meaningful; this exercises the loop only.
