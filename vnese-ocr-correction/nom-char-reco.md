# Nôm Character Recognition

This is a single-GPU, checkpointed image-classification pipeline for the Nôm character task.

It follows the challenge format:

```text
dataset/
├── train/
│   ├── images/
│   └── labels.csv      # image,label
└── test/
    └── images/
```

The final upload format is:

```text
submission.zip
└── submission.csv      # image,label
```

## 1. Inspect the dataset

```bash
python nom_character.py inspect \
  --data-dir CV/dataset
```

The script can also accept a parent folder and will try to find `CV/dataset` automatically.

## 2. Safe strong training run

Recommended first serious run:

```bash
python nom_character.py train \
  --data-dir CV/dataset \
  --output-dir runs/nom_convnext_seed42 \
  --model convnext_tiny \
  --img-size 224 \
  --epochs 40 \
  --batch-size 32 \
  --lr 3e-4 \
  --seed 42
```

If VRAM is low:

```bash
python nom_character.py train \
  --data-dir CV/dataset \
  --output-dir runs/nom_convnext_seed42 \
  --model convnext_tiny \
  --img-size 224 \
  --epochs 40 \
  --batch-size 16 \
  --accum-steps 2 \
  --lr 3e-4 \
  --seed 42
```

Outputs:

```text
runs/nom_convnext_seed42/
├── best.pt
├── last.pt
├── epoch_005.pt
├── epoch_010.pt
├── ...
├── train_log.csv
├── label_vocab.json
└── config.json
```

## 3. Resume training

```bash
python nom_character.py train \
  --data-dir CV/dataset \
  --output-dir runs/nom_convnext_seed42 \
  --resume runs/nom_convnext_seed42/last.pt \
  --epochs 60
```

## 4. Generate submission

Greedy/low TTA:

```bash
python nom_character.py infer \
  --data-dir CV/dataset \
  --checkpoints runs/nom_convnext_seed42/best.pt \
  --submission-csv submission.csv \
  --zip submission.zip \
  --tta 5
```

More serious TTA:

```bash
python nom_character.py infer \
  --data-dir CV/dataset \
  --checkpoints runs/nom_convnext_seed42/best.pt \
  --submission-csv submission_tta9.csv \
  --zip submission_tta9.zip \
  --tta 9
```

## 5. Ensemble checkpoints or seeds

You can average logits from multiple checkpoints/runs at inference:

```bash
python nom_character.py infer \
  --data-dir CV/dataset \
  --checkpoints \
    runs/nom_convnext_seed42/best.pt \
    runs/nom_convnext_seed3407/best.pt \
  --submission-csv submission_ensemble.csv \
  --zip submission_ensemble.zip \
  --tta 5
```

## 6. Submission strategy with only 30 attempts

Do not spam random submissions. Suggested order:

1. `convnext_tiny`, seed 42, `tta=5`
2. same checkpoint, `tta=9`
3. `convnext_tiny`, seed 3407, `tta=5`
4. ensemble seed 42 + seed 3407, `tta=5`
5. optional: `efficientnet_v2_s`, seed 42, `tta=5`
6. ensemble ConvNeXt + EfficientNet if local validation suggests both are strong

Macro F1 rewards rare classes, so the script uses:

- stratified validation
- class-balanced sampler
- class-weighted focal cross entropy
- EMA checkpointing
- no flip augmentation, because flipping changes characters
- no external data and no test pseudo-labeling
