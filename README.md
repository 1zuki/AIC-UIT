# AIC-UIT

Collection of my AI Challenge projects and experiments within VNUHCM.

This repo contains multiple challenge tracks (VQA, OCR correction, Nôm character recognition, sign language video classification, and translation), each in its own folder with separate data and outputs.

## Repository Layout

```text
AIC-UIT/
├── adversarial-attack/
├── visual-question-answering/
├── sign-lang-reco/
├── trans/
├── vnese-ocr-correction/
└── LICENSE
```

## Quick Setup

Use Python 3.10+ (3.11 recommended for newer tooling).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- CUDA support depends on your local PyTorch install.
- Some projects are notebook-first and already include generated outputs in their folders.

## Projects

### 1) Adversarial VQA Attack

Path: `adversarial-attack/`

Main entrypoints:
- `adversarial_vqa_attack_worker.py` (local full-run script)
- `adversarial_vqa_attack_worker_local.ipynb` (local notebook)
- `adversarial_vqa_attack_worker_kaggle_tpu.ipynb` (Kaggle TPU notebook)

Expected local assets:
- `dataset/` (contains `images/` and `questions/test.json`)
- `Model/` (contains `vit_model.py`, `model.pth`, `vocab.pth`, `ans_vocab.pth`)

Smoke run:

```bash
cd adversarial-attack
python adversarial_vqa_attack_worker.py --preset FAST --limit-images 64
```

Full run:

```bash
cd adversarial-attack
python adversarial_vqa_attack_worker.py --preset COOK
```

Main outputs:
- `adv_outputs_smooth/`
- `submission.zip`

### 2) Visual Question Answering (SynVQA)

Path: `visual-question-answering/`

Main entrypoint:
- `synvqa_cooked_film_ensemble.ipynb`

Expected local data:
- `datasets/images/`
- `datasets/questions/train.json`
- `datasets/questions/test.json`

Main outputs:
- `outputs/submission.csv`
- `outputs/submission.zip`
- `outputs/models/*.pth` (best checkpoints)
- `outputs/models/*.last.pth` (resume checkpoints)

### 3) Sign Language Recognition

Path: `sign-lang-reco/`

Main scripts:
- `src/train.py`
- `src/infer.py`

Expected local data:
- `data/train/` (class folders with `.mp4`)
- `data/test/` (test `.mp4`)
- `data/label_mapping.pkl`

Train:

```bash
cd sign-lang-reco/src
python train.py
```

Infer and export submission:

```bash
cd sign-lang-reco/src
python infer.py
```

Outputs:
- `../outputs/best.pth`
- `../outputs/submission.csv`
- `../outputs/submission.zip`

### 4) Chinese -> Vietnamese Translation

Path: `trans/`

Main entrypoint:
- `baseline.py`

Expected local data:
- `dataset/train/train.zh`
- `dataset/train/train.vi`
- `dataset/test/test.zh`

Run:

```bash
cd trans
python baseline.py
```

Outputs:
- `submission.csv`
- `submission.zip`
- `checkpoints/`

### 5) Vietnamese OCR Correction + Nôm Character Recognition

Path: `vnese-ocr-correction/`

This folder contains 2 separate challenge pipelines.

#### A. Vietnamese OCR Text Correction

Main script:
- `vn_ocr_transducer.py`

Expected data:
- `NLP/train.csv`
- `NLP/test.csv`

Inspect:

```bash
cd vnese-ocr-correction
python vn_ocr_transducer.py inspect --train-csv NLP/train.csv --test-csv NLP/test.csv
```

Train:

```bash
cd vnese-ocr-correction
python vn_ocr_transducer.py train --train-csv NLP/train.csv --test-csv NLP/test.csv --output-dir runs/ocr_transducer_seed42
```

Infer:

```bash
cd vnese-ocr-correction
python vn_ocr_transducer.py infer --checkpoint runs/ocr_transducer_seed42/best.pt --test-csv NLP/test.csv --submission submission.csv --zip submission.zip
```

#### B. Nôm Character Recognition

Main script:
- `nom_character_cooker.py`

Expected data:
- `CV/dataset/train/images`
- `CV/dataset/train/labels.csv`
- `CV/dataset/test/images`

Inspect:

```bash
cd vnese-ocr-correction
python nom_character_cooker.py inspect --data-dir CV/dataset
```

Train:

```bash
cd vnese-ocr-correction
python nom_character_cooker.py train --data-dir CV/dataset --output-dir runs/nom_convnext_seed42 --model convnext_tiny
```

Infer:

```bash
cd vnese-ocr-correction
python nom_character_cooker.py infer --data-dir CV/dataset --checkpoints runs/nom_convnext_seed42/best.pt --submission-csv submission.csv --zip submission.zip --tta 5
```

## Tips

- Keep each project isolated when running experiments to avoid mixing outputs.
- Most folders already contain examples of generated `submission.csv` and `submission.zip`.
- If a run is interrupted, check each project's `runs/` or `outputs/` folder for resumable checkpoints.
