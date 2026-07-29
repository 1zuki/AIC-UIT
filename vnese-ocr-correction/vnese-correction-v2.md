# Vietnamese OCR Transducer V2

This is a safer copy-edit OCR correction pipeline for the Vietnamese OCR Text Correction contest.

## What changed from v1

V1 already fixed the big seq2seq hallucination issue by predicting one edit segment per input character. V2 keeps that idea, but adds:

- base-character features, e.g. `ầ/á/a` share an accentless base signal
- unicode character-type features: whitespace, digit, lower, upper, punctuation, symbol
- local depthwise convolution blocks before the Transformer encoder
- auxiliary edit-gate head, so the model must decide whether a character should be edited before the edit is accepted
- EMA checkpointing
- validation grid search over threshold, copy margin, gate threshold, and lexicon on/off
- conservative fallback if a prediction edits too much text or changes length too much
- same-run checkpoint ensemble at inference

It still uses only `train.csv` and `test.csv`. No external data, pretrained model, pretrained embeddings, test pseudo-labeling, or manual test editing.

## Inspect

```bash
python vn_ocr_transducer_v2.py inspect \
  --train-csv NLP/train.csv \
  --test-csv NLP/test.csv
```

## Serious single-GPU training

```bash
python vn_ocr_transducer_v2.py train \
  --train-csv NLP/train.csv \
  --test-csv NLP/test.csv \
  --output-dir runs/ocr_v2_seed3407 \
  --seed 3407 \
  --max-chars 896 \
  --hard-align-chars 1400 \
  --d-model 448 \
  --layers 7 \
  --conv-layers 2 \
  --ffn-dim 1792 \
  --batch-size 16 \
  --accum-steps 2 \
  --epochs 45 \
  --lr 4e-4 \
  --eval-samples 1200 \
  --eval-thresholds "0.55,0.65,0.75,0.85,0.90" \
  --eval-copy-margins "0.04,0.08,0.12" \
  --eval-gate-thresholds "0.20,0.35,0.50" \
  --eval-use-lexicon "0,1"
```

If VRAM is tight:

```bash
--batch-size 8 --accum-steps 4
```

## More aggressive run

Use this only if the normal run is stable:

```bash
python vn_ocr_transducer_v2.py train \
  --train-csv NLP/train.csv \
  --test-csv NLP/test.csv \
  --output-dir runs/ocr_v2_seed777_big \
  --seed 777 \
  --max-chars 1024 \
  --hard-align-chars 1600 \
  --d-model 512 \
  --layers 8 \
  --conv-layers 3 \
  --ffn-dim 2048 \
  --batch-size 8 \
  --accum-steps 4 \
  --epochs 50 \
  --lr 3.5e-4 \
  --eval-samples 1500 \
  --eval-thresholds "0.60,0.65,0.70,0.75,0.80,0.85,0.90" \
  --eval-copy-margins "0.04,0.08,0.12" \
  --eval-gate-thresholds "0.20,0.35,0.50"
```

## Inference using saved best decode params

```bash
python vn_ocr_transducer_v2.py infer \
  --checkpoint runs/ocr_v2_seed3407/best.pt \
  --test-csv NLP/test.csv \
  --submission submission_v2.csv \
  --zip submission_v2.zip
```

## Threshold variants

Since your public leaderboard likes around 0.65-0.85, try only a few carefully:

```bash
python vn_ocr_transducer_v2.py infer \
  --checkpoint runs/ocr_v2_seed3407/best.pt \
  --test-csv NLP/test.csv \
  --submission submission_v2_th075.csv \
  --zip submission_v2_th075.zip \
  --threshold 0.75 \
  --copy-margin 0.08 \
  --gate-threshold 0.35
```

Safer/copy-more:

```bash
--threshold 0.85 --copy-margin 0.12 --gate-threshold 0.50
```

More aggressive:

```bash
--threshold 0.65 --copy-margin 0.04 --gate-threshold 0.20
```

## Same-run checkpoint ensemble

This only safely works for checkpoints from the same output directory, because they share the same char/edit vocab.

```bash
python vn_ocr_transducer_v2.py infer \
  --run-dir runs/ocr_v2_seed3407 \
  --checkpoints \
    runs/ocr_v2_seed3407/best.pt \
    runs/ocr_v2_seed3407/epoch_040.pt \
    runs/ocr_v2_seed3407/last.pt \
  --test-csv NLP/test.csv \
  --submission submission_v2_ens.csv \
  --zip submission_v2_ens.zip
```

## Submission strategy

With limited submissions, do not brute-force thresholds forever. Recommended order:

1. `best.pt` with saved best params
2. `best.pt` threshold `0.75`
3. `best.pt` threshold `0.85`
4. same-run ensemble if validation CER improves
5. second seed only if its validation CER is similar or better

