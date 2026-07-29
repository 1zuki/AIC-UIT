# Vietnamese OCR Correction

## Approach

For OCR correction, most characters should stay copied. So this pipeline predicts a small edit for every input character:

```text
input char -> output segment
n -> nh
u -> ư
x -> ""   # deletion
c -> c    # copy
```

The final text is the concatenation of those predicted segments. This is much safer for OCR text than generating the whole paragraph from scratch.

## Files

Use:

```text
NLP/train.csv
NLP/test.csv
```

Submission format:

```text
submission.zip
└── submission.csv
```

## Test

```bash
python vn_ocr_transducer.py inspect \
  --train-csv NLP/train.csv \
  --test-csv NLP/test.csv
```

## Train locally on one GPU

Good starting run:

```bash
python vn_ocr_transducer.py train \
  --train-csv NLP/train.csv \
  --test-csv NLP/test.csv \
  --output-dir runs/ocr_transducer_seed42 \
  --seed 42 \
  --max-chars 768 \
  --d-model 384 \
  --layers 6 \
  --batch-size 24 \
  --epochs 30 \
  --eval-samples 1000
```

If VRAM is not enough:

```bash
python vn_ocr_transducer.py train \
  --train-csv NLP/train.csv \
  --test-csv NLP/test.csv \
  --output-dir runs/ocr_transducer_seed42 \
  --seed 42 \
  --max-chars 768 \
  --d-model 384 \
  --layers 6 \
  --batch-size 12 \
  --accum-steps 2 \
  --epochs 30
```

## Resume

```bash
python vn_ocr_transducer.py train \
  --train-csv NLP/train.csv \
  --test-csv NLP/test.csv \
  --output-dir runs/ocr_transducer_seed42 \
  --resume runs/ocr_transducer_seed42/last.pt \
  --epochs 45
```

## Inference

Use the best checkpoint:

```bash
python vn_ocr_transducer.py infer \
  --checkpoint runs/ocr_transducer_seed42/best.pt \
  --test-csv NLP/test.csv \
  --submission submission.csv \
  --zip submission.zip
```

Try threshold variants before spending too many submissions:

```bash
python vn_ocr_transducer.py infer \
  --checkpoint runs/ocr_transducer_seed42/best.pt \
  --test-csv NLP/test.csv \
  --submission submission_th035.csv \
  --zip submission_th035.zip \
  --threshold 0.35

python vn_ocr_transducer.py infer \
  --checkpoint runs/ocr_transducer_seed42/best.pt \
  --test-csv NLP/test.csv \
  --submission submission_th055.csv \
  --zip submission_th055.zip \
  --threshold 0.55
```

Lower threshold = more aggressive correction. Higher threshold = safer/copy more.

## Recommended submission strategy

Because the contest has limited submissions:

1. Train seed 42.
2. Submit best checkpoint with default threshold from validation.
3. Submit one conservative threshold, usually `0.55`.
4. Submit one aggressive threshold, usually `0.35`.
5. Train another seed only if local validation is promising.
