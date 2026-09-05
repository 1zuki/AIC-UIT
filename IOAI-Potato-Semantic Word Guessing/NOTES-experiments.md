# Potato Semantic Word Guessing — experiment notes

Written 2026-08-27. Everything here comes from artifacts in `experiments/`,
`baselines/`, and the root submission package. Local public scores are not
evidence about the private judge.

## What is submitted right now

| Item | Value |
| --- | --- |
| Root notebook | `solution.ipynb`, SHA-256 `7f6e901b2001f668f3bc5007e643b954c57fccefd03c014fc6599d138482caac` |
| Root archive | `solution.zip`, SHA-256 `6561c1a25bc665d5733c01f5b4b26dfd8db1fb29988e5f3e590b2a029ae8ddb7` |
| Strategy | `consistency_base`: 3-expert mixture posterior over word hypotheses, contradiction tracking, consistency-triggered switch from expert 0 to the full mixture |
| Local public judge | 120/120 wins, 98.00/100, 15.33 s |
| Archived predecessor | `baselines/score_57_02/`, SHA-256 `2cc93b01…`, private 57.02 with 82/120 wins |

The predecessor is the only artifact with a known private score. The current root
package was promoted by explicit user override after the sealed gate said
`retain_incumbent`, so its private performance is unknown.

## Method that framed all the work

- Clean room: the notebook reads only `dataset/vocabulary.json` and
  `dataset/public_embeddings.npy`, never `test_public.json`, never the network.
- Single pure code cell, JSON-only stdout, fresh state on `new_game`.
- Candidates were screened against surrogate private oracles: the public
  embedding matrix transformed to imitate a judge we cannot see (raw, masked
  dimensions, centered, CSLS, spectral truncation, sharpened, smoothed, mixed).
- Secret index sets were salted and, in the later rounds, cell-disjoint, so the
  same games could not be used both to pick a config and to justify it.
- A single-use sealed promotion gate (`experiments/final_promotion_gate.py`)
  with pre-committed thresholds decided promotion. It is now spent.

## What was tried

### Likelihood and posterior families

| Family | Verdict |
| --- | --- |
| Single sharp logistic (`control_sharp`, scale 0.005, no flip) | Strong on dev screens (81.5 mean), fragile on the wider blind set |
| Adaptive flip with reliability gate (`control_adaptive`) | The 57.02 baseline shape; consistently mid-pack, never worst |
| Error floors 0.01 / 0.03 / 0.05 | All below `control_sharp`; floor 0.01 clearly worst (72.96) |
| Cauchy and heavy-tailed links (`latent_cauchy`, `single_cauchy`) | Best on the orthogonal screen (76.78) but did not survive blind-v2 |
| Sign-only and noise-mixture links | Weaker than the logistic family |
| Preference / Bradley-Terry style (`preference*`) | Never reached the shortlist |
| 3-expert mixtures with priors (0.50, 0.30, 0.20) | Basis of both finalists |
| Prior variants 0.70 / 0.80 on expert 0 | 79.88 / 79.00; no improvement over the base mixture |
| Contradiction budget 1, violation tolerance 3.2e-5 | Equal or slightly worse than budget 0 |

### Query and candidate selection

| Idea | Verdict |
| --- | --- |
| Binary-split entropy plus a growing direct-hit bonus | Kept; this is the incumbent selection rule |
| Mutual-information query over the mixture (`query="mi"`) | Best dev config (`ensemble_mi_base13`, 82.79) but lost every stress round |
| Fixed coverage switches (`*_cover10/13/16`) | Rejected; coverage 10 collapsed the worst case to 46.5 |
| MAP-turn late modes (`map10`, `map13`) | No gain over the plain late switch at turn 30 |
| Switch thresholds 0.005 / 0.010 / 0.020 / 0.04 | Indistinguishable on the dev screen |
| Block-gated geometry over 5 embedding blocks | Identical to `control_adaptive`; the gate never fired usefully |
| Committee and rejuvenation variants | `latent_rejuvenate` was the worst config tested (53.94) |

### Engineering

- Chunked cosine matrix construction, float32 similarities, single-threaded BLAS.
- int16 similarity quantization and 32-wide chunks: correct, but bought no
  resource headroom worth the added complexity.
- Five hard-limit resource repeats on the finalist: about 34 s solution time,
  about 54 MB VmHWM, about 24 MB cgroup peak, against limits of 45 s and 58 MB.
  All five passed.

### Stress oracles, 64 games each

| Oracle | 57.02 baseline | `consistency_base` | `mi_base13` |
| --- | --- | --- | --- |
| raw (120 games) | 98.53 | 98.52 | 97.80 |
| CSLS | 96.09 | 96.28 | 93.31 |
| half-center | 97.41 | 97.50 | 97.19 |
| mask60 | 95.66 | 95.59 | 94.44 |
| sharpen16 | 96.88 | 96.63 | 94.69 |
| smooth25 | 95.88 | 95.34 | 95.13 |

`mi_base13` won the dev screens and lost every full-view stress cell. That is the
clearest single lesson from the whole search: dev-screen wins on a small salted
oracle set did not transfer.

### The sealed gate, 256 games, 8 cells

Decision: `retain_incumbent`, `promote=false`, no retry allowed.

- Overall: baseline 244 wins / 88.375, candidate 243 wins / 88.250.
- Paired: 196 identical games, 13 major gains, 13 major harms, 3 rescues, 4 harms.
- Bootstrap 99% lower bound on the score delta: -1.98.
- Thresholds demanded at least +2.0 score and +8 wins, so the gate could only
  fire on a large effect. It measured roughly zero.
- Per cell: `held_center75` +2.69 and `held_mix60` +0.69 for the candidate;
  `full_mask50` -1.56, `held_raw` -1.06, `held_csls48` -0.88.

Read plainly: the two finalists are statistically tied under every surrogate we
built, and the candidate trades robustness on masked and CSLS geometry for gains
on centered geometry.

## Ideas thought about but not tried

Ordered roughly by expected value per unit of work.

1. Score-aware utility. Selection currently maximizes an entropy proxy plus a
   hand-tuned hit bonus. The real objective is a turn-indexed score curve with
   p10/p20 cliffs. Optimizing expected normalized score directly, including a
   genuine one- or two-step lookahead, is the most principled unexplored gain.
2. Transitivity constraints. Every comparison implies an order relation. The
   posterior currently treats observations independently and only counts
   violations. Maintaining the induced partial order and pruning hypotheses that
   violate it could be much sharper than soft likelihood reweighting.
3. Rank-based likelihood. Use the rank of a similarity within its row instead of
   the raw margin. Ranks are invariant to any monotone rescaling of the judge's
   similarity function, which is exactly the unknown the surrogate oracles were
   built to guess at.
4. Hubness and anisotropy correction inside the model. CSLS and centering were
   used only as adversarial stress transforms. Making them experts, or removing
   the top principal components once in an all-but-the-top style, is a different
   bet: that the private judge is closer to a corrected geometry than to raw.
5. Online transform identification. Put a posterior over a handful of candidate
   similarity geometries and do Bayesian model averaging over them as evidence
   arrives, instead of committing to a fixed 3-expert mixture with fixed priors.
6. Informative "same" verdicts. Ties are currently only renormalized. A tie is a
   real constraint (two words nearly equidistant from the target) and could be
   used as evidence rather than a no-op.
7. Embedding-density priors. A non-uniform prior from local neighborhood density
   is clean-room legal and might beat the uniform prior in the first few turns,
   where most of the score is decided.
8. Wider shortlists. `query_shortlist=96` and `hit_shortlist=32` were never
   swept, and there is runtime headroom (34 s of a 45 s budget).
9. Gate design. The thresholds (+2.0 score, +8 wins) are strict enough that only
   a large effect could ever promote. If the truth is a real but small
   improvement, this protocol will never detect it. A larger gate with calibrated
   thresholds would be a better instrument, but only 45 reserve indices and 301
   unspent indices remain, and the v2 gate is spent.

## Deliberately not done

Reading `test_public.json` as part of the strategy, tuning against the
leaderboard, runtime downloads, or bundling anything beyond `solution.ipynb`.
All were available and all were ruled out as cheating.

## Honest caveats

- Local public score 98.00 against a known private 57.02 for the predecessor.
  The public/private gap is enormous; treat local numbers as a smoke test for
  protocol correctness, not as a performance estimate.
- No surrogate oracle built here is known to resemble the real judge.
- The current root package was promoted against the gate's recommendation. The
  57.02 artifact is preserved byte-identical under `baselines/score_57_02/` and
  can be restored at any time.
