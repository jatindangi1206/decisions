# Findings — do decision models converge as they get more competent?

**Short answer: no.** In this experiment, independently-trained decision models (BC and CQL) do
**not** converge to a shared internal representation as they become more competent. The
pre-registered trend test is **not supported** for either decision objective; between-seed
representational similarity is *highest at random initialization* and *declines* with training.

This document records the process, the numbers, and the scope of that claim.

---

## The claim being tested (pre-registered)

Platonic Representation Hypothesis, decision/RL form — **Claim A**: as a decision model gets more
capable, independently-trained versions of it become more similar inside. Deliverable is a **trend**
(similarity vs competence), not a one-shot comparison.

Pre-registered success rule (frozen in `config.yaml` before running, unchanged after):

> Claim A is supported if between-seed mutual-kNN **rises with competence**
> (**Spearman r > 0 AND permutation p ≤ 0.05**) for the decision objectives (BC, CQL).

## Process (how the number is produced)

- **Environment / data.** Minari `D4RL/pointmaze/medium-v2` — an 8×8 PointMaze **with interior
  walls**, so the walls-aware geodesic distance-to-goal genuinely differs from straight-line
  distance. Real data only; the pipeline hard-fails if the dataset can't be resolved. Offline
  buffer: **416,840 transitions / 418,840 states** (2,000 episodes). Provenance stamped on every
  report (`provenance_sha = 24b3a6377a010f68`, probe seed 12345, probe size 4000).
- **One shared encoder.** Same architecture (CNN, latent 256, ~2.76M params) for every objective;
  each `(objective, seed)` trains its own copy. Objective-specific parts live in heads. Observations
  are **pixels** (64×64 RGB rendered from state) — the primary study; the 6-number "state" mode is a
  plumbing check only.
- **Objectives.** BC, CQL (both decision/policy), JEPA and reward (predictors). All consume the one
  encoder.
- **Competence ladder (x-axis).** Each objective trained to 100k steps, **8 seeds**, checkpointed at
  12 rungs `[0, 1k, 2k, 3.5k, 5k, 7.5k, 10k, 15k, 25k, 40k, 65k, 100k]`. Competence measured at each
  rung: policies → mean return over 25 env episodes; predictors → held-out predictive score.
- **Geometry (y-axis).** Freeze encoder, extract on the fixed 4000-state probe. **Between-seed
  mutual-kNN** (primary, local) is the headline; CKA (global) and participation ratio logged too.
- **Anchors (reality vs value).** Z = physical state `[x,y,vx,vy]`; V = walls-aware geodesic
  distance-to-goal. Alignment of each encoder to Z and to V via mutual-kNN.
- **Nulls.** Random-init (untrained, step-0) and Gaussian (chance floor).
- **Trend test.** Spearman(competence, between-seed similarity) across checkpoints + permutation test
  (10,000 shuffles) for p. Applied per the pre-registered rule.

## Result: Claim A is not supported

`verdict.json` → `claim_A_supported: false`. Floors: Gaussian null mutual-kNN = **0.0027**;
untrained between-seed similarity = **0.873**.

| objective | type | competence (0→100k) | between-seed mutual-kNN | Spearman r | perm p | convergence? |
|---|---|---|---|---|---|---|
| **BC**  | decision | ~0.20, flat from step 1k | 0.66 → **0.54** (steady decline) | **−0.34** | 0.29 | **no** |
| **CQL** | decision | ~0.22, flat/noisy | 0.78 → 0.82 → **0.66** (rise then fall) | **−0.55** | 0.066 | **no** |
| JEPA | predictor | metric saturated (see below) | 0.35 → 0.22 (strong divergence) | −0.49 | 0.11 | no |
| reward | predictor | 0 → 0.22 (real ladder) | 0.50 → 0.69 → 0.53 (rise then fall) | +0.07 | 0.83 | no |

**Reading:** for both decision objectives the trend is *negative* — independently-trained seeds get
**less** similar as training proceeds, not more. Similarity is maximal at random init (0.873, driven
by shared CNN inductive bias) and erodes as each seed specializes. No objective, decision or
predictor, satisfies the pre-registered convergence rule.

### The convergence signal is non-monotone (important, honest detail)

CQL and reward *do* converge briefly early (through ~3.5–5k steps) before diverging out to 100k.
The earlier, shorter run (archived at `runs/archive/pixels_schedule_10k/`, latent 64, 5 seeds,
stopped at 10k) caught only that early-rising phase and showed CQL r = +0.71 — the reverse sign.
So the honest characterization is **"convergence peaks early, then reverses with continued
training,"** and the net effect over a full competence sweep is no convergence.

## Two robust side-results

- **Reality over value — decisively, at every checkpoint and both model scales.** Encoder neighbor
  structure aligns with physical state Z (align→Z ≈ 0.13–0.16) but **not** with value/geodesic V
  (align→V ≈ 0.0038, pinned at the 0.0027 null floor — 35–60× smaller). The collapse test agrees:
  far-in-reality/near-in-value pairs and their mirror sit at near-equal encoder distances at every
  rung. Decision and predictive encoders in this maze organize by *where you physically are*, not
  *how close to the goal you are*.
- **JEPA anti-collapse fix worked.** Participation ratio now *grows* with training (4.5 → 12.9;
  `non_collapse: true`) instead of collapsing (2.4 → 1.2 in the pre-fix run) — the representation
  genuinely enriches, yet the seeds still diverge. (Its multi-step retrieval competence metric
  saturated at chance and needs recalibration — a measurement issue, not a representation issue.)
- **CKA hides the effect.** Global CKA sits at 0.92–0.97 for BC/CQL and barely moves while mutual-kNN
  falls — exactly the "washes out under calibration" caution that made mutual-kNN the headline.

## Scope and limitations of the "they do not converge" claim

What is clean: **the pre-registered test fails for both decision objectives, and between-seed
similarity does not rise with competence — it declines with training.** That is a valid,
frozen-in-advance negative result and is the basis for the conclusion below.

What tempers the *strength* of a blanket "capable decision models never converge":

1. **Competence saturated early.** BC/CQL hit ~0.2 return by the first checkpoint, so the competence
   axis had little range; strictly, we observe divergence with *training*, and the *competence*-
   specific null is underpowered. The task/data (PointMaze-medium offline), not model capacity, caps
   competence here.
2. **Non-monotonicity** (above): the sign of the trend depends on how long you train.
3. **Overtraining a 2.76M-param encoder on 416k transitions** plausibly contributes to the divergent
   tail via seed-specific specialization.

**Recommended confirmation** (to make the claim reviewer-proof): densely checkpoint the sub-1000-step
window where competence actually climbs (e.g. 50/100/200/350/500/750). If similarity still does not
rise with competence *there*, non-convergence is established on a valid competence axis rather than
inferred from overtraining.

## Conclusion

Within the design tested — walled PointMaze, pixel observations, one shared encoder, BC/CQL/JEPA/
reward, competence measured directly, mutual-kNN across 8 independent seeds — **this experiment finds
no evidence that decision models converge to a shared representation as they get more competent.**
The pre-registered convergence test is not supported for either BC or CQL; between-seed similarity is
largest at initialization and decreases with training. Independently-trained decision models do not,
here, become more alike inside as they improve — and separately, to the extent their representations
are organized at all, they align with physical reality, not with proximity to the goal.

_Artifacts: `runs/prh_rl_claimA_pixels/` (`verdict.json`, `competence.jsonl`, `geometry.jsonl`,
`plots/`); prior truncated run `runs/archive/pixels_schedule_10k/`. Reproduce: `python run.py
--obs-type pixels`._
