# Findings — do decision models converge as they get more competent?

**Short answer: no — and now cleanly.** With a validly sampled competence axis, independently-trained
decision models (BC, CQL) do **not** converge to a shared internal representation as they become more
competent: the pre-registered trend test is **not supported**, and between-seed similarity does not
rise with competence even in the regime where competence genuinely climbs.

A second, separate question — do they converge toward **reality** (physical state) or **value**
(closeness to goal)? — turns out to be **unanswerable in this maze**: a pre-registered
environment-validity gate rejects `medium-v2` because reality and value are too entangled to
separate. This **retracts** the earlier "organizes by reality, not value" claim, which was an
artifact of a degenerate value anchor (see below).

This is the instrument-fixed re-run; it supersedes the powered run archived at
`runs/archive/pixels_powered_100k/`.

---

## The claim (pre-registered, unchanged)

Platonic Representation Hypothesis, decision/RL form — **Claim A**: as a decision model gets more
capable, independently-trained versions of it become more similar inside. The deliverable is a
**trend** (similarity vs competence), not a one-shot comparison.

Frozen success rule (`config.yaml`, unchanged across all runs):

> Claim A is supported iff between-seed mutual-kNN **rises with competence** — **Spearman r > 0 AND
> permutation p ≤ 0.05** — for the decision objectives BC and CQL, on the full sweep with mean
> return as competence.

## Process

- **Data.** Minari `D4RL/pointmaze/medium-v2` — 8×8 PointMaze with interior walls. Real data only,
  hard-fail otherwise. Offline buffer **416,840 transitions / 418,840 states** (2,000 episodes).
  Fixed 4,000-state probe, stamped (`provenance_sha = bf727bc044a372a2`).
- **One shared encoder.** CNN, latent 256, ~2.76M params, identical for every objective; each
  `(objective, seed)` trains its own copy. Observations are **pixels** (64×64 RGB).
- **Objectives.** BC, CQL (decision/policy); JEPA, reward (predictors). All consume the one encoder.
- **Competence ladder.** 8 seeds, **16 rungs** with dense sub-1000 sampling
  `[0, 50, 100, 200, 350, 500, 750, 1k, 1.5k, 2k, 3.5k, 5k, 7.5k, 10k, 20k, 40k]` so the
  competence-**rising** phase is actually captured. Competence: policies → mean return (25 episodes);
  JEPA → horizon-2 retrieval **MRR**; reward → held-out R².
- **Geometry.** Frozen encoder on the probe. Headline y-axis = **between-seed mutual-kNN**. CKA and
  participation ratio logged.
- **Anchors (reality vs value).** Z = physical state `[x, y, vx, vy]`. V_emb = continuous,
  de-quantized walls-aware geodesic to **8 fixed landmarks** — a fair, multi-dim value/reachability
  anchor. Alignment measured by **distance correlation (dCor)** (primary; tie-robust,
  dimensionality-fair), mutual-kNN secondary.
- **Environment-validity gate.** Reject the maze for reality-vs-value if **dCor(Z, V_emb) ≥ 0.30**;
  the value kernel must also be non-degenerate. Both stamped in provenance.
- **Trend test.** Spearman(competence, between-seed similarity) + 10,000-shuffle permutation p, on
  the full sweep (frozen rule) and split at the competence peak into **rising** vs **plateau**
  regimes (diagnostic, so an overtrained tail is not read as a competence effect).

## Result 1 — Claim A is not supported (clean)

`claim_A_supported: false`. Untrained between-seed similarity = **0.873** (shared CNN inductive
bias); Gaussian null ≈ 0.003.

| objective | type | competence span | between-seed kNN | full r (p) | **rising-regime r (p)** | converges? |
|---|---|---|---|---|---|---|
| **BC**  | decision | 0.025 → 0.23 | 0.873 → 0.59 | +0.20 (0.46) | **−0.03 (0.94)** | **no** |
| **CQL** | decision | 0.040 → 0.29 | 0.873 → 0.76 | +0.07 (0.79) | **+0.16 (0.62)** | **no** |
| JEPA | predictor | 0.013 → 0.14 | 0.873 → 0.25 | −0.73 (**0.001**) | −0.73 (0.001) | no (diverges) |
| reward | predictor | 0.000 → 0.22 | 0.873 → 0.64 | −0.42 (0.11) | −0.42 (0.11) | no |

**Why this is the clean version.** The dense sub-1000 rungs gave the decision objectives a real
competence range (BC 0.025→0.23, CQL 0.04→0.29), fixing the flat-axis weakness of the prior run.
Even in the competence-**rising** regime, between-seed similarity does not rise with competence
(BC r = −0.03, CQL r = +0.16 — both far from significant). Between-seed similarity is highest at
random initialization and falls with training, and that fall does **not** track competence. Per the
pre-registered outcome table this is the *"full fail + rising fail → clean non-convergence"* cell:
competence genuinely climbs, convergence does not follow.

(The full-sweep r for BC/CQL is now mildly positive rather than the strongly negative value of the
100k run — because the moderate 40k tail excludes most of the overtraining-divergence tail. Neither
is significant. The honest statement is *no reliable convergence trend*, not *divergence*.)

## Result 2 — reality vs value is unanswerable in this maze (prior claim retracted)

The environment-validity gate **fired**: **dCor(Z, V_emb) = 0.545 ≥ 0.30 → `env_reject: true`.**
The value kernel is healthy (participation ratio 2.89, non-degenerate), so the anchor is fine — but
in `medium-v2`, physical state Z and geodesic value V_emb are too distance-correlated to be
separated. Both are dominated by position, and the medium maze's walls don't bend geodesics far
enough from straight-line distance. With Z and V_emb entangled at 0.545, "aligns more with value"
(every objective shows dCor→V_emb > dCor→Z here) **cannot be interpreted** — so no reality-vs-value
conclusion is drawn.

**This retracts the earlier "organizes by reality, not value" result.** That used a 1-D *quantized*
geodesic whose mutual-kNN alignment sat near the floor (~0.004) simply because a 1-D anchor has
almost no neighbor structure — a degenerate-anchor artifact, not evidence encoders ignore value.
A fair, continuous, multi-dim anchor shows the opposite raw ordering, and the gate then shows the
maze can't support either claim. **Verdict: needs a larger / more-walled maze** (`large-v2`) where
dCor(Z, V_emb) drops below 0.30.

## Result 3 — instrument fixes verified

- **JEPA competence metric now has range.** Horizon-2 retrieval MRR (pool 512) spans 0.013 → 0.138
  (the prior horizon-5 top-1 sat pinned at chance ~0.0005). On this valid axis JEPA seeds diverge
  **significantly** (r = −0.73, p = 0.001).
- **JEPA anti-collapse holds.** Participation ratio *grows* 1.2 → 34.8 over training
  (`non_collapse: true`) — the representation enriches while seeds diverge.
- **CKA still hides the effect** (secondary metric): saturates high for BC/CQL while mutual-kNN
  moves — the reason mutual-kNN is the headline.

## Scope

- **Clean:** the frozen test fails, and non-convergence holds in the competence-rising regime on a
  validly sampled competence axis. This is a valid, pre-registered negative result.
- **Bounded:** BC/CQL competence still caps ~0.2–0.3 return (task/data ceiling of PointMaze-medium
  offline), so "capable" here means "as capable as this task allows," not arbitrarily strong.
- **Open:** reality-vs-value is undetermined until a maze passes the env-validity gate.

## Conclusion

Within the design tested — walled PointMaze, pixels, one shared encoder, BC/CQL/JEPA/reward,
competence measured directly and sampled densely across the rising phase, mutual-kNN across 8
independent seeds — **this experiment finds no evidence that decision models converge to a shared
representation as they get more competent.** The pre-registered test is not supported for BC or CQL,
and between-seed similarity fails to rise with competence even where competence genuinely climbs;
JEPA diverges significantly. The reality-vs-value question is left open: `medium-v2` cannot separate
the two anchors (env-reject at dCor 0.545), and the earlier reality-leaning result was an
anchor artifact.

_Artifacts: `runs/prh_rl_claimA_pixels/` (`verdict.json`, `competence.jsonl`, `geometry.jsonl`,
`plots/`, `probe_provenance.json`). Prior runs: `runs/archive/pixels_powered_100k/` (powered 100k,
pre-fix), `runs/archive/pixels_schedule_10k/` (truncated). Frozen analysis choices:
`PREREGISTRATION.md`. Reproduce: `python run.py --obs-type pixels`._
