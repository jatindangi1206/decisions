# Pre-registration — instrument fixes for the re-run

Frozen **before** the instrument-fixed re-run. These are **instrument fixes and coverage
improvements**, not a change of hypothesis or success criterion. The success rule below is
**unchanged** from the original `config.yaml`; nothing here relaxes or moves it.

## Success rule (UNCHANGED, frozen)

> Claim A is supported iff between-seed mutual-kNN **rises with competence**
> — **Spearman r > 0 AND permutation p ≤ 0.05** — for the decision objectives **BC and CQL**,
> on the **full-sweep** trend using **mean return** as competence.

Permutation test: 10,000 shuffles of competence order. This is the only test that decides
`claim_A_supported`.

## Frozen analysis choices (new, fixed now before seeing re-run results)

1. **Reality-vs-value — PRIMARY statistic = distance correlation (dCor).** Székely distance
   correlation between each encoder and the reality anchor **Z** vs the value anchor **V_emb**,
   subsampled to 2000 probe points with a fixed seed. dCor is tie-robust and dimensionality-fair,
   unlike mutual-kNN (which stays as the SECONDARY readout). `drifts_toward = value` iff
   mean dCor(enc, V_emb) > mean dCor(enc, Z), else `reality`.
2. **Value anchor rebuilt to be fair.** V = continuous, de-quantized walls-aware geodesic to each
   state's own goal (sub-cell gradient projection, no integer ties). **V_emb** = continuous
   geodesic to **8 fixed farthest-point landmarks** — a multi-dim reachability embedding
   comparable in dimensionality to 4-D Z.
3. **Environment-validity gate.** The maze is only a valid reality-vs-value testbed if the two
   anchors are sufficiently distinct: **reject if dCor(Z, V_emb) ≥ 0.30** (independent baseline at
   this N ≈ 0.10). The value kernel must be **non-degenerate** (participation ratio ≥ 1.5 and
   pairwise-distance CV ≥ 0.1). Both are stamped in `probe_provenance.json`.
4. **Competence axis.** Primary competence = mean return (policies) / predictor score, as before.
   **success_rate** is logged as an ALTERNATIVE competence measure and a parallel trend is
   reported (not used by the frozen rule). **JEPA competence = horizon-2 retrieval MRR** over a
   512-candidate pool (rank-based, dynamic range) — replacing the chance-pinned horizon-5 top-1.
5. **Regime split (diagnostic, pre-registered).** Split checkpoints at the step of **peak measured
   competence** into a competence-**rising** regime and a **plateau/overtrained** regime; report
   Spearman separately for each. This prevents a declining overtrained tail from being read as a
   competence effect. The frozen rule still uses the full sweep — the split only interprets it.
6. **Schedule.** Dense sub-1000 rungs (50/100/200/350/500/750) to sample the competence-rising
   phase, then a moderate tail to 40k. (JEPA participation-ratio non-collapse check retained.)

## Pre-registered outcome table

| full-sweep rule (frozen) | rising-regime trend | interpretation |
|---|---|---|
| PASS (r>0, p≤0.05) | — | Claim A supported: decision models converge with competence |
| fail | rising PASS (r>0, p≤0.05) | convergence exists in the competence-rising phase but reverses under overtraining (non-monotone) — Claim A not supported as a sustained trend |
| fail | rising fail | **no convergence even where competence genuinely rises** — clean non-convergence |
| fail | rising n<3 | inconclusive: competence axis still too flat to test |

Reality-vs-value (separate axis, reported for every outcome): `reality` if mean dCor(enc,Z) >
mean dCor(enc,V_emb), else `value`; valid only if the environment-validity gate passes.

## Status of the prior run

The powered 100k-schedule run (no dCor, quantized 1-D value, chance-pinned JEPA competence) is
archived under `runs/archive/` for comparison. It returned `claim_A_supported: false` with the
decision objectives diverging over training; see `FINDINGS.md`. This re-run re-tests the same
frozen rule with the fixed instruments and denser competence coverage.
