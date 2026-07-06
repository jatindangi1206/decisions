# PRH-RL — the Platonic Representation Hypothesis in the decision/RL domain

**The question is a TREND, not a comparison.** As a decision model gets more *competent*, do
independently-trained versions of it become more *similar inside*? The deliverable per objective
is a **curve** — representational similarity (y) vs measured competence (x) — with a trend-test
verdict, not a one-shot side-by-side of two models.

And because we work in a simulator, we have something the original PRH work never had: the **true
world state**. So we can ask the harder question underneath convergence —

> Do better decision models converge toward **REALITY** (physical state) or toward **VALUE**
> (closeness to the goal)?

That second question is the point of the project.

---

## Why a walled maze

We use **Minari PointMaze with interior walls** (`medium`/`large`, never `umaze`). Walls are not
decoration: they make the **geodesic** (walls-aware) shortest-path distance-to-goal diverge from
straight-line distance. Two states can be *far apart in reality* yet *equally close to the goal in
value* — separated by a wall but the same number of steps away. Those pairs are what let
"reality geometry" (**Z**) and "value geometry" (**V**) genuinely differ. With no walls the two
collapse into each other and the central question becomes untestable.

- **Z (reality)** = true physical state `[x, y, vx, vy]`.
- **V (value)** = geodesic, walls-aware shortest-path length from a state's cell to its goal's cell
  (BFS on the free-cell grid). `data.geodesic_to_goal`.

`tools/check_render.py` prints the count of far-in-Z / near-in-V pairs — if it's zero, the maze or
goal is wrong and the study is pointless, so check it first.

## Why pixels (and why "state" is only a plumbing check)

With the ~6 raw numbers, the input already *is* the answer: an untrained encoder's seeds already
agree at ~0.87, so competence has no room to reshape anything. **Pixels force the model to choose
what matters.** So:

- `obs_type: state` → raw 6-vector + MLP encoder. Run **once** as a fast plumbing check.
- `obs_type: pixels` → MuJoCo RGB render + small CNN encoder. This is **the real study.**

Same latent dim, same shared-encoder rule, same probe, same anchors for both. The two modes differ
in exactly one function (`Ctx.make_input`).

## The disciplines (carried over from the pilot)

1. **ONE shared encoder** across all objectives — same architecture, each (objective, seed) trains
   its own copy. Objective-specific parts live in **heads** (`models.py`, `objectives.py`). This is
   what lets us attribute representational change to the *objective* and to *competence*, not the net.
2. **ONE fixed probe set**, sampled once and **stamped with data provenance** (dataset id, minari
   version, maze shape/scaling, probe seed, SHA of the probe). Every objective/seed/checkpoint is
   measured on the identical probe. `data.build_probe` → `runs/.../probe_provenance.json`.
3. **Nulls**: a **random-init** null (untrained step-0 encoder) and a **Gaussian** null (chance-level
   mutual-kNN floor). Real convergence must clear these.
4. **Real data only, hard failure.** If Minari can't resolve a walled PointMaze dataset, the code
   **raises** — there is **no synthetic data and no fake/fallback path anywhere**. The only random
   arrays in the repo are the Gaussian *null baseline* and the geometry self-test.
5. **Success rule pre-registered in `config.yaml` BEFORE running** — see below. Do not edit it after
   seeing results.

## Competence = how good it is, not how big

Competence is the **x-axis**, measured at every checkpoint, keyed by `(objective, seed, step)`:

- **Policies (BC, CQL)** → roll out ~20 episodes in the env; competence = mean return (success rate
  logged too).
- **Predictors (JEPA, reward)** → held-out prediction error, reported as an R²-like score in [0,1].

Each objective is trained long and checkpointed across a schedule (`train.schedule`), so it becomes
a sequence of encoders weak→strong. (Width can be a second axis later; checkpoints only for now.)

## Geometry (every checkpoint)

Freeze encoder, extract on the fixed probe:

- **mutual-kNN** — *primary, local, the headline.* Between-seed agreement = the y-axis.
- **CKA** — secondary, global. Logged, but **not** what the verdict is built on: global CKA
  convergence tends to wash out under calibration (the Aristotelian critique).
- **participation ratio** — effective dimensionality.

## The novel part: alignment to Z vs V

For every encoder at every competence rung we measure mutual-kNN alignment to **Z** and to **V**, and
run `geometry.collapse_test`: it finds probe pairs far apart in Z but close in V and reports the
encoder's distance on them vs the mirror set. If, as competence rises, the encoder collapses the
far-in-reality / near-in-value pairs, its neighbor structure is drifting toward **value**.

---

## Pre-registered success rule

From `config.yaml` (frozen before running):

> **Claim A** is supported if between-seed mutual-kNN **rises with competence**
> (**Spearman r > 0 AND permutation p ≤ 0.05**) for the **decision objectives** (BC, CQL).

`analyze.py` computes Spearman(competence, between-seed similarity) across checkpoints, a permutation
test (shuffle competence order) for p, writes `runs/.../verdict.json`, and prints a verdict table.

---

## Run it

```bash
pip install -r requirements.txt

# Headless Linux GPU box: MuJoCo needs an offscreen GL backend for pixel rendering.
export MUJOCO_GL=egl          # or: osmesa

# 0) validate the pixels path + the Z/V anchors before any long run
python tools/check_render.py

# 1) fast plumbing check on raw state (should train + produce curves quickly)
python run.py --obs-type state

# 2) the real study on pixels
python run.py --obs-type pixels

# stages are resumable (useful for long runs):
python run.py --obs-type pixels --stage train
python run.py --obs-type pixels --stage geometry
python run.py --obs-type pixels --stage analyze
```

Set `train.device: auto` (default) to use CUDA when present. For **paper scale**, widen
`train.schedule` to `[1000, 2000, 5000, 10000, 20000, 50000, 100000]` in `config.yaml`.

**Outputs** (`runs/<name>_<obs_type>/`): `probe_provenance.json`, `competence.jsonl`,
`geometry.jsonl`, `verdict.json`, and per-objective `plots/<obj>.png` (similarity-vs-competence with
the null line + trend test, and the reality-vs-value alignment curves).

### Environment notes
- **MuJoCo needs a native-arch Python** (arm64 on Apple Silicon; x86_64 fails). A Linux x86_64 GPU
  box is exactly right. Headless rendering needs `MUJOCO_GL=egl` (GPU) or `osmesa` (CPU).
- Tested with the pinned versions in `requirements.txt` (minari 0.5.3, gymnasium-robotics 1.4.2,
  mujoco 3.10.0). The env/maze/render API is version-sensitive; `tools/check_render.py` catches drift.

---

## Files

| file | role |
|------|------|
| `config.yaml` | the whole experiment + the **pre-registered success rule** |
| `models.py` | the ONE shared encoder (MLP + CNN) and the head |
| `data.py` | Minari loader (hard-fail), stamped probe, **Z** and geodesic **V**, rendering |
| `objectives.py` | BC / CQL / JEPA / reward — the extension point (add IQL = one class) |
| `train.py` | generic ladder training + checkpointing + competence logging |
| `geometry.py` | mutual-kNN, CKA, PR, Z/V alignment, nulls, collapse test (self-checks in `__main__`) |
| `analyze.py` | trend test, verdict, curves |
| `run.py` | orchestrator (`train` → `geometry` → `analyze`, resumable) |
| `tools/check_render.py` | validate pixels + anchors before a long run |

## Roadmap (documented, not built)

- **Phase 1 (this build):** do decision models converge as competence rises, and toward reality or
  value? BC, CQL, JEPA, reward.
- **Phase 2 — cross-algorithm convergence:** add IQL / world-model / IRL. Extension point: write one
  class in `objectives.py` with the same interface and list it in `config.objectives`. Nothing else
  changes.
- **Phase 3 (Claim B):** compare large decision models to GPT / CLIP / DINO / MAE. The probe +
  mutual-kNN + anchor machinery already generalizes to externally-supplied embedding matrices.
