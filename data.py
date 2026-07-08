"""Real data only, hard failure. Minari PointMaze (interior walls) -> offline buffer,
a fixed stamped probe set, and the two ground-truth anchors:

    Z = true physical state  [x, y, vx, vy]           (reality geometry)
    V = geodesic, walls-aware shortest-path-to-goal    (value / reachability geometry)

Everything downstream is index-based: states live once in `states_phys`; transitions and the
probe reference them by index, so "state" and "pixels" modes differ only at the input layer.
"""
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

# Action discretization for the value-based objective (CQL). PointMaze action is a 2-D force
# in [-1, 1]; a 3x3 grid -> 9 discrete actions keeps CQL a clean argmax objective on the shared
# encoder without dragging in a continuous actor-critic. ponytail: 3x3 grid; widen if CQL underfits.
DISC_GRID = 3
_DISC_VALS = np.linspace(-1.0, 1.0, DISC_GRID)

PIX_SIZE = 64  # rendered image side for "pixels" mode; keep in sync with the CNN in models.py


def cont_to_idx(a):
    ij = np.abs(a[..., None] - _DISC_VALS).argmin(-1)  # nearest grid value per dim
    return ij[..., 0] * DISC_GRID + ij[..., 1]


def idx_to_cont(i):
    return np.stack([_DISC_VALS[i // DISC_GRID], _DISC_VALS[i % DISC_GRID]], -1)


@dataclass
class Buffer:
    states_phys: np.ndarray   # (M, 4)  [x, y, vx, vy]  -> anchor Z
    states_goal: np.ndarray   # (M, 2)  desired goal per state
    i_obs: np.ndarray         # (T,) index into states_phys
    i_next: np.ndarray        # (T,)
    act: np.ndarray           # (T, 2) continuous
    a_idx: np.ndarray         # (T,) discretized action
    rew: np.ndarray           # (T,)
    done: np.ndarray          # (T,)
    rtg: np.ndarray           # (T,) discounted return-to-go
    ep_end: np.ndarray        # (T,) exclusive transition-index end of each transition's episode
    dataset_id: str
    maze: object              # gymnasium_robotics Maze (for geodesic V + rendering)
    minari_version: str

    @property
    def n_states(self):
        return len(self.states_phys)

    @property
    def n_trans(self):
        return len(self.i_obs)


def resolve_and_load(cfg):
    import minari
    last = None
    for did in cfg["dataset"]["candidates"]:
        try:
            try:
                return did, minari.load_dataset(did, download=True), minari.__version__
            except TypeError:  # older minari: separate download step
                minari.download_dataset(did)
                return did, minari.load_dataset(did), minari.__version__
        except Exception as e:  # noqa: BLE001 — try next candidate, remember the reason
            last = f"{did}: {e}"
    raise RuntimeError(
        "REAL DATA REQUIRED — no PointMaze candidate resolved. Last error: " + str(last)
    )


def load_buffer(cfg):
    dataset_id, ds, ver = resolve_and_load(cfg)
    gamma = cfg["train"]["gamma"]
    max_ep = cfg["dataset"]["max_episodes"]

    sp, sg = [], []           # per-state physical + goal
    io, ino, ac, rw, dn, rt = [], [], [], [], [], []
    ee = []                   # per-transition episode-end (exclusive), for multi-step targets
    base = 0
    for k, ep in enumerate(ds.iterate_episodes()):
        if k >= max_ep:
            break
        obs = ep.observations
        o = np.asarray(obs["observation"] if isinstance(obs, dict) else obs, np.float32)
        g = np.asarray(obs["desired_goal"], np.float32) if isinstance(obs, dict) else o[:, :2]
        L = len(ep.rewards)
        if len(o) < L + 1:
            continue
        sp.append(o[: L + 1])
        sg.append(g[: L + 1])
        term = np.asarray(ep.terminations, bool)
        rew = np.asarray(ep.rewards, np.float32)
        # discounted return-to-go, computed backwards within the episode
        rtg = np.zeros(L, np.float32)
        acc = 0.0
        for t in range(L - 1, -1, -1):
            acc = rew[t] + gamma * acc * (0.0 if term[t] else 1.0)
            rtg[t] = acc
        for t in range(L):
            io.append(base + t)
            ino.append(base + t + 1)
        ee.extend([len(io)] * L)  # len(io) now == this episode's exclusive transition end
        ac.append(np.asarray(ep.actions, np.float32)[:L])
        rw.append(rew)
        dn.append(term.astype(np.float32))
        rt.append(rtg)
        base += L + 1

    if not sp:
        raise RuntimeError("REAL DATA REQUIRED — dataset produced zero usable transitions.")

    states_phys = np.concatenate(sp)
    act = np.concatenate(ac)
    buf = Buffer(
        states_phys=states_phys,
        states_goal=np.concatenate(sg),
        i_obs=np.asarray(io, np.int64),
        i_next=np.asarray(ino, np.int64),
        act=act,
        a_idx=cont_to_idx(act).astype(np.int64),
        rew=np.concatenate(rw),
        done=np.concatenate(dn),
        rtg=np.concatenate(rt),
        ep_end=np.asarray(ee, np.int64),
        dataset_id=dataset_id,
        maze=ds.recover_environment().unwrapped.maze,
        minari_version=ver,
    )
    return buf, ds


# ---- ground-truth anchors --------------------------------------------------------------

def _wall(cell):
    return str(cell) == "1"


def _bfs_from(maze, goal_rc):
    """Cell-distance BFS over free cells (4-connected). Returns dict[(r,c)] -> steps."""
    grid = maze.maze_map
    H, W = len(grid), len(grid[0])
    from collections import deque
    dist = {goal_rc: 0}
    q = deque([goal_rc])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not _wall(grid[nr][nc]) and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _free_cells(maze):
    grid = maze.maze_map
    return [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if not _wall(grid[r][c])]


def _continuous_geo(maze, bfs, positions):
    """De-quantized walls-aware geodesic from a BFS source to each continuous position.

    Cell-level BFS gives integer distances (lots of ties -> bad for kNN and distance-correlation).
    We de-quantize by projecting the sub-cell offset onto the local goal-ward gradient of the BFS
    field: moving toward the closer neighbour cell smoothly reduces distance. Result is continuous
    and tie-free, a fair anchor against continuous reality Z.
    """
    s = float(maze.maze_size_scaling)
    out = np.empty(len(positions), np.float64)
    dcache = {}
    for i, p in enumerate(positions):
        p = np.asarray(p, np.float64)
        rc = tuple(int(x) for x in maze.cell_xy_to_rowcol(p))
        d = bfs.get(rc)
        if d is None:
            out[i] = np.nan
            continue
        if rc not in dcache:
            cc = np.asarray(maze.cell_rowcol_to_xy(np.asarray(rc, np.float64)), np.float64)
            grad = np.zeros(2)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dn = bfs.get((rc[0] + dr, rc[1] + dc))
                if dn is not None:
                    vec = np.asarray(maze.cell_rowcol_to_xy(np.asarray((rc[0] + dr, rc[1] + dc),
                                     np.float64)), np.float64) - cc
                    nv = np.linalg.norm(vec)
                    if nv > 0:
                        grad += (d - dn) * vec / nv   # points toward decreasing distance (goal-ward)
            ng = np.linalg.norm(grad)
            dcache[rc] = (cc, grad / ng if ng > 0 else np.zeros(2))
        cc, u = dcache[rc]
        off = float(np.clip(np.dot(p - cc, u), -s / 2, s / 2))
        out[i] = d * s - off
    if np.isnan(out).any():
        out[np.isnan(out)] = np.nanmax(out) + s
    return out.astype(np.float32)


def _farthest_point_landmarks(maze, n, seed):
    """Deterministic farthest-point sampling of free cells (geodesic) -> spread landmarks."""
    free = _free_cells(maze)
    rng = np.random.default_rng(seed)
    chosen = [free[int(rng.integers(len(free)))]]
    dmin = {c: v for c, v in _bfs_from(maze, chosen[0]).items()}
    dmin = {c: dmin.get(c, 1e9) for c in free}
    while len(chosen) < min(n, len(free)):
        nxt = max(free, key=lambda c: dmin[c])
        chosen.append(nxt)
        b = _bfs_from(maze, nxt)
        for c in free:
            dmin[c] = min(dmin[c], b.get(c, 1e9))
    return chosen


def build_value_anchors(maze, positions, goals, n_landmarks, seed):
    """Value / reachability geometry as a fair anchor comparable to reality Z:

      V     = continuous walls-aware geodesic to each state's OWN goal (the scalar "closeness to
              goal" value).
      V_emb = continuous walls-aware geodesic to `n_landmarks` fixed, spread landmarks — a
              multi-dim embedding of the walls-aware reachability manifold, dimensionality-fair
              against Z. This is the anchor the reality-vs-value statistic runs on.
    """
    from collections import defaultdict
    positions = np.asarray(positions, np.float64)
    V = np.empty(len(positions), np.float32)
    groups = defaultdict(list)
    for i, g in enumerate(goals):
        groups[tuple(int(x) for x in maze.cell_xy_to_rowcol(np.asarray(g, np.float64)))].append(i)
    for grc, idxs in groups.items():
        V[idxs] = _continuous_geo(maze, _bfs_from(maze, grc), positions[idxs])
    landmarks = _farthest_point_landmarks(maze, n_landmarks, seed)
    V_emb = np.empty((len(positions), len(landmarks)), np.float32)
    for j, lc in enumerate(landmarks):
        V_emb[:, j] = _continuous_geo(maze, _bfs_from(maze, lc), positions)
    return V, V_emb, [list(c) for c in landmarks]


# ---- fixed, stamped probe --------------------------------------------------------------

def build_probe(buf, cfg, run_dir):
    """Sample the probe ONCE and stamp provenance. If it already exists on disk, reuse it
    verbatim — the probe must be identical across every objective, seed, and checkpoint."""
    from geometry import distance_correlation, kernel_health
    npz = os.path.join(run_dir, "probe.npz")
    meta_path = os.path.join(run_dir, "probe_provenance.json")
    if os.path.exists(npz) and os.path.exists(meta_path):
        d = np.load(npz)
        if "V_emb" in d.files:  # else it's a stale pre-V_emb probe -> regenerate below
            return d["idx"], d["Z"], d["V"], d["V_emb"], json.load(open(meta_path))

    rng = np.random.default_rng(cfg["probe"]["seed"])
    idx = rng.choice(buf.n_states, size=min(cfg["probe"]["size"], buf.n_states), replace=False)
    idx.sort()
    Z = buf.states_phys[idx].astype(np.float32)                          # reality
    n_lm = cfg.get("anchors", {}).get("n_landmarks", 8)
    V, V_emb, landmarks = build_value_anchors(buf.maze, Z[:, :2], buf.states_goal[idx],
                                              n_lm, cfg["probe"]["seed"])  # value + value embedding

    rv_dcor = float(distance_correlation(Z, V_emb))   # reality vs value: must stay well below 0.3
    kh = kernel_health(V_emb)                          # value kernel must be non-degenerate
    stamp = hashlib.sha256(Z.tobytes() + V_emb.tobytes() + buf.dataset_id.encode()).hexdigest()[:16]
    meta = {
        "dataset_id": buf.dataset_id,
        "minari_version": buf.minari_version,
        "n_states": int(buf.n_states),
        "n_transitions": int(buf.n_trans),
        "probe_seed": cfg["probe"]["seed"],
        "probe_size": int(len(idx)),
        "maze_shape": [len(buf.maze.maze_map), len(buf.maze.maze_map[0])],
        "maze_scaling": float(buf.maze.maze_size_scaling),
        "n_landmarks": len(landmarks),
        "landmark_cells": landmarks,
        "reality_value_dcor": rv_dcor,
        "env_reject": bool(rv_dcor >= 0.3),           # Z and V too aligned -> maze can't separate them
        "value_kernel": kh,
        "provenance_sha": stamp,
    }
    if meta["env_reject"]:
        print(f"[probe] ENV-REJECT: reality-value dCor={rv_dcor:.3f} >= 0.3 — Z and V too aligned "
              f"to test reality-vs-value in this maze.")
    if kh["degenerate"]:
        print(f"[probe] WARNING: value kernel degenerate {kh} — V_emb lacks real structure.")
    os.makedirs(run_dir, exist_ok=True)
    np.savez(npz, idx=idx, Z=Z, V=V, V_emb=V_emb)
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[probe] reality-value dCor={rv_dcor:.3f} (env-reject>=0.3: {meta['env_reject']}); "
          f"value kernel PR={kh['participation_ratio']:.2f} degenerate={kh['degenerate']}")
    return idx, Z, V, V_emb, meta


# ---- encoder inputs (the only place the two obs modes differ) --------------------------

def state_vectors(buf, idx):
    """"state" mode input: the ~6 raw numbers [x, y, vx, vy, goal_x, goal_y]."""
    return np.concatenate([buf.states_phys[idx], buf.states_goal[idx]], 1).astype(np.float32)


def preprocess_frame(img, size=64):
    """RGB HxWx3 -> uint8 3xSIZExSIZE: center-crop to square, nearest-resize, channels-first.
    Shared by offline pre-render and live rollout so training and eval see identical pixels."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    ys = np.arange(size) * s // size
    return img[ys][:, ys].transpose(2, 0, 1).astype(np.uint8)


def render_states(ds, states_phys, idx, size=64):
    """"pixels" mode input: render each physical state to an RGB image via MuJoCo.

    This is the primary study's input. Rendering is version-sensitive, so it is isolated here
    and validated by tools/check_render.py before any pixel run. ponytail: renders one state at
    a time; batch/offscreen-pool it only if pixel pre-render becomes the bottleneck.
    """
    env = ds.recover_environment(render_mode="rgb_array")
    env.reset(seed=0)  # satisfy OrderEnforcing + init the renderer; set_state below sets the pose
    u = env.unwrapped
    point = getattr(u, "point_env", u)  # PointMaze wraps an inner Point MuJoCo env
    frames = np.empty((len(idx), 3, size, size), np.uint8)
    for j, i in enumerate(idx):
        x, y, vx, vy = states_phys[i]
        point.set_state(np.array([x, y], np.float64), np.array([vx, vy], np.float64))
        frames[j] = preprocess_frame(u.render(), size)  # unwrapped: skip per-frame wrapper re-gating
    return frames


if __name__ == "__main__":
    # self-check the value anchors on a hand-built walled maze (no MuJoCo).
    class _FakeMaze:
        maze_size_scaling = 1.0

        def __init__(self, grid):
            self.maze_map = grid
            self._xc = len(grid[0]) / 2 * self.maze_size_scaling
            self._yc = len(grid) / 2 * self.maze_size_scaling

        def cell_rowcol_to_xy(self, rc):
            return np.array([(rc[1] + 0.5) * self.maze_size_scaling - self._xc,
                             self._yc - (rc[0] + 0.5) * self.maze_size_scaling])

        def cell_xy_to_rowcol(self, xy):
            return np.array([int(np.floor((self._yc - xy[1]) / self.maze_size_scaling)),
                             int(np.floor((xy[0] + self._xc) / self.maze_size_scaling))])

    # the real PointMaze medium-v2 layout (1=wall) — complex interior walls, a realistic fixture.
    rows = ["########", "#..##..#", "#..#...#", "##...###", "#..#...#", "#.#..#.#", "#...#..#", "########"]
    grid = [[1 if ch == "#" else 0 for ch in row] for row in rows]
    m = _FakeMaze(grid)
    free = _free_cells(m)
    start = free[0]
    goal = m.cell_rowcol_to_xy(start)
    bfs = _bfs_from(m, start)
    pos = np.array([m.cell_rowcol_to_xy(c) for c in free])

    V = _continuous_geo(m, bfs, pos)
    assert V.max() - V.min() > 3, "geodesic value must span a real range in a walled maze"
    # de-quantization: jittered positions yield many more distinct values than integer cell distances
    jit = pos + np.random.default_rng(0).uniform(-0.3, 0.3, pos.shape)
    cell_d = [bfs[tuple(int(x) for x in m.cell_xy_to_rowcol(p))] for p in pos]
    assert len(np.unique(np.round(_continuous_geo(m, bfs, jit), 4))) > len(np.unique(cell_d)), \
        "continuous V must de-quantize (more distinct values than cell distances)"
    # multi-landmark embedding: distinct landmarks, non-degenerate kernel
    _, V_emb, lms = build_value_anchors(m, pos, [goal] * len(free), 6, 0)
    from geometry import kernel_health
    kh = kernel_health(V_emb)
    assert len({tuple(x) for x in lms}) == len(lms), "landmarks must be distinct"
    assert not kh["degenerate"], f"V_emb must be a non-degenerate anchor, got {kh}"
    print(f"data value-anchor self-check OK  (V_emb {V_emb.shape}, {len(lms)} landmarks, kernel PR "
          f"{kh['participation_ratio']:.2f})")
