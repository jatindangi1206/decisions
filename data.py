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


def geodesic_to_goal(maze, positions, goals):
    """Walls-aware shortest-path length from each position's cell to its goal's cell.

    This is V: two states equally close to the goal in Euclidean terms can be far apart here
    if a wall sits between them — that separation is the whole point of using a walled maze.
    """
    scaling = float(maze.maze_size_scaling)
    caches = {}
    out = np.empty(len(positions), np.float32)
    for i, (p, g) in enumerate(zip(positions, goals)):
        grc = tuple(int(x) for x in maze.cell_xy_to_rowcol(np.asarray(g, np.float64)))
        if grc not in caches:
            caches[grc] = _bfs_from(maze, grc)
        prc = tuple(int(x) for x in maze.cell_xy_to_rowcol(np.asarray(p, np.float64)))
        d = caches[grc].get(prc)
        out[i] = (d * scaling) if d is not None else np.nan
    # unreachable cells (walls / rounding) -> max finite distance, so V stays a usable metric
    if np.isnan(out).any():
        out[np.isnan(out)] = np.nanmax(out) + scaling
    return out


# ---- fixed, stamped probe --------------------------------------------------------------

def build_probe(buf, cfg, run_dir):
    """Sample the probe ONCE and stamp provenance. If it already exists on disk, reuse it
    verbatim — the probe must be identical across every objective, seed, and checkpoint."""
    npz = os.path.join(run_dir, "probe.npz")
    meta_path = os.path.join(run_dir, "probe_provenance.json")
    if os.path.exists(npz) and os.path.exists(meta_path):
        d = np.load(npz)
        return d["idx"], d["Z"], d["V"], json.load(open(meta_path))

    rng = np.random.default_rng(cfg["probe"]["seed"])
    idx = rng.choice(buf.n_states, size=min(cfg["probe"]["size"], buf.n_states), replace=False)
    idx.sort()
    Z = buf.states_phys[idx].astype(np.float32)                       # reality
    V = geodesic_to_goal(buf.maze, Z[:, :2], buf.states_goal[idx])    # value
    stamp = hashlib.sha256(Z.tobytes() + buf.dataset_id.encode()).hexdigest()[:16]
    meta = {
        "dataset_id": buf.dataset_id,
        "minari_version": buf.minari_version,
        "n_episodes_used": None,  # filled by caller if wanted
        "n_states": int(buf.n_states),
        "n_transitions": int(buf.n_trans),
        "probe_seed": cfg["probe"]["seed"],
        "probe_size": int(len(idx)),
        "maze_shape": [len(buf.maze.maze_map), len(buf.maze.maze_map[0])],
        "maze_scaling": float(buf.maze.maze_size_scaling),
        "provenance_sha": stamp,
    }
    os.makedirs(run_dir, exist_ok=True)
    np.savez(npz, idx=idx, Z=Z, V=V)
    json.dump(meta, open(meta_path, "w"), indent=2)
    return idx, Z, V, meta


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
    point = getattr(env.unwrapped, "point_env", env.unwrapped)  # PointMaze wraps inner Point env
    frames = np.empty((len(idx), 3, size, size), np.uint8)
    for j, i in enumerate(idx):
        x, y, vx, vy = states_phys[i]
        point.set_state(np.array([x, y], np.float64), np.array([vx, vy], np.float64))
        frames[j] = preprocess_frame(env.render(), size)
    return frames
