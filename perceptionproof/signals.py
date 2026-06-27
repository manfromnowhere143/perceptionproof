"""The four label-free signals. Each function is the exact equation in
docs/MATHEMATICS.md sec 2 — that document is the spec, this is its implementation.

All signals: input is per-segment model outputs; output is a non-negative float
where larger = more predicted risk/uncertainty. No RFS label is ever read here.

Status: S1 implemented + tested (pure CPU). S2-S4 are gated stubs implemented at
P3 once real multi-frame / occupancy / VLA outputs are wired (they need real model
structure to test meaningfully), but their math is fully specified in MATHEMATICS.md.
"""

from __future__ import annotations

import numpy as np

from .types import ModelOutput, TrajectoryMode


def trajectory_distance(a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> float:
    """d(tau, tau') — horizon-discounted mean L2 over waypoints (MATHEMATICS sec 1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"trajectory shape mismatch: {a.shape} vs {b.shape}")
    t = a.shape[0]
    w = gamma ** np.arange(1, t + 1)
    step = np.linalg.norm(a - b, axis=1)
    return float((w * step).sum() / w.sum())


def _rbf(a: np.ndarray, b: np.ndarray, sigma: float, gamma: float) -> float:
    """Trajectory RBF kernel kappa(tau,tau') = exp(-d^2 / 2 sigma^2)."""
    d = trajectory_distance(a, b, gamma)
    return float(np.exp(-(d * d) / (2.0 * sigma * sigma)))


def _representative_trajectory(output: ModelOutput) -> np.ndarray:
    """Weighted mean over a model's trajectory modes (single mode -> itself)."""
    if not output.trajectory_modes:
        raise ValueError(f"model {output.model_id} has no trajectory modes")
    weights = np.array([m.weight for m in output.trajectory_modes], dtype=float)
    weights = weights / weights.sum()
    return sum(w * np.asarray(m.waypoints, dtype=float) for w, m in zip(weights, output.trajectory_modes))


def _se2_into_prev(traj_next: np.ndarray, dtheta: float, dx: float, dy: float) -> np.ndarray:
    """Map points expressed in ego frame (k+1) into ego frame k via the SE(2) transform
    p_k = R(dtheta) p_{k+1} + (dx, dy)."""
    c, s = np.cos(dtheta), np.sin(dtheta)
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(traj_next, dtype=float) @ rot.T + np.array([dx, dy])


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """H(p) = -p log p - (1-p) log(1-p), nats, numerically safe at 0/1."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def _weighted_mmd2(
    p: list[TrajectoryMode], q: list[TrajectoryMode], sigma: float, gamma: float
) -> float:
    """Squared MMD between two weighted trajectory-mode sets (MATHEMATICS sec 2.1)."""
    xp = [m.waypoints for m in p]
    xq = [m.waypoints for m in q]
    wp = np.array([m.weight for m in p], dtype=float)
    wq = np.array([m.weight for m in q], dtype=float)
    wp = wp / wp.sum()
    wq = wq / wq.sum()

    pp = sum(wp[i] * wp[j] * _rbf(xp[i], xp[j], sigma, gamma) for i in range(len(xp)) for j in range(len(xp)))
    qq = sum(wq[i] * wq[j] * _rbf(xq[i], xq[j], sigma, gamma) for i in range(len(xq)) for j in range(len(xq)))
    pq = sum(wp[i] * wq[j] * _rbf(xp[i], xq[j], sigma, gamma) for i in range(len(xp)) for j in range(len(xq)))
    return float(pp - 2.0 * pq + qq)


def s1_ensemble_disagreement(outputs: list[ModelOutput], sigma: float, gamma: float = 1.0) -> float:
    """S1 — mean pairwise MMD^2 between models' trajectory-mode sets (MATHEMATICS sec 2.1).

    Multimodality-aware via the RBF/MMD kernel; reduces to mean pairwise distance when
    each model is unimodal. Returns 0.0 when fewer than two models predicted.
    """
    mode_sets = [o.trajectory_modes for o in outputs if o.trajectory_modes]
    m = len(mode_sets)
    if m < 2:
        return 0.0
    vals = [
        _weighted_mmd2(mode_sets[i], mode_sets[j], sigma, gamma)
        for i in range(m)
        for j in range(i + 1, m)
    ]
    return float(np.mean(vals))


def s2_temporal_inconsistency(
    per_frame_outputs: list[list[ModelOutput]],
    ego_motions: list[tuple[float, float, float]],
    gamma: float = 1.0,
) -> float:
    """S2 — forecast flicker between forecasts at k and k+1, SE(2)-aligned (MATHEMATICS sec 2.2).

    per_frame_outputs[k] is the list of M model outputs at scene-time k. ego_motions[k]
    = (dtheta, dx, dy) maps ego frame k+1 into frame k. A temporally stable model's
    forecast at k+1, advanced one step and ego-aligned, matches its forecast at k on the
    overlapping horizon; the residual is the flicker. Returns 0.0 with fewer than 2 frames.
    """
    L = len(per_frame_outputs)
    if L < 2:
        return 0.0
    if len(ego_motions) < L - 1:
        raise ValueError("need an ego motion for each consecutive frame pair")
    residuals: list[float] = []
    for k in range(L - 1):
        dtheta, dx, dy = ego_motions[k]
        outs_k, outs_k1 = per_frame_outputs[k], per_frame_outputs[k + 1]
        for o_k, o_k1 in zip(outs_k, outs_k1):
            tk = _representative_trajectory(o_k)
            tk1 = _representative_trajectory(o_k1)
            overlap_k = tk[1:]  # horizons 2..T at frame k
            overlap_k1 = _se2_into_prev(tk1[:-1], dtheta, dx, dy)  # horizons 1..T-1 at k+1, into frame k
            residuals.append(trajectory_distance(overlap_k, overlap_k1, gamma))
    return float(np.mean(residuals)) if residuals else 0.0


def s3_occupancy_conflict(outputs: list[ModelOutput], theta_occ: float, alpha: float = 0.5) -> float:
    """S3 — corridor occupancy entropy + planner-vs-occupancy conflict (MATHEMATICS sec 2.3).

    Restricts to the ego-corridor voxels. The conflict term fires where occupancy says
    'risk' (p > theta_occ) yet the planner is confident the corridor is clear — the
    'scene pretending to be safe' / hidden-actor case. Needs one occupancy-bearing output.
    """
    occ = next((o.occupancy for o in outputs if o.occupancy is not None), None)
    if occ is None:
        raise ValueError("S3 requires a model output carrying an occupancy field")
    mask = np.asarray(occ.corridor_mask, dtype=bool)
    corridor_p = np.asarray(occ.prob, dtype=float)[mask]
    if corridor_p.size == 0:
        return 0.0
    entropy = float(_binary_entropy(corridor_p).mean())
    free_conf = next((o.free_space_confidence for o in outputs if o.free_space_confidence is not None), 0.0)
    conflict = float(((corridor_p > theta_occ).astype(float) * free_conf).mean())
    return alpha * entropy + (1.0 - alpha) * conflict


def s4_semantic_entropy(outputs: list[ModelOutput], cluster_eps: float = 1.0, gamma: float = 1.0) -> float:
    """S4 — semantic entropy over K VLA reasoning rollouts (Kuhn et al. 2023; MATHEMATICS sec 2.4).

    Clusters the K sampled rollout trajectories by semantic equivalence (greedy, distance
    <= cluster_eps = same maneuver), then returns the entropy (nats) of the cluster-mass
    distribution. High entropy = the model's reasoning disagrees with itself. Needs one
    output carrying reasoning_rollouts; returns 0.0 with <= 1 rollout.
    """
    rollouts = next((o.reasoning_rollouts for o in outputs if o.reasoning_rollouts), [])
    if len(rollouts) <= 1:
        return 0.0
    clusters: list[list[np.ndarray]] = []
    for r in rollouts:
        wp = np.asarray(r.waypoints, dtype=float)
        for cluster in clusters:
            if trajectory_distance(wp, cluster[0], gamma) <= cluster_eps:
                cluster.append(wp)
                break
        else:
            clusters.append([wp])
    k = len(rollouts)
    pis = np.array([len(c) / k for c in clusters], dtype=float)
    return float(-(pis * np.log(pis)).sum())


SIGNALS = {
    "g1": s1_ensemble_disagreement,
    "g2": s2_temporal_inconsistency,
    "g3": s3_occupancy_conflict,
    "g4": s4_semantic_entropy,
}
