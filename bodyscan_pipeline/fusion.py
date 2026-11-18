"""Multi-view shape/scale fusion, canonical A-pose reconstruction, and arm masking."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import weakref

import numpy as np
import torch

from .config import (
    ARM_JOINT_IDS,
    ARM_WEIGHT_THRESHOLD,
    ELBOW_CANDIDATES,
    MIN_OUTLIER_DIST_THRESHOLD,
    OUTLIER_MEDIAN_MULTIPLIER,
)


@dataclass
class FusedSubjectParams:
    """Aggregated shape/scale parameters and per-view outlier weights."""

    shape_params: torch.Tensor  # (1, 45)
    scale_params: torch.Tensor  # (1, 28)
    weights: np.ndarray  # (N,)
    distances: np.ndarray  # (N,)
    images_used: int
    images_total: int
    mean_confidence: float
    expr_params: Optional[torch.Tensor] = None  # (1, 72)


def _extract_valid_expr(raw_expr: Any) -> Optional[np.ndarray]:
    """Convert an optional expression vector to a finite 1D float64 array of length 72."""
    if raw_expr is None:
        return None
    if isinstance(raw_expr, torch.Tensor):
        arr = raw_expr.detach().cpu().numpy()
    else:
        arr = np.asarray(raw_expr)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0 or arr.size > 72 or not np.all(np.isfinite(arr)):
        return None
    out = np.zeros(72, dtype=np.float64)
    out[: arr.size] = arr
    return out


def fuse_multiview_params(raw_results: List[Dict[str, Any]]) -> FusedSubjectParams:
    """Fuse multi-view shape, scale, and expression parameters with median outlier rejection."""
    if not raw_results:
        raise ValueError("No valid person detections found across input images.")

    shapes = np.stack([np.asarray(r["output"]["shape_params"], dtype=np.float64).reshape(-1) for r in raw_results])
    scales = np.stack([np.asarray(r["output"]["scale_params"], dtype=np.float64).reshape(-1) for r in raw_results])

    med_shape = np.median(shapes, axis=0)
    med_scale = np.median(scales, axis=0)
    shape_dists = np.linalg.norm(shapes - med_shape, axis=1)
    scale_dists = np.linalg.norm(scales - med_scale, axis=1)
    # Combine shape and scale deviations so scale-only outliers are also rejected cleanly.
    dists = np.hypot(shape_dists, scale_dists)

    med_dist = float(np.median(dists)) + 1e-6
    cutoff = max(OUTLIER_MEDIAN_MULTIPLIER * med_dist, MIN_OUTLIER_DIST_THRESHOLD)

    weights = np.where(dists <= cutoff, 1.0 / (1.0 + dists), 0.0)
    if weights.sum() == 0:
        weights = np.ones(len(shapes), dtype=np.float64)
    weights = weights / weights.sum()

    used = int((weights > 0).sum())
    mean_conf = float(np.mean(weights[weights > 0])) if used > 0 else 0.0

    fused_shape = torch.tensor((shapes * weights[:, None]).sum(axis=0), dtype=torch.float32)[None]
    fused_scale = torch.tensor((scales * weights[:, None]).sum(axis=0), dtype=torch.float32)[None]

    expr_vecs = [_extract_valid_expr(r["output"].get("expr_params")) for r in raw_results]
    valid_expr_idx = [i for i, e in enumerate(expr_vecs) if e is not None and weights[i] > 0.0]
    if not valid_expr_idx:
        valid_expr_idx = [i for i, e in enumerate(expr_vecs) if e is not None]
    fused_expr: Optional[torch.Tensor] = None
    if valid_expr_idx:
        expr_stack = np.stack([expr_vecs[i] for i in valid_expr_idx])  # type: ignore[arg-type]
        expr_w = np.array([weights[i] for i in valid_expr_idx], dtype=np.float64)
        if expr_w.sum() <= 0.0:
            expr_w = np.ones(len(valid_expr_idx), dtype=np.float64)
        expr_w = expr_w / expr_w.sum()
        fused_expr = torch.tensor((expr_stack * expr_w[:, None]).sum(axis=0), dtype=torch.float32)[None]

    return FusedSubjectParams(
        shape_params=fused_shape,
        scale_params=fused_scale,
        weights=weights,
        distances=dists,
        images_used=used,
        images_total=len(raw_results),
        mean_confidence=mean_conf,
        expr_params=fused_expr,
    )


def _normalize_expr_tensor(
    expr_t: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize optional expression tensor to shape (1, 72) on target device/dtype."""
    if expr_t is None:
        return torch.zeros(1, 72, device=device, dtype=dtype)
    expr_dev = expr_t.to(device=device, dtype=dtype)
    if expr_dev.ndim == 1:
        expr_dev = expr_dev.unsqueeze(0)
    return expr_dev


def evaluate_mhr_pose(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    shape_t: torch.Tensor,
    scale_t: torch.Tensor,
    body_pose_t: torch.Tensor,
    expr_t: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run MHR forward kinematics in canonical pose (+Y up, +Z front, meters)."""
    dev = head_assets["scale_mean"].device
    with torch.no_grad():
        scale_vec = head_assets["scale_mean"][None, :] + scale_t.to(dev) @ head_assets["scale_comps"]
        full_pose = torch.cat([torch.zeros(1, 6, device=dev), body_pose_t[:, :130].to(dev)], dim=1)
        model_params = torch.cat([full_pose, scale_vec], dim=1)
        expr_vec = _normalize_expr_tensor(expr_t, dev, scale_vec.dtype)
        verts, skel = mhr_model(shape_t.to(dev), model_params, expr_vec)
    return (verts[0] / 100.0).cpu().numpy(), (skel[0, :, :3] / 100.0).cpu().numpy()


_APOSE_CACHE_MAX_ENTRIES = 16
_APOSE_CACHE: "weakref.WeakKeyDictionary[Any, OrderedDict[Tuple[Any, ...], torch.Tensor]]" = (
    weakref.WeakKeyDictionary()
)


def _tensor_fingerprint(tensor: Optional[torch.Tensor]) -> Optional[Tuple[Tuple[int, ...], str, bytes]]:
    """Compact deterministic fingerprint for caching MHR parameter tensors."""
    if tensor is None:
        return None
    t_cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.blake2b(t_cpu.numpy().tobytes(), digest_size=16).digest()
    return (tuple(t_cpu.shape), str(t_cpu.dtype), digest)


def _apose_cache_key(
    head_assets: Dict[str, torch.Tensor],
    shape_t: torch.Tensor,
    scale_t: torch.Tensor,
    expr_t: Optional[torch.Tensor],
) -> Tuple[Any, ...]:
    dev = head_assets["scale_mean"].device
    dtype = head_assets["scale_mean"].dtype
    norm_expr = _normalize_expr_tensor(expr_t, torch.device("cpu"), dtype)
    return (
        str(dev),
        str(dtype),
        _tensor_fingerprint(head_assets.get("scale_mean")),
        _tensor_fingerprint(head_assets.get("scale_comps")),
        _tensor_fingerprint(shape_t.to(dtype=dtype)),
        _tensor_fingerprint(scale_t.to(dtype=dtype)),
        _tensor_fingerprint(norm_expr),
    )


def solve_straight_apose(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    shape_t: torch.Tensor,
    scale_t: torch.Tensor,
    expr_t: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Find elbow angles that straighten MHR's default 45-degree bent elbows into an A-pose."""
    cache_key: Optional[Tuple[Any, ...]] = None
    model_cache: Optional["OrderedDict[Tuple[Any, ...], torch.Tensor]"] = None
    if use_cache:
        try:
            cache_key = _apose_cache_key(head_assets, shape_t, scale_t, expr_t)
            model_cache = _APOSE_CACHE.get(mhr_model)
            if model_cache is not None and cache_key in model_cache:
                model_cache.move_to_end(cache_key)
                return model_cache[cache_key].clone()
        except TypeError:
            # Unweakrefable mhr_model mock/proxy; compute without caching.
            cache_key = None

    best_pose = torch.zeros(1, 130)

    def _eval_arm_angle(pose_cand: torch.Tensor, j_sh: int, j_el: int, j_wr: int) -> float:
        _, joints = evaluate_mhr_pose(
            mhr_model, head_assets, shape_t, scale_t, pose_cand, expr_t=expr_t
        )
        v1 = joints[j_el] - joints[j_sh]
        v2 = joints[j_wr] - joints[j_el]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))

    for _, elbow_cand, j_sh, j_el, j_wr in ELBOW_CANDIDATES:
        best_idx = elbow_cand[0]
        best_val = 0.0
        best_ang = _eval_arm_angle(best_pose, j_sh, j_el, j_wr)
        if best_ang >= 1.5:
            # Coarse 7-point scan (0.0 already evaluated above, check remaining 6 points).
            for idx in elbow_cand:
                for val in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
                    candidate = best_pose.clone()
                    candidate[0, idx] = float(val)
                    ang = _eval_arm_angle(candidate, j_sh, j_el, j_wr)
                    if ang < best_ang:
                        best_ang, best_idx, best_val = ang, idx, float(val)
            # Two-stage local refinement around winning coordinate (step 0.15 then 0.04 rad).
            tested = {0.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5}
            for step in (0.15, 0.04):
                center = best_val
                for delta in (-2.0 * step, -step, step, 2.0 * step):
                    val = round(float(np.clip(center + delta, -1.5, 1.5)), 4)
                    if val in tested:
                        continue
                    tested.add(val)
                    candidate = best_pose.clone()
                    candidate[0, best_idx] = val
                    ang = _eval_arm_angle(candidate, j_sh, j_el, j_wr)
                    if ang < best_ang:
                        best_ang, best_val = ang, val
        best_pose[0, best_idx] = best_val

    if use_cache and cache_key is not None:
        try:
            if model_cache is None:
                model_cache = OrderedDict()
                _APOSE_CACHE[mhr_model] = model_cache
            model_cache[cache_key] = best_pose.detach().cpu().clone()
            model_cache.move_to_end(cache_key)
            while len(model_cache) > _APOSE_CACHE_MAX_ENTRIES:
                model_cache.popitem(last=False)
        except TypeError:
            pass
    return best_pose


def build_arm_vertex_mask(mhr_model: Any, num_verts: int) -> np.ndarray:
    """Return boolean mask keeping non-arm vertices (torso, head, legs) via MHR skinning weights."""
    char = getattr(mhr_model, "character_torch", None)
    lbs = getattr(char, "linear_blend_skinning", None) if char is not None else None
    if lbs is None or not hasattr(lbs, "skin_indices_flattened"):
        return np.ones(num_verts, dtype=bool)
    skin_idx = lbs.skin_indices_flattened.cpu().numpy().astype(np.int64, copy=False)
    skin_w = lbs.skin_weights_flattened.cpu().numpy().astype(np.float32, copy=False)
    vert_idx = lbs.vert_indices_flattened.cpu().numpy().astype(np.int64, copy=False)

    arm_weights = np.zeros(num_verts, dtype=np.float32)
    is_arm = np.isin(skin_idx, list(ARM_JOINT_IDS))
    if np.any(is_arm):
        valid_v = vert_idx[is_arm]
        valid_w = skin_w[is_arm]
        in_bounds = (valid_v >= 0) & (valid_v < num_verts)
        np.add.at(arm_weights, valid_v[in_bounds], valid_w[in_bounds])
    return arm_weights < ARM_WEIGHT_THRESHOLD


def build_scaled_apose_mesh(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    fused: FusedSubjectParams,
    height_cm: float,
    apose_params: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reconstruct canonical A-pose mesh, ground feet to y=0, and scale to height_cm."""
    if height_cm <= 0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")

    if apose_params is None:
        apose_params = solve_straight_apose(
            mhr_model, head_assets, fused.shape_params, fused.scale_params, expr_t=fused.expr_params
        )
    verts, joints = evaluate_mhr_pose(
        mhr_model,
        head_assets,
        fused.shape_params,
        fused.scale_params,
        apose_params,
        expr_t=fused.expr_params,
    )

    min_y, max_y = float(verts[:, 1].min()), float(verts[:, 1].max())
    raw_height_m = max_y - min_y
    if raw_height_m <= 0:
        raise RuntimeError("Degenerate mesh height encountered.")

    scale_factor = (height_cm / 100.0) / raw_height_m
    verts[:, 1] -= min_y
    verts *= scale_factor
    joints[:, 1] -= min_y
    joints *= scale_factor

    keep_torso_verts = build_arm_vertex_mask(mhr_model, len(verts))
    return verts, joints, keep_torso_verts, raw_height_m * 100.0
