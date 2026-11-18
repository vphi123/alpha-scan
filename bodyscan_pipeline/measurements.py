"""3D mesh slicing, tape-measure convex hull circumferences, loft profile, and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
try:
    import trimesh
except ImportError:
    trimesh = None  # type: ignore[assignment]
from scipy.spatial import ConvexHull, QhullError

from .config import (
    CHEST_FRAC_RANGE,
    HIP_FRAC_RANGE,
    LIMB_SPECS,
    LOFT_PCT_MAX,
    LOFT_PCT_MIN,
    NECK_BASE_JOINT,
    NECK_HEAD_JOINT,
    NECK_T_END,
    NECK_T_START,
    NECK_T_STEPS,
    WAIST_FEMALE_FRAC_RANGE,
    WAIST_MALE_NAVEL_FRAC,
)

UP_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)

LIMB_MAX_RADIUS_M: Dict[str, float] = {
    "bicep_l": 0.24,
    "bicep_r": 0.24,
    "forearm_l": 0.18,
    "forearm_r": 0.18,
    "wrist_l": 0.14,
    "wrist_r": 0.14,
    "thigh_l": 0.24,
    "thigh_r": 0.24,
    "calf_l": 0.18,
    "calf_r": 0.18,
    "ankle_l": 0.14,
    "ankle_r": 0.14,
}


@dataclass
class SimpleTriMesh:
    """Lightweight pure-NumPy triangle mesh container used when trimesh is unavailable."""

    vertices: np.ndarray
    faces: np.ndarray


def _extract_verts_faces(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (vertices, faces) arrays from a Trimesh, SimpleTriMesh, tuple, or dict."""
    if isinstance(mesh, (tuple, list)) and len(mesh) >= 2:
        return np.asarray(mesh[0], dtype=np.float64), np.asarray(mesh[1], dtype=np.int64)
    if isinstance(mesh, dict):
        v = mesh.get("vertices") if "vertices" in mesh else mesh.get("verts")
        f = mesh.get("faces")
        if v is not None and f is not None:
            return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    v_attr = getattr(mesh, "vertices", None)
    if v_attr is None:
        v_attr = getattr(mesh, "verts", None)
    f_attr = getattr(mesh, "faces", None)
    if v_attr is not None and f_attr is not None:
        return np.asarray(v_attr, dtype=np.float64), np.asarray(f_attr, dtype=np.int64)
    raise TypeError("mesh must provide vertices and faces (e.g., trimesh.Trimesh, SimpleTriMesh, or (verts, faces)).")


def _slice_mesh_plane_numpy(
    verts: np.ndarray,
    faces: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Intersect a triangle mesh with a plane in pure NumPy and return (M, 2, 3) 3D segments."""
    v_arr = np.asarray(verts, dtype=np.float64)
    f_arr = np.asarray(faces, dtype=np.int64)
    if len(f_arr) == 0 or len(v_arr) == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)
    n_arr = np.asarray(plane_normal, dtype=np.float64)
    n_len = float(np.linalg.norm(n_arr))
    if n_len < 1e-12:
        return np.zeros((0, 2, 3), dtype=np.float64)
    n_unit = n_arr / n_len
    origin = np.asarray(plane_origin, dtype=np.float64)

    d = (v_arr - origin) @ n_unit
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    df = d[f_arr]
    all_pos = np.all(df > 0.0, axis=1)
    all_neg = np.all(df < 0.0, axis=1)
    hit_idx = np.where(~(all_pos | all_neg))[0]
    if len(hit_idx) == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)

    segs: List[np.ndarray] = []
    for fi in hit_idx:
        tri = f_arr[fi]
        pts: List[np.ndarray] = []
        for a_loc, b_loc in ((0, 1), (1, 2), (2, 0)):
            ia, ib = int(tri[a_loc]), int(tri[b_loc])
            da, db = float(d[ia]), float(d[ib])
            if (da > 0.0) != (db > 0.0):
                t = da / (da - db)
                pts.append(v_arr[ia] + t * (v_arr[ib] - v_arr[ia]))
        if len(pts) == 2:
            segs.append(np.stack([pts[0], pts[1]], axis=0))
    return np.asarray(segs, dtype=np.float64) if segs else np.zeros((0, 2, 3), dtype=np.float64)


def _intersect_mesh_plane(
    mesh: Any,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
) -> np.ndarray:
    """Compute mesh-plane intersection segments via trimesh when available or pure-NumPy fallback."""
    if trimesh is not None and isinstance(mesh, trimesh.Trimesh):
        return np.asarray(
            trimesh.intersections.mesh_plane(mesh, plane_normal=plane_normal, plane_origin=plane_origin),
            dtype=np.float64,
        )
    verts, faces = _extract_verts_faces(mesh)
    return _slice_mesh_plane_numpy(verts, faces, plane_origin=plane_origin, plane_normal=plane_normal)


def _infer_limb_max_radius_m(
    limb_name: Optional[str] = None,
    j_start: Optional[int] = None,
    j_end: Optional[int] = None,
    mode: Optional[str] = None,
) -> float:
    """Resolve anatomical limb search radius cap (0.14 wrist/ankle, 0.18 forearm/calf, 0.24 bicep/thigh)."""
    if limb_name:
        key = str(limb_name).strip().lower()
        if key in LIMB_MAX_RADIUS_M:
            return LIMB_MAX_RADIUS_M[key]
        if "wrist" in key or "ankle" in key:
            return 0.14
        if "forearm" in key or "calf" in key:
            return 0.18
        if "bicep" in key or "thigh" in key:
            return 0.24
    for spec in LIMB_SPECS:
        if spec.joint_start == j_start and spec.joint_end == j_end and (mode is None or spec.mode == mode):
            return LIMB_MAX_RADIUS_M.get(spec.name, 0.24)
    if mode == "min":
        return 0.14
    return 0.24


def _polygon_area_2d(pts: np.ndarray) -> float:
    """Compute unsigned 2D polygon area via the shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _order_cycle_points(
    graph: nx.Graph,
    comp: set,
    coord_map: Dict[Tuple[float, ...], np.ndarray],
) -> Tuple[np.ndarray, bool]:
    """Trace an ordered closed 2D ring from a slice component; return (pts, is_closed)."""
    sub = graph.subgraph(comp).copy()
    while True:
        spurs = [n for n, deg in sub.degree() if deg < 2]
        if not spurs:
            break
        sub.remove_nodes_from(spurs)

    if sub.number_of_nodes() >= 3:
        if all(deg == 2 for _, deg in sub.degree()):
            start = min(sub.nodes())
            ordered = [start]
            prev = None
            curr = start
            for _ in range(sub.number_of_nodes() - 1):
                nbrs = [n for n in sub.neighbors(curr) if n != prev]
                if not nbrs:
                    break
                nxt = min(nbrs) if prev is None else nbrs[0]
                ordered.append(nxt)
                prev, curr = curr, nxt
            if len(ordered) >= 3:
                return np.asarray([coord_map[k] for k in ordered], dtype=np.float64), True

        cycles = nx.cycle_basis(sub)
        valid_cycles = [c for c in cycles if len(c) >= 3]
        if valid_cycles:
            best_cycle = max(
                valid_cycles,
                key=lambda c: _polygon_area_2d(np.asarray([coord_map[k] for k in c], dtype=np.float64)),
            )
            return np.asarray([coord_map[k] for k in best_cycle], dtype=np.float64), True

    raw_pts = np.asarray([coord_map[k] for k in sorted(comp)], dtype=np.float64)
    return raw_pts, False


def _smooth_closed_ring(
    pts: np.ndarray,
    iters: int = 4,
    lam: float = 0.5,
    mu: float = -0.52,
) -> np.ndarray:
    """Apply volume-preserving Taubin smoothing on dense closed loops to suppress vertex noise."""
    if len(pts) < 32:
        return pts
    ring = pts.astype(np.float64, copy=True)
    for _ in range(iters):
        avg = 0.5 * (np.roll(ring, 1, axis=0) + np.roll(ring, -1, axis=0))
        ring = ring + lam * (avg - ring)
        avg = 0.5 * (np.roll(ring, 1, axis=0) + np.roll(ring, -1, axis=0))
        ring = ring + mu * (avg - ring)
    return ring


def _hull_fallback_ring(pts: np.ndarray) -> Optional[np.ndarray]:
    """Compute ordered convex hull boundary, returning None on degenerate collinear inputs."""
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    return pts[hull.vertices]


def slice_loop_2d(
    mesh: Any,
    origin: np.ndarray,
    normal: np.ndarray,
    pick: str = "closest",
    max_radius_m: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Slice mesh with a plane and compute the ordered 2D perimeter (cm) of the selected loop."""
    normal_arr = np.asarray(normal, dtype=np.float64)
    norm_len = np.linalg.norm(normal_arr)
    if norm_len < 1e-12:
        return None, 0.0
    normal_arr = normal_arr / norm_len
    origin_arr = np.asarray(origin, dtype=np.float64)

    segs = _intersect_mesh_plane(mesh, plane_normal=normal_arr, plane_origin=origin_arr)
    if len(segs) < 3:
        return None, 0.0

    # Build orthonormal 2D basis (u, v) on the slicing plane.
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal_arr[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, normal_arr)
    u /= np.linalg.norm(u)
    v = np.cross(normal_arr, u)
    v /= np.linalg.norm(v)

    rel = segs - origin_arr
    pts_2d_segs = np.stack([rel @ u, rel @ v], axis=-1)

    graph = nx.Graph()
    coord_map: Dict[Tuple[float, float, ...], np.ndarray] = {}
    for seg in pts_2d_segs:
        k0 = tuple(np.round(seg[0], 4))
        k1 = tuple(np.round(seg[1], 4))
        coord_map[k0] = seg[0]
        coord_map[k1] = seg[1]
        if k0 != k1:
            graph.add_edge(k0, k1)

    min_comp_len = 3 if pick == "closest" else 5
    comps = [c for c in nx.connected_components(graph) if len(c) >= min_comp_len]
    if not comps:
        return None, 0.0

    def _centroid_dist(comp: set) -> float:
        arr = np.asarray([coord_map[k] for k in comp], dtype=np.float64)
        return float(np.linalg.norm(np.mean(arr, axis=0)))

    if max_radius_m is not None and float(max_radius_m) > 0.0:
        comps = [c for c in comps if _centroid_dist(c) <= float(max_radius_m)]
        if not comps:
            return None, 0.0

    if pick == "closest":
        best_comp = min(comps, key=_centroid_dist)
    else:
        best_comp = max(comps, key=len)

    ring, is_closed = _order_cycle_points(graph, best_comp, coord_map)
    if not is_closed:
        hull_pts = _hull_fallback_ring(ring)
        if hull_pts is None or len(hull_pts) < 3:
            return None, 0.0
        ring = hull_pts

    if len(ring) < 3:
        return None, 0.0

    eval_ring = _smooth_closed_ring(ring) if is_closed else ring
    perim_m = float(np.sum(np.linalg.norm(np.roll(eval_ring, -1, axis=0) - eval_ring, axis=1)))
    return ring, perim_m * 100.0


def _best_slice_in_band(
    mesh: Any,
    y_min: float,
    y_max: float,
    steps: int,
    agg: str,
) -> Tuple[Optional[np.ndarray], float]:
    """Scan horizontal planes across [y_min, y_max] and return the min or max loop."""
    candidates = [
        slice_loop_2d(mesh, np.array([0.0, y, 0.0]), UP_AXIS, pick="largest")
        for y in np.linspace(y_min, y_max, steps)
    ]
    valid = [c for c in candidates if c[0] is not None and c[1] > 0.0]
    if not valid:
        return None, 0.0
    return min(valid, key=lambda x: x[1]) if agg == "min" else max(valid, key=lambda x: x[1])


def measure_neck_perpendicular(
    torso_mesh: Any,
    joints: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """Measure neck orthogonal to the cervical spine vector (J110 -> J113) below chin."""
    p_base = joints[NECK_BASE_JOINT]
    p_head = joints[NECK_HEAD_JOINT]
    neck_axis = p_head - p_base

    valid: List[Tuple[Optional[np.ndarray], float]] = []
    for t in np.linspace(NECK_T_START, NECK_T_END, NECK_T_STEPS):
        pts, circ = slice_loop_2d(
            torso_mesh, p_base + t * neck_axis, neck_axis, pick="closest", max_radius_m=0.14
        )
        if pts is not None and circ > 0.0:
            valid.append((pts, circ))

    if not valid:
        return None, 0.0
    # Prefer anatomically plausible neck loops (>15 cm) over tiny stray mesh fragments;
    # fall back to min(valid) if a synthetic test mesh uses sub-human scale.
    plausible = [item for item in valid if item[1] > 15.0]
    return min(plausible if plausible else valid, key=lambda x: x[1])


def measure_limb(
    full_mesh: Any,
    joints: np.ndarray,
    j_start: int,
    j_end: int,
    t_values: Tuple[float, ...],
    mode: str,
    max_radius_m: Optional[float] = None,
    limb_name: Optional[str] = None,
) -> float:
    """Measure a limb segment perpendicular to its bone axis with anatomical radius filtering."""
    p0, p1 = joints[j_start], joints[j_end]
    axis = p1 - p0
    eff_radius = (
        float(max_radius_m)
        if max_radius_m is not None
        else _infer_limb_max_radius_m(limb_name=limb_name, j_start=j_start, j_end=j_end, mode=mode)
    )
    vals: List[float] = []
    for t in t_values:
        _, circ = slice_loop_2d(
            full_mesh,
            p0 + t * axis,
            axis,
            pick="closest",
            max_radius_m=eff_radius,
        )
        if circ > 0.0:
            vals.append(circ)
    if not vals:
        return 0.0
    if mode == "max":
        return float(max(vals))
    if mode == "min":
        return float(min(vals))
    return float(vals[0])


def measure_all_circumferences(
    full_mesh: Any,
    torso_mesh: Any,
    joints: np.ndarray,
    height_m: float,
    sex: str,
) -> Tuple[Dict[str, float], Dict[str, Optional[np.ndarray]]]:
    """Compute ~16 body circumferences (cm) and return torso 2D slice points for QA plotting."""
    neck_pts, neck_cm = measure_neck_perpendicular(torso_mesh, joints)

    chest_pts, chest_cm = _best_slice_in_band(
        torso_mesh, CHEST_FRAC_RANGE[0] * height_m, CHEST_FRAC_RANGE[1] * height_m, CHEST_FRAC_RANGE[2], "max"
    )

    if sex.lower() == "female":
        waist_pts, waist_cm = _best_slice_in_band(
            torso_mesh,
            WAIST_FEMALE_FRAC_RANGE[0] * height_m,
            WAIST_FEMALE_FRAC_RANGE[1] * height_m,
            WAIST_FEMALE_FRAC_RANGE[2],
            "min",
        )
    else:
        waist_pts, waist_cm = slice_loop_2d(
            torso_mesh, np.array([0.0, WAIST_MALE_NAVEL_FRAC * height_m, 0.0]), UP_AXIS, pick="largest"
        )

    hip_pts, hip_cm = _best_slice_in_band(
        torso_mesh, HIP_FRAC_RANGE[0] * height_m, HIP_FRAC_RANGE[1] * height_m, HIP_FRAC_RANGE[2], "max"
    )

    circumferences: Dict[str, float] = {
        "neck": round(neck_cm, 1),
        "chest": round(chest_cm, 1),
        "waist": round(waist_cm, 1),
        "hip": round(hip_cm, 1),
    }
    for spec in LIMB_SPECS:
        circumferences[spec.name] = round(
            measure_limb(
                full_mesh,
                joints,
                spec.joint_start,
                spec.joint_end,
                spec.t_values,
                spec.mode,
                max_radius_m=LIMB_MAX_RADIUS_M.get(spec.name),
                limb_name=spec.name,
            ),
            1,
        )

    debug_slices = {
        "neck": neck_pts,
        "chest": chest_pts,
        "waist": waist_pts,
        "hip": hip_pts,
    }
    return circumferences, debug_slices


def extract_16_measurements(
    full_mesh_or_verts: Any,
    torso_mesh_or_faces: Any,
    joints_or_keep_torso: np.ndarray,
    height_m_or_joints: Any,
    sex_or_height_m: Any = "female",
    sex: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Optional[np.ndarray]]]:
    """Extract 16 body circumferences from either (full_mesh, torso_mesh, joints, height_m, sex) or raw arrays."""
    if sex is not None:
        verts_a = np.asarray(full_mesh_or_verts, dtype=np.float64)
        faces = np.asarray(torso_mesh_or_faces, dtype=np.int64)
        keep_torso = np.asarray(joints_or_keep_torso, dtype=bool)
        joints = np.asarray(height_m_or_joints, dtype=np.float64)
        height_m = float(sex_or_height_m)
        torso_mask = keep_torso[faces].all(axis=1)
        full_m = SimpleTriMesh(vertices=verts_a, faces=faces)
        torso_m = SimpleTriMesh(vertices=verts_a, faces=faces[torso_mask])
        return measure_all_circumferences(full_m, torso_m, joints, height_m, str(sex))
    return measure_all_circumferences(
        full_mesh_or_verts,
        torso_mesh_or_faces,
        np.asarray(joints_or_keep_torso, dtype=np.float64),
        float(height_m_or_joints),
        str(sex_or_height_m),
    )


def _interval_union_length(intervals: List[Tuple[float, float]]) -> float:
    """Return total covered length of 1D intervals without double-counting overlaps or gaps."""
    if not intervals:
        return 0.0
    sorted_iv = sorted(intervals, key=lambda iv: (iv[0], iv[1]))
    total = 0.0
    cur_s, cur_e = sorted_iv[0]
    for s, e in sorted_iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += max(0.0, cur_e - cur_s)
            cur_s, cur_e = s, e
    total += max(0.0, cur_e - cur_s)
    return float(total)


def compute_loft_profile(torso_mesh: Any, height_m: float) -> List[List[float]]:
    """Compute [height_frac, width_m, depth_m] every 1% from 0.30 to 0.86 of height."""
    loft: List[List[float]] = []
    for pct in range(LOFT_PCT_MIN, LOFT_PCT_MAX + 1):
        frac = pct / 100.0
        segs = _intersect_mesh_plane(
            torso_mesh, plane_normal=UP_AXIS, plane_origin=np.array([0.0, frac * height_m, 0.0])
        )
        if len(segs) > 0:
            xz_segs = segs[:, :, [0, 2]]
            graph = nx.Graph()
            coord_map: Dict[Tuple[float, ...], np.ndarray] = {}
            for seg in xz_segs:
                k0 = tuple(np.round(seg[0], 4))
                k1 = tuple(np.round(seg[1], 4))
                coord_map[k0] = seg[0]
                coord_map[k1] = seg[1]
                if k0 != k1:
                    graph.add_edge(k0, k1)
            comps = [c for c in nx.connected_components(graph) if len(c) >= 3]
            if comps:
                x_intervals: List[Tuple[float, float]] = []
                z_depths: List[float] = []
                for comp in comps:
                    arr = np.asarray([coord_map[k] for k in comp], dtype=np.float64)
                    x_intervals.append((float(arr[:, 0].min()), float(arr[:, 0].max())))
                    z_depths.append(float(arr[:, 1].max() - arr[:, 1].min()))
                width_m = _interval_union_length(x_intervals)
                depth_m = max(z_depths)
            else:
                pts3 = segs.reshape(-1, 3)
                width_m = float(pts3[:, 0].max() - pts3[:, 0].min())
                depth_m = float(pts3[:, 2].max() - pts3[:, 2].min())
        else:
            width_m, depth_m = 0.0, 0.0
        loft.append([round(frac, 2), round(width_m, 4), round(depth_m, 4)])
    return loft


def validate_sanity(circumferences: Dict[str, float], sex: str) -> List[str]:
    """Run domain sanity checks defined in the technical specification."""
    warnings: List[str] = []
    if sex.lower() == "female" and (circumferences.get("chest", 0.0) - circumferences.get("hip", 0.0) > 10.0):
        warnings.append("Chest exceeds hip by >10 cm for female subject (check arm removal).")
    if sex.lower() == "female" and not (27.0 <= circumferences.get("neck", 0.0) <= 38.0):
        warnings.append(f"Female neck {circumferences.get('neck')} cm is outside typical 27–38 cm range.")
    zero_keys = [k for k, v in circumferences.items() if v <= 0.0]
    if zero_keys:
        warnings.append(f"Zero/missing measurements: {zero_keys}")
    return warnings
