"""Stage 2 high-resolution head subdivision, RAM-only facial/nose sculpting, and 4-view hair shell."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (
    ARM_JOINT_IDS,
    ARM_WEIGHT_THRESHOLD,
    HEAD_JOINT_IDS,
    MHR_HEAD_SUBDIV_STEPS,
    NECK_BASE_JOINT,
    NECK_HEAD_JOINT,
)
from .inference import verify_license_compliance
from .segmentation import SegmentedView
from .texturing import build_head_face_vertex_mask

# Canonical MediaPipe FaceMesh 478 indices used ephemerally in RAM only (S2).
_MP_NOSE_TIP = 1
_MP_NOSE_BRIDGE = 6
_MP_FOREHEAD = 10
_MP_R_EYE_OUTER = 33
_MP_L_EYE_OUTER = 263
_MP_CHIN = 152
_MP_R_MOUTH = 61
_MP_L_MOUTH = 291
_MP_R_CHEEK = 234
_MP_L_CHEEK = 454


@dataclass
class HeadPartition:
    """Localized neck-collar partition and watertight subdivided cranium buffers."""

    head_vert_idx: np.ndarray
    collar_vert_idx: np.ndarray
    body_face_mask: np.ndarray
    head_face_mask: np.ndarray
    subdiv_verts: np.ndarray
    subdiv_faces: np.ndarray
    collar_subdiv_idx: np.ndarray
    subdiv_collar_ring_dist: np.ndarray
    cranium_verts: np.ndarray


@dataclass
class HighResHeadOutput:
    """Stitched watertight humanoid mesh with high-res head/nose/hair and 3b semantic masks."""

    full_verts: np.ndarray
    full_faces: np.ndarray
    head_vert_mask: np.ndarray
    hair_vert_mask: np.ndarray
    face_skin_vert_mask: np.ndarray
    front_face_vert_mask: np.ndarray
    warnings: List[str]
    undisplaced_verts: Optional[np.ndarray] = None
    body_only_idx: Optional[np.ndarray] = None


def _build_vertex_adjacency(num_verts: int, faces: np.ndarray) -> List[List[int]]:
    """Build deduplicated symmetric vertex adjacency lists from triangle faces."""
    neighbors: List[set[int]] = [set() for _ in range(num_verts)]
    if len(faces) == 0:
        return [[] for _ in range(num_verts)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [sorted(s) for s in neighbors]


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted unit vertex normals with safe zero-length fallback."""
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    v_normals = np.zeros_like(v, dtype=np.float32)
    if len(f) == 0 or len(v) == 0:
        return v_normals

    tris = v[f]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    np.add.at(v_normals, f[:, 0], cross)
    np.add.at(v_normals, f[:, 1], cross)
    np.add.at(v_normals, f[:, 2], cross)
    norms = np.linalg.norm(v_normals, axis=1, keepdims=True)
    safe = norms > 1e-10
    fallback = np.zeros_like(v_normals, dtype=np.float32)
    fallback[:, 1] = 1.0
    v_normals = np.where(safe, v_normals / np.maximum(norms, 1e-10), fallback).astype(np.float32)
    return v_normals


def _laplacian_smooth_field(
    field: np.ndarray,
    neighbors: List[List[int]],
    fixed_mask: np.ndarray,
    iterations: int = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """Smooth a scalar or vector vertex field across edges while clamping fixed vertices."""
    out = np.asarray(field, dtype=np.float32).copy()
    if iterations <= 0 or len(out) == 0:
        return out
    for _ in range(iterations):
        nxt = out.copy()
        for idx, nbrs in enumerate(neighbors):
            if fixed_mask[idx] or not nbrs:
                continue
            mean_nbr = np.mean(out[nbrs], axis=0)
            nxt[idx] = (1.0 - alpha) * out[idx] + alpha * mean_nbr
        out = nxt
    return out


def _select_largest_face_component(faces: np.ndarray, seed_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Keep the connected triangle component containing seed vertices (or largest component)."""
    n_f = len(faces)
    if n_f == 0:
        return np.zeros(0, dtype=bool)
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi, f in enumerate(faces):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(e, []).append(fi)

    adj: List[List[int]] = [[] for _ in range(n_f)]
    for flist in edge_to_faces.values():
        if len(flist) == 2:
            f0, f1 = flist
            adj[f0].append(f1)
            adj[f1].append(f0)

    visited = np.zeros(n_f, dtype=bool)
    best_comp: List[int] = []
    best_score = -1
    for start in range(n_f):
        if visited[start]:
            continue
        comp: List[int] = []
        q: deque[int] = deque([start])
        visited[start] = True
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        if seed_mask is not None:
            comp_verts = np.unique(faces[comp])
            seed_hits = int(np.sum(seed_mask[comp_verts]))
            score = seed_hits * 1_000_000 + len(comp)
        else:
            score = len(comp)
        if score > best_score:
            best_score = score
            best_comp = comp

    keep = np.zeros(n_f, dtype=bool)
    if best_comp:
        keep[best_comp] = True
    return keep


def _identify_head_face_mask(
    verts_a: np.ndarray,
    faces: np.ndarray,
    joints_a: np.ndarray,
    mhr_model: Optional[Any],
) -> np.ndarray:
    """Partition triangles above NECK_BASE_JOINT=110 cleanly without capturing shoulders or arms."""
    num_verts = len(verts_a)
    core_head_mask, _ = build_head_face_vertex_mask(verts_a, joints_a, mhr_model=mhr_model)

    neck_base = np.asarray(joints_a[NECK_BASE_JOINT], dtype=np.float32)
    neck_head = np.asarray(joints_a[NECK_HEAD_JOINT], dtype=np.float32)
    neck_base_y = float(neck_base[1])
    neck_head_y = float(neck_head[1])

    above_collar = verts_a[:, 1] >= neck_base_y - 1e-5
    if mhr_model is not None and hasattr(mhr_model, "character_torch"):
        try:
            lbs = mhr_model.character_torch.linear_blend_skinning
            skin_idx = lbs.skin_indices_flattened.cpu().numpy().astype(np.int64)
            skin_w = lbs.skin_weights_flattened.cpu().numpy().astype(np.float32)
            vert_idx = lbs.vert_indices_flattened.cpu().numpy().astype(np.int64)
            arm_w = np.zeros(num_verts, dtype=np.float32)
            arm_and_clav = ARM_JOINT_IDS | {38, 74}
            is_arm = np.isin(skin_idx, list(arm_and_clav))
            np.add.at(arm_w, vert_idx[is_arm], skin_w[is_arm])
            above_collar = above_collar & (arm_w < ARM_WEIGHT_THRESHOLD)

            head_neck_w = np.zeros(num_verts, dtype=np.float32)
            is_hn = np.isin(skin_idx, list(HEAD_JOINT_IDS | {NECK_BASE_JOINT, 111, 112}))
            np.add.at(head_neck_w, vert_idx[is_hn], skin_w[is_hn])
            above_collar = above_collar & ((head_neck_w >= 0.15) | core_head_mask)
        except Exception:
            pass
    elif len(joints_a) > 75:
        sh_r = np.asarray(joints_a[39], dtype=np.float32)
        sh_l = np.asarray(joints_a[75], dtype=np.float32)
        sh_span = float(np.linalg.norm(sh_l - sh_r))
        if sh_span > 0.15:
            neck_cx = 0.5 * float(neck_base[0] + neck_head[0])
            neck_cz = 0.5 * float(neck_base[2] + neck_head[2])
            max_neck_rad = max(0.11, 0.30 * sh_span)
            in_neck_band = (verts_a[:, 1] >= neck_base_y - 1e-5) & (verts_a[:, 1] < neck_head_y)
            radial_xz = np.hypot(verts_a[:, 0] - neck_cx, verts_a[:, 2] - neck_cz)
            above_collar = above_collar & (~in_neck_band | (radial_xz <= max_neck_rad))

    cand_verts = above_collar | core_head_mask
    cand_faces = np.all(cand_verts[faces], axis=1)
    if not np.any(cand_faces):
        cand_faces = np.all(core_head_mask[faces], axis=1)
    if not np.any(cand_faces):
        raise ValueError("No head triangles found above NECK_BASE_JOINT.")

    comp_local = _select_largest_face_component(faces[cand_faces], seed_mask=core_head_mask)
    head_face_mask = np.zeros(len(faces), dtype=bool)
    head_face_mask[np.where(cand_faces)[0][comp_local]] = True
    return head_face_mask


def _subdivide_head_watertight(
    verts: np.ndarray,
    faces: np.ndarray,
    collar_local_idx: np.ndarray,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform 1-to-4 midpoint subdivision on head faces while locking collar boundary edges."""
    cur_verts = np.asarray(verts, dtype=np.float32).copy()
    cur_faces = np.asarray(faces, dtype=np.int64).copy()
    collar_set = set(int(i) for i in collar_local_idx.tolist())

    for _ in range(max(0, int(steps))):
        edge_counts: Dict[Tuple[int, int], int] = {}
        for f in cur_faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            for u, v in ((a, b), (b, c), (c, a)):
                e = (u, v) if u < v else (v, u)
                edge_counts[e] = edge_counts.get(e, 0) + 1

        locked_edges = {
            e
            for e, cnt in edge_counts.items()
            if cnt == 1 and (e[0] in collar_set and e[1] in collar_set)
        }

        edge_midpoints: Dict[Tuple[int, int], int] = {}
        new_vert_list: List[np.ndarray] = [cur_verts]
        next_v_idx = len(cur_verts)
        added_verts: List[np.ndarray] = []

        for e in sorted(edge_counts.keys()):
            if e in locked_edges:
                continue
            mid = 0.5 * (cur_verts[e[0]] + cur_verts[e[1]])
            edge_midpoints[e] = next_v_idx
            added_verts.append(mid)
            next_v_idx += 1

        if added_verts:
            new_vert_list.append(np.asarray(added_verts, dtype=np.float32))
            cur_verts = np.vstack(new_vert_list)

        new_faces: List[Tuple[int, int, int]] = []
        for f in cur_faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            m_ab = edge_midpoints.get((a, b) if a < b else (b, a), -1)
            m_bc = edge_midpoints.get((b, c) if b < c else (c, b), -1)
            m_ca = edge_midpoints.get((c, a) if c < a else (a, c), -1)
            split_flags = (m_ab >= 0, m_bc >= 0, m_ca >= 0)
            n_splits = sum(split_flags)

            if n_splits == 0:
                new_faces.append((a, b, c))
            elif n_splits == 3:
                new_faces.extend(
                    [
                        (a, m_ab, m_ca),
                        (m_ab, b, m_bc),
                        (m_ca, m_bc, c),
                        (m_ab, m_bc, m_ca),
                    ]
                )
            elif n_splits == 1:
                # Rotate (a, b, c) so the single split edge is (b, c).
                if m_ab >= 0:
                    va, vb, vc, m_split = c, a, b, m_ab
                elif m_bc >= 0:
                    va, vb, vc, m_split = a, b, c, m_bc
                else:
                    va, vb, vc, m_split = b, c, a, m_ca
                new_faces.extend([(va, vb, m_split), (va, m_split, vc)])
            else:
                # Two split edges: rotate so the single unsplit edge is (a, b).
                if m_ab < 0:
                    va, vb, vc, mbc, mca = a, b, c, m_bc, m_ca
                elif m_bc < 0:
                    va, vb, vc, mbc, mca = b, c, a, m_ca, m_ab
                else:
                    va, vb, vc, mbc, mca = c, a, b, m_ab, m_bc
                d_a_mbc = float(np.sum((cur_verts[va] - cur_verts[mbc]) ** 2))
                d_b_mca = float(np.sum((cur_verts[vb] - cur_verts[mca]) ** 2))
                if d_a_mbc <= d_b_mca:
                    new_faces.extend([(va, vb, mbc), (va, mbc, mca), (mca, mbc, vc)])
                else:
                    new_faces.extend([(va, vb, mca), (mca, vb, mbc), (mca, mbc, vc)])

        cur_faces = np.asarray(new_faces, dtype=np.int64)

    return cur_verts, cur_faces


def _compute_collar_ring_distances(
    num_verts: int,
    faces: np.ndarray,
    collar_subdiv_idx: np.ndarray,
) -> np.ndarray:
    """Compute BFS geodesic edge-ring distance upward from collar_subdiv_idx."""
    dist = np.full(num_verts, -1, dtype=np.int32)
    if num_verts == 0:
        return dist
    if len(collar_subdiv_idx) == 0:
        return np.full(num_verts, 10, dtype=np.int32)

    neighbors = _build_vertex_adjacency(num_verts, faces)
    q: deque[int] = deque()
    for c_idx in collar_subdiv_idx.tolist():
        dist[int(c_idx)] = 0
        q.append(int(c_idx))
    while q:
        cur = q.popleft()
        nxt_d = int(dist[cur]) + 1
        for nb in neighbors[cur]:
            if dist[nb] < 0:
                dist[nb] = nxt_d
                q.append(nb)

    if np.any(dist < 0):
        max_d = int(dist[dist >= 0].max()) if np.any(dist >= 0) else 0
        dist[dist < 0] = max_d + 1
    return dist


def _collar_blend_weights(collar_ring_dist: np.ndarray) -> np.ndarray:
    """Return smooth [0, 1] weights with exact 0.0 at ring_dist == 0 and collar blend damping."""
    rings = np.asarray(collar_ring_dist, dtype=np.float32)
    if len(rings) == 0:
        return rings
    max_ring = float(np.max(rings))
    if max_ring >= 8.0:
        weights = np.clip((rings - 1.0) / 3.0, 0.0, 1.0)
    else:
        weights = np.clip(rings / 3.0, 0.0, 1.0)
    weights[collar_ring_dist <= 0] = 0.0
    return weights.astype(np.float32)


def partition_and_subdivide_head(
    verts_a: np.ndarray,
    faces: np.ndarray,
    joints_a: np.ndarray,
    mhr_model: Optional[Any] = None,
    subdiv_steps: int = MHR_HEAD_SUBDIV_STEPS,
) -> HeadPartition:
    """Partition head above NECK_BASE_JOINT=110 and subdivide interior head triangles watertightly."""
    verify_license_compliance()
    verts_f = np.asarray(verts_a, dtype=np.float32)
    faces_i = np.asarray(faces, dtype=np.int64)
    joints_f = np.asarray(joints_a, dtype=np.float32)

    head_face_mask = _identify_head_face_mask(verts_f, faces_i, joints_f, mhr_model=mhr_model)
    body_face_mask = ~head_face_mask

    head_vert_idx = np.unique(faces_i[head_face_mask]).astype(np.int64)
    body_vert_idx = (
        np.unique(faces_i[body_face_mask]).astype(np.int64)
        if np.any(body_face_mask)
        else np.zeros(0, dtype=np.int64)
    )
    collar_vert_idx = np.intersect1d(head_vert_idx, body_vert_idx).astype(np.int64)

    orig_to_local = np.full(len(verts_f), -1, dtype=np.int64)
    orig_to_local[head_vert_idx] = np.arange(len(head_vert_idx), dtype=np.int64)
    local_head_verts = verts_f[head_vert_idx].copy()
    local_head_faces = orig_to_local[faces_i[head_face_mask]]
    collar_subdiv_idx = orig_to_local[collar_vert_idx].astype(np.int64)

    subdiv_verts, subdiv_faces = _subdivide_head_watertight(
        local_head_verts,
        local_head_faces,
        collar_subdiv_idx,
        steps=subdiv_steps,
    )
    ring_dist = _compute_collar_ring_distances(len(subdiv_verts), subdiv_faces, collar_subdiv_idx)

    return HeadPartition(
        head_vert_idx=head_vert_idx,
        collar_vert_idx=collar_vert_idx,
        body_face_mask=body_face_mask,
        head_face_mask=head_face_mask,
        subdiv_verts=subdiv_verts,
        subdiv_faces=subdiv_faces,
        collar_subdiv_idx=collar_subdiv_idx,
        subdiv_collar_ring_dist=ring_dist,
        cranium_verts=subdiv_verts.copy(),
    )


def _skull_geometry_summary(
    head_verts: np.ndarray,
    joints_a: Optional[np.ndarray] = None,
    collar_ring_dist: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute canonical skull center, crown height, neck cut height, and face/cranium spans."""
    y_crown = float(head_verts[:, 1].max())
    y_min = float(head_verts[:, 1].min())
    if joints_a is not None and len(joints_a) > NECK_HEAD_JOINT:
        neck_base_y = float(joints_a[NECK_BASE_JOINT, 1])
        neck_head_y = float(joints_a[NECK_HEAD_JOINT, 1])
    else:
        neck_base_y = y_min
        neck_head_y = y_min + 0.32 * max(0.05, y_crown - y_min)

    if collar_ring_dist is not None and np.any(collar_ring_dist == 0):
        y_collar = float(np.mean(head_verts[collar_ring_dist == 0, 1]))
    else:
        y_collar = neck_base_y

    upper_mask = head_verts[:, 1] >= neck_head_y - 1e-4
    upper_pts = head_verts[upper_mask] if np.any(upper_mask) else head_verts
    skull_center = np.array(
        [
            float(np.mean(upper_pts[:, 0])),
            0.5 * (float(upper_pts[:, 1].min()) + y_crown),
            float(np.mean(upper_pts[:, 2])),
        ],
        dtype=np.float32,
    )
    return {
        "y_crown": y_crown,
        "y_collar": y_collar,
        "neck_base_y": neck_base_y,
        "neck_head_y": neck_head_y,
        "head_h": max(0.08, y_crown - y_collar),
        "upper_pts": upper_pts,
        "skull_center": skull_center,
    }


def _extract_raw_cam_px_per_m(rec: Optional[Dict[str, Any]]) -> Optional[float]:
    """Compute metric pixels-per-meter (focal_length / cam_t.z) from a raw view record if valid."""
    if not isinstance(rec, dict):
        return None
    out = rec.get("output") or {}
    focal = out.get("focal_length")
    if focal is None:
        focal = rec.get("focal_length")
    cam_t = out.get("pred_cam_t")
    if cam_t is None:
        cam_t = rec.get("pred_cam_t")
    if focal is None or cam_t is None:
        return None
    try:
        focal_f = float(np.asarray(focal, dtype=np.float64).reshape(-1)[0])
        tz = float(np.asarray(cam_t, dtype=np.float64).reshape(-1)[2])
    except (IndexError, TypeError, ValueError):
        return None
    if not (np.isfinite(focal_f) and np.isfinite(tz)) or focal_f <= 10.0 or tz <= 0.25:
        return None
    return max(20.0, focal_f / tz)


def _estimate_view_frame_single(
    sv: SegmentedView,
    geom: Dict[str, Any],
    cam_px_per_m: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Estimate single-view 2D pixel frame (cx_px, cy_skull_px, y_neck_px, px_per_m, reliable)."""
    az_rad = float(np.deg2rad(sv.azimuth_deg))
    u_axis = np.array([np.cos(az_rad), 0.0, np.sin(az_rad)], dtype=np.float32)
    cam_dir = np.array([-np.sin(az_rad), 0.0, np.cos(az_rad)], dtype=np.float32)

    pmask = np.asarray(sv.person_mask, dtype=bool)
    bmask = np.asarray(sv.body_mask, dtype=bool)
    fskin = np.asarray(sv.face_skin_mask, dtype=bool)
    h_img, w_img = pmask.shape[:2] if pmask.ndim == 2 and pmask.size > 0 else (0, 0)

    upper_pts = geom["upper_pts"]
    u_proj = upper_pts @ u_axis
    cranium_u_span = max(0.06, float(u_proj.max() - u_proj.min()))

    if h_img == 0 or w_img == 0 or (not np.any(pmask) and not np.any(bmask)):
        if sv.nose_profile_curve is not None and len(sv.nose_profile_curve) >= 3:
            curve = np.asarray(sv.nose_profile_curve, dtype=np.float32)
            cy_mid = 0.5 * float(curve[:, 1].min() + curve[:, 1].max())
            face_span_px = max(12.0, float(curve[:, 1].max() - curve[:, 1].min()))
            px_per_m = face_span_px / max(0.08, 0.56 * geom["head_h"])
            return {
                "u_axis": u_axis,
                "cam_dir": cam_dir,
                "cx_px": float(np.median(curve[:, 0])),
                "cy_skull_px": cy_mid,
                "y_neck_px": float(curve[:, 1].max()) + 0.15 * face_span_px,
                "px_per_m": float(px_per_m),
                "img_shape": (h_img, w_img),
                "reliable": False,
            }
        return None

    ref_mask = bmask if np.any(bmask) else pmask
    ys_p, _ = np.where(pmask if np.any(pmask) else bmask)
    y_top_fg = int(ys_p.min())
    y_bot_fg = int(ys_p.max())

    # Scan downward from y_top_fg to locate cranium equator and neck/shoulder transition cleanly
    # (works identically on head crops and full-body silhouettes without capturing shoulder/hip widths).
    if np.any(fskin):
        ys_s, _ = np.where(fskin)
        y_s_min, y_s_max = int(ys_s.min()), int(ys_s.max())
        skin_h = max(8, y_s_max - y_s_min)
        hard_limit = min(y_bot_fg, y_s_max + int(0.45 * max(skin_h, y_s_max - y_top_fg)))
    else:
        y_s_max = y_top_fg
        skin_h = 0
        hard_limit = min(y_bot_fg, y_top_fg + max(32, int(0.35 * (y_bot_fg - y_top_fg))))

    min_face_stop_y = (y_s_max - int(0.15 * skin_h)) if np.any(fskin) else y_top_fg + 12

    def _scan_head_rows(scan_mask: np.ndarray) -> List[Tuple[int, float, float]]:
        rows_out: List[Tuple[int, float, float]] = []
        head_peak_w = 0.0
        head_peak_y = y_top_fg
        post_peak_min_w = float("inf")
        seen_neck_dip = False
        for y in range(y_top_fg, hard_limit + 1):
            xs = np.where(scan_mask[y])[0]
            if len(xs) < 4:
                continue
            w_cur = float(xs.max() - xs.min() + 1)
            c_cur = 0.5 * float(xs.min() + xs.max())
            if w_cur > head_peak_w and not seen_neck_dip:
                head_peak_w = w_cur
                head_peak_y = y
                post_peak_min_w = w_cur
            elif y > head_peak_y + max(4, int(0.25 * head_peak_w)):
                if w_cur < post_peak_min_w:
                    post_peak_min_w = w_cur
                if post_peak_min_w <= 0.85 * head_peak_w:
                    seen_neck_dip = True
                if w_cur > 1.25 * post_peak_min_w and y >= min_face_stop_y:
                    break
            rows_out.append((y, w_cur, c_cur))
        return rows_out

    row_widths = _scan_head_rows(ref_mask)
    if len(row_widths) < 3 and np.any(pmask) and ref_mask is not pmask:
        row_widths = _scan_head_rows(pmask)
    if not row_widths:
        return None

    rw_arr = np.asarray(row_widths, dtype=np.float32)
    max_w = float(rw_arr[:, 1].max())
    peak_rows = rw_arr[rw_arr[:, 1] >= 0.85 * max_w]
    y_equator = float(np.median(peak_rows[:, 0]))
    cx_px = float(np.median(peak_rows[:, 2]))

    # Detect neck constriction below equator (width < 0.72 * max_w) or widening into shoulders.
    below_eq = rw_arr[rw_arr[:, 0] > y_equator]
    if len(below_eq) >= 3:
        narrow_rows = below_eq[below_eq[:, 1] <= 0.72 * max_w]
        if len(narrow_rows) > 0:
            y_neck_px = float(narrow_rows[0, 0])
        else:
            wide_sh = below_eq[below_eq[:, 1] >= 1.15 * max_w]
            y_neck_px = float(wide_sh[0, 0]) if len(wide_sh) > 0 else float(below_eq[-1, 0])
    else:
        y_neck_px = float(y_bot_fg)

    # Skull cranium width excluding sharp anterior nose spike (~90th percentile of equator band).
    head_rows = rw_arr[rw_arr[:, 0] <= y_neck_px]
    if len(head_rows) >= 3:
        # Exclude the top 15% crown tip and bottom 15% neck taper when estimating skull diameter.
        cranium_band = head_rows[(head_rows[:, 1] >= 0.65 * max_w)]
        if len(cranium_band) >= 3:
            w_skull_px = float(np.percentile(cranium_band[:, 1], 65))
            cy_skull_px = float(np.median(cranium_band[:, 0]))
            cx_px = float(np.median(cranium_band[:, 2]))
        else:
            w_skull_px = max_w
            cy_skull_px = y_equator
    else:
        w_skull_px = max_w
        cy_skull_px = y_equator

    px_per_m_mask = max(20.0, w_skull_px / cranium_u_span)
    # Check if hair covers most of body_mask on this view (e.g. back view with full hair).
    body_head_px = int(np.sum(bmask[y_top_fg : int(round(y_neck_px)) + 1, :]))
    hair_head_px = int(np.sum(sv.hair_mask[y_top_fg : int(round(y_neck_px)) + 1, :]))
    reliable = (body_head_px > 0.35 * max(1, body_head_px + hair_head_px)) and (len(head_rows) >= 6)

    if cam_px_per_m is not None and cam_px_per_m > 20.0:
        # Camera focal/z avoids px_per_m underestimation when hair_mask subtracts temples/crown from body_mask.
        px_per_m = float(cam_px_per_m)
    else:
        px_per_m = px_per_m_mask
        if (not reliable) and np.any(pmask):
            # Fallback to person_mask vertical head span when hair obscures skull width in body_mask.
            head_span_px = max(12.0, float(y_neck_px - y_top_fg))
            px_per_m = max(px_per_m, min(1.35 * px_per_m, head_span_px / max(0.14, float(geom["head_h"]))))

    # Refine cx_px symmetrically from person_mask around mid-head if hair displaced body_mask center.
    mid_lo = max(0, int(round(cy_skull_px - 0.15 * w_skull_px)))
    mid_hi = min(h_img, int(round(cy_skull_px + 0.15 * w_skull_px)) + 1)
    if mid_hi > mid_lo:
        _, xs_sym = np.where(bmask[mid_lo:mid_hi, :] if np.any(bmask[mid_lo:mid_hi, :]) else pmask[mid_lo:mid_hi, :])
        if len(xs_sym) > 0:
            cx_px = 0.5 * float(xs_sym.min() + xs_sym.max())

    return {
        "u_axis": u_axis,
        "cam_dir": cam_dir,
        "cx_px": float(cx_px),
        "cy_skull_px": float(cy_skull_px),
        "y_neck_px": float(y_neck_px),
        "px_per_m": float(px_per_m),
        "img_shape": (h_img, w_img),
        "reliable": bool(reliable),
    }


def _estimate_head_frames(
    segmented_views: Sequence[SegmentedView],
    head_verts: np.ndarray,
    joints_a: Optional[np.ndarray] = None,
    collar_ring_dist: Optional[np.ndarray] = None,
    raw_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Optional[Dict[str, Any]]]]:
    """Build per-view 2D-3D head frames with consensus scale/center fallback for hair-occluded views."""
    geom = _skull_geometry_summary(head_verts, joints_a=joints_a, collar_ring_dist=collar_ring_dist)
    frames: List[Optional[Dict[str, Any]]] = []
    for i, sv in enumerate(segmented_views):
        rec = None
        if raw_results is not None:
            v_idx = getattr(sv, "view_index", i)
            if 0 <= v_idx < len(raw_results):
                rec = raw_results[v_idx]
            elif i < len(raw_results):
                rec = raw_results[i]
        cam_ppm = _extract_raw_cam_px_per_m(rec)
        frames.append(_estimate_view_frame_single(sv, geom, cam_px_per_m=cam_ppm))

    reliable_frames = [f for f in frames if f is not None and f["reliable"]]
    if reliable_frames:
        med_scale = float(np.median([f["px_per_m"] for f in reliable_frames]))
        med_cy = float(np.median([f["cy_skull_px"] for f in reliable_frames]))
        med_neck = float(np.median([f["y_neck_px"] for f in reliable_frames]))
        med_shape = reliable_frames[0]["img_shape"]
        for f in frames:
            if f is not None and (not f["reliable"]) and f["img_shape"] == med_shape:
                f["px_per_m"] = med_scale
                f["cy_skull_px"] = med_cy
                f["y_neck_px"] = med_neck
    return geom, frames


def _fit_ridge_tps_3d(
    src_pts: np.ndarray,
    deltas: np.ndarray,
    query_pts: np.ndarray,
    reg: float = 1e-3,
) -> np.ndarray:
    """Evaluate ridge-regularized 3D thin-plate-spline RBF interpolation cleanly in NumPy."""
    n = len(src_pts)
    if n < 4 or len(query_pts) == 0:
        return np.zeros((len(query_pts), 3), dtype=np.float32)

    diff_ss = src_pts[:, None, :] - src_pts[None, :, :]
    r_ss = np.linalg.norm(diff_ss, axis=-1)
    k_ss = (r_ss ** 2) * np.log(r_ss + 1e-6)
    p_s = np.hstack([np.ones((n, 1), dtype=np.float64), src_pts.astype(np.float64)])

    top = np.hstack([k_ss + reg * np.eye(n, dtype=np.float64), p_s])
    bot = np.hstack([p_s.T, np.zeros((4, 4), dtype=np.float64)])
    sys_mat = np.vstack([top, bot])
    rhs = np.vstack([deltas.astype(np.float64), np.zeros((4, 3), dtype=np.float64)])
    # Add tiny stabilization on affine diagonal block so degenerate collinear landmarks never fail.
    sys_mat[n:, n:] += 1e-6 * np.eye(4, dtype=np.float64)
    coeffs = np.linalg.solve(sys_mat, rhs)

    diff_qs = query_pts[:, None, :] - src_pts[None, :, :]
    r_qs = np.linalg.norm(diff_qs, axis=-1)
    k_qs = (r_qs ** 2) * np.log(r_qs + 1e-6)
    p_q = np.hstack([np.ones((len(query_pts), 1), dtype=np.float64), query_pts.astype(np.float64)])
    pred = np.hstack([k_qs, p_q]) @ coeffs
    return pred.astype(np.float32)


def align_face_landmarks_3d(
    head_verts: np.ndarray,
    joints_a: np.ndarray,
    segmented_views: List[SegmentedView],
    raw_results: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ephemeral RAM-only landmark face warp and facial skin mask (never persisted)."""
    verify_license_compliance()
    verts_f = np.asarray(head_verts, dtype=np.float32)
    joints_f = np.asarray(joints_a, dtype=np.float32)
    n_verts = len(verts_f)
    delta_out = np.zeros((n_verts, 3), dtype=np.float32)
    if n_verts == 0:
        return delta_out, np.zeros(0, dtype=bool)

    geom = _skull_geometry_summary(verts_f, joints_a=joints_f)
    y_crown = geom["y_crown"]
    neck_base_y = geom["neck_base_y"]
    neck_head_y = geom["neck_head_y"]
    sc = geom["skull_center"]

    # Facial skin below forehead hairline and above neck base (excludes scalp/hair per 3b).
    hairline_y = y_crown - 0.34 * max(0.05, y_crown - neck_head_y)
    face_skin_subdiv_mask = (
        (verts_f[:, 1] > neck_base_y + 0.015)
        & (verts_f[:, 1] <= hairline_y)
        & (verts_f[:, 2] >= float(sc[2]) - 0.01)
    )
    if not np.any(face_skin_subdiv_mask):
        face_skin_subdiv_mask = (verts_f[:, 1] > neck_base_y + 1e-4) & (verts_f[:, 2] >= float(sc[2]))

    face_idx = np.where(face_skin_subdiv_mask)[0]
    if len(face_idx) == 0:
        return delta_out, face_skin_subdiv_mask

    # Keep all 478 MediaPipe landmarks strictly in local variables inside this function (S2).
    face_pts = verts_f[face_idx]
    f_x_span = max(0.04, float(face_pts[:, 0].max() - face_pts[:, 0].min()))
    f_y_min = float(face_pts[:, 1].min())
    f_y_max = float(face_pts[:, 1].max())
    lm_y_lo = max(f_y_min, neck_head_y)
    if f_y_max - lm_y_lo < 0.04:
        lm_y_lo = f_y_min
    lm_y_span = max(0.04, f_y_max - lm_y_lo)

    accum_delta = np.zeros((len(face_idx), 3), dtype=np.float32)
    accum_w = 0.0
    anchor_ids = (
        _MP_NOSE_TIP,
        _MP_NOSE_BRIDGE,
        _MP_FOREHEAD,
        _MP_R_EYE_OUTER,
        _MP_L_EYE_OUTER,
        _MP_CHIN,
        _MP_R_MOUTH,
        _MP_L_MOUTH,
        _MP_R_CHEEK,
        _MP_L_CHEEK,
    )

    for sv in segmented_views:
        az_rad = float(np.deg2rad(sv.azimuth_deg))
        if sv.view_label not in ("front", "oblique") or float(np.cos(az_rad)) <= 0.35:
            continue
        lm_local = sv.face_landmarks_ram
        if lm_local is None:
            continue
        lm_arr = np.asarray(lm_local, dtype=np.float32)
        if lm_arr.ndim != 2 or lm_arr.shape[0] <= max(anchor_ids) or lm_arr.shape[1] < 3:
            continue
        anchors = lm_arr[list(anchor_ids), :3]
        if not np.isfinite(anchors).all():
            continue

        # Procrustes-like normalized anchor placement projected along each view's (u_axis, v_axis).
        u_axis = np.array([np.cos(az_rad), 0.0, np.sin(az_rad)], dtype=np.float32)
        v_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cam_dir = np.array([-np.sin(az_rad), 0.0, np.cos(az_rad)], dtype=np.float32)
        rel_pts = face_pts - sc[None, :]
        face_u = rel_pts @ u_axis
        face_v = face_pts @ v_axis
        f_u_span = max(f_x_span, float(face_u.max() - face_u.min()))

        eye_mid_xy = 0.5 * (anchors[3, :2] + anchors[4, :2])
        eye_mid_z = 0.5 * float(anchors[3, 2] + anchors[4, 2])
        iod = max(0.05, float(np.linalg.norm(anchors[4, :2] - anchors[3, :2])))
        face_v_span = max(0.08, float(anchors[5, 1] - anchors[2, 1]))
        scale_xy = min(0.18, 0.55 * f_u_span) / iod

        grouped_deltas: Dict[int, List[np.ndarray]] = {}
        for k_idx in range(len(anchor_ids)):
            ax, ay, az = float(anchors[k_idx, 0]), float(anchors[k_idx, 1]), float(anchors[k_idx, 2])
            cand_u = (ax - float(eye_mid_xy[0])) * scale_xy
            # anchors[5] is chin (largest image y), anchors[2] is forehead (smallest image y).
            norm_up = (float(anchors[5, 1]) - ay) / face_v_span
            rel_y = lm_y_lo + (0.05 + 0.90 * norm_up) * lm_y_span
            cand_y = float(np.clip(rel_y, lm_y_lo, f_y_max))
            d2 = (face_u - cand_u) ** 2 + (face_v - cand_y) ** 2
            nearest_idx = int(np.argmin(d2))
            nearest_u = float(face_u[nearest_idx])
            nearest_v = float(face_v[nearest_idx])

            depth_push = float(np.clip(-(az - eye_mid_z) * iod * 0.85, -0.020, 0.022))
            lat_shift = float(np.clip((cand_u - nearest_u) * 0.25, -0.012, 0.012))
            dy_relief = float(np.clip((cand_y - nearest_v) * 0.20, -0.012, 0.012))
            delta_vec = lat_shift * u_axis + dy_relief * v_axis + depth_push * cam_dir
            grouped_deltas.setdefault(nearest_idx, []).append(delta_vec)

        src_pts = np.asarray([face_pts[vi] for vi in grouped_deltas.keys()], dtype=np.float32)
        tgt_deltas = np.asarray([np.mean(vs, axis=0) for vs in grouped_deltas.values()], dtype=np.float32)
        if len(src_pts) < 4:
            continue

        view_w = 1.0 if sv.view_label == "front" else 0.45
        warped = _fit_ridge_tps_3d(
            src_pts,
            tgt_deltas,
            face_pts,
            reg=1e-3,
        )
        warped = np.clip(warped, -0.025, 0.025)
        # Smooth ramp down near lower jaw/neck boundary.
        y_ramp = np.clip((face_pts[:, 1] - neck_head_y) / 0.04, 0.0, 1.0)[:, None]
        accum_delta += view_w * (warped * y_ramp.astype(np.float32))
        accum_w += view_w

    if accum_w > 0.0:
        delta_out[face_idx] = np.clip(accum_delta / accum_w, -0.025, 0.025)
    return delta_out, face_skin_subdiv_mask


def _apply_moge2_facial_relief(
    verts: np.ndarray,
    faces: np.ndarray,
    segmented_views: Sequence[SegmentedView],
    moge2_depths: Optional[Sequence[Optional[np.ndarray]]],
    frames: Sequence[Optional[Dict[str, Any]]],
    geom: Dict[str, Any],
) -> np.ndarray:
    """Apply metric-aligned local MoGe2 depth relief onto anterior facial vertices."""
    relief = np.zeros_like(verts, dtype=np.float32)
    if not moge2_depths:
        return relief

    sc = geom["skull_center"]
    v_normals = _compute_vertex_normals(verts, faces)
    face_roi = (verts[:, 1] > geom["neck_head_y"] - 0.01) & (verts[:, 2] >= float(sc[2]) - 0.01)
    if not np.any(face_roi):
        return relief

    total_w = np.zeros(len(verts), dtype=np.float32)
    accum_disp = np.zeros_like(verts, dtype=np.float32)

    for idx, sv in enumerate(segmented_views):
        if idx >= len(moge2_depths) or moge2_depths[idx] is None:
            continue
        frame = frames[idx] if idx < len(frames) else None
        if frame is None or frame["img_shape"][0] == 0:
            continue

        raw_d = np.asarray(moge2_depths[idx], dtype=np.float32).squeeze()
        if raw_d.ndim != 2 or raw_d.size == 0:
            continue
        h_img, w_img = frame["img_shape"]
        if raw_d.shape != (h_img, w_img):
            if np.isfinite(raw_d).all():
                raw_d = cv2.resize(raw_d, (w_img, h_img), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            else:
                fin = np.isfinite(raw_d).astype(np.float32)
                num = cv2.resize(
                    np.where(np.isfinite(raw_d), raw_d, 0.0),
                    (w_img, h_img),
                    interpolation=cv2.INTER_LINEAR,
                )
                den = cv2.resize(fin, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                raw_d = np.where(den > 1e-4, num / np.maximum(den, 1e-4), np.nan).astype(np.float32)

        valid_mask = np.asarray(sv.face_skin_mask, dtype=bool)
        if not np.any(valid_mask):
            valid_mask = np.asarray(sv.body_mask, dtype=bool)
        if not np.any(valid_mask):
            continue

        u_axis = frame["u_axis"]
        cam_dir = frame["cam_dir"]
        px_per_m = frame["px_per_m"]
        cx_px, cy_px = frame["cx_px"], frame["cy_skull_px"]

        vis_dots = v_normals @ cam_dir
        vis = face_roi & (vis_dots > 0.25)
        if int(np.sum(vis)) < 6:
            continue

        pts_v = verts[vis]
        u_px = cx_px + ((pts_v - sc) @ u_axis) * px_per_m
        v_px = cy_px - (pts_v[:, 1] - sc[1]) * px_per_m
        ui = np.round(u_px).astype(np.int64)
        vi = np.round(v_px).astype(np.int64)
        inb = (ui >= 0) & (ui < w_img) & (vi >= 0) & (vi < h_img)
        if int(np.sum(inb)) < 6:
            continue
        inb_mask = valid_mask[np.clip(vi, 0, h_img - 1), np.clip(ui, 0, w_img - 1)] & inb
        if int(np.sum(inb_mask)) < 6:
            continue

        raw_samples = raw_d[vi[inb_mask], ui[inb_mask]]
        finite = np.isfinite(raw_samples) & (raw_samples > 0.0)
        if int(np.sum(finite)) < 6:
            continue

        # Orthographic relative depth in camera direction (smaller = closer to camera).
        d_mesh = 2.0 - (pts_v[inb_mask][finite] - sc) @ cam_dir
        r_samp = raw_samples[finite]
        med_r = float(np.median(r_samp))
        med_m = float(np.median(d_mesh))
        denom = float(np.sum((r_samp - med_r) ** 2))
        if denom > 1e-8:
            scale = max(0.05, float(np.sum((r_samp - med_r) * (d_mesh - med_m)) / denom))
        else:
            scale = 1.0
        shift = med_m - scale * med_r
        aligned_d = scale * raw_d + shift

        # High-pass facial relief against smoothed depth field (zero NaN/inf).
        clean_d = np.where(np.isfinite(aligned_d), aligned_d, med_m).astype(np.float32)
        k_blur = max(7, (min(h_img, w_img) // 16) | 1)
        base_d = cv2.GaussianBlur(clean_d, (k_blur, k_blur), 0)
        hp_d = clean_d - base_d

        sampled_hp = hp_d[vi[inb_mask], ui[inb_mask]]
        local_push = np.clip(-sampled_hp, -0.015, 0.015)

        vis_indices = np.where(vis)[0][inb_mask]
        w_view = np.maximum(vis_dots[vis_indices], 0.0)
        accum_disp[vis_indices] += (w_view * local_push)[:, None] * v_normals[vis_indices]
        total_w[vis_indices] += w_view

    nonzero = total_w > 1e-5
    relief[nonzero] = np.clip(accum_disp[nonzero] / total_w[nonzero, None], -0.015, 0.015)
    return relief


def fit_nose_profile_and_depth(
    head_verts: np.ndarray,
    head_faces: np.ndarray,
    collar_ring_dist: np.ndarray,
    segmented_views: List[SegmentedView],
    moge2_depths: Optional[List[Optional[np.ndarray]]] = None,
) -> np.ndarray:
    """Sculpt nose bridge/tip/chin profile from side silhouettes and optional MoGe2 depth relief."""
    verify_license_compliance()
    verts_f = np.asarray(head_verts, dtype=np.float32).copy()
    faces_i = np.asarray(head_faces, dtype=np.int64)
    rings = np.asarray(collar_ring_dist, dtype=np.int32)
    if len(verts_f) == 0:
        return verts_f

    geom, frames = _estimate_head_frames(segmented_views, verts_f, collar_ring_dist=rings)
    sc = geom["skull_center"]
    y_crown = geom["y_crown"]
    y_collar = geom["y_collar"]
    head_h = geom["head_h"]

    # Anterior region candidates for sagittal nose/chin profile enforcement.
    x_span = max(0.06, float(geom["upper_pts"][:, 0].max() - geom["upper_pts"][:, 0].min()))
    mid_x = float(sc[0])
    ant_mask = (verts_f[:, 2] >= float(sc[2]) + 0.01) & (verts_f[:, 1] > y_collar + 0.15 * head_h)
    if not np.any(ant_mask):
        ant_mask = (verts_f[:, 2] >= float(sc[2])) & (rings > 0)

    # Measure current sagittal mesh front envelope z_mesh(y) along midline band.
    mid_band = ant_mask & (np.abs(verts_f[:, 0] - mid_x) <= 0.35 * x_span)
    if not np.any(mid_band):
        mid_band = ant_mask
    y_samples = verts_f[mid_band, 1]
    z_samples = verts_f[mid_band, 2]
    y_grid = np.linspace(y_collar + 0.15 * head_h, y_crown - 0.02 * head_h, 48, dtype=np.float32)
    z_mesh_grid = np.zeros_like(y_grid)
    bw_y = max(0.015, 0.08 * head_h)
    for gi, yg in enumerate(y_grid):
        near = np.abs(y_samples - yg) <= bw_y
        if np.any(near):
            z_mesh_grid[gi] = float(np.max(z_samples[near]))
        else:
            z_mesh_grid[gi] = float(z_samples[np.argmin(np.abs(y_samples - yg))])

    delta_z_total = np.zeros(len(verts_f), dtype=np.float32)
    profile_count = 0

    for idx, sv in enumerate(segmented_views):
        if sv.nose_profile_curve is None or len(sv.nose_profile_curve) < 3:
            continue
        label = sv.view_label.lower()
        if label not in ("right", "left"):
            continue
        frame = frames[idx] if idx < len(frames) else None
        if frame is None:
            continue

        curve = np.asarray(sv.nose_profile_curve, dtype=np.float32)
        finite_c = np.isfinite(curve).all(axis=1)
        curve = curve[finite_c]
        if len(curve) < 3:
            continue

        dir_sign = 1.0 if label == "right" else -1.0
        px_per_m = float(frame["px_per_m"])
        cx_px = float(frame["cx_px"])
        cy_skull_px = float(frame["cy_skull_px"])

        y_3d_curve = sc[1] - (curve[:, 1] - cy_skull_px) / px_per_m
        order = np.argsort(y_3d_curve)
        y_c_sorted = y_3d_curve[order]
        z_obs_raw = dir_sign * (curve[order, 0] - cx_px) / px_per_m

        # Align baseline offset using lower quartile of residual (forehead/chin baseline)
        # so flat controls matching the base skull stay near 0.0 while protruding noses sculpt cleanly.
        z_mesh_at_c = np.interp(y_c_sorted, y_grid, z_mesh_grid)
        res_c = z_obs_raw - z_mesh_at_c
        baseline_shift = float(np.percentile(res_c, 25))
        dz_curve = np.clip(res_c - baseline_shift, -0.020, 0.055)

        # Interpolate profile delta onto anterior head vertices within curve vertical span.
        v_y = verts_f[ant_mask, 1]
        in_y = (v_y >= float(y_c_sorted[0]) - 0.01) & (v_y <= float(y_c_sorted[-1]) + 0.01)
        dz_v = np.zeros(int(np.sum(ant_mask)), dtype=np.float32)
        dz_v[in_y] = np.interp(v_y[in_y], y_c_sorted, dz_curve).astype(np.float32)

        # Narrow Gaussian lateral falloff around sagittal midline keeps nose ridge crisp.
        sigma_x = max(0.025, 0.22 * x_span)
        lat_w = np.exp(-(((verts_f[ant_mask, 0] - mid_x) / sigma_x) ** 2)).astype(np.float32)
        # Taper smoothly toward zero near lateral curve vertical boundaries.
        y_lo_w = np.clip((v_y - float(y_c_sorted[0]) + 0.01) / 0.02, 0.0, 1.0)
        y_hi_w = np.clip((float(y_c_sorted[-1]) + 0.01 - v_y) / 0.02, 0.0, 1.0)
        delta_z_total[ant_mask] += dz_v * lat_w * np.minimum(y_lo_w, y_hi_w).astype(np.float32)
        profile_count += 1

    disp = np.zeros_like(verts_f, dtype=np.float32)
    if profile_count > 0:
        disp[:, 2] = delta_z_total / float(profile_count)

    # Add optional metric-scaled MoGe2 depth relief on anterior face.
    disp += _apply_moge2_facial_relief(verts_f, faces_i, segmented_views, moge2_depths, frames, geom)

    # Regularize displacement smoothly while keeping collar rings strictly damped to zero.
    neighbors = _build_vertex_adjacency(len(verts_f), faces_i)
    fixed_collar = rings <= 0
    disp = _laplacian_smooth_field(disp, neighbors, fixed_mask=fixed_collar, iterations=1, alpha=0.35)
    collar_w = _collar_blend_weights(rings)[:, None]
    disp = disp * collar_w
    disp[fixed_collar] = 0.0

    out_verts = verts_f + disp
    out_verts[fixed_collar] = verts_f[fixed_collar]
    return out_verts.astype(np.float32)


def _extract_scalp_patch(
    cranium_verts: np.ndarray,
    cranium_faces: np.ndarray,
    joints_a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract clean connected scalp/crown patch strictly above NECK_BASE_JOINT=110."""
    geom = _skull_geometry_summary(cranium_verts, joints_a=joints_a)
    sc = geom["skull_center"]
    y_crown = geom["y_crown"]
    neck_base_y = geom["neck_base_y"]
    neck_head_y = geom["neck_head_y"]

    upper_pts = geom["upper_pts"]
    z_max = float(upper_pts[:, 2].max())
    ap_half = max(0.05, z_max - float(sc[2]))

    # Forehead hairline on anterior side + occipital/nape above neck_base_y on back/sides.
    hairline_front_y = y_crown - 0.36 * max(0.05, y_crown - neck_head_y)
    back_y_floor = max(neck_base_y + 0.02, neck_head_y)
    scalp_v_mask = (
        (cranium_verts[:, 1] >= hairline_front_y)
        | ((cranium_verts[:, 1] >= back_y_floor) & (cranium_verts[:, 2] <= float(sc[2]) + 0.08 * ap_half))
    )
    scalp_face_mask = np.all(scalp_v_mask[cranium_faces], axis=1)
    if not np.any(scalp_face_mask):
        fallback_mask = cranium_verts[:, 1] >= float(sc[1])
        scalp_face_mask = np.all(fallback_mask[cranium_faces], axis=1)

    cand_faces = cranium_faces[scalp_face_mask]
    crown_seed = cranium_verts[:, 1] >= y_crown - 0.03
    comp_keep = _select_largest_face_component(cand_faces, seed_mask=crown_seed)
    patch_faces_global = cand_faces[comp_keep]

    # Prune degree-1 corner fin triangles so the boundary forms clean closed hairline rim(s).
    while len(patch_faces_global) > 4:
        edge_cnt: Dict[Tuple[int, int], int] = {}
        for f in patch_faces_global:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            for u, v in ((a, b), (b, c), (c, a)):
                e = (u, v) if u < v else (v, u)
                edge_cnt[e] = edge_cnt.get(e, 0) + 1
        keep_rows = []
        removed = False
        for f in patch_faces_global:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            b_edges = sum(
                1 for u, v in ((a, b), (b, c), (c, a)) if edge_cnt[(u, v) if u < v else (v, u)] == 1
            )
            if b_edges >= 2:
                removed = True
            else:
                keep_rows.append(f)
        if not removed or len(keep_rows) < 4:
            break
        patch_faces_global = np.asarray(keep_rows, dtype=np.int64)

    orig_scalp_idx = np.unique(patch_faces_global).astype(np.int64)
    g2l = np.full(len(cranium_verts), -1, dtype=np.int64)
    g2l[orig_scalp_idx] = np.arange(len(orig_scalp_idx), dtype=np.int64)
    local_faces = g2l[patch_faces_global]
    local_verts = cranium_verts[orig_scalp_idx].copy()

    edge_cnt_local: Dict[Tuple[int, int], int] = {}
    for f in local_faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_cnt_local[e] = edge_cnt_local.get(e, 0) + 1
    rim_set: set[int] = set()
    for (u, v), cnt in edge_cnt_local.items():
        if cnt == 1:
            rim_set.update((u, v))
    rim_mask = np.zeros(len(local_verts), dtype=bool)
    if rim_set:
        rim_mask[sorted(rim_set)] = True

    return local_verts, local_faces, rim_mask, orig_scalp_idx, sc


def _sample_hair_envelope_radius_px(
    hair_mask: np.ndarray,
    cx_px: float,
    cy_px: float,
    du: float,
    dv: float,
    r_min_px: float,
    r_max_px: float,
    y_neck_px: float,
) -> Optional[float]:
    """Ray-march outward in hair_mask (above y_neck_px) stopping at the first contiguous hair boundary."""
    h_img, w_img = hair_mask.shape
    norm_2d = float(np.hypot(du, dv))
    if norm_2d < 1e-5 or r_max_px <= r_min_px:
        return None
    ux, vy = du / norm_2d, dv / norm_2d
    px, py = -vy, ux

    steps = max(10, int(np.ceil((r_max_px - r_min_px) * 2.0)))
    rs = np.linspace(r_min_px, r_max_px, steps)
    max_gap_steps = 4  # ~2 px gap tolerance along half-pixel march; prevents jumping across background voids.
    best_r: Optional[float] = None
    for off in (0.0, -1.5, 1.5):
        xs = np.round(cx_px + rs * ux + off * px).astype(np.int64)
        ys = np.round(cy_px + rs * vy + off * py).astype(np.int64)
        valid = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img) & (ys <= int(np.ceil(y_neck_px)))
        if not np.any(valid):
            continue
        hits = np.zeros(len(rs), dtype=bool)
        hits[valid] = hair_mask[ys[valid], xs[valid]]
        if not np.any(hits):
            continue

        started = False
        gap = 0
        last_hit_idx = -1
        for i in range(len(rs)):
            if hits[i]:
                started = True
                gap = 0
                last_hit_idx = i
            elif started:
                gap += 1
                if gap > max_gap_steps:
                    break
        if last_hit_idx >= 0:
            hit_r = float(rs[last_hit_idx])
            if best_r is None or hit_r > best_r:
                best_r = hit_r
    return best_r


def build_hair_volume_shell(
    cranium_verts: np.ndarray,
    cranium_faces: np.ndarray,
    joints_a: np.ndarray,
    segmented_views: List[SegmentedView],
    raw_results: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a closed watertight collar-clear hair shell clamped >= cranium radius."""
    verify_license_compliance()
    verts_f = np.asarray(cranium_verts, dtype=np.float32)
    faces_i = np.asarray(cranium_faces, dtype=np.int64)
    joints_f = np.asarray(joints_a, dtype=np.float32)
    if len(verts_f) == 0 or len(faces_i) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    scalp_verts, scalp_faces, rim_mask, _, sc = _extract_scalp_patch(verts_f, faces_i, joints_f)
    if len(scalp_verts) == 0 or len(scalp_faces) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    _, frames = _estimate_head_frames(
        segmented_views,
        verts_f,
        joints_a=joints_f,
        raw_results=raw_results,
    )

    rel_vecs = scalp_verts - sc[None, :]
    r_base = np.linalg.norm(rel_vecs, axis=1).astype(np.float32)
    dirs = rel_vecs / np.maximum(r_base[:, None], 1e-8)

    # Smooth anatomical envelope limit: taller allowance on crown/back bun (up to 4.2cm), tighter on temples/sides (2.2cm max, 1.5cm uncorroborated).
    bun_bias = np.clip(0.65 * np.maximum(dirs[:, 1], 0.0) + 0.65 * np.maximum(-dirs[:, 2], 0.0), 0.0, 1.0)
    max_dr_per_vert = (0.022 + 0.020 * bun_bias).astype(np.float32)

    per_vert_obs: List[List[Tuple[float, float, int]]] = [[] for _ in range(len(scalp_verts))]
    has_any_hair = False

    for idx, sv in enumerate(segmented_views):
        hmask = np.asarray(sv.hair_mask, dtype=bool)
        if not np.any(hmask):
            continue
        frame = frames[idx] if idx < len(frames) else None
        if frame is None or frame["img_shape"][0] == 0:
            continue

        # Prune hair_mask pixels that match outer wall or facial skin when RGB is available.
        rec = None
        if raw_results is not None:
            v_idx = getattr(sv, "view_index", idx)
            if 0 <= v_idx < len(raw_results):
                rec = raw_results[v_idx]
            elif idx < len(raw_results):
                rec = raw_results[idx]
        if isinstance(rec, dict) and rec.get("image_rgb") is not None:
            rgb_img = np.asarray(rec["image_rgb"], dtype=np.float32)
            if rgb_img.ndim == 3 and rgb_img.shape[:2] == hmask.shape[:2] and rgb_img.shape[2] >= 3:
                rgb_3 = rgb_img[..., :3]
                pmask_v = np.asarray(sv.person_mask, dtype=bool)
                upper_lim = max(8, int(round(0.35 * hmask.shape[0])))
                bg_sample_mask = (~pmask_v[:upper_lim, :]) & (~hmask[:upper_lim, :])
                if int(np.sum(bg_sample_mask)) >= 32:
                    bg_med = np.median(rgb_3[:upper_lim, :][bg_sample_mask], axis=0)
                    bg_bright = float(np.mean(bg_med)) > 180.0
                    if bg_bright:
                        bg_dist = np.linalg.norm(rgb_3 - bg_med.reshape(1, 1, 3), axis=2)
                        hmask = hmask & (bg_dist >= 20.0)
                fskin_v = np.asarray(sv.face_skin_mask, dtype=bool)
                if fskin_v.shape[:2] == hmask.shape[:2] and int(np.sum(fskin_v)) >= 24:
                    skin_med = np.median(rgb_3[fskin_v], axis=0)
                    skin_dist = np.linalg.norm(rgb_3 - skin_med.reshape(1, 1, 3), axis=2)
                    # Only strip hair pixels that look almost identical to face skin and bright (>115).
                    is_skin_match = (skin_dist < 16.0) & (np.mean(rgb_3, axis=2) > 115.0)
                    hmask = hmask & (~is_skin_match)
        if not np.any(hmask):
            continue

        px_per_m = float(frame["px_per_m"])
        skull_area_est = max(16.0, float(np.pi) * ((0.105 * px_per_m) ** 2))
        hair_area = float(np.sum(hmask))
        # Downweight or reject wall/ceiling flood masks starting from 1.8x skull area.
        ys_hm, xs_hm = np.where(hmask)
        box_area = float(max(1, (int(ys_hm.max()) - int(ys_hm.min()) + 1) * (int(xs_hm.max()) - int(xs_hm.min()) + 1)))
        rect_fill = hair_area / max(1.0, box_area)
        area_ratio = hair_area / max(1.0, skull_area_est)
        view_quality = 1.0
        if area_ratio > 3.5 or rect_fill > 0.80:
            view_quality = 0.12
        elif area_ratio > 1.8:
            t_damp = (area_ratio - 1.8) / (3.5 - 1.8)
            view_quality = float(1.0 - 0.88 * np.clip(t_damp, 0.0, 1.0))

        has_any_hair = True
        u_axis = frame["u_axis"]
        cam_dir = frame["cam_dir"]
        cx_px, cy_px = float(frame["cx_px"]), float(frame["cy_skull_px"])
        y_neck_px = float(frame["y_neck_px"])

        # Project ray directions onto this view's image plane (u_axis, -Y_up).
        proj_u = dirs @ u_axis
        proj_v = -dirs[:, 1]
        proj_norm = np.hypot(proj_u, proj_v)
        cam_dot = dirs @ cam_dir

        for vi in range(len(scalp_verts)):
            if rim_mask[vi] or proj_norm[vi] < 0.28 or abs(float(cam_dot[vi])) > 0.90:
                continue
            r_b_px = float(r_base[vi] * proj_norm[vi] * px_per_m)
            max_dr_v = float(max_dr_per_vert[vi])
            r_max_px = max(r_b_px + 6.0, r_b_px + max_dr_v * float(proj_norm[vi]) * px_per_m * 1.15)
            hit_r_px = _sample_hair_envelope_radius_px(
                hair_mask=hmask,
                cx_px=cx_px,
                cy_px=cy_px,
                du=float(proj_u[vi]),
                dv=float(proj_v[vi]),
                r_min_px=max(2.0, 0.65 * r_b_px),
                r_max_px=r_max_px,
                y_neck_px=y_neck_px,
            )
            if hit_r_px is None:
                continue
            r_target_3d = (hit_r_px / px_per_m) / max(float(proj_norm[vi]), 0.45)
            dr = float(np.clip(r_target_3d - float(r_base[vi]), 0.0, max_dr_v))
            if view_quality < 1.0:
                dr *= view_quality
            w = view_quality * float(np.exp(-((float(cam_dot[vi]) / 0.50) ** 2)))
            per_vert_obs[vi].append((dr, w, idx))

    delta_r = np.zeros(len(scalp_verts), dtype=np.float32)
    observed = np.zeros(len(scalp_verts), dtype=bool)
    for vi in range(len(scalp_verts)):
        obs_list = per_vert_obs[vi]
        if rim_mask[vi] or not obs_list:
            continue
        drs = np.array([x[0] for x in obs_list], dtype=np.float32)
        ws = np.array([x[1] for x in obs_list], dtype=np.float32)
        v_ids = {x[2] for x in obs_list}
        # Multi-view hull consistency: shave single-view outlier spikes above consensus median + 12mm.
        if len(drs) >= 2:
            med_dr = float(np.median(drs))
            inlier = drs <= (med_dr + 0.012)
            if np.any(inlier):
                drs_use, ws_use = drs[inlier], ws[inlier]
            else:
                drs_use, ws_use = drs, ws
        else:
            drs_use, ws_use = drs, ws
        w_sum = float(np.sum(ws_use))
        if w_sum <= 1e-6:
            continue
        dr_est = float(np.sum(ws_use * drs_use) / w_sum)
        # Tighter temple/lateral cap (<=15mm) unless >=2 views corroborate larger hair volume.
        if float(bun_bias[vi]) < 0.35 and dr_est > 0.015:
            corroborating = len({x[2] for x in obs_list if x[0] > 0.012})
            if corroborating < 2 or len(v_ids) < 2:
                dr_est = 0.015
        delta_r[vi] = dr_est
        observed[vi] = True

    neighbors = _build_vertex_adjacency(len(scalp_verts), scalp_faces)

    # Fill unobserved blind-spot interior vertices from observed neighbor field before smoothing.
    if np.any(observed):
        known = observed | rim_mask
        for _ in range(10):
            missing_idx = np.where(~known)[0]
            if len(missing_idx) == 0:
                break
            next_vals: List[Tuple[int, float]] = []
            for vi in missing_idx:
                nbrs = [n for n in neighbors[vi] if known[n] and not rim_mask[n]]
                if nbrs:
                    next_vals.append((int(vi), float(np.mean(delta_r[nbrs]))))
            if not next_vals:
                break
            for vi, val in next_vals:
                delta_r[vi] = val
                known[vi] = True

    # Two-pass neighbor median outlier shaving suppresses spiky/boxy isolated ray jumps.
    for _ in range(2):
        despiked = delta_r.copy()
        for vi in range(len(scalp_verts)):
            if rim_mask[vi] or not neighbors[vi]:
                continue
            nbr_med = float(np.median(delta_r[neighbors[vi]]))
            despiked[vi] = min(float(delta_r[vi]), nbr_med + 0.012)
        delta_r = despiked

    delta_r = _laplacian_smooth_field(delta_r, neighbors, fixed_mask=rim_mask, iterations=8, alpha=0.48)
    delta_r = np.clip(delta_r, 0.0, max_dr_per_vert).astype(np.float32)

    # Smooth 5-ring smoothstep collar feathering inward from rim_mask to eliminate hard step cliffs.
    ring_dist = np.full(len(scalp_verts), -1, dtype=np.int32)
    ring_dist[rim_mask] = 0
    frontier = np.where(rim_mask)[0].tolist()
    for d_step in range(1, 6):
        next_frontier: List[int] = []
        for cur in frontier:
            for nb in neighbors[int(cur)]:
                if ring_dist[nb] < 0:
                    ring_dist[nb] = d_step
                    next_frontier.append(int(nb))
        frontier = next_frontier
        if not frontier:
            break
    taper_mask = (ring_dist > 0) & (ring_dist <= 5)
    if np.any(taper_mask):
        t_vals = np.clip(ring_dist[taper_mask].astype(np.float32) / 5.0, 0.0, 1.0)
        smooth_w = t_vals * t_vals * (3.0 - 2.0 * t_vals)
        delta_r[taper_mask] *= smooth_w

    delta_r[rim_mask] = 0.0

    # Ensure interior shell keeps tiny positive clearance when hair exists so closed cap is non-degenerate.
    min_thick = 0.002 if has_any_hair else 0.0005
    delta_r[~rim_mask] = np.maximum(delta_r[~rim_mask], min_thick)

    outer_verts = (sc[None, :] + dirs * (r_base + delta_r)[:, None]).astype(np.float32)
    neck_floor_y = float(joints_f[NECK_BASE_JOINT, 1]) if len(joints_f) > NECK_BASE_JOINT else float(verts_f[:, 1].min())
    outer_verts[:, 1] = np.maximum(outer_verts[:, 1], neck_floor_y)
    outer_verts[rim_mask] = scalp_verts[rim_mask]

    # Build a closed watertight double-walled cap: outer surface + inner skull-conforming cap joined at rim.
    interior_idx = np.where(~rim_mask)[0]
    inner_verts = scalp_verts[interior_idx].copy()
    inner_verts[:, 1] = np.maximum(inner_verts[:, 1], neck_floor_y)

    hair_verts = np.vstack([outer_verts, inner_verts]).astype(np.float32)
    inner_map = np.arange(len(scalp_verts), dtype=np.int64)
    inner_map[interior_idx] = len(outer_verts) + np.arange(len(interior_idx), dtype=np.int64)
    inner_faces = inner_map[scalp_faces[:, [0, 2, 1]]]
    hair_faces = np.vstack([scalp_faces, inner_faces]).astype(np.int64)
    return hair_verts, hair_faces


def reconstruct_highres_head_and_hair(
    verts_a: np.ndarray,
    faces: np.ndarray,
    joints_a: np.ndarray,
    segmented_views: List[SegmentedView],
    raw_results: List[Dict[str, Any]],
    mhr_model: Optional[Any] = None,
    subdiv_steps: int = MHR_HEAD_SUBDIV_STEPS,
) -> HighResHeadOutput:
    """Run full Stage 2 head subdivision, facial/nose/depth sculpting, hair shell, and 3b mask split."""
    verify_license_compliance()
    verts_f = np.asarray(verts_a, dtype=np.float32)
    faces_i = np.asarray(faces, dtype=np.int64)
    joints_f = np.asarray(joints_a, dtype=np.float32)

    warnings: List[str] = []
    seen_warn: set[str] = set()

    def _add_warn(msg: str) -> None:
        if msg and msg not in seen_warn:
            seen_warn.add(msg)
            warnings.append(msg)

    for sv in segmented_views:
        for w_msg in sv.warnings:
            _add_warn(w_msg)

    if not segmented_views:
        _add_warn("No segmented views provided; returning subdivided cranium without facial or hair sculpting.")

    has_landmarks = any(sv.face_landmarks_ram is not None for sv in segmented_views)
    if not has_landmarks:
        _add_warn(
            "MediaPipe 478 facial landmarks unavailable; skipping landmark TPS warp and using silhouette sculpting."
        )

    moge2_depths: List[Optional[np.ndarray]] = []
    if raw_results:
        for idx, sv in enumerate(segmented_views):
            rec_idx = sv.view_index if 0 <= sv.view_index < len(raw_results) else idx
            rec = raw_results[rec_idx] if 0 <= rec_idx < len(raw_results) else {}
            moge2_depths.append(rec.get("moge2_depth") if isinstance(rec, dict) else None)
    if not any(d is not None for d in moge2_depths):
        _add_warn("MoGe2 depth maps unavailable; skipping metric facial depth relief.")

    part = partition_and_subdivide_head(
        verts_a=verts_f,
        faces=faces_i,
        joints_a=joints_f,
        mhr_model=mhr_model,
        subdiv_steps=subdiv_steps,
    )

    face_delta, face_skin_subdiv_mask = align_face_landmarks_3d(
        head_verts=part.subdiv_verts,
        joints_a=joints_f,
        segmented_views=segmented_views,
        raw_results=raw_results,
    )
    sculpted_head = (part.subdiv_verts + face_delta).astype(np.float32)
    sculpted_head[part.collar_subdiv_idx] = verts_f[part.collar_vert_idx]

    sculpted_head = fit_nose_profile_and_depth(
        head_verts=sculpted_head,
        head_faces=part.subdiv_faces,
        collar_ring_dist=part.subdiv_collar_ring_dist,
        segmented_views=segmented_views,
        moge2_depths=moge2_depths if moge2_depths else None,
    )
    sculpted_head[part.collar_subdiv_idx] = verts_f[part.collar_vert_idx]

    hair_verts, hair_faces = build_hair_volume_shell(
        cranium_verts=part.cranium_verts,
        cranium_faces=part.subdiv_faces,
        joints_a=joints_f,
        segmented_views=segmented_views,
        raw_results=raw_results,
    )

    # Assemble unified watertight mesh: body_only (bit-exact to verts_a) + sculpted_head + hair_shell.
    body_faces_orig = faces_i[part.body_face_mask]
    body_all_idx = np.unique(body_faces_orig).astype(np.int64) if len(body_faces_orig) > 0 else np.zeros(0, dtype=np.int64)
    body_only_idx = np.setdiff1d(body_all_idx, part.collar_vert_idx).astype(np.int64)

    n_body_only = len(body_only_idx)
    n_head = len(sculpted_head)
    n_hair = len(hair_verts)

    orig_to_full = np.full(len(verts_f), -1, dtype=np.int64)
    orig_to_full[body_only_idx] = np.arange(n_body_only, dtype=np.int64)
    orig_to_full[part.collar_vert_idx] = n_body_only + part.collar_subdiv_idx

    body_faces_full = orig_to_full[body_faces_orig] if len(body_faces_orig) > 0 else np.zeros((0, 3), dtype=np.int64)
    head_faces_full = part.subdiv_faces + n_body_only
    base_head_verts = part.subdiv_verts.astype(np.float32).copy()
    base_head_verts[part.collar_subdiv_idx] = verts_f[part.collar_vert_idx]
    if n_hair > 0 and len(hair_faces) > 0:
        scalp_verts_b, _, rim_mask_b, _, _ = _extract_scalp_patch(
            part.cranium_verts, part.subdiv_faces, joints_f
        )
        neck_floor_y = (
            float(joints_f[NECK_BASE_JOINT, 1])
            if len(joints_f) > NECK_BASE_JOINT
            else float(verts_f[:, 1].min())
        )
        base_outer = scalp_verts_b.copy()
        base_outer[:, 1] = np.maximum(base_outer[:, 1], neck_floor_y)
        interior_idx_b = np.where(~rim_mask_b)[0]
        base_inner = scalp_verts_b[interior_idx_b].copy()
        base_inner[:, 1] = np.maximum(base_inner[:, 1], neck_floor_y)
        hair_base_verts = np.vstack([base_outer, base_inner]).astype(np.float32)
        if len(hair_base_verts) != n_hair:
            hair_base_verts = hair_verts.copy()

        hair_faces_full = hair_faces + (n_body_only + n_head)
        full_verts = np.vstack([verts_f[body_only_idx], sculpted_head, hair_verts]).astype(np.float32)
        undisplaced_verts = np.vstack(
            [verts_f[body_only_idx], base_head_verts, hair_base_verts]
        ).astype(np.float32)
        full_faces = np.vstack([body_faces_full, head_faces_full, hair_faces_full]).astype(np.int64)
    else:
        full_verts = np.vstack([verts_f[body_only_idx], sculpted_head]).astype(np.float32)
        undisplaced_verts = np.vstack([verts_f[body_only_idx], base_head_verts]).astype(np.float32)
        full_faces = np.vstack([body_faces_full, head_faces_full]).astype(np.int64)

    n_total = len(full_verts)
    head_vert_mask = np.zeros(n_total, dtype=bool)
    hair_vert_mask = np.zeros(n_total, dtype=bool)
    face_skin_vert_mask = np.zeros(n_total, dtype=bool)

    head_above_collar = part.subdiv_collar_ring_dist > 0
    head_vert_mask[n_body_only : n_body_only + n_head] = head_above_collar
    if n_hair > 0:
        head_vert_mask[n_body_only + n_head :] = True
        hair_vert_mask[n_body_only + n_head :] = True

    face_skin_subdiv_clean = face_skin_subdiv_mask & head_above_collar
    face_skin_vert_mask[n_body_only : n_body_only + n_head] = face_skin_subdiv_clean
    face_skin_vert_mask &= ~hair_vert_mask

    neck_head_z = float(joints_f[NECK_HEAD_JOINT, 2]) if len(joints_f) > NECK_HEAD_JOINT else 0.0
    front_face_vert_mask = face_skin_vert_mask & (full_verts[:, 2] >= neck_head_z + 0.01)
    if not np.any(front_face_vert_mask) and np.any(face_skin_vert_mask):
        front_face_vert_mask = face_skin_vert_mask.copy()

    return HighResHeadOutput(
        full_verts=full_verts,
        full_faces=full_faces,
        head_vert_mask=head_vert_mask,
        hair_vert_mask=hair_vert_mask,
        face_skin_vert_mask=face_skin_vert_mask,
        front_face_vert_mask=front_face_vert_mask,
        warnings=warnings,
        undisplaced_verts=undisplaced_verts,
        body_only_idx=body_only_idx,
    )
