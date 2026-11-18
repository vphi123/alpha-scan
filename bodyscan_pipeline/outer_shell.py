"""Stage 3 dual-layer clothed/outer body shell reconstruction with inner MHR barrier & hull cap."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from .config import (
    INNER_BARRIER_M,
    MHR70_L_ANKLE,
    MHR70_L_ELBOW,
    MHR70_L_WRIST,
    MHR70_R_ANKLE,
    MHR70_R_ELBOW,
    MHR70_R_WRIST,
    MHR127_ANKLE_JOINT_IDS,
    MHR127_FOREARM_ELBOW_PAIRS,
    MHR127_WRIST_ELBOW_PAIRS,
    NECK_BASE_JOINT,
)
from .fitting import (
    _compute_mask_iou,
    _extract_focal_length,
    _resolve_raw_record,
    _rotation_y_and_cv_flip,
    _solve_initial_view_camera,
    compute_soft_silhouette_and_edge_loss,
)
from .head_hair import HighResHeadOutput, _compute_vertex_normals
from .inference import verify_license_compliance
from .segmentation import SegmentedView
from .texturing import render_view_depth_buffer


@dataclass
class OuterShellOutput:
    """Dual-layer reconstruction with clothed outer surface and unmodified inner MHR core."""

    outer_verts: np.ndarray
    outer_faces: np.ndarray
    inner_mhr_verts: np.ndarray
    inner_mhr_faces: np.ndarray
    head_vert_mask: np.ndarray
    hair_vert_mask: np.ndarray
    face_skin_vert_mask: np.ndarray
    front_face_vert_mask: np.ndarray
    outer_silhouette_iou: float
    warnings: List[str] = field(default_factory=list)
    undisplaced_verts: Optional[np.ndarray] = None


def align_metric_depth_map(
    raw_depth: np.ndarray,
    mhr_rendered_depth: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Fit robust Huber affine transform (a > 0, b) aligning monocular raw_depth to metric MHR depth."""
    verify_license_compliance()
    raw_f = np.asarray(raw_depth, dtype=np.float32).squeeze()
    mhr_f = np.asarray(mhr_rendered_depth, dtype=np.float32).squeeze()
    valid_b = np.asarray(valid_mask, dtype=bool).squeeze()

    if raw_f.ndim != 2 or mhr_f.ndim != 2 or valid_b.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for depth alignment, got shapes {raw_f.shape}, {mhr_f.shape}, {valid_b.shape}"
        )
    if raw_f.shape != mhr_f.shape:
        raw_f = cv2.resize(raw_f, (mhr_f.shape[1], mhr_f.shape[0]), interpolation=cv2.INTER_NEAREST)
    if valid_b.shape != mhr_f.shape:
        valid_b = (
            cv2.resize(valid_b.astype(np.uint8), (mhr_f.shape[1], mhr_f.shape[0]), interpolation=cv2.INTER_NEAREST)
            > 0
        )

    fit_mask = valid_b & np.isfinite(raw_f) & np.isfinite(mhr_f) & (mhr_f > 0.0)
    if int(np.sum(fit_mask)) < 6:
        fallback_mask = valid_b & np.isfinite(raw_f) & np.isfinite(mhr_f)
        if int(np.sum(fallback_mask)) >= 2:
            fit_mask = fallback_mask
        else:
            out_fallback = np.full(raw_f.shape, np.nan, dtype=np.float32)
            fin = np.isfinite(raw_f)
            out_fallback[fin] = raw_f[fin]
            return out_fallback

    x = raw_f[fit_mask].astype(np.float64)
    y = mhr_f[fit_mask].astype(np.float64)
    xm = float(np.median(x))
    ym = float(np.median(y))
    dx = x - xm
    dy = y - ym
    var_x = float(np.var(x))

    if var_x < 1e-12:
        a = 1.0
        b = ym - xm
    else:
        mad_x = float(np.median(np.abs(dx)))
        far = np.abs(dx) >= 0.5 * mad_x if mad_x >= 1e-9 else np.zeros(len(x), dtype=bool)
        if int(np.sum(far)) >= 4:
            slopes = dy[far] / dx[far]
            pos_slopes = slopes[slopes > 0.0]
            if len(pos_slopes) >= max(2, len(slopes) // 3):
                a = float(np.median(pos_slopes))
            else:
                a = float(np.dot(dx, dy) / max(float(np.dot(dx, dx)), 1e-12))
        else:
            a = float(np.dot(dx, dy) / max(float(np.dot(dx, dx)), 1e-12))
        a = max(a, 1e-4)
        b = ym - a * xm

        # Huber IRLS refinement to resist localized garment folds & boundary outliers.
        for _ in range(15):
            res = y - (a * x + b)
            med_r = float(np.median(res))
            mad_r = float(np.median(np.abs(res - med_r)))
            scale_r = max(1.4826 * mad_r, 1e-4)
            thresh = 1.345 * scale_r
            abs_r = np.abs(res)
            w = np.where(abs_r <= thresh, 1.0, thresh / np.maximum(abs_r, 1e-12))
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                break
            xw = float(np.sum(w * x) / w_sum)
            yw = float(np.sum(w * y) / w_sum)
            dxw = x - xw
            dyw = y - yw
            denom = float(np.sum(w * dxw * dxw))
            if denom <= 1e-12:
                break
            a_new = max(1e-4, float(np.sum(w * dxw * dyw) / denom))
            b_new = yw - a_new * xw
            if abs(a_new - a) < 1e-9 and abs(b_new - b) < 1e-9:
                a, b = a_new, b_new
                break
            a, b = a_new, b_new

    aligned = np.full(raw_f.shape, np.nan, dtype=np.float32)
    finite_raw = np.isfinite(raw_f)
    aligned[finite_raw] = (a * raw_f[finite_raw].astype(np.float64) + b).astype(np.float32)
    return aligned


def _rasterize_binary_silhouette_fast(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Project 3D camera-frame triangles into an (img_h, img_w) boolean silhouette mask on CPU."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(verts_cam) == 0 or len(faces) == 0:
        return mask > 0

    z = np.maximum(verts_cam[:, 2], 1e-4)
    u = float(focal_length) * (verts_cam[:, 0] / z) + (img_w / 2.0)
    v = float(focal_length) * (verts_cam[:, 1] / z) + (img_h / 2.0)
    pts_2d = np.stack([u, v], axis=-1)

    tris_z = verts_cam[faces, 2]
    tris_2d = pts_2d[faces]
    valid = (
        np.all(tris_z > 0.05, axis=1)
        & np.all(np.isfinite(tris_2d), axis=(1, 2))
        & (np.max(tris_2d[:, :, 0], axis=1) >= 0.0)
        & (np.min(tris_2d[:, :, 0], axis=1) < float(img_w))
        & (np.max(tris_2d[:, :, 1], axis=1) >= 0.0)
        & (np.min(tris_2d[:, :, 1], axis=1) < float(img_h))
    )
    if not np.any(valid):
        return mask > 0

    tris_int = np.round(tris_2d[valid]).astype(np.int32)
    batch = 4096
    for start in range(0, len(tris_int), batch):
        cv2.fillPoly(mask, list(tris_int[start : start + batch]), 1)
    return mask > 0


def _compute_oriented_outward_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unit outward vertex normals, flipping orientation if mesh signed volume is negative."""
    normals = _compute_vertex_normals(verts, faces)
    if len(faces) == 0 or len(verts) == 0:
        return normals
    v_f = np.asarray(verts, dtype=np.float64)
    v_c = v_f - np.mean(v_f, axis=0, keepdims=True)
    tris = v_c[faces]
    signed_vol6 = float(np.sum(tris[:, 0] * np.cross(tris[:, 1], tris[:, 2])))
    if signed_vol6 < -1e-9:
        normals = -normals
    return normals.astype(np.float32)


def _subdivide_body_watertight(
    highres_head: HighResHeadOutput,
    max_body_faces_to_subdiv: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subdivide torso/limb triangles once watertightly while locking collar edges and head/hair."""
    full_verts = np.asarray(highres_head.full_verts, dtype=np.float32).copy()
    full_faces = np.asarray(highres_head.full_faces, dtype=np.int64).copy()
    hr_head = np.asarray(highres_head.head_vert_mask, dtype=bool)
    hr_hair = np.asarray(highres_head.hair_vert_mask, dtype=bool)
    hr_face_skin = np.asarray(highres_head.face_skin_vert_mask, dtype=bool)
    hr_front_face = np.asarray(highres_head.front_face_vert_mask, dtype=bool)

    n_hr = len(full_verts)
    head_side_face_mask = np.any(hr_head[full_faces], axis=1) if len(full_faces) > 0 else np.zeros(0, dtype=bool)
    body_side_face_mask = ~head_side_face_mask

    if np.any(head_side_face_mask):
        head_side_verts = np.unique(full_faces[head_side_face_mask])
        collar_idx = head_side_verts[~hr_head[head_side_verts]].astype(np.int64)
    else:
        collar_idx = np.zeros(0, dtype=np.int64)

    body_faces = full_faces[body_side_face_mask]
    if len(body_faces) == 0 or len(body_faces) > max_body_faces_to_subdiv:
        collar_mask = np.zeros(n_hr, dtype=bool)
        if len(collar_idx) > 0:
            collar_mask[collar_idx] = True
        return (
            full_verts,
            full_faces,
            hr_head.copy(),
            hr_hair.copy(),
            hr_face_skin.copy(),
            hr_front_face.copy(),
            collar_mask,
        )

    base_normals = _compute_oriented_outward_normals(full_verts, full_faces)
    collar_set = set(int(i) for i in collar_idx.tolist())

    edge_counts: Dict[Tuple[int, int], int] = {}
    for f in body_faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_counts[e] = edge_counts.get(e, 0) + 1

    locked_edges = {e for e in edge_counts if e[0] in collar_set and e[1] in collar_set}

    edge_midpoints: Dict[Tuple[int, int], int] = {}
    added_verts: List[np.ndarray] = []
    next_idx = n_hr
    for e in sorted(edge_counts.keys()):
        if e in locked_edges:
            continue
        u, v = e
        vu, vv = full_verts[u], full_verts[v]
        nu, nv = base_normals[u], base_normals[v]
        edge_len = float(np.linalg.norm(vv - vu))
        n_sum = nu + nv
        n_norm = float(np.linalg.norm(n_sum))
        if n_norm > 1e-8 and edge_len > 1e-8:
            n_mid = n_sum / n_norm
            # Non-negative quadratic sagitta lift prevents inward chord shrinkage on convex limbs/torso.
            sagitta_3d = 0.125 * float(np.dot(vu - vv, nu) + np.dot(vv - vu, nv))
            nu_xz = nu[[0, 2]]
            nv_xz = nv[[0, 2]]
            nu_xz_n = float(np.linalg.norm(nu_xz))
            nv_xz_n = float(np.linalg.norm(nv_xz))
            sagitta_xz = 0.0
            n_xz_mid: Optional[np.ndarray] = None
            if nu_xz_n > 1e-4 and nv_xz_n > 1e-4:
                nu_xz_u = nu_xz / nu_xz_n
                nv_xz_u = nv_xz / nv_xz_n
                n_xz_sum = nu_xz_u + nv_xz_u
                n_xz_len = float(np.linalg.norm(n_xz_sum))
                if n_xz_len > 1e-4:
                    n_xz_mid = np.array(
                        [n_xz_sum[0] / n_xz_len, 0.0, n_xz_sum[1] / n_xz_len],
                        dtype=np.float32,
                    )
                    dxz = vv[[0, 2]] - vu[[0, 2]]
                    sagitta_xz = 0.125 * float(np.dot(-dxz, nu_xz_u) + np.dot(dxz, nv_xz_u))
            if sagitta_3d >= sagitta_xz:
                lift = float(np.clip(sagitta_3d, 0.0, 0.15 * edge_len))
                mid = 0.5 * (vu + vv) + lift * n_mid
            elif n_xz_mid is not None and sagitta_xz > 0.0:
                # Preserve exact cone Y while lifting convex XZ chord to the circumscribed circle.
                lift_xz = float(np.clip(sagitta_xz, 0.0, 0.15 * edge_len))
                mid = 0.5 * (vu + vv) + lift_xz * n_xz_mid
            else:
                mid = 0.5 * (vu + vv)
        else:
            mid = 0.5 * (vu + vv)
        edge_midpoints[e] = next_idx
        added_verts.append(mid.astype(np.float32))
        next_idx += 1

    new_body_faces: List[Tuple[int, int, int]] = []
    for f in body_faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        m_ab = edge_midpoints.get((a, b) if a < b else (b, a), -1)
        m_bc = edge_midpoints.get((b, c) if b < c else (c, b), -1)
        m_ca = edge_midpoints.get((c, a) if c < a else (a, c), -1)
        n_splits = int(m_ab >= 0) + int(m_bc >= 0) + int(m_ca >= 0)

        if n_splits == 0:
            new_body_faces.append((a, b, c))
        elif n_splits == 3:
            new_body_faces.extend(
                [
                    (a, m_ab, m_ca),
                    (m_ab, b, m_bc),
                    (m_ca, m_bc, c),
                    (m_ab, m_bc, m_ca),
                ]
            )
        elif n_splits == 1:
            if m_ab >= 0:
                va, vb, vc, m_split = c, a, b, m_ab
            elif m_bc >= 0:
                va, vb, vc, m_split = a, b, c, m_bc
            else:
                va, vb, vc, m_split = b, c, a, m_ca
            new_body_faces.extend([(va, vb, m_split), (va, m_split, vc)])
        else:
            if m_ab < 0:
                va, vb, vc, mbc, mca = a, b, c, m_bc, m_ca
            elif m_bc < 0:
                va, vb, vc, mbc, mca = b, c, a, m_ca, m_ab
            else:
                va, vb, vc, mbc, mca = c, a, b, m_ab, m_bc
            d_a_mbc = float(np.sum((full_verts[va] - added_verts[mbc - n_hr]) ** 2))
            d_b_mca = float(np.sum((full_verts[vb] - added_verts[mca - n_hr]) ** 2))
            if d_a_mbc <= d_b_mca:
                new_body_faces.extend([(va, vb, mbc), (va, mbc, mca), (mca, mbc, vc)])
            else:
                new_body_faces.extend([(va, vb, mca), (mca, vb, mbc), (mca, mbc, vc)])

    if added_verts:
        out_verts = np.vstack([full_verts, np.asarray(added_verts, dtype=np.float32)]).astype(np.float32)
    else:
        out_verts = full_verts
    n_out = len(out_verts)

    body_faces_arr = np.asarray(new_body_faces, dtype=np.int64)
    head_faces_arr = full_faces[head_side_face_mask]
    out_faces = (
        np.vstack([body_faces_arr, head_faces_arr]).astype(np.int64)
        if len(head_faces_arr) > 0
        else body_faces_arr
    )

    head_mask = np.zeros(n_out, dtype=bool)
    hair_mask = np.zeros(n_out, dtype=bool)
    face_skin_mask = np.zeros(n_out, dtype=bool)
    front_face_mask = np.zeros(n_out, dtype=bool)
    collar_mask = np.zeros(n_out, dtype=bool)

    head_mask[:n_hr] = hr_head
    hair_mask[:n_hr] = hr_hair
    face_skin_mask[:n_hr] = hr_face_skin
    front_face_mask[:n_hr] = hr_front_face
    if len(collar_idx) > 0:
        collar_mask[collar_idx] = True

    return out_verts, out_faces, head_mask, hair_mask, face_skin_mask, front_face_mask, collar_mask


def _build_directed_edge_laplacian(
    num_verts: int, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deduplicated directed edge indices (src, dst) and inverse vertex degrees."""
    if num_verts == 0 or len(faces) == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(num_verts, dtype=np.float32),
        )
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
            faces[:, [1, 0]],
            faces[:, [2, 1]],
            faces[:, [0, 2]],
        ]
    ).astype(np.int64)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(num_verts, dtype=np.float32),
        )
    src = edges[:, 0]
    dst = edges[:, 1]
    deg = np.bincount(src, minlength=num_verts).astype(np.float32)
    inv_deg = np.where(deg > 0.0, 1.0 / np.maximum(deg, 1.0), 0.0).astype(np.float32)
    return src, dst, inv_deg


def _apply_graph_laplacian(
    field_t: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    inv_deg: torch.Tensor,
) -> torch.Tensor:
    """Evaluate umbrella graph Laplacian (L d)_i = d_i - mean_{j in N(i)} d_j without sparse warnings."""
    if edge_src.numel() == 0:
        return torch.zeros_like(field_t)
    nbr_sum = torch.zeros_like(field_t).index_add(0, edge_src, field_t[edge_dst])
    return field_t - nbr_sum * inv_deg


def _closest_triangle_barycentric(
    pts: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    k_cand: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bind 3D query points to nearest triangles via exact Voronoi barycentric coords + normal height."""
    tri_pts = verts[faces]  # (F, 3, 3)
    centroids = tri_pts.mean(axis=1)
    k = min(max(1, k_cand), len(faces))
    _, cand_idx = cKDTree(centroids).query(pts, k=k)
    if k == 1:
        cand_idx = cand_idx[:, None]

    # (N, K, 3)
    p = pts[:, None, :]
    a = tri_pts[cand_idx, 0, :]
    b = tri_pts[cand_idx, 1, :]
    c = tri_pts[cand_idx, 2, :]

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    bp = p - b
    d3 = np.sum(ab * bp, axis=-1)
    d4 = np.sum(ac * bp, axis=-1)
    cp = p - c
    d5 = np.sum(ab * cp, axis=-1)
    d6 = np.sum(ac * cp, axis=-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    eps = 1e-12

    denom_f = np.maximum(va + vb + vc, eps)
    u = va / denom_f
    v = vb / denom_f
    w = 1.0 - u - v

    # Override exterior regions using exact Voronoi conditions.
    m_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), eps)
    u = np.where(m_bc, 0.0, u)
    v = np.where(m_bc, 1.0 - w_bc, v)
    w = np.where(m_bc, w_bc, w)

    m_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w_ac = d2 / np.maximum(d2 - d6, eps)
    u = np.where(m_ac, 1.0 - w_ac, u)
    v = np.where(m_ac, 0.0, v)
    w = np.where(m_ac, w_ac, w)

    m_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v_ab = d1 / np.maximum(d1 - d3, eps)
    u = np.where(m_ab, 1.0 - v_ab, u)
    v = np.where(m_ab, v_ab, v)
    w = np.where(m_ab, 0.0, w)

    m_c = (d6 >= 0.0) & (d5 <= d6)
    u = np.where(m_c, 0.0, u)
    v = np.where(m_c, 0.0, v)
    w = np.where(m_c, 1.0, w)

    m_b = (d3 >= 0.0) & (d4 <= d3)
    u = np.where(m_b, 0.0, u)
    v = np.where(m_b, 1.0, v)
    w = np.where(m_b, 0.0, w)

    m_a = (d1 <= 0.0) & (d2 <= 0.0)
    u = np.where(m_a, 1.0, u)
    v = np.where(m_a, 0.0, v)
    w = np.where(m_a, 0.0, w)

    closest = u[..., None] * a + v[..., None] * b + w[..., None] * c
    dist2 = np.sum((p - closest) ** 2, axis=-1)
    best_k = np.argmin(dist2, axis=1)
    row_i = np.arange(len(pts))

    best_tri = cand_idx[row_i, best_k].astype(np.int64)
    bary = np.stack([u[row_i, best_k], v[row_i, best_k], w[row_i, best_k]], axis=1).astype(np.float32)

    face_edges1 = verts[faces[best_tri, 1]] - verts[faces[best_tri, 0]]
    face_edges2 = verts[faces[best_tri, 2]] - verts[faces[best_tri, 0]]
    fn = np.cross(face_edges1, face_edges2)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    best_closest = closest[row_i, best_k]
    h_off = np.clip(np.sum((pts - best_closest) * fn, axis=1), -0.09, 0.09).astype(np.float32)
    return best_tri, bary, h_off


def _eval_bound_verts(
    binding: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_verts: np.ndarray,
    target_faces: np.ndarray,
) -> np.ndarray:
    """Evaluate bound query points on a view's posed base MHR mesh."""
    best_tri, bary, h_off = binding
    tri = target_verts[target_faces[best_tri]]  # (N, 3, 3)
    p_base = (bary[:, 0:1] * tri[:, 0, :]) + (bary[:, 1:2] * tri[:, 1, :]) + (bary[:, 2:3] * tri[:, 2, :])
    fn = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    return (p_base + h_off[:, None] * fn).astype(np.float32)


def _despike_scalar_field_subset(
    values: np.ndarray,
    faces: np.ndarray,
    subset_idx: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
    max_dev: float = 0.012,
    passes: int = 1,
) -> np.ndarray:
    """Clamp isolated 1-ring scalar spikes against neighbor medians while keeping smooth ridges intact."""
    if len(subset_idx) == 0 or len(values) == 0:
        return values
    n_verts = len(subset_idx)
    rev_map = np.full(int(faces.max()) + 1 if len(faces) else n_verts, -1, dtype=np.int64)
    if np.max(subset_idx) < len(rev_map):
        rev_map[subset_idx] = np.arange(n_verts, dtype=np.int64)
    else:
        rev_map = np.full(int(np.max(subset_idx)) + 1, -1, dtype=np.int64)
        rev_map[subset_idx] = np.arange(n_verts, dtype=np.int64)

    nbrs: List[List[int]] = [[] for _ in range(n_verts)]
    for f in faces:
        ids = [rev_map[int(v)] if int(v) < len(rev_map) else -1 for v in f]
        for ia, ib in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            if ia >= 0 and ib >= 0 and ia != ib:
                nbrs[ia].append(ib)
                nbrs[ib].append(ia)

    out = values.copy()
    for _ in range(passes):
        nxt = out.copy()
        for i in range(n_verts):
            if active_mask is not None and not active_mask[i]:
                continue
            nb = [n for n in nbrs[i] if active_mask is None or active_mask[n]]
            if len(nb) < 2:
                continue
            med = float(np.median(out[nb]))
            nxt[i] = float(np.clip(out[i], med - max_dev, med + max_dev))
        out = nxt
    return out


def _resolve_outer_view_camera(
    inner_verts_a: np.ndarray,
    inner_faces: np.ndarray,
    joints_a: np.ndarray,
    view: SegmentedView,
    rec: Dict[str, Any],
    fitted_camera: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resolve OpenCV camera (R_v, t_v, focal) aligning canonical/view mesh to view."""
    if fitted_camera is not None and len(fitted_camera) == 3:
        return (
            np.asarray(fitted_camera[0], dtype=np.float32),
            np.asarray(fitted_camera[1], dtype=np.float32).reshape(3),
            float(fitted_camera[2]),
        )
    out = rec.get("output") or {}
    if isinstance(rec, dict) and "camera" in rec and rec["camera"] is not None:
        cam = rec["camera"]
        if len(cam) == 3:
            return (
                np.asarray(cam[0], dtype=np.float32),
                np.asarray(cam[1], dtype=np.float32).reshape(3),
                float(cam[2]),
            )
    if isinstance(out, dict) and "fitted_camera" in out and out["fitted_camera"] is not None:
        cam = out["fitted_camera"]
        if len(cam) == 3:
            return (
                np.asarray(cam[0], dtype=np.float32),
                np.asarray(cam[1], dtype=np.float32).reshape(3),
                float(cam[2]),
            )

    person_mask = np.asarray(view.person_mask, dtype=bool)
    body_mask = np.asarray(view.body_mask, dtype=bool)
    ref_mask = body_mask if np.any(body_mask) else person_mask
    img_h, img_w = ref_mask.shape[:2]
    r_v = _rotation_y_and_cv_flip(view.azimuth_deg)
    focal = _extract_focal_length(out, img_h, img_w)

    pred_cam_t = out.get("pred_cam_t") if isinstance(out, dict) else None
    if pred_cam_t is not None and np.any(ref_mask):
        t_cand = np.asarray(pred_cam_t, dtype=np.float32).reshape(3)
        if np.isfinite(t_cand).all():
            verts_cam = (inner_verts_a @ r_v.T) + t_cand.reshape(1, 3)
            if float(np.mean(verts_cam[:, 2] > 0.2)) >= 0.90:
                proj = _rasterize_binary_silhouette_fast(verts_cam, inner_faces, focal, img_h, img_w)
                iou = _compute_mask_iou(proj, ref_mask)
                area_p = int(np.sum(proj))
                recall = float(np.sum(proj & ref_mask) / max(area_p, 1)) if area_p > 0 else 0.0
                if iou >= 0.40 or (recall >= 0.80 and iou >= 0.25):
                    return r_v, t_cand, focal

    return _solve_initial_view_camera(inner_verts_a, joints_a, view, out if isinstance(out, dict) else {})


def _extract_raw_depth(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract optional 2D monocular depth map from inference record or output dict."""
    if not isinstance(rec, dict):
        return None
    out = rec.get("output") or {}
    for container in (rec, out if isinstance(out, dict) else {}):
        for key in ("moge2_depth", "depth_map"):
            val = container.get(key)
            if val is not None:
                arr = val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)
                arr_2d = np.asarray(arr, dtype=np.float32).squeeze()
                if arr_2d.ndim == 2 and arr_2d.size > 0:
                    return arr_2d
    return None


def _bilinear_sample_2d(grid: np.ndarray, u_px: np.ndarray, v_px: np.ndarray) -> np.ndarray:
    """Sample a 2D float32 array bilinearly at floating-point pixel coords (u, v)."""
    h, w = grid.shape
    uc = np.clip(u_px, 0.0, float(w - 1))
    vc = np.clip(v_px, 0.0, float(h - 1))
    x0 = np.floor(uc).astype(np.int64)
    y0 = np.floor(vc).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (uc - x0).astype(np.float32)
    fy = (vc - y0).astype(np.float32)

    v00 = grid[y0, x0]
    v10 = grid[y0, x1]
    v01 = grid[y1, x0]
    v11 = grid[y1, x1]
    top = v00 * (1.0 - fx) + v10 * fx
    bot = v01 * (1.0 - fx) + v11 * fx
    val = top * (1.0 - fy) + bot * fy
    oob = (u_px < -0.5) | (u_px > float(w) - 0.5) | (v_px < -0.5) | (v_px > float(h) - 0.5)
    return np.where(oob, -1.0, val).astype(np.float32)


def _compute_view_hull_and_sil_targets(
    base_mov: np.ndarray,
    norm_mov: np.ndarray,
    r_v: np.ndarray,
    t_v: np.ndarray,
    focal: float,
    person_mask_w: np.ndarray,
    base_zbuf: np.ndarray,
    d_grid: np.ndarray,
    extremity_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-vertex visual-hull upper bound and unoccluded silhouette limb targets for one view."""
    n_mov = len(base_mov)
    ren_h, ren_w = person_mask_w.shape
    d_max_v = np.full(n_mov, float(d_grid[-1]), dtype=np.float32)
    d_sil_v = np.zeros(n_mov, dtype=np.float32)
    w_sil_v = np.zeros(n_mov, dtype=np.float32)
    if n_mov == 0 or not np.any(person_mask_w):
        return d_max_v, d_sil_v, w_sil_v

    mask_u8 = person_mask_w.astype(np.uint8)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    sdf = (dist_in - dist_out).astype(np.float32)

    p_cam0 = (base_mov @ r_v.T) + t_v.reshape(1, 3)
    n_cam = norm_mov @ r_v.T
    z0 = np.maximum(p_cam0[:, 2], 0.1)
    u0 = focal * (p_cam0[:, 0] / z0) + ren_w / 2.0
    v0 = focal * (p_cam0[:, 1] / z0) + ren_h / 2.0

    # Measure 2D silhouette motion sensitivity per unit outward normal displacement.
    p_step = p_cam0 + 0.02 * n_cam
    zs = np.maximum(p_step[:, 2], 0.1)
    us = focal * (p_step[:, 0] / zs) + ren_w / 2.0
    vs = focal * (p_step[:, 1] / zs) + ren_h / 2.0
    px_per_m = focal / z0
    sin_angle = np.hypot((us - u0) / 0.02, (vs - v0) / 0.02) / np.maximum(px_per_m, 1.0)

    # Sample SDF along ray base_mov[i] + d * norm_mov[i] across d_grid.
    pts_d_cam = p_cam0[:, None, :] + d_grid[None, :, None] * n_cam[:, None, :]
    zd = np.maximum(pts_d_cam[..., 2], 0.1)
    ud = focal * (pts_d_cam[..., 0] / zd) + ren_w / 2.0
    vd = focal * (pts_d_cam[..., 1] / zd) + ren_h / 2.0
    sdf_samples = _bilinear_sample_2d(sdf, ud, vd)  # (N_mov, K)

    # Find first exit where sdf transitions >= margin -> < margin while moving laterally into mask boundary.
    margin = 0.25
    active_lat = (sin_angle >= 0.15) & (p_cam0[:, 2] > 0.2)
    s_adj = sdf_samples - margin
    inside_any = s_adj >= 0.0
    exits = inside_any[:, :-1] & (~inside_any[:, 1:])

    has_exit = np.any(exits, axis=1) & active_lat
    if np.any(has_exit):
        first_k = np.argmax(exits[has_exit], axis=1)
        rows_e = np.where(has_exit)[0]
        s0 = s_adj[rows_e, first_k]
        s1 = s_adj[rows_e, first_k + 1]
        frac = np.clip(s0 / np.maximum(s0 - s1, 1e-6), 0.0, 1.0)
        d_exit = d_grid[first_k] + frac * (d_grid[first_k + 1] - d_grid[first_k])
        d_max_v[rows_e] = np.maximum(float(-INNER_BARRIER_M), d_exit.astype(np.float32))

    # Smooth ramp outside mask instead of discontinuous +60/-3 mm cliff jump.
    sdf0 = sdf_samples[:, 0]
    always_out = active_lat & (~np.any(sdf_samples >= 0.0, axis=1)) & (sdf_samples[:, -1] < sdf0 - 0.1)
    if np.any(always_out):
        out_t = np.clip(-sdf0[always_out] / 3.0, 0.0, 1.0)
        d_max_v[always_out] = (-float(INNER_BARRIER_M) * out_t).astype(np.float32)

    # Unoccluded silhouette limb attraction toward person_mask contour (strictly avoiding +/-45 deg corners).
    z_min_win = cv2.erode(
        np.where(base_zbuf > 0.0, base_zbuf, 1e6).astype(np.float32),
        np.ones((5, 5), dtype=np.uint8),
    )
    ui0 = np.clip(np.round(u0).astype(np.int64), 0, ren_w - 1)
    vi0 = np.clip(np.round(v0).astype(np.int64), 0, ren_h - 1)
    z_occ = z_min_win[vi0, ui0]
    unoccluded = (z0 <= z_occ + 0.05) | (z_occ > 1e5)

    abs_nz = np.abs(n_cam[:, 2])
    limb_ok = has_exit & unoccluded & (abs_nz < 0.35) & (sin_angle >= 0.25) & (sdf0 >= -2.5)
    if extremity_mask is not None and len(extremity_mask) == n_mov:
        limb_ok = limb_ok & (~np.asarray(extremity_mask, dtype=bool))
    if np.any(limb_ok):
        w_limb = (((0.35 - abs_nz[limb_ok]) / 0.35) ** 2) * np.clip((sin_angle[limb_ok] - 0.25) / 0.30, 0.0, 1.0)
        raw_sil = np.clip(d_max_v[limb_ok], float(-INNER_BARRIER_M), float(d_grid[-1]))
        # Smooth distance-tapered saturation suppresses spikes on noisy 20px mask lobes.
        pos_sil = np.maximum(raw_sil, 0.0)
        tapered_pos = 0.045 * np.tanh(pos_sil / 0.045)
        d_sil_v[limb_ok] = np.where(raw_sil >= 0.0, tapered_pos, raw_sil).astype(np.float32)
        w_sil_v[limb_ok] = w_limb.astype(np.float32)

    return d_max_v, d_sil_v, w_sil_v


def _align_region_depth_map(
    raw_d_w: np.ndarray,
    mhr_zbuf: np.ndarray,
    reg_mask: np.ndarray,
    pmask_w: np.ndarray,
    inner_verts_a: np.ndarray,
    joints_a: np.ndarray,
    r_v: np.ndarray,
    t_v: np.ndarray,
    focal_w: float,
) -> np.ndarray:
    """Align raw depth per torso/limb region and blend smoothly across the hip boundary."""
    ren_h, ren_w = reg_mask.shape
    y_foot = float(np.min(inner_verts_a[:, 1]))
    if len(joints_a) > NECK_BASE_JOINT:
        y_neck = float(joints_a[NECK_BASE_JOINT, 1])
    else:
        y_neck = float(np.max(inner_verts_a[:, 1])) * 0.82
    y_hip = y_foot + 0.52 * (y_neck - y_foot)

    hip_world = np.array(
        [[float(np.mean(inner_verts_a[:, 0])), y_hip, float(np.mean(inner_verts_a[:, 2]))]],
        dtype=np.float32,
    )
    hip_cam = (hip_world @ r_v.T) + t_v.reshape(1, 3)
    if float(hip_cam[0, 2]) > 0.2:
        v_hip = float(focal_w * (hip_cam[0, 1] / hip_cam[0, 2]) + ren_h / 2.0)
    else:
        ys_reg = np.nonzero(reg_mask)[0]
        v_hip = float(np.median(ys_reg)) if len(ys_reg) > 0 else float(ren_h * 0.55)

    # Reconstruct 3D world Y on valid mhr_zbuf pixels so region split is camera-tilt invariant.
    fg_y, fg_x = np.nonzero(reg_mask)
    z_fg = mhr_zbuf[fg_y, fg_x]
    u_fg = fg_x.astype(np.float32) + 0.5 - (ren_w / 2.0)
    v_fg = fg_y.astype(np.float32) + 0.5 - (ren_h / 2.0)
    p_cam_fg = np.stack([(u_fg / focal_w) * z_fg, (v_fg / focal_w) * z_fg, z_fg], axis=1)
    p_world_fg = (p_cam_fg - t_v.reshape(1, 3)) @ r_v
    upper_fg = p_world_fg[:, 1] >= (y_hip - 0.03)
    lower_fg = p_world_fg[:, 1] <= (y_hip + 0.03)

    if int(np.sum(upper_fg)) >= 30 and int(np.sum(lower_fg)) >= 30:
        upper_mask = np.zeros_like(reg_mask, dtype=bool)
        lower_mask = np.zeros_like(reg_mask, dtype=bool)
        upper_mask[fg_y[upper_fg], fg_x[upper_fg]] = True
        lower_mask[fg_y[lower_fg], fg_x[lower_fg]] = True

        aligned_upper = align_metric_depth_map(raw_d_w, mhr_zbuf, upper_mask)
        aligned_lower = align_metric_depth_map(raw_d_w, mhr_zbuf, lower_mask)

        rows_grid = np.arange(ren_h, dtype=np.float32)[:, None]
        if float(np.median(fg_y[upper_fg])) < float(np.median(fg_y[lower_fg])):
            w_upper_col = np.clip(0.5 + (v_hip - rows_grid) / 24.0, 0.0, 1.0)
        else:
            w_upper_col = np.clip(0.5 + (rows_grid - v_hip) / 24.0, 0.0, 1.0)
        w_upper = np.broadcast_to(w_upper_col, reg_mask.shape).astype(np.float32)

        both_ok = np.isfinite(aligned_upper) & np.isfinite(aligned_lower)
        aligned = np.where(
            both_ok,
            w_upper * aligned_upper + (1.0 - w_upper) * aligned_lower,
            np.where(np.isfinite(aligned_upper), aligned_upper, aligned_lower),
        ).astype(np.float32)
        return np.where(pmask_w, aligned, np.nan).astype(np.float32)

    return align_metric_depth_map(raw_d_w, mhr_zbuf, reg_mask)


def _prepare_view_payloads(
    inner_verts_a: np.ndarray,
    inner_faces: np.ndarray,
    joints_a: np.ndarray,
    base_verts: np.ndarray,
    outer_faces: np.ndarray,
    v_normals: np.ndarray,
    mov_idx: np.ndarray,
    segmented_views: Sequence[SegmentedView],
    raw_results: Sequence[Dict[str, Any]],
    prefer_pytorch3d: bool,
    fitted_cameras: Optional[Sequence[Tuple[np.ndarray, np.ndarray, float]]] = None,
    fitted_view_verts: Optional[Sequence[np.ndarray]] = None,
    collar_cap: Optional[np.ndarray] = None,
    extremity_mask_mov: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Prepare per-view cameras, work-res masks, aligned MoGe2 depths, hull caps, and limb targets."""
    n_mov = len(mov_idx)
    d_grid = np.linspace(-INNER_BARRIER_M, 0.060, 49, dtype=np.float32)
    d_hull_max = np.full(n_mov, float(d_grid[-1]), dtype=np.float32)
    if collar_cap is not None and len(collar_cap) == n_mov:
        d_hull_max = np.minimum(d_hull_max, collar_cap.astype(np.float32))
    sil_w_sum = np.zeros(n_mov, dtype=np.float32)
    sil_d_sum = np.zeros(n_mov, dtype=np.float32)
    max_obs_cos = np.zeros(len(base_verts), dtype=np.float32)

    binding: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if fitted_view_verts is not None:
        has_posed_diff = any(
            isinstance(vv, np.ndarray)
            and vv.shape == inner_verts_a.shape
            and float(np.max(np.abs(vv - inner_verts_a))) > 1e-5
            for vv in fitted_view_verts
            if vv is not None
        )
        if has_posed_diff and len(inner_faces) > 0:
            binding = _closest_triangle_barycentric(base_verts, inner_verts_a, inner_faces)

    payloads: List[Dict[str, Any]] = []
    has_any_depth = False

    for idx, sv in enumerate(segmented_views):
        pmask = np.asarray(sv.person_mask, dtype=bool)
        if pmask.ndim != 2 or pmask.size == 0 or not np.any(pmask):
            continue
        orig_h, orig_w = pmask.shape
        bmask = np.asarray(sv.body_mask, dtype=bool)
        if bmask.shape != pmask.shape:
            bmask = pmask.copy()

        vi_key = int(getattr(sv, "view_index", idx))
        fit_cam = (
            fitted_cameras[vi_key]
            if (fitted_cameras is not None and 0 <= vi_key < len(fitted_cameras))
            else None
        )
        rec = _resolve_raw_record(sv, idx, list(raw_results))
        r_v, t_v, focal_orig = _resolve_outer_view_camera(
            inner_verts_a, inner_faces, joints_a, sv, rec, fitted_camera=fit_cam
        )

        v_inner_v = inner_verts_a
        v_base_v = base_verts
        v_norm_v = v_normals
        base_v_payload: Optional[np.ndarray] = None
        norm_v_payload: Optional[np.ndarray] = None
        if (
            binding is not None
            and fitted_view_verts is not None
            and 0 <= vi_key < len(fitted_view_verts)
            and fitted_view_verts[vi_key] is not None
        ):
            cand_inner = np.asarray(fitted_view_verts[vi_key], dtype=np.float32)
            if cand_inner.shape == inner_verts_a.shape and float(np.max(np.abs(cand_inner - inner_verts_a))) > 1e-5:
                v_inner_v = cand_inner
                v_base_v = _eval_bound_verts(binding, v_inner_v, inner_faces)
                v_norm_v = _compute_oriented_outward_normals(v_base_v, outer_faces)
                base_v_payload = v_base_v
                norm_v_payload = v_norm_v

        base_mov = v_base_v[mov_idx]
        norm_mov = v_norm_v[mov_idx]

        max_dim = max(orig_h, orig_w)
        scale_w = min(1.0, 512.0 / float(max_dim)) if max_dim > 512 else 1.0
        ren_h = max(32, int(round(orig_h * scale_w)))
        ren_w = max(32, int(round(orig_w * scale_w)))
        focal_w = focal_orig * 0.5 * ((ren_w / float(orig_w)) + (ren_h / float(orig_h)))

        if ren_h != orig_h or ren_w != orig_w:
            pmask_w = cv2.resize(pmask.astype(np.uint8), (ren_w, ren_h), interpolation=cv2.INTER_NEAREST) > 0
            bmask_w = cv2.resize(bmask.astype(np.uint8), (ren_w, ren_h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            pmask_w = pmask
            bmask_w = bmask

        inner_cam = (v_inner_v @ r_v.T) + t_v.reshape(1, 3)
        mhr_zbuf, _ = render_view_depth_buffer(
            inner_cam, inner_faces, focal_w, ren_w, ren_h, prefer_pytorch3d=prefer_pytorch3d
        )
        base_cam = (v_base_v @ r_v.T) + t_v.reshape(1, 3)
        base_zbuf, _ = render_view_depth_buffer(
            base_cam, outer_faces, focal_w, ren_w, ren_h, prefer_pytorch3d=prefer_pytorch3d
        )

        # Track maximum orthogonal view cosine per vertex for 45-deg grazing regularization.
        cam_dir_world = -r_v[2]
        max_obs_cos = np.maximum(max_obs_cos, np.maximum(0.0, v_norm_v @ cam_dir_world))

        d_max_v, d_sil_v, w_sil_v = _compute_view_hull_and_sil_targets(
            base_mov=base_mov,
            norm_mov=norm_mov,
            r_v=r_v,
            t_v=t_v,
            focal=focal_w,
            person_mask_w=pmask_w,
            base_zbuf=base_zbuf,
            d_grid=d_grid,
            extremity_mask=extremity_mask_mov,
        )
        d_hull_max = np.minimum(d_hull_max, d_max_v)
        sil_d_sum += w_sil_v * d_sil_v
        sil_w_sum += w_sil_v

        aligned_depth_clean: Optional[np.ndarray] = None
        aligned_depth_valid: Optional[np.ndarray] = None
        raw_d = _extract_raw_depth(rec)
        if raw_d is not None:
            if raw_d.shape != (ren_h, ren_w):
                raw_d_w = cv2.resize(raw_d, (ren_w, ren_h), interpolation=cv2.INTER_NEAREST)
            else:
                raw_d_w = raw_d
            reg_mask = pmask_w & (mhr_zbuf > 0.0) & np.isfinite(raw_d_w)
            if int(np.sum(reg_mask & bmask_w)) >= 30:
                reg_mask = reg_mask & bmask_w
            eroded = cv2.erode(reg_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
            if int(np.sum(eroded)) >= 50:
                reg_mask = eroded
            if int(np.sum(reg_mask)) >= 6:
                aligned = _align_region_depth_map(
                    raw_d_w=raw_d_w,
                    mhr_zbuf=mhr_zbuf,
                    reg_mask=reg_mask,
                    pmask_w=pmask_w,
                    inner_verts_a=v_inner_v,
                    joints_a=joints_a,
                    r_v=r_v,
                    t_v=t_v,
                    focal_w=focal_w,
                )
                # Reject boundary silhouette halos and sharp foreground/background depth cliffs.
                er_p5 = cv2.erode(pmask_w.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
                er_p3 = cv2.erode(pmask_w.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
                pmask_count = max(1, int(np.sum(pmask_w)))
                if int(np.sum(er_p5)) >= 0.50 * pmask_count:
                    depth_roi = er_p5
                elif int(np.sum(er_p3)) >= 0.45 * pmask_count:
                    depth_roi = er_p3
                else:
                    depth_roi = pmask_w

                fin_valid = depth_roi & np.isfinite(aligned) & (aligned > 0.1)
                depth_conf = np.zeros(aligned.shape, dtype=np.float32)
                if np.any(fin_valid):
                    safe_a = np.where(np.isfinite(aligned), aligned, 0.0).astype(np.float32)
                    gx = cv2.Sobel(safe_a, cv2.CV_32F, 1, 0, ksize=3) * 0.25
                    gy = cv2.Sobel(safe_a, cv2.CV_32F, 0, 1, ksize=3) * 0.25
                    grad_mag = np.hypot(gx, gy)
                    grad_ok = grad_mag < 0.08
                    if int(np.sum(fin_valid & grad_ok)) >= 0.45 * int(np.sum(fin_valid)):
                        fin_valid = fin_valid & grad_ok
                    dist_in = cv2.distanceTransform(pmask_w.astype(np.uint8), cv2.DIST_L2, 3)
                    edge_w = np.clip((dist_in - 1.5) / 4.5, 0.0, 1.0).astype(np.float32)
                    grad_w = np.exp(-((grad_mag / 0.035) ** 2)).astype(np.float32)
                    depth_conf = np.where(fin_valid, edge_w * grad_w, 0.0).astype(np.float32)

                if np.any(depth_conf > 0.05):
                    has_any_depth = True
                    aligned_depth_valid = depth_conf
                    aligned_depth_clean = np.where(fin_valid, aligned, 0.0).astype(np.float32)

        payloads.append(
            {
                "view": sv,
                "r_v": r_v,
                "t_v": t_v,
                "focal_orig": focal_orig,
                "focal_w": focal_w,
                "orig_shape": (orig_h, orig_w),
                "work_shape": (ren_h, ren_w),
                "pmask_w": pmask_w,
                "base_zbuf": base_zbuf,
                "base_v": base_v_payload,
                "norm_v": norm_v_payload,
                "aligned_depth_clean": aligned_depth_clean,
                "aligned_depth_valid": aligned_depth_valid,
            }
        )

    d_hull_max = np.maximum(d_hull_max, float(-INNER_BARRIER_M)).astype(np.float32)
    sil_target = np.where(sil_w_sum > 1e-6, sil_d_sum / np.maximum(sil_w_sum, 1e-6), 0.0).astype(np.float32)
    sil_weight = np.clip(sil_w_sum, 0.0, 1.0).astype(np.float32)
    sil_target = _despike_scalar_field_subset(
        sil_target, outer_faces, mov_idx, active_mask=(sil_weight > 0.08), max_dev=0.012, passes=1
    )
    sil_target = np.clip(sil_target, float(-INNER_BARRIER_M), d_hull_max).astype(np.float32)
    return payloads, d_hull_max, sil_target, sil_weight, max_obs_cos, has_any_depth


def _sample_aligned_depth_loss_torch(
    verts_w: torch.Tensor,
    normals_w: torch.Tensor,
    r_t: torch.Tensor,
    t_t: torch.Tensor,
    focal: float,
    ren_h: int,
    ren_w: int,
    depth_t: torch.Tensor,
    valid_t: torch.Tensor,
    base_zbuf_t: torch.Tensor,
    mov_idx_t: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable multi-view SmoothL1 depth loss against aligned MoGe2 depth map."""
    v_mov = verts_w[mov_idx_t]
    n_mov = normals_w[mov_idx_t]
    v_cam = (v_mov @ r_t.T) + t_t.unsqueeze(0)
    n_cam = n_mov @ r_t.T

    z_cam = v_cam[:, 2]
    z_safe = torch.clamp(z_cam, min=0.1)
    u_px = float(focal) * (v_cam[:, 0] / z_safe) + (ren_w / 2.0)
    v_px = float(focal) * (v_cam[:, 1] / z_safe) + (ren_h / 2.0)

    grid_x = (2.0 * (u_px + 0.5) / float(ren_w)) - 1.0
    grid_y = (2.0 * (v_px + 0.5) / float(ren_h)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)

    sampled_d = F.grid_sample(depth_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False).view(-1)
    sampled_val = F.grid_sample(valid_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False).view(-1)
    sampled_occ = F.grid_sample(base_zbuf_t, grid, mode="nearest", padding_mode="zeros", align_corners=False).view(-1)

    front_cos = torch.clamp(-n_cam[:, 2], min=0.0)
    unocc = ((sampled_occ <= 0.0) | (z_cam <= sampled_occ + 0.04)).float()
    in_bounds = (
        (z_cam > 0.2) & (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    ).float()
    vis_w = (front_cos ** 2) * torch.clamp(sampled_val, min=0.0, max=1.0) * unocc * in_bounds
    diff_ok = (torch.abs(z_cam - sampled_d) < 0.065).float()
    vis_w = vis_w * diff_ok

    # Multiply beta=0.01 so small residuals stay quadratic in m^2 while outliers clamp cleanly.
    per_vert_loss = 0.01 * F.smooth_l1_loss(z_cam, sampled_d, reduction="none", beta=0.01)
    return torch.mean(vis_w * per_vert_loss)


def _optimize_outer_shell_displacements(
    base_verts: np.ndarray,
    outer_faces: np.ndarray,
    v_normals: np.ndarray,
    mov_idx: np.ndarray,
    payloads: List[Dict[str, Any]],
    d_hull_max: np.ndarray,
    sil_target: np.ndarray,
    sil_weight: np.ndarray,
    max_obs_cos: np.ndarray,
    prefer_pytorch3d: bool,
    collar_ring_boost: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """Optimize normal displacements with PyTorch3D SoftSilhouette+RGB-D or CPU fallback."""
    n_verts = len(base_verts)
    n_mov = len(mov_idx)
    if n_mov == 0 or not payloads:
        return np.zeros(n_verts, dtype=np.float32), None

    src_np, dst_np, inv_deg_np = _build_directed_edge_laplacian(n_verts, outer_faces)
    grazing = np.clip((0.92 - max_obs_cos) / 0.35, 0.0, 1.0) ** 2
    w_smooth_np = (1.0 + 3.0 * grazing).astype(np.float32)
    if collar_ring_boost is not None and len(collar_ring_boost) == n_verts:
        w_smooth_np = (w_smooth_np * collar_ring_boost).astype(np.float32)
    w_anchor_np = (0.015 + 0.06 * grazing).astype(np.float32)

    p3d_mod = None
    p3d_warning: Optional[str] = None
    if prefer_pytorch3d:
        try:
            from pytorch3d.renderer import (  # type: ignore[import-not-found]
                MeshRasterizer,
                MeshRenderer,
                RasterizationSettings,
                SoftSilhouetteShader,
            )
            from pytorch3d.structures import Meshes  # type: ignore[import-not-found]
            from pytorch3d.utils import cameras_from_opencv_projection  # type: ignore[import-not-found]

            p3d_mod = (
                MeshRasterizer,
                MeshRenderer,
                RasterizationSettings,
                SoftSilhouetteShader,
                Meshes,
                cameras_from_opencv_projection,
            )
        except (ImportError, OSError, RuntimeError):
            p3d_mod = None

    def _run_solver(use_p3d: bool) -> np.ndarray:
        dev = torch.device("cuda" if (use_p3d and torch.cuda.is_available()) else "cpu")
        base_t = torch.tensor(base_verts, dtype=torch.float32, device=dev)
        norm_t = torch.tensor(v_normals, dtype=torch.float32, device=dev)
        faces_t = torch.tensor(outer_faces, dtype=torch.int64, device=dev)
        mov_idx_t = torch.tensor(mov_idx, dtype=torch.int64, device=dev)

        d_hull_t = torch.tensor(d_hull_max, dtype=torch.float32, device=dev)
        d_sil_t = torch.tensor(sil_target, dtype=torch.float32, device=dev)
        w_sil_t = torch.tensor(sil_weight, dtype=torch.float32, device=dev)
        w_smooth_t = torch.tensor(w_smooth_np, dtype=torch.float32, device=dev)
        w_anchor_t = torch.tensor(w_anchor_np, dtype=torch.float32, device=dev)
        edge_src_t = torch.tensor(src_np, dtype=torch.int64, device=dev)
        edge_dst_t = torch.tensor(dst_np, dtype=torch.int64, device=dev)
        edge_w_t = 0.5 * (w_smooth_t[edge_src_t] + w_smooth_t[edge_dst_t])
        inv_deg_t = torch.tensor(inv_deg_np, dtype=torch.float32, device=dev)

        view_tensors: List[Dict[str, Any]] = []
        for p in payloads:
            ren_h, ren_w = p["work_shape"]
            r_t = torch.tensor(p["r_v"], dtype=torch.float32, device=dev)
            t_t = torch.tensor(p["t_v"], dtype=torch.float32, device=dev)
            base_zbuf_t = torch.tensor(p["base_zbuf"], dtype=torch.float32, device=dev).view(1, 1, ren_h, ren_w)
            depth_t = (
                torch.tensor(p["aligned_depth_clean"], dtype=torch.float32, device=dev).view(1, 1, ren_h, ren_w)
                if p["aligned_depth_clean"] is not None
                else None
            )
            valid_t = (
                torch.tensor(p["aligned_depth_valid"], dtype=torch.float32, device=dev).view(1, 1, ren_h, ren_w)
                if p["aligned_depth_valid"] is not None
                else None
            )
            base_v_t = (
                torch.tensor(p["base_v"], dtype=torch.float32, device=dev)
                if p.get("base_v") is not None
                else None
            )
            norm_v_t = (
                torch.tensor(p["norm_v"], dtype=torch.float32, device=dev)
                if p.get("norm_v") is not None
                else None
            )
            entry: Dict[str, Any] = {
                "r_t": r_t,
                "t_t": t_t,
                "focal_w": p["focal_w"],
                "ren_h": ren_h,
                "ren_w": ren_w,
                "base_zbuf_t": base_zbuf_t,
                "depth_t": depth_t,
                "valid_t": valid_t,
                "base_v_t": base_v_t,
                "norm_v_t": norm_v_t,
            }
            if use_p3d and p3d_mod is not None:
                MeshRasterizer, MeshRenderer, RasterizationSettings, SoftSilhouetteShader, _, cams_fn = p3d_mod
                max_d = max(ren_h, ren_w)
                p3d_scale = min(1.0, 256.0 / float(max_d)) if max_d > 256 else 1.0
                sh = max(32, int(round(ren_h * p3d_scale)))
                sw = max(32, int(round(ren_w * p3d_scale)))
                sf = p["focal_w"] * 0.5 * ((sw / float(ren_w)) + (sh / float(ren_h)))
                pm_ren = (
                    cv2.resize(p["pmask_w"].astype(np.float32), (sw, sh), interpolation=cv2.INTER_NEAREST)
                    if (sh != ren_h or sw != ren_w)
                    else p["pmask_w"].astype(np.float32)
                )
                k_mat = torch.tensor(
                    [[[sf, 0.0, sw / 2.0], [0.0, sf, sh / 2.0], [0.0, 0.0, 1.0]]],
                    dtype=torch.float32,
                    device=dev,
                )
                im_sz = torch.tensor([[sh, sw]], dtype=torch.float32, device=dev)
                cams = cams_fn(r_t.unsqueeze(0), t_t.unsqueeze(0), k_mat, im_sz)
                raster_cfg = RasterizationSettings(
                    image_size=(sh, sw),
                    blur_radius=math.log(1.0 / 1e-4 - 1.0) * 1e-4,
                    faces_per_pixel=8,
                )
                entry["p3d_renderer"] = MeshRenderer(
                    rasterizer=MeshRasterizer(raster_settings=raster_cfg),
                    shader=SoftSilhouetteShader(),
                )
                entry["p3d_cams"] = cams
                entry["p3d_target"] = torch.tensor(pm_ren, dtype=torch.float32, device=dev)
            view_tensors.append(entry)

        min_barrier_t = torch.full_like(d_hull_t, float(-INNER_BARRIER_M))
        init_full = torch.zeros(n_verts, dtype=torch.float32, device=dev)
        init_seed = torch.tensor(
            np.clip(sil_target * sil_weight, float(-INNER_BARRIER_M), d_hull_max).astype(np.float32),
            dtype=torch.float32,
            device=dev,
        )
        init_full.index_copy_(0, mov_idx_t, init_seed)
        with torch.no_grad():
            for _ in range(2):
                ld_i = _apply_graph_laplacian(init_full, edge_src_t, edge_dst_t, inv_deg_t)
                init_full = init_full - 0.45 * ld_i
            init_mov = torch.clamp(init_full[mov_idx_t], min=min_barrier_t, max=d_hull_t)

        d_mov = init_mov.clone().detach().requires_grad_(True)
        n_steps = 45 if use_p3d else 65
        optimizer = torch.optim.Adam([d_mov], lr=0.004)

        for step_idx in range(n_steps):
            if step_idx == int(n_steps * 0.65):
                for grp in optimizer.param_groups:
                    grp["lr"] = 0.0015
            optimizer.zero_grad()

            d_full = torch.zeros(n_verts, dtype=torch.float32, device=dev)
            d_full = d_full.index_copy(0, mov_idx_t, d_mov)
            cur_verts = base_t + d_full.unsqueeze(1) * norm_t

            l_depth = torch.zeros((), dtype=torch.float32, device=dev)
            n_depth_views = 0
            l_p3d_sil = torch.zeros((), dtype=torch.float32, device=dev)
            has_shared_p3d = use_p3d and p3d_mod is not None and all(vt["base_v_t"] is None for vt in view_tensors)
            shared_mesh_p3d = p3d_mod[4](verts=[cur_verts], faces=[faces_t]) if has_shared_p3d else None

            for vt in view_tensors:
                if vt["base_v_t"] is not None and vt["norm_v_t"] is not None:
                    v_verts = vt["base_v_t"] + d_full.unsqueeze(1) * vt["norm_v_t"]
                    v_norms = vt["norm_v_t"]
                else:
                    v_verts = cur_verts
                    v_norms = norm_t

                if vt["depth_t"] is not None and vt["valid_t"] is not None:
                    l_depth = l_depth + _sample_aligned_depth_loss_torch(
                        v_verts,
                        v_norms,
                        vt["r_t"],
                        vt["t_t"],
                        vt["focal_w"],
                        vt["ren_h"],
                        vt["ren_w"],
                        vt["depth_t"],
                        vt["valid_t"],
                        vt["base_zbuf_t"],
                        mov_idx_t,
                    )
                    n_depth_views += 1
                if use_p3d and p3d_mod is not None and "p3d_renderer" in vt:
                    v_mesh = shared_mesh_p3d if shared_mesh_p3d is not None else p3d_mod[4](verts=[v_verts], faces=[faces_t])
                    rgba = vt["p3d_renderer"](v_mesh, cameras=vt["p3d_cams"])
                    l_sil_v, l_edge_v = compute_soft_silhouette_and_edge_loss(rgba[0, :, :, 3], vt["p3d_target"])
                    l_p3d_sil = l_p3d_sil + (l_sil_v + 0.25 * l_edge_v)

            if n_depth_views > 0:
                l_depth = l_depth / float(n_depth_views)

            l_sil_limb = torch.mean(w_sil_t * ((d_mov - d_sil_t) ** 2))

            ld = _apply_graph_laplacian(d_full, edge_src_t, edge_dst_t, inv_deg_t)
            l2d = _apply_graph_laplacian(ld, edge_src_t, edge_dst_t, inv_deg_t)
            l_edge_diff = (
                torch.mean(edge_w_t * ((d_full[edge_src_t] - d_full[edge_dst_t]) ** 2))
                if len(edge_src_t) > 0
                else torch.zeros((), dtype=torch.float32, device=dev)
            )
            l_smooth = torch.mean(w_smooth_t * ((ld ** 2) + 0.5 * (l2d ** 2))) + 0.45 * l_edge_diff
            l_anchor = torch.mean(w_anchor_t * (d_full ** 2))
            l_bounds = torch.mean(
                F.relu(d_mov - d_hull_t) ** 2 + F.relu(min_barrier_t - d_mov) ** 2
            )

            total = 4.0 * l_depth + 2.5 * l_sil_limb + 1.6 * l_smooth + 0.08 * l_anchor + 10.0 * l_bounds
            if use_p3d and p3d_mod is not None and len(view_tensors) > 0:
                total = total + 0.03 * (l_p3d_sil / float(len(view_tensors)))

            total.backward()
            optimizer.step()
            with torch.no_grad():
                d_mov.clamp_(min=min_barrier_t, max=d_hull_t)

        d_mov_np = d_mov.clamp(min=min_barrier_t, max=d_hull_t).detach().cpu().numpy().astype(np.float32)
        d_mov_np = _despike_scalar_field_subset(d_mov_np, outer_faces, mov_idx, max_dev=0.014, passes=1)
        d_mov_np = np.clip(d_mov_np, -float(INNER_BARRIER_M), d_hull_max).astype(np.float32)

        d_final_np = np.zeros(n_verts, dtype=np.float32)
        d_final_np[mov_idx] = d_mov_np
        return d_final_np

    if p3d_mod is not None:
        try:
            return _run_solver(use_p3d=True), None
        except (RuntimeError, ValueError, TypeError) as exc:
            torch.cuda.empty_cache()
            p3d_warning = f"PyTorch3D outer-shell optimization failed ({exc}); used CPU depth/silhouette fallback."

    return _run_solver(use_p3d=False), p3d_warning


def _compute_collar_ring_limits(
    n_verts: int,
    outer_faces: np.ndarray,
    fixed_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-vertex max displacement cap & smoothness boost across 3 geodesic collar rings."""
    caps = np.full(n_verts, 0.060, dtype=np.float32)
    smooth_boost = np.ones(n_verts, dtype=np.float32)
    if n_verts == 0 or len(outer_faces) == 0 or not np.any(fixed_mask):
        return caps, smooth_boost

    nbrs: List[List[int]] = [[] for _ in range(n_verts)]
    for f in outer_faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        nbrs[a].extend((b, c))
        nbrs[b].extend((a, c))
        nbrs[c].extend((a, b))

    ring_caps = (0.022, 0.042, 0.054)
    ring_boosts = (2.2, 1.5, 1.15)
    visited = fixed_mask.copy()
    frontier = set(int(i) for i in np.where(fixed_mask)[0])
    for ring_cap, ring_boost in zip(ring_caps, ring_boosts):
        nxt_frontier: set[int] = set()
        for vi in frontier:
            for nb in nbrs[vi]:
                if not visited[nb]:
                    visited[nb] = True
                    nxt_frontier.add(nb)
        if not nxt_frontier:
            break
        nxt_idx = np.fromiter(nxt_frontier, dtype=np.int64)
        caps[nxt_idx] = np.minimum(caps[nxt_idx], float(ring_cap))
        smooth_boost[nxt_idx] = np.maximum(smooth_boost[nxt_idx], float(ring_boost))
        frontier = nxt_frontier

    return caps, smooth_boost


def _compute_extremity_limits(
    base_verts: np.ndarray,
    joints_a: Optional[np.ndarray] = None,
    head_vert_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify hand/finger/foot/toe (<= 4 mm) and wrist/ankle (<= 10 mm) extremity vertices."""
    n_verts = len(base_verts)
    caps = np.full(n_verts, 0.060, dtype=np.float32)
    smooth_boost = np.ones(n_verts, dtype=np.float32)
    hand_foot_mask = np.zeros(n_verts, dtype=bool)
    wrist_ankle_mask = np.zeros(n_verts, dtype=bool)
    if n_verts == 0:
        return caps, smooth_boost, hand_foot_mask, wrist_ankle_mask

    v_arr = np.asarray(base_verts, dtype=np.float32)
    y_min = float(np.min(v_arr[:, 1]))
    y_max = float(np.max(v_arr[:, 1]))
    height_m = max(y_max - y_min, 1e-4)
    y_rel = v_arr[:, 1] - y_min

    not_head = (
        ~np.asarray(head_vert_mask, dtype=bool)
        if (head_vert_mask is not None and len(head_vert_mask) == n_verts)
        else np.ones(n_verts, dtype=bool)
    )

    coord_hand_foot = np.zeros(n_verts, dtype=bool)
    coord_wrist_ankle = np.zeros(n_verts, dtype=bool)
    if height_m >= 0.45:
        foot_coord = (y_rel <= 0.045 * height_m) & not_head
        ankle_coord = (y_rel > 0.045 * height_m) & (y_rel <= 0.090 * height_m) & not_head
        abs_x = np.abs(v_arr[:, 0])
        hand_coord = (abs_x >= 0.41 * height_m) & (y_rel >= 0.25 * height_m) & not_head
        wrist_coord = (
            (abs_x >= 0.35 * height_m)
            & (abs_x < 0.41 * height_m)
            & (y_rel >= 0.25 * height_m)
            & not_head
        )
        coord_hand_foot |= foot_coord | hand_coord
        coord_wrist_ankle |= ankle_coord | wrist_coord

    joint_hand_foot = np.zeros(n_verts, dtype=bool)
    joint_wrist_ankle = np.zeros(n_verts, dtype=bool)
    if joints_a is not None and len(joints_a) > 0:
        j_arr = np.asarray(joints_a, dtype=np.float32)
        if j_arr.ndim == 2 and j_arr.shape[1] == 3:
            nonzero_joints = int(np.sum(np.linalg.norm(j_arr, axis=1) > 1e-4))
            sparse_test_joints = nonzero_joints <= 16

            # Wrist & hand/finger detection: canonical MHR-127 ((42,40),(78,76) + distal (41,40),(77,76))
            # with explicit MHR-70 ((41,8),(62,7)) fallback when <=70 joints or sparse 70-indexed mock.
            if len(j_arr) <= 70:
                wrist_pairs: Tuple[Tuple[int, int], ...] = (
                    (MHR70_R_WRIST, MHR70_R_ELBOW),
                    (MHR70_L_WRIST, MHR70_L_ELBOW),
                )
            else:
                wrist_pairs = MHR127_WRIST_ELBOW_PAIRS + MHR127_FOREARM_ELBOW_PAIRS
                if sparse_test_joints:
                    has_127_l = any(
                        wi < len(j_arr) and float(np.linalg.norm(j_arr[wi])) > 1e-4
                        for wi, _ in (MHR127_WRIST_ELBOW_PAIRS[1], MHR127_FOREARM_ELBOW_PAIRS[1])
                    )
                    if (not has_127_l) and MHR70_L_WRIST < len(j_arr) and float(np.linalg.norm(j_arr[MHR70_L_WRIST])) > 1e-4:
                        wrist_pairs = wrist_pairs + ((MHR70_L_WRIST, MHR70_L_ELBOW),)

            for wi, ei in wrist_pairs:
                if wi >= len(j_arr):
                    continue
                w_pos = j_arr[wi]
                if not (abs(float(w_pos[0])) > 0.02 or (sparse_test_joints and float(np.linalg.norm(w_pos)) > 1e-4)):
                    continue
                if ei < len(j_arr) and (
                    abs(float(j_arr[ei, 0])) > 0.01 or (sparse_test_joints and float(np.linalg.norm(j_arr[ei])) > 1e-4)
                ):
                    arm_dir = w_pos - j_arr[ei]
                    arm_norm = float(np.linalg.norm(arm_dir))
                else:
                    arm_dir = np.zeros(3, dtype=np.float32)
                    arm_norm = 0.0
                if arm_norm > 1e-4:
                    arm_unit = arm_dir / arm_norm
                else:
                    sign_x = 1.0 if float(w_pos[0]) >= 0.0 else -1.0
                    arm_unit = np.array([sign_x, 0.0, 0.0], dtype=np.float32)

                rel_w = v_arr - w_pos.reshape(1, 3)
                dist_w = np.linalg.norm(rel_w, axis=1)
                proj_w = rel_w @ arm_unit
                is_hand = (dist_w <= 0.24) & (proj_w >= 0.015) & not_head
                is_wrist = (dist_w <= 0.075) & (~is_hand) & not_head
                joint_hand_foot |= is_hand
                joint_wrist_ankle |= is_wrist

            # Ankle & foot/toe detection: canonical MHR-127 (4, 20), with MHR-70 (13, 14) or sparse mock fallback.
            if len(j_arr) <= 70:
                ankle_ids: Tuple[int, ...] = (MHR70_L_ANKLE, MHR70_R_ANKLE)
            else:
                ankle_ids = MHR127_ANKLE_JOINT_IDS
                if sparse_test_joints:
                    has_127_ankles = any(
                        ai < len(j_arr) and float(np.linalg.norm(j_arr[ai])) > 1e-4
                        for ai in MHR127_ANKLE_JOINT_IDS
                    )
                    if not has_127_ankles:
                        ankle_ids = (MHR70_R_ELBOW, MHR70_L_ANKLE, MHR70_R_ANKLE)

            for ai in ankle_ids:
                if ai >= len(j_arr):
                    continue
                a_pos = j_arr[ai]
                if not (abs(float(a_pos[0])) > 0.01 or (sparse_test_joints and float(np.linalg.norm(a_pos)) > 1e-4)):
                    continue
                if sparse_test_joints and ai == MHR70_R_ELBOW and float(a_pos[1]) > y_min + 0.25 * height_m:
                    continue
                rel_a = v_arr - a_pos.reshape(1, 3)
                dist_a = np.linalg.norm(rel_a, axis=1)
                is_foot = (
                    (dist_a <= 0.24)
                    & ((v_arr[:, 1] <= float(a_pos[1]) - 0.012) | (np.abs(v_arr[:, 2] - float(a_pos[2])) >= 0.045))
                    & not_head
                )
                is_ankle = (dist_a <= 0.075) & (~is_foot) & not_head
                joint_hand_foot |= is_foot
                joint_wrist_ankle |= is_ankle

    joint_wrist_ankle &= ~joint_hand_foot
    hand_foot_mask = joint_hand_foot | (coord_hand_foot & ~joint_wrist_ankle)
    wrist_ankle_mask = (joint_wrist_ankle | coord_wrist_ankle) & ~hand_foot_mask
    caps[wrist_ankle_mask] = np.minimum(caps[wrist_ankle_mask], 0.010)
    caps[hand_foot_mask] = np.minimum(caps[hand_foot_mask], 0.004)
    smooth_boost[wrist_ankle_mask] = np.maximum(smooth_boost[wrist_ankle_mask], 2.5)
    smooth_boost[hand_foot_mask] = np.maximum(smooth_boost[hand_foot_mask], 4.0)
    return caps, smooth_boost, hand_foot_mask, wrist_ankle_mask


def _enforce_edge_stretch_guard(
    base_verts: np.ndarray,
    outer_faces: np.ndarray,
    v_normals: np.ndarray,
    d_final: np.ndarray,
    fixed_mask: Optional[np.ndarray] = None,
    max_ratio: float = 1.35,
    max_grad: float = 0.65,
) -> np.ndarray:
    """Clamp normal displacements for edge stretch <= max_ratio and 1-ring cliff |d_u-d_v| <= max_grad * edge_len."""
    d = np.asarray(d_final, dtype=np.float32).copy()
    if len(base_verts) == 0 or len(outer_faces) == 0:
        return d

    f_arr = np.asarray(outer_faces, dtype=np.int64)
    edges = np.vstack([f_arr[:, [0, 1]], f_arr[:, [1, 2]], f_arr[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) == 0:
        return d

    u_idx = edges[:, 0]
    v_idx = edges[:, 1]
    base_f = np.asarray(base_verts, dtype=np.float32)
    norm_f = np.asarray(v_normals, dtype=np.float32)
    orig_len = np.linalg.norm(base_f[u_idx] - base_f[v_idx], axis=1)
    max_len = float(max_ratio) * np.maximum(orig_len, 1e-7)
    max_jump = float(max_grad) * np.maximum(orig_len, 1e-5)

    fix = (
        np.asarray(fixed_mask, dtype=bool)
        if (fixed_mask is not None and len(fixed_mask) == len(d))
        else np.zeros(len(d), dtype=bool)
    )
    d[fix] = 0.0
    mov_edge = ~(fix[u_idx] & fix[v_idx])

    for _ in range(45):
        cur_v = base_f + d[:, None] * norm_f
        new_len = np.linalg.norm(cur_v[u_idx] - cur_v[v_idx], axis=1)
        bad_stretch = mov_edge & (new_len > max_len)
        du = d[u_idx]
        dv = d[v_idx]
        diff = du - dv
        bad_grad = mov_edge & (np.abs(diff) > max_jump)
        if not (np.any(bad_stretch) or np.any(bad_grad)):
            break

        if np.any(bad_stretch):
            bu = u_idx[bad_stretch]
            bv = v_idx[bad_stretch]
            scale_edge = np.clip((max_len[bad_stretch] / np.maximum(new_len[bad_stretch], 1e-7)) * 0.85, 0.0, 0.90).astype(np.float32)
            vert_scale = np.ones(len(d), dtype=np.float32)
            np.minimum.at(vert_scale, bu, scale_edge)
            np.minimum.at(vert_scale, bv, scale_edge)
            d *= vert_scale
            d[fix] = 0.0

        if np.any(bad_grad):
            # Pull higher-magnitude endpoint toward lower-magnitude neighbor to shave fin/wedge cliffs.
            du = d[u_idx]
            dv = d[v_idx]
            diff = du - dv
            bad_grad = mov_edge & (np.abs(diff) > max_jump)
            if np.any(bad_grad):
                gu = u_idx[bad_grad]
                gv = v_idx[bad_grad]
                gdu = du[bad_grad]
                gdv = dv[bad_grad]
                mj = max_jump[bad_grad]
                u_higher = (np.abs(gdu) >= np.abs(gdv)) & (~fix[gu])
                target_u = gdv + np.sign(gdu - gdv) * mj
                target_v = gdu + np.sign(gdv - gdu) * mj
                hi_idx = np.where(u_higher, gu, gv)
                hi_target = np.where(u_higher, target_u, target_v).astype(np.float32)
                # Blend toward target strictly reducing peak magnitude without overshoot.
                hi_cur = d[hi_idx]
                safe_abs = np.where(hi_target * hi_cur >= 0.0, np.abs(hi_target), 0.0).astype(np.float32)
                vert_abs_cap = np.full(len(d), np.inf, dtype=np.float32)
                np.minimum.at(vert_abs_cap, hi_idx, safe_abs)
                capped = np.isfinite(vert_abs_cap) & (np.abs(d) > vert_abs_cap)
                d[capped] = np.sign(d[capped]) * vert_abs_cap[capped]
                d[fix] = 0.0

    return d


def _compute_outer_silhouette_iou(
    outer_verts: np.ndarray,
    outer_faces: np.ndarray,
    payloads: Sequence[Dict[str, Any]],
    d_final: Optional[np.ndarray] = None,
    fixed_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute mean multi-view silhouette IoU of outer_verts against full person_mask M_v."""
    ious: List[float] = []
    for p in payloads:
        sv: SegmentedView = p["view"]
        target_mask = np.asarray(sv.person_mask, dtype=bool)
        if target_mask.ndim != 2 or not np.any(target_mask):
            continue
        img_h, img_w = target_mask.shape
        if d_final is not None and p.get("base_v") is not None and p.get("norm_v") is not None:
            v_eval = (p["base_v"] + d_final[:, None] * p["norm_v"]).astype(np.float32)
            if fixed_mask is not None:
                v_eval[fixed_mask] = p["base_v"][fixed_mask]
        else:
            v_eval = outer_verts
        verts_cam = (v_eval @ p["r_v"].T) + p["t_v"].reshape(1, 3)
        pred_mask = _rasterize_binary_silhouette_fast(
            verts_cam, outer_faces, float(p["focal_orig"]), img_h, img_w
        )
        ious.append(_compute_mask_iou(pred_mask, target_mask))
    return float(np.mean(ious)) if ious else 0.0


def build_dual_layer_outer_shell(
    inner_verts_a: np.ndarray,
    inner_faces: np.ndarray,
    joints_a: np.ndarray,
    highres_head: HighResHeadOutput,
    segmented_views: List[SegmentedView],
    raw_results: List[Dict[str, Any]],
    prefer_pytorch3d: bool = True,
    fitted_cameras: Optional[Sequence[Tuple[np.ndarray, np.ndarray, float]]] = None,
    fitted_view_verts: Optional[Sequence[np.ndarray]] = None,
) -> OuterShellOutput:
    """Reconstruct clothed/outer body surface paired with highres_head and unmodified inner MHR core."""
    verify_license_compliance()
    inner_verts_copy = np.array(inner_verts_a, copy=True)
    inner_faces_copy = np.array(inner_faces, copy=True)
    inner_verts_f = np.asarray(inner_verts_a, dtype=np.float32)
    inner_faces_i = np.asarray(inner_faces, dtype=np.int64)
    joints_f = np.asarray(joints_a, dtype=np.float32)

    warnings: List[str] = []
    seen_warn: set[str] = set()

    def _add_warn(msg: str) -> None:
        if msg and msg not in seen_warn:
            seen_warn.add(msg)
            warnings.append(msg)

    for w in getattr(highres_head, "warnings", ()):
        _add_warn(w)

    (
        base_verts,
        outer_faces,
        head_vert_mask,
        hair_vert_mask,
        face_skin_vert_mask,
        front_face_vert_mask,
        collar_mask,
    ) = _subdivide_body_watertight(highres_head)

    fixed_mask = head_vert_mask | collar_mask
    mov_idx = np.where(~fixed_mask)[0].astype(np.int64)
    v_normals = _compute_oriented_outward_normals(base_verts, outer_faces)
    collar_caps_full, collar_boost_full = _compute_collar_ring_limits(len(base_verts), outer_faces, fixed_mask)
    ext_caps_full, ext_boost_full, hand_foot_mask, wrist_ankle_mask = _compute_extremity_limits(
        base_verts, joints_f, head_vert_mask
    )
    collar_caps_full = np.minimum(collar_caps_full, ext_caps_full)
    collar_boost_full = np.maximum(collar_boost_full, ext_boost_full)
    collar_cap_mov = collar_caps_full[mov_idx]
    extremity_mask_full = hand_foot_mask | wrist_ankle_mask
    extremity_mask_mov = extremity_mask_full[mov_idx]

    undisplaced_verts = base_verts.astype(np.float32).copy()
    head_undisp = getattr(highres_head, "undisplaced_verts", None)
    if head_undisp is not None:
        head_undisp_arr = np.asarray(head_undisp, dtype=np.float32)
        if head_undisp_arr.shape == undisplaced_verts.shape:
            undisplaced_verts = head_undisp_arr.copy()
        elif (
            head_undisp_arr.ndim == 2
            and head_undisp_arr.shape[1] == 3
            and len(head_undisp_arr) == len(highres_head.full_verts)
            and len(undisplaced_verts) >= len(head_undisp_arr)
        ):
            undisplaced_verts[: len(head_undisp_arr)] = head_undisp_arr

    if not segmented_views:
        _add_warn("No segmented views provided; returning outer shell without clothing/depth expansion.")
        return OuterShellOutput(
            outer_verts=base_verts.astype(np.float32),
            outer_faces=outer_faces,
            inner_mhr_verts=inner_verts_copy,
            inner_mhr_faces=inner_faces_copy,
            head_vert_mask=head_vert_mask,
            hair_vert_mask=hair_vert_mask,
            face_skin_vert_mask=face_skin_vert_mask,
            front_face_vert_mask=front_face_vert_mask,
            outer_silhouette_iou=0.0,
            warnings=warnings,
            undisplaced_verts=undisplaced_verts,
        )

    payloads, d_hull_max, sil_target, sil_weight, max_obs_cos, has_any_depth = _prepare_view_payloads(
        inner_verts_a=inner_verts_f,
        inner_faces=inner_faces_i,
        joints_a=joints_f,
        base_verts=base_verts,
        outer_faces=outer_faces,
        v_normals=v_normals,
        mov_idx=mov_idx,
        segmented_views=segmented_views,
        raw_results=raw_results,
        prefer_pytorch3d=prefer_pytorch3d,
        fitted_cameras=fitted_cameras,
        fitted_view_verts=fitted_view_verts,
        collar_cap=collar_cap_mov,
        extremity_mask_mov=extremity_mask_mov,
    )

    if not payloads:
        _add_warn("All segmented views had empty person_mask; returning unexpanded outer shell.")
        return OuterShellOutput(
            outer_verts=base_verts.astype(np.float32),
            outer_faces=outer_faces,
            inner_mhr_verts=inner_verts_copy,
            inner_mhr_faces=inner_faces_copy,
            head_vert_mask=head_vert_mask,
            hair_vert_mask=hair_vert_mask,
            face_skin_vert_mask=face_skin_vert_mask,
            front_face_vert_mask=front_face_vert_mask,
            outer_silhouette_iou=0.0,
            warnings=warnings,
            undisplaced_verts=undisplaced_verts,
        )

    if len(payloads) < 4:
        _add_warn(
            f"Only {len(payloads)} segmented view(s) active for outer shell reconstruction (< 4 cardinal views)."
        )
    if not has_any_depth:
        _add_warn("MoGe2 depth maps unavailable; reconstructing outer body shell from silhouette boundaries only.")

    d_final, p3d_warn = _optimize_outer_shell_displacements(
        base_verts=base_verts,
        outer_faces=outer_faces,
        v_normals=v_normals,
        mov_idx=mov_idx,
        payloads=payloads,
        d_hull_max=d_hull_max,
        sil_target=sil_target,
        sil_weight=sil_weight,
        max_obs_cos=max_obs_cos,
        prefer_pytorch3d=prefer_pytorch3d,
        collar_ring_boost=collar_boost_full,
    )
    if p3d_warn:
        _add_warn(p3d_warn)

    d_final[fixed_mask] = 0.0
    if np.any(wrist_ankle_mask):
        d_final[wrist_ankle_mask] = np.clip(d_final[wrist_ankle_mask], -float(INNER_BARRIER_M), 0.010)
    if np.any(hand_foot_mask):
        d_final[hand_foot_mask] = np.clip(d_final[hand_foot_mask], -float(INNER_BARRIER_M), 0.004)
    d_final = _enforce_edge_stretch_guard(
        base_verts=base_verts,
        outer_faces=outer_faces,
        v_normals=v_normals,
        d_final=d_final,
        fixed_mask=fixed_mask,
        max_ratio=1.35,
    )
    outer_verts = (base_verts + d_final[:, None] * v_normals).astype(np.float32)
    outer_verts[fixed_mask] = base_verts[fixed_mask]

    outer_iou = _compute_outer_silhouette_iou(
        outer_verts, outer_faces, payloads, d_final=d_final, fixed_mask=fixed_mask
    )
    return OuterShellOutput(
        outer_verts=outer_verts,
        outer_faces=outer_faces,
        inner_mhr_verts=inner_verts_copy,
        inner_mhr_faces=inner_faces_copy,
        head_vert_mask=head_vert_mask,
        hair_vert_mask=hair_vert_mask,
        face_skin_vert_mask=face_skin_vert_mask,
        front_face_vert_mask=front_face_vert_mask,
        outer_silhouette_iou=outer_iou,
        warnings=warnings,
        undisplaced_verts=undisplaced_verts,
    )
