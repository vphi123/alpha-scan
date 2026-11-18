"""Step 3 UV parameterization, multi-view PyTorch3D texture baking, inpainting, and privacy tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .config import (
    DEFAULT_TEXTURE_SIZE,
    FACE_VIEW_WEIGHT_BOOST,
    FOREGROUND_MASK_ERODE_PX,
    HEAD_JOINT_IDS,
    HEAD_SKIN_WEIGHT_THRESHOLD,
    HEAD_TEXTURE_SIZE,
    NECK_BASE_JOINT,
    NECK_HEAD_JOINT,
    TEXTURE_ANGLE_POWER,
    TEXTURE_DEPTH_TOLERANCE_M,
    TEXTURE_MIN_VIEW_COS,
    UV_GUTTER_DILATE_PX,
)
from .inference import prepare_image_rgb, verify_license_compliance


@dataclass
class UVLayout:
    """Canonical UV atlas coordinates aligned with original mesh triangles."""

    uv: np.ndarray  # (V_uv, 2) float32 in [0, 1], v=0 bottom, v=1 top (OpenGL/Trimesh GLB order)
    uv_faces: np.ndarray  # (F, 3) int64 indices into uv and vmapping
    vmapping: np.ndarray  # (V_uv,) int64 indices into original vertex array
    orig_faces: np.ndarray  # (F, 3) int64 original mesh triangle indices
    source: str  # "mhr" or "xatlas"


@dataclass
class ViewObservation:
    """Per-view posed MHR prediction and camera parameters aligned to its RGB frame."""

    image_rgb: np.ndarray  # (H, W, 3) uint8 RGB
    pred_vertices: np.ndarray  # (V, 3) float32 camera-frame vertices
    pred_cam_t: np.ndarray  # (3,) float32 camera translation
    focal_length: float
    pred_keypoints_3d: Optional[np.ndarray] = None  # (K, 3)
    pred_keypoints_2d: Optional[np.ndarray] = None  # (K, 2)
    weight: float = 1.0
    is_closeup_face: bool = False
    view_name: str = ""
    foreground_mask: Optional[np.ndarray] = None  # (H, W) bool/float foreground silhouette
    hair_mask: Optional[np.ndarray] = None  # (H, W) bool/float hair silhouette
    face_skin_mask: Optional[np.ndarray] = None  # (H, W) bool/float facial skin mask
    arm_occlusion_mask: Optional[np.ndarray] = None  # (H, W) bool 2D hanging-arm self-occlusion mask in side views
    mask_erode_px: int = FOREGROUND_MASK_ERODE_PX
    calibrated_camera: bool = False


@dataclass
class TexturedScanOutputs:
    """In-memory and on-disk texture baking outputs for Alpha Scan privacy levels."""

    uv_layout: UVLayout
    tex_face_rgb: np.ndarray  # (H, W, 3) uint8
    tex_blur_rgb: np.ndarray  # (H, W, 3) uint8
    head_uv_mask: np.ndarray  # (H, W) bool
    coverage_mask: np.ndarray  # (H, W) bool (before inpainting)
    used_pytorch3d: bool
    face_blur_uv_mask: Optional[np.ndarray] = None  # (H_head, W_head) bool
    hair_uv_mask: Optional[np.ndarray] = None  # (H_head, W_head) bool
    head_uv_layout: Optional[UVLayout] = None
    head_tex_face_rgb: Optional[np.ndarray] = None  # (H_head, W_head, 3) uint8
    head_tex_blur_rgb: Optional[np.ndarray] = None  # (H_head, W_head, 3) uint8
    coverage_pct: float = 0.0
    head_coverage_pct: float = 0.0
    head_coverage_mask: Optional[np.ndarray] = None  # (H_head, W_head) bool


def build_head_face_vertex_mask(
    verts_a: np.ndarray,
    joints_a: np.ndarray,
    mhr_model: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (head_mask, front_face_mask) on canonical A-pose vertices (+Y up, +Z front)."""
    num_verts = len(verts_a)
    head_anchor = joints_a[NECK_HEAD_JOINT]
    geom_head_mask = verts_a[:, 1] >= float(head_anchor[1])

    if mhr_model is not None and hasattr(mhr_model, "character_torch"):
        try:
            lbs = mhr_model.character_torch.linear_blend_skinning
            skin_idx = lbs.skin_indices_flattened.cpu().numpy().astype(np.int64)
            skin_w = lbs.skin_weights_flattened.cpu().numpy().astype(np.float32)
            vert_idx = lbs.vert_indices_flattened.cpu().numpy().astype(np.int64)
            is_head_j = np.isin(skin_idx, list(HEAD_JOINT_IDS))
            head_weights = np.zeros(num_verts, dtype=np.float32)
            np.add.at(head_weights, vert_idx[is_head_j], skin_w[is_head_j])
            lbs_head = head_weights >= HEAD_SKIN_WEIGHT_THRESHOLD
            if np.any(lbs_head):
                geom_head_mask = geom_head_mask | lbs_head
        except Exception:
            pass

    front_face_mask = geom_head_mask & (verts_a[:, 2] >= float(head_anchor[2]) + 0.02)
    return geom_head_mask, front_face_mask


def _to_numpy_array(val: Any) -> Optional[np.ndarray]:
    if val is None:
        return None
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        return val
    return np.asarray(val)


def extract_mhr_uvs(
    mhr_model: Optional[Any],
    head_assets: Optional[Dict[str, Any]],
    num_faces: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Inspect MHR model or cached head assets for built-in UV coordinates and UV face indices."""
    containers: List[Any] = []
    if head_assets:
        containers.append(head_assets)
    if mhr_model is not None:
        containers.append(mhr_model)
        if hasattr(mhr_model, "character_torch"):
            char = mhr_model.character_torch
            containers.append(char)
            if hasattr(char, "mesh"):
                containers.append(char.mesh)

    uv_names = ("uv", "uvs", "texcoords", "vt", "verts_uvs")
    uv_face_names = ("uv_faces", "faces_uvs", "face_uvs", "ft", "tex_faces")

    for obj in containers:
        uv_arr = None
        for name in uv_names:
            raw = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            if raw is not None:
                arr = _to_numpy_array(raw)
                if arr is not None and arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    uv_arr = arr.astype(np.float32)
                    break
        if uv_arr is None:
            continue

        uv_faces_arr = None
        for name in uv_face_names:
            raw_f = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            if raw_f is not None:
                arr_f = _to_numpy_array(raw_f)
                if arr_f is not None and arr_f.shape == (num_faces, 3):
                    uv_faces_arr = arr_f.astype(np.int64)
                    break
        return uv_arr, uv_faces_arr
    return None, None


def build_uv_layout(
    verts: np.ndarray,
    faces: np.ndarray,
    mhr_uvs: Optional[np.ndarray] = None,
    mhr_uv_faces: Optional[np.ndarray] = None,
) -> UVLayout:
    """Resolve canonical UVs from MHR assets if present, otherwise unwrap seamlessly with xatlas."""
    faces_i64 = np.asarray(faces, dtype=np.int64)
    num_verts = len(verts)

    if mhr_uvs is not None:
        uvs = np.asarray(mhr_uvs, dtype=np.float32)
        if mhr_uv_faces is not None:
            uv_faces = np.asarray(mhr_uv_faces, dtype=np.int64)
            vmapping = np.zeros(len(uvs), dtype=np.int64)
            vmapping[uv_faces.reshape(-1)] = faces_i64.reshape(-1)
            return UVLayout(uv=uvs, uv_faces=uv_faces, vmapping=vmapping, orig_faces=faces_i64, source="mhr")
        if len(uvs) == num_verts:
            return UVLayout(
                uv=uvs,
                uv_faces=faces_i64.copy(),
                vmapping=np.arange(num_verts, dtype=np.int64),
                orig_faces=faces_i64,
                source="mhr",
            )

    try:
        import xatlas
    except ImportError as exc:
        raise ImportError(
            "xatlas is required to unwrap MHR mesh UVs when built-in UVs are absent. "
            "Install it via `pip install xatlas`."
        ) from exc

    vmapping, indices, uvs = xatlas.parametrize(
        np.asarray(verts, dtype=np.float32),
        np.asarray(faces_i64, dtype=np.int32),
    )
    uvs_clipped = np.clip(np.asarray(uvs, dtype=np.float32), 0.0, 1.0)
    return UVLayout(
        uv=uvs_clipped,
        uv_faces=np.asarray(indices, dtype=np.int64),
        vmapping=np.asarray(vmapping, dtype=np.int64),
        orig_faces=faces_i64,
        source="xatlas",
    )


def _rasterize_2d_triangles_with_bary(
    verts_px: np.ndarray,
    faces: np.ndarray,
    height: int,
    width: int,
    draw_order: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize 2D projected triangles into (tri_id_map, bary_map) with 1/16px subpixel precision."""
    id_buf = np.zeros((height, width, 3), dtype=np.uint8)
    tris_px = verts_px[faces]  # (F, 3, 2)
    tris_fixed = np.round(tris_px * 16.0).astype(np.int32)
    face_ids = np.arange(len(faces), dtype=np.int32) if draw_order is None else draw_order

    for f_idx in face_ids:
        tri_val = int(f_idx) + 1
        color = (tri_val & 255, (tri_val >> 8) & 255, (tri_val >> 16) & 255)
        cv2.fillConvexPoly(id_buf, tris_fixed[f_idx], color=color, lineType=cv2.LINE_8, shift=4)

    tri_id_map = (
        id_buf[..., 0].astype(np.int32)
        | (id_buf[..., 1].astype(np.int32) << 8)
        | (id_buf[..., 2].astype(np.int32) << 16)
    ) - 1

    bary_map = np.zeros((height, width, 3), dtype=np.float32)
    rows, cols = np.where(tri_id_map >= 0)
    if len(rows) == 0:
        return tri_id_map, bary_map

    f_hits = tri_id_map[rows, cols]
    tri_corners = tris_px[f_hits]
    p0, p1, p2 = tri_corners[:, 0, :], tri_corners[:, 1, :], tri_corners[:, 2, :]
    pts = np.stack([cols.astype(np.float32), rows.astype(np.float32)], axis=-1)

    v0 = p1 - p0
    v1 = p2 - p0
    v2 = pts - p0
    denom = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    safe = np.abs(denom) > 1e-8
    inv_denom = np.where(safe, 1.0 / np.where(safe, denom, 1.0), 0.0)

    w1 = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) * inv_denom
    w2 = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) * inv_denom
    w0 = 1.0 - w1 - w2

    bary = np.clip(np.stack([w0, w1, w2], axis=-1), 0.0, 1.0)
    bary_sum = bary.sum(axis=-1, keepdims=True)
    bary = np.where(bary_sum > 1e-8, bary / np.maximum(bary_sum, 1e-8), 1.0 / 3.0)
    bary_map[rows, cols] = bary.astype(np.float32)
    return tri_id_map, bary_map


def rasterize_uv_atlas_map(
    uv_layout: UVLayout,
    tex_size: int = DEFAULT_TEXTURE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize mesh triangles into UV pixel space returning (tri_id_map, bary_map)."""
    uv_px = np.empty_like(uv_layout.uv, dtype=np.float32)
    uv_px[:, 0] = uv_layout.uv[:, 0] * (tex_size - 1)
    uv_px[:, 1] = (1.0 - uv_layout.uv[:, 1]) * (tex_size - 1)
    return _rasterize_2d_triangles_with_bary(uv_px, uv_layout.uv_faces, tex_size, tex_size)


def project_points_pinhole(
    pts_cam: np.ndarray,
    focal_length: float,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Project 3D OpenCV camera-frame points (+X right, +Y down, +Z forward) to pixel (u, v)."""
    z = np.maximum(pts_cam[..., 2], 1e-4)
    u = float(focal_length) * (pts_cam[..., 0] / z) + (img_width / 2.0)
    v = float(focal_length) * (pts_cam[..., 1] / z) + (img_height / 2.0)
    return np.stack([u, v], axis=-1).astype(np.float32)


def transform_vertices_to_camera(
    verts: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    img_width: int,
    img_height: int,
    kpts_3d: Optional[np.ndarray] = None,
    kpts_2d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply camera translation and self-calibrate axis signs against pred_keypoints_2d if present."""
    verts_f = np.asarray(verts, dtype=np.float32)
    t_f = np.asarray(cam_t, dtype=np.float32).reshape(1, 3)

    if kpts_3d is None or kpts_2d is None or len(kpts_3d) == 0 or len(kpts_2d) == 0:
        return verts_f + t_f

    k3 = np.asarray(kpts_3d, dtype=np.float32)
    k2_raw = np.asarray(kpts_2d, dtype=np.float32)
    k2 = k2_raw[..., :2]
    n_cmp = min(len(k3), len(k2))
    k3, k2, k2_raw = k3[:n_cmp], k2[:n_cmp], k2_raw[:n_cmp]
    valid_kp = np.isfinite(k3).all(axis=1) & np.isfinite(k2).all(axis=1)
    if k2_raw.ndim == 2 and k2_raw.shape[1] >= 3:
        conf_valid = valid_kp & np.isfinite(k2_raw[:, 2]) & (k2_raw[:, 2] > 0.05)
        if int(conf_valid.sum()) >= 3:
            valid_kp = conf_valid
    in_bounds = (
        valid_kp
        & (k2[:, 0] > 0.0)
        & (k2[:, 0] < float(img_width))
        & (k2[:, 1] > 0.0)
        & (k2[:, 1] < float(img_height))
    )
    if int(in_bounds.sum()) >= 3:
        valid_kp = in_bounds
    if int(valid_kp.sum()) < 3:
        return verts_f + t_f
    k3, k2 = k3[valid_kp], k2[valid_kp]

    best_err = float("inf")
    best_cfg = (np.array([1.0, 1.0, 1.0], dtype=np.float32), True)
    candidates = [
        (np.array([1.0, 1.0, 1.0], dtype=np.float32), True),
        (np.array([1.0, -1.0, 1.0], dtype=np.float32), True),
        (np.array([-1.0, -1.0, 1.0], dtype=np.float32), True),
        (np.array([1.0, -1.0, -1.0], dtype=np.float32), True),
        (np.array([1.0, -1.0, 1.0], dtype=np.float32), False),
        (np.array([-1.0, -1.0, 1.0], dtype=np.float32), False),
        (np.array([1.0, -1.0, -1.0], dtype=np.float32), False),
    ]
    for sign, add_before in candidates:
        pts_k = (k3 + t_f) * sign if add_before else (k3 * sign + t_f)
        if float(np.median(pts_k[:, 2])) <= 0.05:
            continue
        proj = project_points_pinhole(pts_k, focal_length, img_width, img_height)
        err = float(np.median(np.linalg.norm(proj - k2, axis=1)))
        if err < best_err:
            best_err = err
            best_cfg = (sign, add_before)

    sign, add_before = best_cfg
    return (verts_f + t_f) * sign if add_before else (verts_f * sign + t_f)


def render_view_depth_buffer(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    img_width: int,
    img_height: int,
    prefer_pytorch3d: bool = True,
    return_tri_id: bool = False,
) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, np.ndarray, bool]]:
    """Render an OpenCV-aligned depth buffer (H, W) with PyTorch3D MeshRasterizer or CPU fallback."""
    verify_license_compliance()
    if prefer_pytorch3d:
        try:
            from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
            from pytorch3d.structures import Meshes
            from pytorch3d.utils import cameras_from_opencv_projection

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            r_eye = torch.eye(3, dtype=torch.float32, device=dev)[None]
            t_zero = torch.zeros((1, 3), dtype=torch.float32, device=dev)
            cam_k = torch.tensor(
                [[
                    [float(focal_length), 0.0, img_width / 2.0],
                    [0.0, float(focal_length), img_height / 2.0],
                    [0.0, 0.0, 1.0],
                ]],
                dtype=torch.float32,
                device=dev,
            )
            im_size = torch.tensor([[img_height, img_width]], dtype=torch.float32, device=dev)
            cameras = cameras_from_opencv_projection(r_eye, t_zero, cam_k, im_size)

            mesh_p3d = Meshes(
                verts=[torch.tensor(verts_cam, dtype=torch.float32, device=dev)],
                faces=[torch.tensor(faces, dtype=torch.int64, device=dev)],
            )
            raster_settings = RasterizationSettings(
                image_size=(img_height, img_width),
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(mesh_p3d)
            zbuf = fragments.zbuf[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
            if return_tri_id:
                tri_buf = fragments.pix_to_face[0, :, :, 0].detach().cpu().numpy().astype(np.int32)
                return zbuf, tri_buf, True
            return zbuf, True
        except (ImportError, OSError, RuntimeError):
            pass

    # Pure-NumPy/OpenCV fallback (sorted painter order + perspective-correct 1/Z depth).
    faces_i64 = np.asarray(faces, dtype=np.int64)
    tris_cam = verts_cam[faces_i64]  # (F, 3, 3)
    tri_z_min = tris_cam[:, :, 2].min(axis=1)
    tri_z_mean = tris_cam[:, :, 2].mean(axis=1)
    valid_faces = np.where((tri_z_min > 0.1) & np.isfinite(tris_cam).all(axis=(1, 2)))[0]
    if len(valid_faces) == 0:
        empty_z = np.full((img_height, img_width), -1.0, dtype=np.float32)
        if return_tri_id:
            return empty_z, np.full((img_height, img_width), -1, dtype=np.int32), False
        return empty_z, False

    verts_2d = project_points_pinhole(verts_cam, focal_length, img_width, img_height)
    order = valid_faces[np.argsort(-tri_z_mean[valid_faces])]
    tri_id_map, bary_map = _rasterize_2d_triangles_with_bary(
        verts_2d, faces_i64, img_height, img_width, draw_order=order
    )

    zbuf = np.full((img_height, img_width), -1.0, dtype=np.float32)
    rows, cols = np.where(tri_id_map >= 0)
    if len(rows) > 0:
        f_hits = tri_id_map[rows, cols]
        b = bary_map[rows, cols]
        z_tri = np.maximum(tris_cam[f_hits, :, 2], 1e-4)
        inv_z = (b / z_tri).sum(axis=-1)
        zbuf[rows, cols] = (1.0 / np.maximum(inv_z, 1e-6)).astype(np.float32)
    if return_tri_id:
        return zbuf, tri_id_map.astype(np.int32, copy=False), False
    return zbuf, False


def _orient_face_normals(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    reference_faces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute outward-pointing unit triangle normals for camera-frame mesh vertices."""
    tris = verts_cam[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / np.maximum(norms, 1e-12)

    ref_faces = faces if reference_faces is None else reference_faces
    ref_tris = tris if reference_faces is None else verts_cam[ref_faces]
    ref_cross = (
        cross
        if reference_faces is None
        else np.cross(ref_tris[:, 1] - ref_tris[:, 0], ref_tris[:, 2] - ref_tris[:, 0])
    )
    centers = ref_tris.mean(axis=1)
    mesh_center = ref_tris.mean(axis=(0, 1), keepdims=True)
    signed_vol6 = float(np.sum((centers - mesh_center) * ref_cross))
    if abs(signed_vol6) > 1e-6:
        if signed_vol6 < 0.0:
            normals = -normals
    else:
        # Open/planar patch: point normals toward camera (-Z) if inverted.
        if float(np.mean(normals[:, 2])) > 0.0:
            normals = -normals
    return normals.astype(np.float32)


def _sample_zbuf_min(zbuf: np.ndarray, u_img: np.ndarray, v_img: np.ndarray) -> np.ndarray:
    """Sample positive z-buffer depth safely across the 2x2 neighborhood around projected coords."""
    h, w = zbuf.shape
    x0 = np.clip(np.floor(u_img).astype(np.int64), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(v_img).astype(np.int64), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    inf_val = 1e9
    samples = np.stack(
        [
            np.where(zbuf[y0, x0] > 0, zbuf[y0, x0], inf_val),
            np.where(zbuf[y0, x1] > 0, zbuf[y0, x1], inf_val),
            np.where(zbuf[y1, x0] > 0, zbuf[y1, x0], inf_val),
            np.where(zbuf[y1, x1] > 0, zbuf[y1, x1], inf_val),
        ],
        axis=0,
    )
    min_z = samples.min(axis=0)
    return np.where(min_z < inf_val * 0.5, min_z, -1.0).astype(np.float32)


def _sample_zbuf_visibility(
    zbuf: np.ndarray,
    zbuf_tri_id: Optional[np.ndarray],
    u_img: np.ndarray,
    v_img: np.ndarray,
    f_hits: np.ndarray,
    z_tex: np.ndarray,
    cos_theta: np.ndarray,
    check_tri_id: bool = True,
) -> np.ndarray:
    """Evaluate depth visibility using nearest-pixel + same-triangle check with grazing-angle bias."""
    h, w = zbuf.shape
    xn = np.clip(np.rint(u_img).astype(np.int64), 0, w - 1)
    yn = np.clip(np.rint(v_img).astype(np.int64), 0, h - 1)
    z_nr = zbuf[yn, xn]
    z_min2 = _sample_zbuf_min(zbuf, u_img, v_img)
    z_ref = np.where(z_nr > 0.0, z_nr, z_min2)

    grazing = np.clip(
        (0.65 - cos_theta) / max(0.65 - float(TEXTURE_MIN_VIEW_COS), 1e-4),
        0.0,
        1.0,
    )
    depth_tol = float(TEXTURE_DEPTH_TOLERANCE_M) + 0.015 * grazing
    depth_ok = (z_ref > 0.0) & (z_tex <= z_ref + depth_tol)

    if check_tri_id and zbuf_tri_id is not None:
        same_tri_nr = (z_nr > 0.0) & (zbuf_tri_id[yn, xn] == f_hits) & (z_tex <= z_nr + 0.030)
        depth_ok = depth_ok | same_tri_nr
    return depth_ok


def _rasterize_vertex_mask_to_uv(
    vert_mask: np.ndarray,
    orig_faces: np.ndarray,
    tri_id_map: np.ndarray,
    bary_map: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Project a per-vertex boolean mask onto valid UV atlas texels via barycentric weights."""
    tex_h, tex_w = tri_id_map.shape
    uv_mask = np.zeros((tex_h, tex_w), dtype=bool)
    if len(rows) == 0:
        return uv_mask
    f_hits = tri_id_map[rows, cols]
    b = bary_map[rows, cols]
    tri_vals = np.asarray(vert_mask, dtype=np.float32)[orig_faces[f_hits]]
    uv_mask[rows, cols] = (tri_vals * b).sum(axis=1) >= float(threshold)
    return uv_mask


def _build_uv_head_masks(
    verts_a: np.ndarray,
    joints_a: np.ndarray,
    uv_layout: UVLayout,
    tri_id_map: np.ndarray,
    bary_map: np.ndarray,
    mhr_model: Optional[Any],
    head_vert_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-texel boolean masks (head_uv_mask, front_face_uv_mask) on the UV atlas."""
    tex_h, tex_w = tri_id_map.shape
    head_uv_mask = np.zeros((tex_h, tex_w), dtype=bool)
    front_face_uv_mask = np.zeros((tex_h, tex_w), dtype=bool)

    rows, cols = np.where(tri_id_map >= 0)
    if len(rows) == 0:
        return head_uv_mask, front_face_uv_mask

    f_hits = tri_id_map[rows, cols]
    b = bary_map[rows, cols]
    tris_a = verts_a[uv_layout.orig_faces[f_hits]]  # (N, 3, 3)
    pts_a = np.einsum("ni,nij->nj", b, tris_a)

    head_anchor = joints_a[NECK_HEAD_JOINT]
    if head_vert_mask is not None:
        tri_head = np.asarray(head_vert_mask, dtype=np.float32)[uv_layout.orig_faces[f_hits]]
        is_head = (tri_head * b).sum(axis=1) >= 0.5
    else:
        is_head = pts_a[:, 1] >= float(head_anchor[1])
        if mhr_model is not None and hasattr(mhr_model, "character_torch"):
            lbs_mask, _ = build_head_face_vertex_mask(verts_a, joints_a, mhr_model)
            tri_lbs = lbs_mask[uv_layout.orig_faces[f_hits]].astype(np.float32)
            is_head = is_head | ((tri_lbs * b).sum(axis=1) >= 0.5)

    is_front_face = is_head & (pts_a[:, 2] >= float(head_anchor[2]) + 0.02)
    head_uv_mask[rows, cols] = is_head
    front_face_uv_mask[rows, cols] = is_front_face
    return head_uv_mask, front_face_uv_mask


def _normalize_binary_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Convert heterogeneous mask array to (target_h, target_w) uint8 {0, 1}."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if arr.size > 0 and float(np.max(arr)) > 1.5:
        return (arr > 127.5).astype(np.uint8)
    return (arr >= 0.5).astype(np.uint8)


def _remap_points(
    src: np.ndarray,
    u_pts: np.ndarray,
    v_pts: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: float = 0.0,
) -> np.ndarray:
    """Safely run cv2.remap over arbitrarily many query points by chunking below OpenCV SHRT_MAX."""
    n_pts = len(u_pts)
    if n_pts == 0:
        out_shape = (0, src.shape[2]) if src.ndim == 3 else (0,)
        return np.empty(out_shape, dtype=np.float32)

    chunk = 16384
    if n_pts <= chunk:
        res = cv2.remap(
            src,
            u_pts.reshape(-1, 1),
            v_pts.reshape(-1, 1),
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        return res.reshape(-1, src.shape[2]) if src.ndim == 3 else res.reshape(-1)

    parts: List[np.ndarray] = []
    for start in range(0, n_pts, chunk):
        end = min(start + chunk, n_pts)
        res = cv2.remap(
            src,
            u_pts[start:end].reshape(-1, 1),
            v_pts[start:end].reshape(-1, 1),
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        parts.append(res.reshape(-1, src.shape[2]) if src.ndim == 3 else res.reshape(-1))
    return np.concatenate(parts, axis=0)


def _resolve_arm_torso_keypoint_indices(
    num_kpts: int,
) -> Optional[Tuple[Tuple[Tuple[int, int, int], ...], Tuple[int, int]]]:
    """Return (((sh, el, wr), ...), (hip_a, hip_b)) for MHR-127, MHR-70, or COCO-17 schemas."""
    if num_kpts >= 127:
        return (((75, 76, 78), (39, 40, 42)), (2, 18))
    if num_kpts == 70:
        return (((5, 7, 62), (6, 8, 41)), (9, 10))
    if num_kpts >= 13:
        return (((5, 7, 9), (6, 8, 10)), (11, 12))
    return None


def _build_non_arm_uv_mask(
    verts_a: np.ndarray,
    joints_a: np.ndarray,
    uv_layout: UVLayout,
    tri_id_map: np.ndarray,
    bary_map: np.ndarray,
) -> Optional[np.ndarray]:
    """Identify non-arm texels (torso/waist/legs) so side-view arm occlusion protects torso without blocking arms."""
    if len(joints_a) <= 75 or len(verts_a) == 0:
        return None
    sh_l = np.asarray(joints_a[75], dtype=np.float32)
    sh_r = np.asarray(joints_a[39], dtype=np.float32)
    sh_dist = float(np.linalg.norm(sh_l - sh_r))
    if sh_dist < 0.15 or not (np.isfinite(sh_l).all() and np.isfinite(sh_r).all()):
        return None
    chest_cx = 0.5 * float(sh_l[0] + sh_r[0])
    torso_half_w = max(0.16, 0.58 * sh_dist)
    hip_y = (
        0.5 * float(joints_a[2, 1] + joints_a[18, 1])
        if len(joints_a) > 18 and np.any(joints_a[[2, 18]] != 0.0)
        else float(sh_l[1]) - 0.55
    )
    neck_y = float(joints_a[NECK_BASE_JOINT, 1]) if len(joints_a) > NECK_BASE_JOINT else float(sh_l[1]) + 0.06
    is_arm_v = (
        (np.abs(verts_a[:, 0] - chest_cx) > torso_half_w)
        & (verts_a[:, 1] >= hip_y - 0.12)
        & (verts_a[:, 1] <= neck_y + 0.05)
    )
    if not np.any(is_arm_v):
        return None
    rows, cols = np.where(tri_id_map >= 0)
    arm_uv = _rasterize_vertex_mask_to_uv(is_arm_v, uv_layout.orig_faces, tri_id_map, bary_map, rows, cols)
    return ~arm_uv


def _infer_side_arm_occlusion_mask(
    kpts_2d: Optional[np.ndarray],
    img_h: int,
    img_w: int,
    view_name: str = "",
    r_mat: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Detect side-view hanging-arm self-occlusion over the torso from 2D keypoints and rasterize an exclusion mask."""
    if kpts_2d is None:
        return None
    k2 = np.asarray(kpts_2d, dtype=np.float32)
    if k2.ndim != 2 or k2.shape[1] < 2:
        return None
    schema = _resolve_arm_torso_keypoint_indices(len(k2))
    if schema is None:
        return None
    arm_chains, (hp_a, hp_b) = schema
    (l_sh, l_el, l_wr), (r_sh, r_el, r_wr) = arm_chains
    core_idx = [l_sh, r_sh, hp_a, hp_b]
    if not np.isfinite(k2[core_idx, :2]).all():
        return None
    if k2.shape[1] >= 3 and np.any(k2[core_idx, 2] <= 0.05):
        return None

    sh_c = 0.5 * (k2[l_sh, :2] + k2[r_sh, :2])
    hip_c = 0.5 * (k2[hp_a, :2] + k2[hp_b, :2])
    torso_h = float(hip_c[1] - sh_c[1])
    if torso_h <= 12.0:
        return None

    sh_w = float(abs(k2[l_sh, 0] - k2[r_sh, 0]))
    vname = (view_name or "").lower()
    is_named_side = any(tok in vname for tok in ("side", "left", "right", "profile"))
    is_rot_side = r_mat is not None and abs(float(np.asarray(r_mat, dtype=np.float32)[0, 2])) > 0.55
    is_kpt_side = sh_w < 0.48 * torso_h
    if not (is_named_side or is_rot_side or is_kpt_side):
        return None

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    thickness = max(8, int(round(0.22 * torso_h)))
    torso_x_tol = max(12.0, 0.45 * torso_h)

    for sh_i, el_i, wr_i in arm_chains:
        if not np.isfinite(k2[[sh_i, el_i, wr_i], :2]).all():
            continue
        if k2.shape[1] >= 3 and (k2[el_i, 2] <= 0.05 or k2[wr_i, 2] <= 0.05):
            continue
        sh_pt = k2[sh_i, :2]
        el_pt = k2[el_i, :2]
        wr_pt = k2[wr_i, :2]
        # Hanging arm: elbow is below shoulder and horizontally overlaps the side-view torso band
        if float(el_pt[1]) <= float(sh_pt[1]) + 0.12 * torso_h:
            continue
        if abs(float(el_pt[0]) - float(sh_c[0])) > torso_x_tol and abs(float(wr_pt[0]) - float(hip_c[0])) > torso_x_tol:
            continue
        p0 = (int(round(float(sh_pt[0]))), int(round(float(sh_pt[1]))))
        p1 = (int(round(float(el_pt[0]))), int(round(float(el_pt[1]))))
        p2 = (int(round(float(wr_pt[0]))), int(round(float(wr_pt[1]))))
        cv2.line(mask, p0, p1, 1, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.line(mask, p1, p2, 1, thickness=thickness, lineType=cv2.LINE_AA)

    if not np.any(mask):
        return None

    y_top = max(0, int(round(float(sh_c[1]) + 0.04 * torso_h)))
    y_bot = min(img_h, int(round(float(hip_c[1]) + 0.12 * torso_h)))
    if y_top > 0:
        mask[:y_top, :] = 0
    if y_bot < img_h:
        mask[y_bot:, :] = 0
    return mask.astype(bool) if np.any(mask) else None


def _bake_multiview_core(
    uv_layout: UVLayout,
    tri_id_map: np.ndarray,
    bary_map: np.ndarray,
    observations: List[ViewObservation],
    front_face_uv_mask: np.ndarray,
    prefer_pytorch3d: bool = True,
    occlusion_faces: Optional[np.ndarray] = None,
    non_arm_uv_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray]:
    """Project views onto UV atlas with eroded foreground masking, occlusion check, and semantic weights."""
    tex_h, tex_w = tri_id_map.shape
    accum_rgb = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    accum_w = np.zeros((tex_h, tex_w), dtype=np.float32)
    accum_hair_w = np.zeros((tex_h, tex_w), dtype=np.float32)
    accum_face_skin_w = np.zeros((tex_h, tex_w), dtype=np.float32)

    fallback_rgb = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    fallback_w = np.zeros((tex_h, tex_w), dtype=np.float32)
    fallback_hair_w = np.zeros((tex_h, tex_w), dtype=np.float32)
    fallback_face_skin_w = np.zeros((tex_h, tex_w), dtype=np.float32)
    used_p3d_any = False

    rows, cols = np.where(tri_id_map >= 0)
    if len(rows) == 0 or not observations:
        return accum_rgb, accum_w, used_p3d_any, accum_hair_w, accum_face_skin_w

    f_hits = tri_id_map[rows, cols]
    b = bary_map[rows, cols]
    front_face_texel = front_face_uv_mask[rows, cols]
    non_arm_texel = (
        non_arm_uv_mask[rows, cols]
        if non_arm_uv_mask is not None
        else np.ones(len(rows), dtype=bool)
    )
    depth_faces = uv_layout.orig_faces if occlusion_faces is None else occlusion_faces

    for obs in observations:
        img = np.asarray(obs.image_rgb, dtype=np.float32)
        img_h, img_w = img.shape[:2]
        if getattr(obs, "calibrated_camera", False):
            verts_cam = (
                np.asarray(obs.pred_vertices, dtype=np.float32)
                + np.asarray(obs.pred_cam_t, dtype=np.float32).reshape(1, 3)
            )
        else:
            verts_cam = transform_vertices_to_camera(
                verts=obs.pred_vertices,
                cam_t=obs.pred_cam_t,
                focal_length=obs.focal_length,
                img_width=img_w,
                img_height=img_h,
                kpts_3d=obs.pred_keypoints_3d,
                kpts_2d=obs.pred_keypoints_2d,
            )
        zbuf, zbuf_tri_id, used_p3d = render_view_depth_buffer(
            verts_cam=verts_cam,
            faces=depth_faces,
            focal_length=obs.focal_length,
            img_width=img_w,
            img_height=img_h,
            prefer_pytorch3d=prefer_pytorch3d,
            return_tri_id=True,
        )
        used_p3d_any = used_p3d_any or used_p3d

        face_normals = _orient_face_normals(
            verts_cam, uv_layout.orig_faces, reference_faces=occlusion_faces
        )
        n_hits = face_normals[f_hits]

        tris_cam = verts_cam[uv_layout.orig_faces[f_hits]]  # (N, 3, 3)
        pts_cam = np.einsum("ni,nij->nj", b, tris_cam)

        view_dir = -pts_cam / np.maximum(np.linalg.norm(pts_cam, axis=1, keepdims=True), 1e-8)
        cos_theta = np.sum(n_hits * view_dir, axis=1)

        proj_uv = project_points_pinhole(pts_cam, obs.focal_length, img_w, img_h)
        u_img, v_img = proj_uv[:, 0], proj_uv[:, 1]
        in_frame = (u_img >= 0.0) & (u_img <= img_w - 1.0) & (v_img >= 0.0) & (v_img <= img_h - 1.0)
        z_tex = pts_cam[:, 2]
        depth_ok = _sample_zbuf_visibility(
            zbuf=zbuf,
            zbuf_tri_id=zbuf_tri_id,
            u_img=u_img,
            v_img=v_img,
            f_hits=f_hits,
            z_tex=z_tex,
            cos_theta=cos_theta,
            check_tri_id=(occlusion_faces is None),
        )

        visible = (
            in_frame
            & (z_tex > 0.1)
            & depth_ok
            & (cos_theta >= TEXTURE_MIN_VIEW_COS)
        )

        # Guard against side-view hanging-arm self-occlusion bleeding arm skin onto torso texels.
        arm_occ = getattr(obs, "arm_occlusion_mask", None)
        if arm_occ is None and obs.pred_keypoints_2d is not None:
            arm_occ = _infer_side_arm_occlusion_mask(
                obs.pred_keypoints_2d,
                img_h=img_h,
                img_w=img_w,
                view_name=getattr(obs, "view_name", ""),
            )
        if arm_occ is None and obs.pred_keypoints_3d is not None:
            k3_cam = (
                np.asarray(obs.pred_keypoints_3d, dtype=np.float32)
                + np.asarray(obs.pred_cam_t, dtype=np.float32).reshape(1, 3)
                if getattr(obs, "calibrated_camera", False)
                else transform_vertices_to_camera(
                    verts=obs.pred_keypoints_3d,
                    cam_t=obs.pred_cam_t,
                    focal_length=obs.focal_length,
                    img_width=img_w,
                    img_height=img_h,
                    kpts_3d=obs.pred_keypoints_3d,
                    kpts_2d=obs.pred_keypoints_2d,
                )
            )
            arm_occ = _infer_side_arm_occlusion_mask(
                project_points_pinhole(k3_cam, obs.focal_length, img_w, img_h),
                img_h=img_h,
                img_w=img_w,
                view_name=getattr(obs, "view_name", ""),
            )
        if arm_occ is not None and np.any(visible):
            arm_u8 = _normalize_binary_mask(arm_occ, img_h, img_w)
            cand_arm = np.where(visible)[0]
            arm_s = _remap_points(
                arm_u8.astype(np.float32),
                u_img[cand_arm],
                v_img[cand_arm],
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0.0,
            )
            visible[cand_arm] = (arm_s < 0.5) | (~non_arm_texel[cand_arm])

        sil_w_full = np.ones(len(rows), dtype=np.float32)
        narrow_fallback_mask = np.zeros(len(rows), dtype=bool)

        fg_raw_u8 = None
        fg_u8 = None
        dist_px = None
        if obs.foreground_mask is not None and np.any(visible):
            fg_raw_u8 = _normalize_binary_mask(obs.foreground_mask, img_h, img_w)
            fg_u8 = fg_raw_u8
            erode_px = max(0, int(getattr(obs, "mask_erode_px", FOREGROUND_MASK_ERODE_PX)))
            eff_erode = (
                max(erode_px, int(round(0.0045 * max(img_h, img_w))))
                if erode_px > 0
                else 0
            )
            if eff_erode > 0:
                k_size = 2 * eff_erode + 1
                erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                fg_u8 = cv2.erode(fg_u8, erode_k, iterations=1)

            dist_px = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 3)
            local_max_dist = cv2.dilate(
                dist_px,
                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            )

            cand_idx = np.where(visible)[0]
            fg_sampled = _remap_points(
                fg_u8.astype(np.float32),
                u_img[cand_idx],
                v_img[cand_idx],
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0.0,
            )
            dist_sampled = _remap_points(
                dist_px.astype(np.float32),
                u_img[cand_idx],
                v_img[cand_idx],
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0.0,
            )
            local_max_sampled = _remap_points(
                local_max_dist.astype(np.float32),
                u_img[cand_idx],
                v_img[cand_idx],
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0.0,
            )

            cos_c = cos_theta[cand_idx]
            cos_denom = max(0.55 - float(TEXTURE_MIN_VIEW_COS), 1e-4)
            req_dist = np.where(
                cos_c < 0.55,
                3.0 + 6.0 * np.clip((0.55 - cos_c) / cos_denom, 0.0, 1.0),
                1.0,
            )
            base_fg_ok = (fg_sampled >= 0.5) & (dist_sampled >= 1.5)
            strict_ok = base_fg_ok & (dist_sampled >= req_dist)

            # Safe fallback when a triangle is only seen inside a genuinely narrow mask feature
            is_narrow_feature = base_fg_ok & (~strict_ok) & (local_max_sampled <= req_dist + 2.0)
            narrow_fallback_mask[cand_idx] = is_narrow_feature

            visible[cand_idx] = strict_ok
            sil_w_cand = np.square(np.clip(dist_sampled / 8.0, 0.05, 1.0)).astype(np.float32)
            sil_w_full[cand_idx] = sil_w_cand

        if np.any(narrow_fallback_mask):
            fb_idx = np.where(narrow_fallback_mask)[0]
            u_fb, v_fb = u_img[fb_idx], v_img[fb_idx]
            fb_rgb = _remap_points(
                img,
                u_fb,
                v_fb,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
            )
            # Reject narrow-feature fallback pixels whose sampled color matches the local exterior
            # background band (e.g., gym wall or blue banner under ankle/leg pose drift).
            if fg_raw_u8 is not None and dist_px is not None:
                bg_ext = fg_raw_u8 == 0
                bg_clean = cv2.erode(bg_ext.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
                if not np.any(bg_clean):
                    bg_clean = bg_ext
                core_fg = dist_px >= 3.5
                if not np.any(core_fg) and fg_u8 is not None:
                    core_fg = fg_u8 > 0
                if np.any(bg_clean) and np.any(core_fg):
                    bg_indices = distance_transform_edt(~bg_clean, return_distances=False, return_indices=True)
                    fg_indices = distance_transform_edt(~core_fg, return_distances=False, return_indices=True)
                    vy_fb = np.clip(np.rint(v_fb).astype(np.int64), 0, img_h - 1)
                    vx_fb = np.clip(np.rint(u_fb).astype(np.int64), 0, img_w - 1)
                    local_bg_rgb = img[bg_indices[0, vy_fb, vx_fb], bg_indices[1, vy_fb, vx_fb]]
                    local_fg_rgb = img[fg_indices[0, vy_fb, vx_fb], fg_indices[1, vy_fb, vx_fb]]
                    d_bg = np.linalg.norm(fb_rgb - local_bg_rgb, axis=1)
                    d_fg = np.linalg.norm(fb_rgb - local_fg_rgb, axis=1)
                    keep_fb = ~((d_bg < 32.0) & (d_fg > d_bg + 12.0))
                    fb_idx = fb_idx[keep_fb]
                    fb_rgb = fb_rgb[keep_fb]
                    u_fb = u_fb[keep_fb]
                    v_fb = v_fb[keep_fb]
            if len(fb_idx) > 0:
                w_fb = (
                    float(obs.weight)
                    * np.power(np.maximum(cos_theta[fb_idx], 0.0), TEXTURE_ANGLE_POWER)
                    * sil_w_full[fb_idx]
                )
                if obs.is_closeup_face:
                    w_fb = w_fb * np.where(front_face_texel[fb_idx], FACE_VIEW_WEIGHT_BOOST, 0.05)
                fbr, fbc = rows[fb_idx], cols[fb_idx]
                fallback_rgb[fbr, fbc] += fb_rgb * w_fb[:, None]
                fallback_w[fbr, fbc] += w_fb
                if obs.hair_mask is not None:
                    hair_u8 = _normalize_binary_mask(obs.hair_mask, img_h, img_w)
                    hair_fb = _remap_points(
                        hair_u8.astype(np.float32),
                        u_fb,
                        v_fb,
                        interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_CONSTANT,
                        border_value=0.0,
                    )
                    fallback_hair_w[fbr, fbc] += hair_fb * w_fb
                if obs.face_skin_mask is not None:
                    face_u8 = _normalize_binary_mask(obs.face_skin_mask, img_h, img_w)
                    face_fb = _remap_points(
                        face_u8.astype(np.float32),
                        u_fb,
                        v_fb,
                        interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_CONSTANT,
                        border_value=0.0,
                    )
                    fallback_face_skin_w[fbr, fbc] += face_fb * w_fb

        if not np.any(visible):
            continue

        vis_idx = np.where(visible)[0]
        u_vis = u_img[vis_idx]
        v_vis = v_img[vis_idx]
        sample_rgb = _remap_points(
            img,
            u_vis,
            v_vis,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
        )

        w_tex = (
            float(obs.weight)
            * np.power(np.maximum(cos_theta[vis_idx], 0.0), TEXTURE_ANGLE_POWER)
            * sil_w_full[vis_idx]
        )
        if obs.is_closeup_face:
            face_mult = np.where(front_face_texel[vis_idx], FACE_VIEW_WEIGHT_BOOST, 0.05)
            w_tex = w_tex * face_mult

        vr, vc = rows[vis_idx], cols[vis_idx]
        accum_rgb[vr, vc] += sample_rgb * w_tex[:, None]
        accum_w[vr, vc] += w_tex

        if obs.hair_mask is not None:
            hair_u8 = _normalize_binary_mask(obs.hair_mask, img_h, img_w)
            hair_s = _remap_points(
                hair_u8.astype(np.float32),
                u_vis,
                v_vis,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0.0,
            )
            accum_hair_w[vr, vc] += hair_s * w_tex

        if obs.face_skin_mask is not None:
            face_u8 = _normalize_binary_mask(obs.face_skin_mask, img_h, img_w)
            face_s = _remap_points(
                face_u8.astype(np.float32),
                u_vis,
                v_vis,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0.0,
            )
            accum_face_skin_w[vr, vc] += face_s * w_tex

    use_fb = (accum_w <= 1e-8) & (fallback_w > 1e-8)
    if np.any(use_fb):
        accum_rgb[use_fb] = fallback_rgb[use_fb]
        accum_w[use_fb] = fallback_w[use_fb]
        accum_hair_w[use_fb] = fallback_hair_w[use_fb]
        accum_face_skin_w[use_fb] = fallback_face_skin_w[use_fb]

    return accum_rgb, accum_w, used_p3d_any, accum_hair_w, accum_face_skin_w


def bake_multiview_texture(
    uv_layout: UVLayout,
    tri_id_map: np.ndarray,
    bary_map: np.ndarray,
    observations: List[ViewObservation],
    front_face_uv_mask: np.ndarray,
    prefer_pytorch3d: bool = True,
    occlusion_faces: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Project all camera views onto UV atlas texels with foreground mask, occlusion & angle weighting."""
    accum_rgb, accum_w, used_p3d_any, _, _ = _bake_multiview_core(
        uv_layout=uv_layout,
        tri_id_map=tri_id_map,
        bary_map=bary_map,
        observations=observations,
        front_face_uv_mask=front_face_uv_mask,
        prefer_pytorch3d=prefer_pytorch3d,
        occlusion_faces=occlusion_faces,
    )
    return accum_rgb, accum_w, used_p3d_any


def _dilate_uv_gutters(
    tex: np.ndarray,
    inside_uv: np.ndarray,
    dilate_px: int = UV_GUTTER_DILATE_PX,
) -> np.ndarray:
    """Dilate valid UV island border pixels into gutter padding to prevent glTF bilinear seam artifacts."""
    if dilate_px <= 0 or not np.any(inside_uv) or not np.any(~inside_uv):
        return tex
    dists, gutter_idx = distance_transform_edt(~inside_uv, return_indices=True)
    pad_mask = (~inside_uv) & (dists <= float(dilate_px))
    out = tex.copy()
    out[pad_mask] = tex[gutter_idx[0][pad_mask], gutter_idx[1][pad_mask]]
    return out


def _fill_deep_cavities_with_local_median(
    tex: np.ndarray,
    deep_holes: np.ndarray,
    valid_source: np.ndarray,
    non_face_uv: np.ndarray,
) -> None:
    """Fill shadowed non-face cavities (e.g. armpits, soles) using per-island observed median RGB."""
    if not np.any(deep_holes) or not np.any(valid_source):
        return
    global_med = np.round(np.median(tex[valid_source], axis=0)).astype(np.uint8)
    _, labels = cv2.connectedComponents(non_face_uv.astype(np.uint8), connectivity=8)
    hole_labels = np.unique(labels[deep_holes])
    for lbl in hole_labels:
        comp_holes = deep_holes & (labels == lbl)
        if lbl == 0:
            tex[comp_holes] = global_med
            continue
        comp_valid = valid_source & (labels == lbl)
        fill_rgb = (
            np.round(np.median(tex[comp_valid], axis=0)).astype(np.uint8)
            if np.any(comp_valid)
            else global_med
        )
        tex[comp_holes] = fill_rgb


def _run_redblack_sor(
    cur: np.ndarray,
    dom: np.ndarray,
    unk: np.ndarray,
    n_iter: int = 36,
    omega: float = 1.78,
) -> np.ndarray:
    """Solve Laplace interpolation on unknown domain texels using checkerboard Red-Black SOR."""
    h, w = dom.shape
    rows_u, cols_u = np.where(unk)
    if len(rows_u) == 0:
        return cur

    dom_f = dom.astype(np.float32)
    dom_pad = np.pad(dom_f, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    cur_pad = np.pad(cur.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0.0)

    is_red = ((rows_u + cols_u) & 1) == 0
    passes = []
    for mask_rb in (is_red, ~is_red):
        rr = rows_u[mask_rb]
        cc = cols_u[mask_rb]
        if len(rr) == 0:
            continue
        rp, cp = rr + 1, cc + 1
        wn = dom_pad[rp - 1, cp]
        ws = dom_pad[rp + 1, cp]
        ww = dom_pad[rp, cp - 1]
        we = dom_pad[rp, cp + 1]
        deg = wn + ws + ww + we
        keep = deg > 1e-5
        if np.any(keep):
            passes.append((
                rp[keep],
                cp[keep],
                wn[keep][:, None],
                ws[keep][:, None],
                ww[keep][:, None],
                we[keep][:, None],
                deg[keep][:, None],
            ))

    if not passes:
        return cur

    for _ in range(n_iter):
        for rp, cp, wn, ws, ww, we, deg in passes:
            avg = (
                wn * cur_pad[rp - 1, cp]
                + ws * cur_pad[rp + 1, cp]
                + ww * cur_pad[rp, cp - 1]
                + we * cur_pad[rp, cp + 1]
            ) / deg
            cur_pad[rp, cp] = np.clip(cur_pad[rp, cp] + omega * (avg - cur_pad[rp, cp]), 0.0, 255.0)

    return cur_pad[1 : h + 1, 1 : w + 1]


def _harmonically_fill_chart(
    tex_f: np.ndarray,
    domain: np.ndarray,
    seed: np.ndarray,
    holes: np.ndarray,
    max_levels: int = 10,
) -> np.ndarray:
    """Smoothly interpolate holes inside an isolated UV chart via Voronoi init + 2-scale harmonic SOR."""
    del max_levels
    if not np.any(holes) or not np.any(seed):
        return tex_f

    dom = np.asarray(domain, dtype=bool)
    sd = np.asarray(seed, dtype=bool) & dom & (~holes)
    if not np.any(sd):
        return tex_f

    ch, cw = dom.shape
    dist_0, idx_0 = distance_transform_edt(~sd, return_distances=True, return_indices=True)
    cur = np.where(sd[..., None], tex_f.astype(np.float32), 0.0)
    unk = dom & (~sd)
    if np.any(unk):
        cur[unk] = tex_f[idx_0[0][unk], idx_0[1][unk]].astype(np.float32)

    # Coarse 4x harmonic SOR pass only for deeper holes (> 4 px) so wide seams blend smoothly
    # without pulling distant chart-edge colors into narrow crevices.
    max_hole_dist = float(dist_0[holes].max())
    if max_hole_dist > 6.0 and ch >= 16 and cw >= 16:
        scale = 4
        nh = (ch + scale - 1) // scale
        nw = (cw + scale - 1) // scale
        ph, pw = nh * scale, nw * scale
        pad_img = np.pad(cur, ((0, ph - ch), (0, pw - cw), (0, 0)), mode="constant")
        pad_sd = np.pad(sd.astype(np.float32), ((0, ph - ch), (0, pw - cw)), mode="constant")
        pad_dom = np.pad(dom, ((0, ph - ch), (0, pw - cw)), mode="constant")
        w_c = pad_sd.reshape(nh, scale, nw, scale).sum(axis=(1, 3))
        sd_c = w_c > 1e-6
        dom_c = pad_dom.reshape(nh, scale, nw, scale).any(axis=(1, 3))
        if np.any(sd_c) and np.any(dom_c & (~sd_c)):
            img_sum_c = (pad_img * pad_sd[..., None]).reshape(nh, scale, nw, scale, 3).sum(axis=(1, 3))
            cur_c = np.where(
                w_c[..., None] > 1e-6,
                img_sum_c / np.maximum(w_c[..., None], 1e-6),
                0.0,
            ).astype(np.float32)
            unk_c = dom_c & (~sd_c)
            _, c_idx = distance_transform_edt(~sd_c, return_distances=True, return_indices=True)
            cur_c[unk_c] = cur_c[c_idx[0][unk_c], c_idx[1][unk_c]]
            cur_c = _run_redblack_sor(cur_c, dom_c, unk_c, n_iter=36, omega=1.78)
            dom_cf = dom_c.astype(np.float32)
            num_up = cv2.resize(cur_c * dom_cf[..., None], (pw, ph), interpolation=cv2.INTER_LINEAR)[:ch, :cw]
            den_up = cv2.resize(dom_cf, (pw, ph), interpolation=cv2.INTER_LINEAR)[:ch, :cw]
            valid_up = unk & (den_up > 1e-5) & (dist_0 > 4.0)
            if np.any(valid_up):
                up_rgb = num_up[valid_up] / den_up[valid_up, None]
                blend = np.clip((dist_0[valid_up] - 4.0) / 4.0, 0.0, 1.0)[:, None]
                cur[valid_up] = (1.0 - blend) * cur[valid_up] + blend * up_rgb

    if np.any(unk):
        cur = _run_redblack_sor(cur, dom, unk, n_iter=36, omega=1.78)

    out = tex_f.copy()
    out[holes] = cur[holes]
    return out


def inpaint_and_dilate_texture(
    accum_rgb: np.ndarray,
    accum_w: np.ndarray,
    tri_id_map: np.ndarray,
    dilate_px: int = UV_GUTTER_DILATE_PX,
    strict_no_invent: bool = True,
    no_inpaint_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize blended colors, smoothly fill unseen UV holes via harmonic chart diffusion, and dilate gutters."""
    h, w = tri_id_map.shape
    has_tex = (tri_id_map >= 0) & (accum_w > 1e-6)
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    if not np.any(has_tex):
        return tex

    tex_f = np.zeros((h, w, 3), dtype=np.float32)
    tex_f[has_tex] = accum_rgb[has_tex] / accum_w[has_tex, None]
    tex = np.clip(np.round(tex_f), 0, 255).astype(np.uint8)

    holes = (tri_id_map >= 0) & (~has_tex)
    protected = (
        np.asarray(no_inpaint_mask, dtype=bool)
        if no_inpaint_mask is not None
        else np.zeros((h, w), dtype=bool)
    )
    fill_holes = holes & (~protected)

    if np.any(fill_holes):
        if strict_no_invent:
            valid_source = has_tex & (~protected)
            if not np.any(valid_source):
                valid_source = has_tex
            non_face_uv = (tri_id_map >= 0) & (~protected)
            num_isl, isl_labels, isl_stats, _ = cv2.connectedComponentsWithStats(
                non_face_uv.astype(np.uint8), connectivity=8
            )
            fallback_deep = np.zeros_like(fill_holes)
            for isl_id in range(1, num_isl):
                x0 = int(isl_stats[isl_id, cv2.CC_STAT_LEFT])
                y0 = int(isl_stats[isl_id, cv2.CC_STAT_TOP])
                bw = int(isl_stats[isl_id, cv2.CC_STAT_WIDTH])
                bh = int(isl_stats[isl_id, cv2.CC_STAT_HEIGHT])
                x1, y1 = x0 + bw, y0 + bh
                isl_sub = isl_labels[y0:y1, x0:x1] == isl_id
                holes_sub = fill_holes[y0:y1, x0:x1] & isl_sub
                if not np.any(holes_sub):
                    continue
                src_sub = valid_source[y0:y1, x0:x1] & isl_sub
                if not np.any(src_sub):
                    fallback_deep[y0:y1, x0:x1][holes_sub] = True
                    continue
                filled_sub = _harmonically_fill_chart(
                    tex_f[y0:y1, x0:x1],
                    domain=isl_sub,
                    seed=src_sub,
                    holes=holes_sub,
                )
                tex_f[y0:y1, x0:x1][holes_sub] = filled_sub[holes_sub]
                tex[y0:y1, x0:x1][holes_sub] = np.clip(
                    np.round(filled_sub[holes_sub]), 0, 255
                ).astype(np.uint8)
            if np.any(fallback_deep):
                _fill_deep_cavities_with_local_median(
                    tex,
                    deep_holes=fallback_deep,
                    valid_source=valid_source,
                    non_face_uv=non_face_uv,
                )
                tex_f[fallback_deep] = tex[fallback_deep].astype(np.float32)
        else:
            dist_from_valid, indices = distance_transform_edt(
                ~has_tex, return_distances=True, return_indices=True
            )
            coarse = tex[indices[0], indices[1]]
            near_holes = fill_holes & (dist_from_valid <= 12.0)
            far_holes = fill_holes & (dist_from_valid > 12.0)

            work = tex.copy()
            work[far_holes] = coarse[far_holes]
            if np.any(near_holes):
                work = cv2.inpaint(
                    work,
                    near_holes.astype(np.uint8) * 255,
                    inpaintRadius=7,
                    flags=cv2.INPAINT_TELEA,
                )
            tex[fill_holes] = work[fill_holes]
            tex_f[fill_holes] = tex[fill_holes].astype(np.float32)

    # Smoothly fill protected/facial crevices from observed neighbor texels inside the same
    # facial/UV chart (zero [0,0,0] black holes, zero cv2.inpaint hallucination).
    prot_holes = holes & protected
    if strict_no_invent and np.any(prot_holes):
        prot_uv = (tri_id_map >= 0) & protected
        num_p, p_labels, p_stats, _ = cv2.connectedComponentsWithStats(
            prot_uv.astype(np.uint8), connectivity=8
        )
        fallback_prot = np.zeros_like(prot_holes)
        for pid in range(1, num_p):
            x0 = int(p_stats[pid, cv2.CC_STAT_LEFT])
            y0 = int(p_stats[pid, cv2.CC_STAT_TOP])
            bw = int(p_stats[pid, cv2.CC_STAT_WIDTH])
            bh = int(p_stats[pid, cv2.CC_STAT_HEIGHT])
            x1, y1 = x0 + bw, y0 + bh
            comp_sub = p_labels[y0:y1, x0:x1] == pid
            holes_sub = prot_holes[y0:y1, x0:x1] & comp_sub
            if not np.any(holes_sub):
                continue
            src_sub = has_tex[y0:y1, x0:x1] & comp_sub
            if not np.any(src_sub):
                fallback_prot[y0:y1, x0:x1][holes_sub] = True
                continue
            filled_sub = _harmonically_fill_chart(
                tex_f[y0:y1, x0:x1],
                domain=comp_sub,
                seed=src_sub,
                holes=holes_sub,
            )
            tex[y0:y1, x0:x1][holes_sub] = np.clip(
                np.round(filled_sub[holes_sub]), 0, 255
            ).astype(np.uint8)
        if np.any(fallback_prot):
            all_uv = tri_id_map >= 0
            num_a, a_labels, a_stats, _ = cv2.connectedComponentsWithStats(
                all_uv.astype(np.uint8), connectivity=8
            )
            for aid in range(1, num_a):
                x0 = int(a_stats[aid, cv2.CC_STAT_LEFT])
                y0 = int(a_stats[aid, cv2.CC_STAT_TOP])
                bw = int(a_stats[aid, cv2.CC_STAT_WIDTH])
                bh = int(a_stats[aid, cv2.CC_STAT_HEIGHT])
                x1, y1 = x0 + bw, y0 + bh
                isl_sub = a_labels[y0:y1, x0:x1] == aid
                holes_sub = fallback_prot[y0:y1, x0:x1] & isl_sub
                if not np.any(holes_sub):
                    continue
                src_sub = (has_tex[y0:y1, x0:x1] | fill_holes[y0:y1, x0:x1]) & isl_sub
                if not np.any(src_sub):
                    continue
                filled_sub = _harmonically_fill_chart(
                    tex_f[y0:y1, x0:x1],
                    domain=isl_sub,
                    seed=src_sub,
                    holes=holes_sub,
                )
                tex[y0:y1, x0:x1][holes_sub] = np.clip(
                    np.round(filled_sub[holes_sub]), 0, 255
                ).astype(np.uint8)
            still_zero = fallback_prot & (tex.sum(axis=-1) == 0)
            if np.any(still_zero) and np.any(has_tex):
                tex[still_zero] = np.round(np.median(tex[has_tex], axis=0)).astype(np.uint8)

    return _dilate_uv_gutters(tex, tri_id_map >= 0, dilate_px=dilate_px)


def generate_blurred_face_texture(
    tex_face_rgb: np.ndarray,
    head_uv_mask: np.ndarray,
    tri_id_map: np.ndarray,
    dilate_px: int = UV_GUTTER_DILATE_PX,
    blur_target_mask: Optional[np.ndarray] = None,
    preserve_sharp_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create Privacy Level 2 (tex_blur) blurring facial skin while preserving hair crisp (3b)."""
    h, w = tex_face_rgb.shape[:2]
    sharp_mask = (
        np.asarray(preserve_sharp_mask, dtype=bool)
        if preserve_sharp_mask is not None
        else np.zeros((h, w), dtype=bool)
    )
    raw_target = (
        np.asarray(blur_target_mask, dtype=bool)
        if blur_target_mask is not None
        else np.asarray(head_uv_mask, dtype=bool)
    )
    target_mask = raw_target & (~sharp_mask)

    tex_blur = tex_face_rgb.copy()
    if not np.any(target_mask):
        return tex_blur

    mask_2d = target_mask.astype(np.float32)

    ys, xs = np.nonzero(target_mask)
    target_span = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))

    # Normalized mosaic downscale inside target facial skin UV mask only (zero hair / gutter bleed).
    block_px = max(12, w // 96, target_span // 14)
    small_w = max(8, w // block_px)
    small_h = max(8, h // block_px)
    small_num = cv2.resize(
        tex_face_rgb.astype(np.float32) * mask_2d[..., None],
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )
    small_den_2d = cv2.resize(mask_2d, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_valid = small_den_2d > 1e-5
    small_clean = np.zeros_like(small_num)
    small_clean[small_valid] = small_num[small_valid] / small_den_2d[small_valid, None]
    if np.any(small_valid) and np.any(~small_valid):
        _, nearest_idx = distance_transform_edt(~small_valid, return_indices=True)
        small_clean[~small_valid] = small_clean[
            nearest_idx[0][~small_valid], nearest_idx[1][~small_valid]
        ]
    pixelated = cv2.resize(small_clean, (w, h), interpolation=cv2.INTER_LINEAR)

    ksize = min(151, max(21, (w // 28) | 1, (target_span // 12) | 1))
    sigma = max(4.0, float(ksize) * 0.28)
    num = cv2.GaussianBlur(pixelated * mask_2d[..., None], (ksize, ksize), sigmaX=sigma)
    den = np.maximum(cv2.GaussianBlur(mask_2d, (ksize, ksize), sigmaX=sigma), 1e-5)[..., None]
    blurred_clean = np.clip(np.round(num / den), 0, 255).astype(np.uint8)

    tex_blur[target_mask] = blurred_clean[target_mask]
    # Re-dilate UV gutters while strictly preserving sharp hair pixels bit-identical.
    tex_blur = _dilate_uv_gutters(tex_blur, tri_id_map >= 0, dilate_px=dilate_px)
    if np.any(sharp_mask):
        tex_blur[sharp_mask] = tex_face_rgb[sharp_mask]
    return tex_blur


def _build_submesh_uv_layout(
    verts_a: np.ndarray,
    all_faces: np.ndarray,
    sub_tri_indices: np.ndarray,
    mhr_uvs: Optional[np.ndarray] = None,
    mhr_uv_faces: Optional[np.ndarray] = None,
    rescale_uvs: bool = False,
) -> UVLayout:
    """Build UVLayout for a triangle subset while keeping vmapping/orig_faces indexed into verts_a."""
    sub_faces = np.asarray(all_faces[sub_tri_indices], dtype=np.int64)
    if mhr_uvs is not None:
        uvs_arr = np.asarray(mhr_uvs, dtype=np.float32)
        if mhr_uv_faces is not None:
            sub_uv_faces_raw = np.asarray(mhr_uv_faces, dtype=np.int64)[sub_tri_indices]
            used_uv_idx, inv = np.unique(sub_uv_faces_raw.reshape(-1), return_inverse=True)
            full_vmapping = np.zeros(len(uvs_arr), dtype=np.int64)
            full_vmapping[sub_uv_faces_raw.reshape(-1)] = sub_faces.reshape(-1)
            sub_vmapping = full_vmapping[used_uv_idx]
            sub_uv = uvs_arr[used_uv_idx].copy()
            sub_uv_faces = inv.reshape(-1, 3).astype(np.int64)
            src = "mhr"
        elif len(uvs_arr) == len(verts_a):
            used_uv_idx, inv = np.unique(sub_faces.reshape(-1), return_inverse=True)
            sub_vmapping = used_uv_idx.astype(np.int64)
            sub_uv = uvs_arr[used_uv_idx].copy()
            sub_uv_faces = inv.reshape(-1, 3).astype(np.int64)
            src = "mhr"
        else:
            src = ""

        if src:
            if rescale_uvs and len(sub_uv) > 0:
                uv_min = sub_uv.min(axis=0)
                uv_max = sub_uv.max(axis=0)
                span = float(max(uv_max[0] - uv_min[0], uv_max[1] - uv_min[1]))
                if 1e-5 < span < 0.85:
                    sub_uv = np.clip(
                        0.04 + (sub_uv - uv_min[None, :]) * (0.92 / span), 0.0, 1.0
                    ).astype(np.float32)
            return UVLayout(
                uv=sub_uv,
                uv_faces=sub_uv_faces,
                vmapping=sub_vmapping,
                orig_faces=sub_faces,
                source=src,
            )

    used_verts, inv = np.unique(sub_faces.reshape(-1), return_inverse=True)
    local_verts = np.asarray(verts_a[used_verts], dtype=np.float32)
    local_faces = inv.reshape(-1, 3).astype(np.int64)
    local_layout = build_uv_layout(local_verts, local_faces)
    return UVLayout(
        uv=local_layout.uv,
        uv_faces=local_layout.uv_faces,
        vmapping=used_verts[local_layout.vmapping],
        orig_faces=sub_faces,
        source=local_layout.source,
    )


def _bake_single_atlas(
    verts_a: np.ndarray,
    joints_a: np.ndarray,
    uv_layout: UVLayout,
    observations: List[ViewObservation],
    mhr_model: Optional[Any],
    tex_size: int,
    prefer_pytorch3d: bool,
    occlusion_faces: Optional[np.ndarray] = None,
    head_vert_mask: Optional[np.ndarray] = None,
    hair_vert_mask: Optional[np.ndarray] = None,
    face_skin_vert_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Rasterize, bake, strictly inpaint, and Level-2 blur a single UV atlas."""
    tri_id_map, bary_map = rasterize_uv_atlas_map(uv_layout, tex_size=tex_size)
    head_uv_mask, front_face_uv_mask = _build_uv_head_masks(
        verts_a=verts_a,
        joints_a=joints_a,
        uv_layout=uv_layout,
        tri_id_map=tri_id_map,
        bary_map=bary_map,
        mhr_model=mhr_model,
        head_vert_mask=head_vert_mask,
    )
    rows, cols = np.where(tri_id_map >= 0)
    if hair_vert_mask is not None:
        hair_uv_mask = _rasterize_vertex_mask_to_uv(
            hair_vert_mask, uv_layout.orig_faces, tri_id_map, bary_map, rows, cols
        )
    else:
        hair_uv_mask = np.zeros((tex_size, tex_size), dtype=bool)

    if face_skin_vert_mask is not None:
        face_blur_uv_mask = (
            _rasterize_vertex_mask_to_uv(
                face_skin_vert_mask, uv_layout.orig_faces, tri_id_map, bary_map, rows, cols
            )
            & (~hair_uv_mask)
        )
    elif hair_vert_mask is not None:
        face_blur_uv_mask = head_uv_mask & (~hair_uv_mask)
    else:
        face_blur_uv_mask = head_uv_mask.copy()

    non_arm_uv_mask = _build_non_arm_uv_mask(
        verts_a=verts_a,
        joints_a=joints_a,
        uv_layout=uv_layout,
        tri_id_map=tri_id_map,
        bary_map=bary_map,
    )
    accum_rgb, accum_w, used_p3d, accum_hair_w, accum_face_skin_w = _bake_multiview_core(
        uv_layout=uv_layout,
        tri_id_map=tri_id_map,
        bary_map=bary_map,
        observations=observations,
        front_face_uv_mask=front_face_uv_mask | face_blur_uv_mask,
        prefer_pytorch3d=prefer_pytorch3d,
        occlusion_faces=occlusion_faces,
        non_arm_uv_mask=non_arm_uv_mask,
    )

    valid_atlas = tri_id_map >= 0
    coverage_mask = valid_atlas & (accum_w > 1e-6)
    if hair_vert_mask is None and np.any(accum_hair_w > 0.0):
        hair_uv_mask = head_uv_mask & coverage_mask & (accum_hair_w > 0.5 * np.maximum(accum_w, 1e-6))
        if face_skin_vert_mask is None:
            if np.any(accum_face_skin_w > 0.0):
                face_blur_uv_mask = (
                    (head_uv_mask & (~hair_uv_mask))
                    | (coverage_mask & (accum_face_skin_w > 0.5 * np.maximum(accum_w, 1e-6)))
                ) & (~hair_uv_mask)
            else:
                face_blur_uv_mask = head_uv_mask & (~hair_uv_mask)

    coverage_pct = float(100.0 * np.sum(coverage_mask) / max(1, int(np.sum(valid_atlas))))
    no_inpaint_mask = face_blur_uv_mask | front_face_uv_mask
    tex_face_rgb = inpaint_and_dilate_texture(
        accum_rgb,
        accum_w,
        tri_id_map,
        strict_no_invent=True,
        no_inpaint_mask=no_inpaint_mask,
    )
    tex_blur_rgb = generate_blurred_face_texture(
        tex_face_rgb,
        head_uv_mask,
        tri_id_map,
        blur_target_mask=face_blur_uv_mask,
        preserve_sharp_mask=hair_uv_mask,
    )
    return {
        "tri_id_map": tri_id_map,
        "head_uv_mask": head_uv_mask,
        "hair_uv_mask": hair_uv_mask,
        "face_blur_uv_mask": face_blur_uv_mask,
        "coverage_mask": coverage_mask,
        "coverage_pct": coverage_pct,
        "tex_face_rgb": tex_face_rgb,
        "tex_blur_rgb": tex_blur_rgb,
        "used_p3d": used_p3d,
    }


def run_step3_texturing(
    verts_a: np.ndarray,
    faces: np.ndarray,
    joints_a: np.ndarray,
    observations: List[ViewObservation],
    mhr_model: Optional[Any] = None,
    head_assets: Optional[Dict[str, Any]] = None,
    tex_size: int = DEFAULT_TEXTURE_SIZE,
    mhr_uvs: Optional[np.ndarray] = None,
    mhr_uv_faces: Optional[np.ndarray] = None,
    prefer_pytorch3d: bool = True,
    head_vert_mask: Optional[np.ndarray] = None,
    hair_vert_mask: Optional[np.ndarray] = None,
    face_skin_vert_mask: Optional[np.ndarray] = None,
    dual_material: bool = True,
    head_tex_size: int = HEAD_TEXTURE_SIZE,
) -> TexturedScanOutputs:
    """Execute Step 3 UV unwrap, masked multi-view baking, strict inpainting, dual-material UVs, and Level 2/3 textures."""
    verify_license_compliance()
    faces_arr = np.asarray(faces, dtype=np.int64)
    if mhr_uvs is None:
        mhr_uvs, mhr_uv_faces = extract_mhr_uvs(mhr_model, head_assets, len(faces_arr))

    has_partition_masks = (
        head_vert_mask is not None or hair_vert_mask is not None or face_skin_vert_mask is not None
    )
    head_tri_mask: Optional[np.ndarray] = None
    eff_head_v: Optional[np.ndarray] = None
    if dual_material and has_partition_masks:
        eff_head_v = np.zeros(len(verts_a), dtype=bool)
        for vm in (head_vert_mask, hair_vert_mask, face_skin_vert_mask):
            if vm is not None:
                eff_head_v |= np.asarray(vm, dtype=bool)
        head_tri_mask = eff_head_v[faces_arr].all(axis=1)
        if not np.any(head_tri_mask) and np.any(eff_head_v):
            head_tri_mask = eff_head_v[faces_arr].any(axis=1)

    if head_tri_mask is not None and 0 < int(np.sum(head_tri_mask)) < len(faces_arr):
        body_tri_idx = np.where(~head_tri_mask)[0]
        head_tri_idx = np.where(head_tri_mask)[0]

        body_uv_layout = _build_submesh_uv_layout(
            verts_a,
            faces_arr,
            body_tri_idx,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=mhr_uv_faces,
            rescale_uvs=False,
        )
        head_uv_layout = _build_submesh_uv_layout(
            verts_a,
            faces_arr,
            head_tri_idx,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=mhr_uv_faces,
            rescale_uvs=True,
        )

        body_res = _bake_single_atlas(
            verts_a=verts_a,
            joints_a=joints_a,
            uv_layout=body_uv_layout,
            observations=observations,
            mhr_model=mhr_model,
            tex_size=tex_size,
            prefer_pytorch3d=prefer_pytorch3d,
            occlusion_faces=faces_arr,
            head_vert_mask=np.zeros(len(verts_a), dtype=bool),
        )
        head_res = _bake_single_atlas(
            verts_a=verts_a,
            joints_a=joints_a,
            uv_layout=head_uv_layout,
            observations=observations,
            mhr_model=mhr_model,
            tex_size=head_tex_size,
            prefer_pytorch3d=prefer_pytorch3d,
            occlusion_faces=faces_arr,
            head_vert_mask=eff_head_v,
            hair_vert_mask=hair_vert_mask,
            face_skin_vert_mask=face_skin_vert_mask,
        )

        n_body = len(body_tri_idx)
        n_head = len(head_tri_idx)
        combined_cov_pct = (
            n_body * body_res["coverage_pct"] + n_head * head_res["coverage_pct"]
        ) / max(1, n_body + n_head)

        return TexturedScanOutputs(
            uv_layout=body_uv_layout,
            tex_face_rgb=body_res["tex_face_rgb"],
            tex_blur_rgb=body_res["tex_blur_rgb"],
            head_uv_mask=body_res["head_uv_mask"],
            coverage_mask=body_res["coverage_mask"],
            used_pytorch3d=bool(body_res["used_p3d"] or head_res["used_p3d"]),
            face_blur_uv_mask=head_res["face_blur_uv_mask"],
            hair_uv_mask=head_res["hair_uv_mask"],
            head_uv_layout=head_uv_layout,
            head_tex_face_rgb=head_res["tex_face_rgb"],
            head_tex_blur_rgb=head_res["tex_blur_rgb"],
            coverage_pct=combined_cov_pct,
            head_coverage_pct=head_res["coverage_pct"],
            head_coverage_mask=head_res["coverage_mask"],
        )

    uv_layout = build_uv_layout(verts_a, faces_arr, mhr_uvs=mhr_uvs, mhr_uv_faces=mhr_uv_faces)
    res = _bake_single_atlas(
        verts_a=verts_a,
        joints_a=joints_a,
        uv_layout=uv_layout,
        observations=observations,
        mhr_model=mhr_model,
        tex_size=tex_size,
        prefer_pytorch3d=prefer_pytorch3d,
        occlusion_faces=faces_arr,
        head_vert_mask=head_vert_mask,
        hair_vert_mask=hair_vert_mask,
        face_skin_vert_mask=face_skin_vert_mask,
    )
    head_cov_mask = res["coverage_mask"] & res["head_uv_mask"]
    head_valid = res["head_uv_mask"] & (res["tri_id_map"] >= 0)
    head_cov_pct = (
        float(100.0 * np.sum(head_cov_mask) / max(1, int(np.sum(head_valid))))
        if np.any(head_valid)
        else 0.0
    )

    return TexturedScanOutputs(
        uv_layout=uv_layout,
        tex_face_rgb=res["tex_face_rgb"],
        tex_blur_rgb=res["tex_blur_rgb"],
        head_uv_mask=res["head_uv_mask"],
        coverage_mask=res["coverage_mask"],
        used_pytorch3d=bool(res["used_p3d"]),
        face_blur_uv_mask=res["face_blur_uv_mask"],
        hair_uv_mask=res["hair_uv_mask"],
        head_uv_layout=None,
        head_tex_face_rgb=None,
        head_tex_blur_rgb=None,
        coverage_pct=res["coverage_pct"],
        head_coverage_pct=head_cov_pct,
        head_coverage_mask=head_cov_mask,
    )


def _scalar_float(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    arr = np.asarray(val, dtype=np.float32).reshape(-1)
    return float(arr[0])


def _extract_seg_field(seg: Any, name: str) -> Optional[np.ndarray]:
    if seg is None:
        return None
    val = seg.get(name) if isinstance(seg, dict) else getattr(seg, name, None)
    return np.asarray(val) if val is not None else None


def _first_not_none(*vals: Any) -> Any:
    for val in vals:
        if val is not None:
            return val
    return None


def _fit_mask_to_shape(mask: Optional[np.ndarray], img_h: int, img_w: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[:2] != (img_h, img_w):
        arr = cv2.resize(arr.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5
    return arr.astype(bool)


def build_view_observations(
    raw_results: List[Dict[str, Any]],
    weights: Optional[np.ndarray] = None,
    closeup_face_result: Optional[Dict[str, Any]] = None,
    max_side: int = 1600,
    segmented_views: Optional[List[Any]] = None,
    fitted_cameras: Optional[Sequence[Optional[Tuple[np.ndarray, np.ndarray, float]]]] = None,
    fitted_view_verts: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> List[ViewObservation]:
    """Convert raw SAM 3D Body inference records and optional fitted cameras/meshes into ViewObservations."""
    observations: List[ViewObservation] = []
    seg_by_idx: Dict[int, Any] = {}
    seg_pos_by_idx: Dict[int, int] = {}
    if segmented_views:
        for pos, sv in enumerate(segmented_views):
            sv_idx = sv.get("view_index", pos) if isinstance(sv, dict) else getattr(sv, "view_index", pos)
            seg_by_idx[int(sv_idx)] = sv
            seg_pos_by_idx[int(sv_idx)] = pos

    def _resolve_override(
        seq: Optional[Sequence[Optional[Any]]],
        rec_idx: Optional[int],
    ) -> Optional[Any]:
        if seq is None or rec_idx is None:
            return None
        if len(seq) == len(raw_results) and 0 <= rec_idx < len(seq):
            return seq[rec_idx]
        seg_pos = seg_pos_by_idx.get(rec_idx)
        if seg_pos is not None and 0 <= seg_pos < len(seq):
            return seq[seg_pos]
        if 0 <= rec_idx < len(seq):
            return seq[rec_idx]
        return None

    def _convert(
        rec: Dict[str, Any],
        w: float,
        is_face: bool,
        rec_idx: Optional[int] = None,
    ) -> Optional[ViewObservation]:
        out = rec.get("output", {})
        cam_override = None if is_face else _resolve_override(fitted_cameras, rec_idx)
        verts_override = None if is_face else _resolve_override(fitted_view_verts, rec_idx)

        has_verts = verts_override is not None or "pred_vertices" in out
        has_cam = cam_override is not None or ("pred_cam_t" in out and "focal_length" in out)
        if not has_verts or not has_cam:
            return None
        img_rgb = rec.get("image_rgb")
        if img_rgb is None:
            path = rec.get("image_path", "")
            if not path:
                return None
            img_rgb = prepare_image_rgb(path, max_side=max_side)

        img_arr = np.asarray(img_rgb, dtype=np.uint8)
        img_h, img_w = img_arr.shape[:2]
        seg = None if is_face else (seg_by_idx.get(rec_idx) if rec_idx is not None else None)
        if seg is None and not is_face:
            seg = rec.get("segmented_view")

        fg_mask = _first_not_none(
            _extract_seg_field(seg, "person_mask"),
            rec.get("foreground_mask"),
            rec.get("person_mask"),
        )
        hair_mask = _first_not_none(
            _extract_seg_field(seg, "hair_mask"),
            rec.get("hair_mask"),
        )
        face_skin_mask = _first_not_none(
            _extract_seg_field(seg, "face_skin_mask"),
            rec.get("face_skin_mask"),
        )

        calibrated = False
        pred_k3 = (
            np.asarray(out["pred_keypoints_3d"], dtype=np.float32)
            if out.get("pred_keypoints_3d") is not None
            else None
        )
        raw_k2 = (
            np.asarray(out["pred_keypoints_2d"], dtype=np.float32)
            if out.get("pred_keypoints_2d") is not None
            else None
        )
        pred_k2 = raw_k2
        r_mat_for_arm = None

        if verts_override is not None and cam_override is not None:
            r_v, t_v, f_v = cam_override
            r_mat = np.asarray(r_v, dtype=np.float32)
            r_mat_for_arm = r_mat
            verts_val = np.asarray(verts_override, dtype=np.float32) @ r_mat.T
            cam_t_val = np.asarray(t_v, dtype=np.float32).reshape(3)
            focal_val = float(f_v)
            calibrated = True
            pred_k3 = None
            pred_k2 = None
        elif verts_override is not None:
            verts_val = np.asarray(verts_override, dtype=np.float32)
            cam_t_val = np.zeros(3, dtype=np.float32)
            focal_val = _scalar_float(out["focal_length"])
            calibrated = True
            pred_k3 = None
            pred_k2 = None
        elif cam_override is not None:
            r_v, t_v, f_v = cam_override
            r_mat_for_arm = np.asarray(r_v, dtype=np.float32)
            verts_val = np.asarray(out["pred_vertices"], dtype=np.float32)
            cam_t_val = np.asarray(t_v, dtype=np.float32).reshape(3)
            focal_val = float(f_v)
        else:
            verts_val = np.asarray(out["pred_vertices"], dtype=np.float32)
            cam_t_val = np.asarray(out["pred_cam_t"], dtype=np.float32).reshape(3)
            focal_val = _scalar_float(out["focal_length"])

        v_name = rec.get("image_name", "")
        arm_occ_mask = _fit_mask_to_shape(rec.get("arm_occlusion_mask"), img_h, img_w)
        if arm_occ_mask is None and not is_face and raw_k2 is not None:
            arm_occ_mask = _infer_side_arm_occlusion_mask(
                raw_k2,
                img_h=img_h,
                img_w=img_w,
                view_name=v_name,
                r_mat=r_mat_for_arm,
            )

        return ViewObservation(
            image_rgb=img_arr,
            pred_vertices=verts_val,
            pred_cam_t=cam_t_val,
            focal_length=focal_val,
            pred_keypoints_3d=pred_k3,
            pred_keypoints_2d=pred_k2,
            weight=float(w),
            is_closeup_face=is_face,
            view_name=v_name,
            foreground_mask=_fit_mask_to_shape(fg_mask, img_h, img_w),
            hair_mask=_fit_mask_to_shape(hair_mask, img_h, img_w),
            face_skin_mask=_fit_mask_to_shape(face_skin_mask, img_h, img_w),
            arm_occlusion_mask=arm_occ_mask,
            mask_erode_px=int(rec.get("mask_erode_px", FOREGROUND_MASK_ERODE_PX)),
            calibrated_camera=calibrated,
        )

    for idx, rec in enumerate(raw_results):
        w = float(weights[idx]) if weights is not None and idx < len(weights) else 1.0
        if w <= 0.0:
            continue
        obs = _convert(rec, w=w, is_face=False, rec_idx=idx)
        if obs is not None:
            observations.append(obs)

    if closeup_face_result is not None:
        face_obs = _convert(closeup_face_result, w=1.0, is_face=True, rec_idx=None)
        if face_obs is not None:
            observations.append(face_obs)

    return observations
