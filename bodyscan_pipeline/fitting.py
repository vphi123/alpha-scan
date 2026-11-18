"""Metric multi-view MHR silhouette/volume fitting and weight/proportion audit."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    MHR70_TO_MHR127_JOINT_MAP,
    MIN_MHR_SILHOUETTE_IOU,
    SEX_BODY_DENSITY_KG_PER_L,
    WEIGHT_MATCH_TOL,
    WEIGHT_MISMATCH_TOL,
)
from .fusion import (
    FusedSubjectParams,
    build_arm_vertex_mask,
    build_scaled_apose_mesh,
    evaluate_mhr_pose,
    solve_straight_apose,
)
from .inference import verify_license_compliance
from .segmentation import SegmentedView


@dataclass
class ProportionAudit:
    """Anthropometric weight consistency and multi-view silhouette overlap audit."""

    declared_weight_kg: float
    estimated_weight_kg: float
    weight_diff_pct: Optional[float]
    mhr_silhouette_iou: float
    outer_silhouette_iou: Optional[float]
    status: str  # Strictly one of: "estimated_only", "matched", "mismatch_warning"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit metrics into a JSON-compatible dictionary."""
        return {
            "declared_weight_kg": float(self.declared_weight_kg),
            "estimated_weight_kg": float(self.estimated_weight_kg),
            "weight_diff_pct": None if self.weight_diff_pct is None else float(self.weight_diff_pct),
            "mhr_silhouette_iou": float(self.mhr_silhouette_iou),
            "outer_silhouette_iou": (
                None if self.outer_silhouette_iou is None else float(self.outer_silhouette_iou)
            ),
            "status": str(self.status),
            "warnings": list(self.warnings),
        }


def compute_closed_mesh_volume_liters(verts_m: Any, faces: Any) -> Any:
    """Compute exact closed mesh volume in liters (works with NumPy arrays and PyTorch tensors)."""
    if isinstance(verts_m, torch.Tensor):
        if verts_m.numel() == 0:
            return torch.zeros((), dtype=verts_m.dtype, device=verts_m.device)
        faces_t = (
            faces.to(device=verts_m.device, dtype=torch.int64)
            if isinstance(faces, torch.Tensor)
            else torch.as_tensor(faces, dtype=torch.int64, device=verts_m.device)
        )
        if faces_t.numel() == 0:
            return torch.zeros((), dtype=verts_m.dtype, device=verts_m.device)

        verts_f = verts_m.float() if not verts_m.is_floating_point() else verts_m
        v_centered = verts_f - verts_f.mean(dim=0, keepdim=True)
        v0 = v_centered[faces_t[:, 0]]
        v1 = v_centered[faces_t[:, 1]]
        v2 = v_centered[faces_t[:, 2]]
        signed_vol_m3 = torch.sum(v0 * torch.linalg.cross(v1, v2, dim=-1)) / 6.0
        return torch.abs(signed_vol_m3) * 1000.0

    verts_arr = np.asarray(verts_m, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)
    if verts_arr.size == 0 or faces_arr.size == 0:
        return 0.0

    v_centered = verts_arr - np.mean(verts_arr, axis=0, keepdims=True)
    v0 = v_centered[faces_arr[:, 0]]
    v1 = v_centered[faces_arr[:, 1]]
    v2 = v_centered[faces_arr[:, 2]]
    signed_vol_m3 = float(np.sum(v0 * np.cross(v1, v2)) / 6.0)
    return float(abs(signed_vol_m3) * 1000.0)


def estimate_weight_kg(verts_m: Any, faces: Any, sex: str) -> Any:
    """Estimate body mass (kg) from closed mesh volume (L) and sex-calibrated tissue density."""
    sex_key = (sex or "").strip().lower()
    fallback_density = float(np.mean(list(SEX_BODY_DENSITY_KG_PER_L.values())))
    density = float(SEX_BODY_DENSITY_KG_PER_L.get(sex_key, fallback_density))
    return compute_closed_mesh_volume_liters(verts_m, faces) * density


def evaluate_proportion_audit(
    verts_a: np.ndarray,
    faces: np.ndarray,
    sex: str,
    declared_weight_kg: float,
    mhr_silhouette_iou: float,
    outer_silhouette_iou: Optional[float] = None,
) -> ProportionAudit:
    """Audit calibrated mesh mass and silhouette IoU against optional declared weight."""
    est_weight_kg = float(estimate_weight_kg(np.asarray(verts_a), np.asarray(faces), sex))
    decl_w = float(declared_weight_kg or 0.0)
    mhr_iou = float(mhr_silhouette_iou)
    outer_iou = None if outer_silhouette_iou is None else float(outer_silhouette_iou)
    warnings: List[str] = []

    if decl_w <= 0.0:
        weight_diff_pct: Optional[float] = None
        status = "estimated_only"
    else:
        diff_ratio = abs(est_weight_kg - decl_w) / decl_w
        weight_diff_pct = float(diff_ratio * 100.0)
        if diff_ratio > WEIGHT_MISMATCH_TOL:
            status = "mismatch_warning"
            warnings.append(
                f"Declared weight ({decl_w:.1f} kg) differs by {weight_diff_pct:.1f}% "
                f"from photo-estimated body mass ({est_weight_kg:.1f} kg)."
            )
        elif diff_ratio > WEIGHT_MATCH_TOL:
            status = "matched"
            warnings.append(
                f"Informational: declared weight ({decl_w:.1f} kg) differs slightly by {weight_diff_pct:.1f}% "
                f"from photo-estimated body mass ({est_weight_kg:.1f} kg), within "
                f"{int(round(WEIGHT_MISMATCH_TOL * 100))}% tolerance."
            )
        else:
            status = "matched"

    if mhr_iou < MIN_MHR_SILHOUETTE_IOU:
        status = "mismatch_warning"
        warnings.append(
            f"Low MHR silhouette overlap (IoU={mhr_iou:.2f} < {MIN_MHR_SILHOUETTE_IOU:.2f}); "
            "check subject pose, framing, or foreground segmentation."
        )

    return ProportionAudit(
        declared_weight_kg=decl_w,
        estimated_weight_kg=est_weight_kg,
        weight_diff_pct=weight_diff_pct,
        mhr_silhouette_iou=mhr_iou,
        outer_silhouette_iou=outer_iou,
        status=status,
        warnings=warnings,
    )


def _to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize 2D/3D/4D mask tensor to shape (B, 1, H, W) float32."""
    if tensor.ndim == 2:
        out = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        out = tensor.unsqueeze(1)
    elif tensor.ndim == 4:
        out = tensor.permute(0, 3, 1, 2) if tensor.shape[1] != 1 and tensor.shape[-1] == 1 else tensor
    else:
        raise ValueError(f"Expected 2D/3D/4D mask tensor, got shape {tuple(tensor.shape)}")
    return out.float()


def compute_soft_silhouette_and_edge_loss(
    rendered_alpha: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable BCE + Dice silhouette loss and multi-scale Sobel edge loss."""
    pred = _to_bchw(rendered_alpha).clamp(0.0, 1.0)
    target = _to_bchw(target_mask.to(device=pred.device)).clamp(0.0, 1.0)

    pred_safe = pred.clamp(1e-6, 1.0 - 1e-6)
    l_bce = -(target * torch.log(pred_safe) + (1.0 - target) * torch.log(1.0 - pred_safe)).mean()

    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    l_dice = (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    l_sil = 0.5 * l_bce + 0.5 * l_dice

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3) / 4.0
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3) / 4.0

    gx_p = F.conv2d(pred, kx, padding=1)
    gy_p = F.conv2d(pred, ky, padding=1)
    gx_t = F.conv2d(target, kx, padding=1)
    gy_t = F.conv2d(target, ky, padding=1)

    edge_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-8)
    edge_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-8)
    edge_p_smooth = F.avg_pool2d(edge_p, kernel_size=5, stride=1, padding=2)
    edge_t_smooth = F.avg_pool2d(edge_t, kernel_size=5, stride=1, padding=2)

    l_edge = 0.5 * F.l1_loss(edge_p, edge_t) + 0.5 * F.l1_loss(edge_p_smooth, edge_t_smooth)
    return l_sil, l_edge


def _rotation_y_and_cv_flip(azimuth_deg: float) -> np.ndarray:
    """Return 3x3 rotation mapping canonical MHR (+Y up, +Z front) at azimuth_deg to OpenCV camera."""
    rad = math.radians(float(azimuth_deg))
    c, s = math.cos(rad), math.sin(rad)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    r_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    return (r_flip @ rot_y).astype(np.float32)


def _axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert 3-vector axis-angle to a (3, 3) rotation matrix via Rodrigues formula."""
    theta2 = torch.sum(rotvec * rotvec)
    theta = torch.sqrt(theta2 + 1e-12)
    k = rotvec / theta
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    kx, ky, kz = k[0], k[1], k[2]
    skew = torch.stack(
        [
            torch.stack([zero, -kz, ky]),
            torch.stack([kz, zero, -kx]),
            torch.stack([-ky, kx, zero]),
        ]
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    sin_t = torch.sin(theta)
    one_minus_cos = 1.0 - torch.cos(theta)
    return eye + sin_t * skew + one_minus_cos * (skew @ skew)


def _extract_focal_length(out: Dict[str, Any], img_h: int, img_w: int) -> float:
    """Extract scalar pinhole focal length from inference output with safe fallback."""
    raw = out.get("focal_length")
    if raw is not None:
        arr = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)
        flat = arr.reshape(-1)
        if len(flat) > 0 and np.isfinite(flat[0]) and float(flat[0]) > 1.0:
            return float(flat[0])
    return float(max(img_h, img_w))


def _resolve_raw_record(
    view: SegmentedView,
    list_idx: int,
    raw_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Match a SegmentedView back to its inference record."""
    if 0 <= view.view_index < len(raw_results):
        return raw_results[view.view_index]
    if 0 <= list_idx < len(raw_results):
        return raw_results[list_idx]
    return {}


def _pair_obs_kpts_to_skel_indices(num_obs_kpts: int, num_skel_joints: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned (obs_kpt_indices, skel_joint_indices) arrays handling MHR-70 vs >=127 schemas."""
    if num_obs_kpts <= 0 or num_skel_joints <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if num_obs_kpts >= 127:
        n_cmp = min(num_obs_kpts, num_skel_joints)
        idx = np.arange(n_cmp, dtype=np.int64)
        return idx, idx
    if num_obs_kpts == 70:
        pairs = [
            (obs_i, skel_j)
            for obs_i, skel_j in MHR70_TO_MHR127_JOINT_MAP
            if obs_i < num_obs_kpts and skel_j < num_skel_joints
        ]
        if not pairs:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        obs_arr = np.asarray([p[0] for p in pairs], dtype=np.int64)
        skel_arr = np.asarray([p[1] for p in pairs], dtype=np.int64)
        return obs_arr, skel_arr
    if num_obs_kpts == num_skel_joints:
        idx = np.arange(num_obs_kpts, dtype=np.int64)
        return idx, idx
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def _extract_body_pose_tensor(
    raw_pose: Any,
    fallback_pose: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Normalize per-view body_pose_params to shape (1, 130) on device, or fall back to canonical pose."""
    if raw_pose is None:
        return fallback_pose.detach().to(device=device, dtype=torch.float32).clone()
    arr = np.asarray(raw_pose, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        return fallback_pose.detach().to(device=device, dtype=torch.float32).clone()
    out_arr = np.zeros(130, dtype=np.float32)
    n_copy = min(130, arr.size)
    out_arr[:n_copy] = arr[:n_copy]
    return torch.from_numpy(out_arr).unsqueeze(0).to(device=device, dtype=torch.float32)


def _quick_vertex_mask_iou(
    verts: np.ndarray,
    r_mat: np.ndarray,
    t_vec: np.ndarray,
    focal: float,
    target_mask: np.ndarray,
) -> float:
    """Fast approximate IoU between subsampled projected vertices and foreground mask."""
    h, w = target_mask.shape[:2]
    if h <= 0 or w <= 0 or len(verts) == 0:
        return 0.0
    scale = min(1.0, 128.0 / float(max(h, w)))
    dh = max(16, int(round(h * scale)))
    dw = max(16, int(round(w * scale)))
    mask_ds = (
        cv2.resize(target_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
        if (dh != h or dw != w)
        else target_mask
    )
    stride = max(1, len(verts) // 1500)
    pts_cam = verts[::stride] @ r_mat.T + t_vec[None, :]
    z = pts_cam[:, 2]
    val = z > 0.25
    if not np.any(val):
        return 0.0
    u = (focal * (pts_cam[val, 0] / z[val]) + 0.5 * float(w)) * (float(dw) / float(w))
    v = (focal * (pts_cam[val, 1] / z[val]) + 0.5 * float(h)) * (float(dh) / float(h))
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_b = (ui >= 0) & (ui < dw) & (vi >= 0) & (vi < dh)
    if not np.any(in_b):
        return 0.0
    pred = np.zeros((dh, dw), dtype=np.uint8)
    pred[vi[in_b], ui[in_b]] = 1
    pred = cv2.dilate(pred, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    inter = int(np.sum(pred & mask_ds))
    union = int(np.sum(pred | mask_ds))
    return float(inter) / float(max(1, union))


def _refine_camera_bounded_pnp(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    r_init: np.ndarray,
    t_init: np.ndarray,
    focal: float,
    img_h: int,
    img_w: int,
    verts: np.ndarray,
    target_mask: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Bounded iterative PnP camera refinement handling smartphone pitch/roll/yaw offsets."""
    if len(obj_pts) < 4:
        return None
    k_mat = np.array(
        [
            [float(focal), 0.0, 0.5 * float(img_w)],
            [0.0, float(focal), 0.5 * float(img_h)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        rvec0, _ = cv2.Rodrigues(r_init.astype(np.float64))
        tvec0 = t_init.astype(np.float64).reshape(3, 1)
        ok, rvec_sol, tvec_sol = cv2.solvePnP(
            np.ascontiguousarray(obj_pts, dtype=np.float64),
            np.ascontiguousarray(img_pts, dtype=np.float64),
            k_mat,
            None,
            rvec0,
            tvec0,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        r_pnp, _ = cv2.Rodrigues(rvec_sol)
        t_pnp = tvec_sol.reshape(3).astype(np.float32)
    except cv2.error:
        return None

    if not (np.isfinite(r_pnp).all() and np.isfinite(t_pnp).all()):
        return None
    r_delta = r_pnp @ r_init.astype(np.float64).T
    cos_theta = float(np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cos_theta)))
    if theta_deg > 28.0:
        return None
    init_norm = max(0.3, float(np.linalg.norm(t_init)))
    if float(np.linalg.norm(t_pnp)) > 3.0 * init_norm:
        return None

    pts_cam = obj_pts @ r_pnp.T + t_pnp.astype(np.float64)[None, :]
    if np.any(pts_cam[:, 2] <= 0.25):
        return None
    u_p = float(focal) * (pts_cam[:, 0] / pts_cam[:, 2]) + 0.5 * float(img_w)
    v_p = float(focal) * (pts_cam[:, 1] / pts_cam[:, 2]) + 0.5 * float(img_h)
    med_err_pnp = float(np.median(np.hypot(u_p - img_pts[:, 0], v_p - img_pts[:, 1])))

    if np.any(target_mask):
        iou_init = _quick_vertex_mask_iou(verts, r_init, t_init.astype(np.float32), focal, target_mask)
        iou_pnp = _quick_vertex_mask_iou(verts, r_pnp.astype(np.float32), t_pnp, focal, target_mask)
        if iou_init > 0.05 and iou_pnp < iou_init - 0.025:
            return None
    return r_pnp.astype(np.float32), t_pnp, med_err_pnp


_CORE_PNP_SKEL_IDS = frozenset({124, 122, 75, 39, 2, 18, 110})


def _solve_linear_translation_pnp(
    j_cam0: np.ndarray,
    uv: np.ndarray,
    focal: float,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, float]:
    """Solve linear least-squares camera translation for fixed rotation and return (t_sol, med_err)."""
    cx, cy = img_w / 2.0, img_h / 2.0
    du = uv[:, 0] - cx
    dv = uv[:, 1] - cy
    n_pts = len(uv)
    a_mat = np.zeros((2 * n_pts, 3), dtype=np.float64)
    b_vec = np.zeros(2 * n_pts, dtype=np.float64)
    a_mat[0::2, 0] = focal
    a_mat[0::2, 2] = -du
    b_vec[0::2] = du * j_cam0[:, 2] - focal * j_cam0[:, 0]
    a_mat[1::2, 1] = focal
    a_mat[1::2, 2] = -dv
    b_vec[1::2] = dv * j_cam0[:, 2] - focal * j_cam0[:, 1]
    t_sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    z_pts = j_cam0[:, 2] + t_sol[2]
    med_err = float("inf")
    if np.all(z_pts > 0.25):
        u_proj = focal * (j_cam0[:, 0] + t_sol[0]) / z_pts + cx
        v_proj = focal * (j_cam0[:, 1] + t_sol[1]) / z_pts + cy
        med_err = float(np.median(np.hypot(u_proj - uv[:, 0], v_proj - uv[:, 1])))
    return t_sol, med_err


def _extract_global_rot_candidate(raw_global_rot: Any, r_v: np.ndarray) -> Optional[np.ndarray]:
    """Convert optional global_rot to a valid camera rotation if within 28 degrees of r_v."""
    if raw_global_rot is None:
        return None
    arr = np.asarray(raw_global_rot, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    if arr.shape == (3, 3):
        r_raw = arr
    elif arr.size == 3:
        r_raw, _ = cv2.Rodrigues(arr.reshape(3, 1))
    else:
        return None
    # Test r_raw directly and cv-flipped variant r_cv @ r_raw.
    cv_flip = np.diag([1.0, -1.0, 1.0])
    cos_lim = math.cos(math.radians(28.0))
    for cand in (r_raw, cv_flip @ r_raw):
        if np.linalg.det(cand) > 0.5:
            cos_a = 0.5 * (float(np.trace(cand.T @ r_v)) - 1.0)
            if cos_a >= cos_lim:
                return cand.astype(np.float32)
    return None


def _solve_initial_view_camera(
    verts_a: np.ndarray,
    joints_a: np.ndarray,
    view: SegmentedView,
    out: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve initial OpenCV (R_v, t_v, focal) aligning mesh onto view's body_mask."""
    target_mask = view.body_mask if np.any(view.body_mask) else view.person_mask
    img_h, img_w = target_mask.shape[:2]
    focal = _extract_focal_length(out, img_h, img_w)
    r_v = _rotation_y_and_cv_flip(view.azimuth_deg)

    # Bounding-box alignment baseline used both as fallback and sanity comparator.
    bbox_cam: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    bbox_iou = 0.0
    if np.any(target_mask):
        ys, xs = np.where(target_mask)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        pts_cam0 = verts_a @ r_v.T
        min_xyz = pts_cam0.min(axis=0)
        max_xyz = pts_cam0.max(axis=0)
        mid_xyz = 0.5 * (min_xyz + max_xyz)
        h_mesh = max(float(max_xyz[1] - min_xyz[1]), 1e-4)
        h_px = max(y1 - y0, 1.0)
        tz = focal * (h_mesh / h_px) - float(mid_xyz[2])
        z_center = max(float(mid_xyz[2]) + tz, 0.3)
        cx_px = 0.5 * (x0 + x1)
        cy_px = 0.5 * (y0 + y1)
        tx = ((cx_px - img_w / 2.0) * z_center / focal) - float(mid_xyz[0])
        ty = ((cy_px - img_h / 2.0) * z_center / focal) - float(mid_xyz[1])
        t_bbox = np.array([tx, ty, tz], dtype=np.float32)
        bbox_cam = (r_v, t_bbox, focal)
        bbox_iou = _quick_vertex_mask_iou(verts_a, r_v, t_bbox, focal, target_mask)

    # Attempt 2D-3D anatomical keypoint least squares + bounded PnP using schema-aware joint mapping.
    k2_raw = out.get("pred_keypoints_2d")
    if k2_raw is not None and joints_a is not None and len(joints_a) >= 12:
        k2 = np.asarray(k2_raw, dtype=np.float32)
        if k2.ndim == 2 and k2.shape[1] >= 2:
            obs_idx, skel_idx = _pair_obs_kpts_to_skel_indices(len(k2), len(joints_a))
            if len(obs_idx) >= 6:
                k2_xy = k2[obs_idx, :2]
                valid = (
                    np.isfinite(k2_xy).all(axis=1)
                    & (k2_xy[:, 0] > 0.0)
                    & (k2_xy[:, 0] < float(img_w))
                    & (k2_xy[:, 1] > 0.0)
                    & (k2_xy[:, 1] < float(img_h))
                )
                if k2.shape[1] >= 3:
                    conf_vals = k2[obs_idx, 2]
                    conf_ok = valid & np.isfinite(conf_vals) & (conf_vals > 0.1)
                    if int(conf_ok.sum()) >= 6:
                        valid = conf_ok
                if int(valid.sum()) >= 6:
                    obj_pts = joints_a[skel_idx][valid].astype(np.float64)
                    uv = k2_xy[valid].astype(np.float64)
                    skel_valid = skel_idx[valid]
                    core_mask = np.isin(skel_valid, list(_CORE_PNP_SKEL_IDS))

                    r_cand_list = [r_v]
                    r_global = _extract_global_rot_candidate(out.get("global_rot"), r_v)
                    if r_global is not None:
                        r_cand_list.append(r_global)

                    best_cam: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
                    best_score = -1.0

                    for r_init in r_cand_list:
                        j_cam0_all = (obj_pts @ r_init.T).astype(np.float64)
                        t_all, err_all = _solve_linear_translation_pnp(j_cam0_all, uv, focal, img_w, img_h)

                        # Core-joint initial translation prevents occluded side-view limbs from skewing tz.
                        t_seed = t_all
                        if int(core_mask.sum()) >= 4:
                            j_cam0_core = j_cam0_all[core_mask]
                            uv_core = uv[core_mask]
                            t_core, err_core = _solve_linear_translation_pnp(
                                j_cam0_core, uv_core, focal, img_w, img_h
                            )
                            if np.all(j_cam0_core[:, 2] + t_core[2] > 0.25) and err_core < 0.12 * max(img_h, img_w):
                                t_seed = t_core

                        t_guess = t_seed.astype(np.float32).copy()
                        min_z = float(np.min(j_cam0_all[:, 2] + t_guess[2]))
                        if min_z <= 0.3:
                            t_guess[2] += 0.5 - min_z

                        pnp_pts = obj_pts[core_mask] if int(core_mask.sum()) >= 5 else obj_pts
                        pnp_uv = uv[core_mask] if int(core_mask.sum()) >= 5 else uv
                        pnp_res = _refine_camera_bounded_pnp(
                            obj_pts=pnp_pts,
                            img_pts=pnp_uv,
                            r_init=r_init,
                            t_init=t_guess,
                            focal=focal,
                            img_h=img_h,
                            img_w=img_w,
                            verts=verts_a,
                            target_mask=target_mask,
                        )
                        if pnp_res is not None:
                            r_pnp, t_pnp, med_err_pnp = pnp_res
                            if med_err_pnp < 0.12 * max(img_h, img_w):
                                iou_pnp = _quick_vertex_mask_iou(verts_a, r_pnp, t_pnp, focal, target_mask)
                                score = iou_pnp - 0.002 * med_err_pnp
                                if score > best_score:
                                    best_score = score
                                    best_cam = (r_pnp, t_pnp, focal)

                        z_pts_all = j_cam0_all[:, 2] + t_all[2]
                        if np.all(z_pts_all > 0.25) and err_all < 0.12 * max(img_h, img_w):
                            t_lin = t_all.astype(np.float32)
                            iou_lin = _quick_vertex_mask_iou(verts_a, r_init, t_lin, focal, target_mask)
                            score_lin = iou_lin - 0.002 * err_all
                            if score_lin > best_score:
                                best_score = score_lin
                                best_cam = (r_init, t_lin, focal)

                    if best_cam is not None:
                        cam_iou = _quick_vertex_mask_iou(verts_a, best_cam[0], best_cam[1], focal, target_mask)
                        if bbox_cam is None or not np.any(target_mask) or cam_iou >= 0.15 or bbox_iou <= 0.35:
                            return best_cam

    if bbox_cam is not None:
        return bbox_cam

    pred_cam_t = out.get("pred_cam_t")
    if pred_cam_t is not None:
        return r_v, np.asarray(pred_cam_t, dtype=np.float32).reshape(3), focal
    return r_v, np.array([0.0, 0.85, 2.8], dtype=np.float32), focal


def _rasterize_binary_silhouette_cpu(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Project 3D camera-frame triangles to an (img_h, img_w) boolean silhouette mask on CPU."""
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
    for tri in tris_int:
        cv2.fillConvexPoly(mask, tri, 1)
    return mask > 0


def _compute_mask_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Compute exact boolean mask Intersection-over-Union."""
    inter = int(np.sum(pred_mask & target_mask))
    union = int(np.sum(pred_mask | target_mask))
    if union == 0:
        return 1.0 if (not np.any(pred_mask) and not np.any(target_mask)) else 0.0
    return float(inter / union)


def _evaluate_multiview_silhouette_iou(
    verts_a: np.ndarray,
    faces_np: np.ndarray,
    cameras: List[Tuple[np.ndarray, np.ndarray, float]],
    segmented_views: List[SegmentedView],
    view_verts_list: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Tuple[float, List[float]]:
    """Compute mean and per-view silhouette IoU against hair-excluded body_mask."""
    per_view: List[float] = []
    for idx, (view, (r_v, t_v, focal)) in enumerate(zip(segmented_views, cameras)):
        target_mask = np.asarray(view.body_mask, dtype=bool)
        if not np.any(target_mask):
            continue
        img_h, img_w = target_mask.shape[:2]
        verts_src = (
            view_verts_list[idx]
            if view_verts_list is not None and idx < len(view_verts_list) and view_verts_list[idx] is not None
            else verts_a
        )
        verts_cam = (verts_src @ r_v.T) + t_v.reshape(1, 3)
        pred_mask = _rasterize_binary_silhouette_cpu(verts_cam, faces_np, focal, img_h, img_w)
        per_view.append(_compute_mask_iou(pred_mask, target_mask))
    mean_iou = float(np.mean(per_view)) if per_view else 0.0
    return mean_iou, per_view


def _forward_mhr_with_scale_torch(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    shape_t: torch.Tensor,
    scale_t: torch.Tensor,
    pose_t: torch.Tensor,
    expr_t: torch.Tensor,
    height_cm: float,
    ground_min_y: Optional[torch.Tensor] = None,
    height_scale_factor: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable MHR forward pass grounded and scaled to height_cm (or canonical reference scale)."""
    dev = shape_t.device
    scale_vec = head_assets["scale_mean"].to(dev)[None, :] + scale_t @ head_assets["scale_comps"].to(dev)
    full_pose = torch.cat([torch.zeros(1, 6, device=dev, dtype=shape_t.dtype), pose_t[:, :130].to(dev)], dim=1)
    model_params = torch.cat([full_pose, scale_vec], dim=1)
    verts_cm, skel_cm = mhr_model(shape_t, model_params, expr_t)
    verts_m = verts_cm[0] / 100.0
    skel_m = skel_cm[0, :, :3] / 100.0

    if ground_min_y is None:
        min_y = verts_m[:, 1].min()
    else:
        min_y = ground_min_y
    if height_scale_factor is None:
        raw_height_m = torch.clamp(verts_m[:, 1].max() - min_y, min=1e-4)
        scale_factor = (float(height_cm) / 100.0) / raw_height_m
    else:
        scale_factor = height_scale_factor

    offset = torch.stack([torch.zeros_like(min_y), min_y, torch.zeros_like(min_y)])
    verts_scaled = (verts_m - offset) * scale_factor
    skel_scaled = (skel_m - offset) * scale_factor
    return verts_scaled, skel_scaled, min_y, scale_factor


def _forward_mhr_canonical_torch(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    shape_t: torch.Tensor,
    scale_t: torch.Tensor,
    apose_t: torch.Tensor,
    expr_t: torch.Tensor,
    height_cm: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable canonical A-pose MHR forward pass grounded at y=0 and scaled to height_cm."""
    verts_scaled, skel_scaled, _, _ = _forward_mhr_with_scale_torch(
        mhr_model, head_assets, shape_t, scale_t, apose_t, expr_t, height_cm
    )
    return verts_scaled, skel_scaled


def _clamp_delta_l2(param: torch.Tensor, init: torch.Tensor, max_norm: float) -> None:
    """Clamp parameter drift from initial fused estimate inside an L2 ball."""
    with torch.no_grad():
        delta = param - init
        norm = torch.linalg.norm(delta)
        if float(norm.item()) > max_norm:
            param.copy_(init + delta * (max_norm / (norm + 1e-8)))


def _prepare_view_weights(
    segmented_views: List[SegmentedView],
    fused: FusedSubjectParams,
) -> List[float]:
    """Compute normalized per-view optimization weights across active views."""
    raw_w: List[float] = []
    for i, view in enumerate(segmented_views):
        if not np.any(view.body_mask):
            raw_w.append(0.0)
            continue
        w = (
            float(fused.weights[view.view_index])
            if 0 <= view.view_index < len(fused.weights) and float(fused.weights[view.view_index]) > 0.0
            else 1.0
        )
        raw_w.append(w)
    total = sum(raw_w)
    if total <= 0.0:
        return [0.0 for _ in segmented_views]
    return [w / total for w in raw_w]


def _run_pytorch3d_optimization(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    fused: FusedSubjectParams,
    segmented_views: List[SegmentedView],
    raw_results: List[Dict[str, Any]],
    init_cameras: List[Tuple[np.ndarray, np.ndarray, float]],
    height_cm: float,
    weight_kg: float,
    sex: str,
    steps: int,
    apose_params: Optional[torch.Tensor] = None,
    extra_outputs: Optional[Dict[str, Any]] = None,
) -> Tuple[FusedSubjectParams, List[Tuple[np.ndarray, np.ndarray, float]], List[float], bool]:
    """Attempt PyTorch3D SoftSilhouetteShader optimization; returns (fused, cameras, loss_hist, used_p3d)."""
    try:
        from pytorch3d.renderer import (  # type: ignore[import-not-found]
            MeshRasterizer,
            MeshRenderer,
            RasterizationSettings,
            SoftSilhouetteShader,
        )
        from pytorch3d.structures import Meshes  # type: ignore[import-not-found]
        from pytorch3d.utils import cameras_from_opencv_projection  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError):
        return fused, init_cameras, [], False

    view_weights = _prepare_view_weights(segmented_views, fused)
    active_indices = [i for i, w in enumerate(view_weights) if w > 0.0]
    if not active_indices or int(steps) <= 0:
        return fused, init_cameras, [], True

    dev = head_assets["scale_mean"].device
    faces_t = (
        head_assets["faces"].to(device=dev, dtype=torch.int64)
        if isinstance(head_assets["faces"], torch.Tensor)
        else torch.as_tensor(head_assets["faces"], dtype=torch.int64, device=dev)
    )
    if apose_params is None:
        apose_params = solve_straight_apose(
            mhr_model, head_assets, fused.shape_params, fused.scale_params, expr_t=fused.expr_params
        )
    apose_dev = apose_params.detach().to(device=dev, dtype=torch.float32)

    shape_init = fused.shape_params.detach().to(device=dev, dtype=torch.float32).clone()
    scale_init = fused.scale_params.detach().to(device=dev, dtype=torch.float32).clone()
    expr_init = (
        fused.expr_params.detach().to(device=dev, dtype=torch.float32).clone()
        if fused.expr_params is not None
        else torch.zeros((1, 72), dtype=torch.float32, device=dev)
    )
    if expr_init.ndim == 1:
        expr_init = expr_init.unsqueeze(0)

    shape_opt = shape_init.clone().requires_grad_(True)
    scale_opt = scale_init.clone().requires_grad_(True)
    expr_opt = expr_init.clone().requires_grad_(True)

    # Prepare per-view targets (downsampled <= 384px for fast silhouette rendering) & camera/pose tensors.
    view_payloads: List[Dict[str, Any]] = []
    cam_t_opts: List[torch.Tensor] = []
    cam_rot_opts: List[torch.Tensor] = []
    delta_pose_opts: List[torch.Tensor] = []
    zero_pose = torch.zeros((1, 130), dtype=torch.float32, device=dev)

    for i in active_indices:
        view = segmented_views[i]
        rec = _resolve_raw_record(view, i, raw_results)
        out = rec.get("output") or {}
        r_base_np, t_init_np, focal_orig = init_cameras[i]

        mask_np = view.body_mask.astype(np.float32)
        orig_h, orig_w = mask_np.shape[:2]
        max_dim = max(orig_h, orig_w)
        render_scale = min(1.0, 384.0 / float(max_dim)) if max_dim > 384 else 1.0
        ren_h = max(32, int(round(orig_h * render_scale)))
        ren_w = max(32, int(round(orig_w * render_scale)))
        scale_x = ren_w / float(orig_w)
        scale_y = ren_h / float(orig_h)
        focal_ren = focal_orig * 0.5 * (scale_x + scale_y)

        mask_ren = (
            cv2.resize(mask_np, (ren_w, ren_h), interpolation=cv2.INTER_NEAREST)
            if (ren_h != orig_h or ren_w != orig_w)
            else mask_np
        )
        tgt_t = torch.from_numpy(np.ascontiguousarray(mask_ren)).to(device=dev, dtype=torch.float32)

        k_mat = torch.tensor(
            [[
                [focal_ren, 0.0, ren_w / 2.0],
                [0.0, focal_ren, ren_h / 2.0],
                [0.0, 0.0, 1.0],
            ]],
            dtype=torch.float32,
            device=dev,
        )
        im_size = torch.tensor([[ren_h, ren_w]], dtype=torch.float32, device=dev)
        raster_settings = RasterizationSettings(
            image_size=(ren_h, ren_w),
            blur_radius=math.log(1.0 / 1e-4 - 1.0) * 1e-4,
            faces_per_pixel=25,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftSilhouetteShader(),
        )

        # Optional 2D keypoints scaled to render grid (paired semantically against MHR skeleton later).
        kpt_obs = None
        k2_raw = out.get("pred_keypoints_2d")
        if k2_raw is not None:
            k2 = np.asarray(k2_raw, dtype=np.float32)
            if k2.ndim == 2 and k2.shape[1] >= 2:
                k2_xy = k2[:, :2]
                valid_kp = (
                    np.isfinite(k2_xy).all(axis=1)
                    & (k2_xy[:, 0] > 0.0)
                    & (k2_xy[:, 0] < float(orig_w))
                    & (k2_xy[:, 1] > 0.0)
                    & (k2_xy[:, 1] < float(orig_h))
                )
                if k2.shape[1] >= 3:
                    conf_ok = valid_kp & np.isfinite(k2[:, 2]) & (k2[:, 2] > 0.1)
                    if int(conf_ok.sum()) >= 4:
                        valid_kp = conf_ok
                if int(valid_kp.sum()) >= 4:
                    safe_xy = np.where(np.isfinite(k2_xy), k2_xy, 0.0)
                    xy_all_t = torch.as_tensor(
                        np.stack([safe_xy[:, 0] * scale_x, safe_xy[:, 1] * scale_y], axis=-1),
                        dtype=torch.float32,
                        device=dev,
                    )
                    kpt_obs = (len(k2), valid_kp, xy_all_t, ren_w, ren_h, focal_ren)

        base_pose_v = _extract_body_pose_tensor(out.get("body_pose_params"), apose_dev, dev)
        d_pose_param = torch.zeros((1, 130), dtype=torch.float32, device=dev, requires_grad=True)
        delta_pose_opts.append(d_pose_param)

        t_param = torch.tensor(t_init_np, dtype=torch.float32, device=dev, requires_grad=True)
        rot_param = torch.zeros(3, dtype=torch.float32, device=dev, requires_grad=True)
        cam_t_opts.append(t_param)
        cam_rot_opts.append(rot_param)
        view_payloads.append(
            {
                "list_idx": i,
                "weight": view_weights[i],
                "r_base": torch.tensor(r_base_np, dtype=torch.float32, device=dev),
                "base_pose": base_pose_v,
                "target": tgt_t,
                "k_mat": k_mat,
                "im_size": im_size,
                "renderer": renderer,
                "kpt_obs": kpt_obs,
                "focal_orig": focal_orig,
            }
        )

    optimizer = torch.optim.Adam(
        [
            {"params": [shape_opt, scale_opt], "lr": 0.02},
            {"params": [expr_opt], "lr": 0.005},
            {"params": cam_t_opts, "lr": 0.008},
            {"params": cam_rot_opts, "lr": 0.004},
            {"params": delta_pose_opts, "lr": 0.004},
        ]
    )

    loss_history: List[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad()
        verts_canon, _, canon_min_y, canon_scale = _forward_mhr_with_scale_torch(
            mhr_model, head_assets, shape_opt, scale_opt, apose_dev, expr_opt, height_cm
        )

        l_sil_acc = torch.zeros((), dtype=torch.float32, device=dev)
        l_edge_acc = torch.zeros((), dtype=torch.float32, device=dev)
        l_kpt_acc = torch.zeros((), dtype=torch.float32, device=dev)
        l_rot_reg = torch.zeros((), dtype=torch.float32, device=dev)
        l_pose_reg = torch.zeros((), dtype=torch.float32, device=dev)

        for payload, t_vec, rot_vec, d_pose in zip(view_payloads, cam_t_opts, cam_rot_opts, delta_pose_opts):
            pose_v = payload["base_pose"] + d_pose
            verts_v, skel_v, _, _ = _forward_mhr_with_scale_torch(
                mhr_model,
                head_assets,
                shape_opt,
                scale_opt,
                pose_v,
                expr_opt,
                height_cm,
                ground_min_y=canon_min_y,
                height_scale_factor=canon_scale,
            )
            mesh_v = Meshes(verts=[verts_v], faces=[faces_t])
            r_cur = _axis_angle_to_matrix(rot_vec) @ payload["r_base"]
            cams = cameras_from_opencv_projection(
                r_cur.unsqueeze(0),
                t_vec.unsqueeze(0),
                payload["k_mat"],
                payload["im_size"],
            )
            rgba = payload["renderer"](mesh_v, cameras=cams)
            alpha = rgba[0, :, :, 3]
            l_sil_v, l_edge_v = compute_soft_silhouette_and_edge_loss(alpha, payload["target"])
            w_v = payload["weight"]
            l_sil_acc = l_sil_acc + w_v * l_sil_v
            l_edge_acc = l_edge_acc + w_v * l_edge_v
            l_rot_reg = l_rot_reg + 0.05 * torch.sum(rot_vec * rot_vec)
            l_pose_reg = l_pose_reg + 0.08 * torch.sum(d_pose * d_pose)

            if payload["kpt_obs"] is not None:
                n_obs, valid_kp_mask, xy_all_t, rw, rh, f_ren = payload["kpt_obs"]
                if "paired_kpt_idx" not in payload:
                    obs_map, skel_map = _pair_obs_kpts_to_skel_indices(n_obs, len(skel_v))
                    keep = valid_kp_mask[obs_map] if len(obs_map) > 0 else np.empty(0, dtype=bool)
                    payload["paired_kpt_idx"] = (
                        torch.as_tensor(obs_map[keep], dtype=torch.int64, device=dev),
                        torch.as_tensor(skel_map[keep], dtype=torch.int64, device=dev),
                    )
                obs_j_t, skel_j_t = payload["paired_kpt_idx"]
                if len(skel_j_t) >= 4:
                    xy_target = xy_all_t[obs_j_t]
                    j_cam = (skel_v[skel_j_t] @ r_cur.T) + t_vec.unsqueeze(0)
                    z_j = torch.clamp(j_cam[:, 2], min=0.1)
                    u_j = f_ren * (j_cam[:, 0] / z_j) + (rw / 2.0)
                    v_j = f_ren * (j_cam[:, 1] / z_j) + (rh / 2.0)
                    norm_scale = float(max(rw, rh))
                    l_kpt_v = F.mse_loss(torch.stack([u_j, v_j], dim=-1) / norm_scale, xy_target / norm_scale)
                    l_kpt_acc = l_kpt_acc + w_v * l_kpt_v

        l_vol = torch.zeros((), dtype=torch.float32, device=dev)
        if float(weight_kg or 0.0) > 0.0:
            est_w = estimate_weight_kg(verts_canon, faces_t, sex)
            l_vol = ((est_w - float(weight_kg)) / max(float(weight_kg), 1.0)) ** 2

        l_prior = (
            0.02 * F.mse_loss(shape_opt, shape_init)
            + 0.04 * F.mse_loss(scale_opt, scale_init)
            + 0.15 * F.mse_loss(expr_opt, expr_init)
            + l_rot_reg
            + l_pose_reg
        )
        # Silhouette & contour fidelity strongly dominate L_vol so contradictory weights never distort geometry.
        l_total = 1.0 * l_sil_acc + 0.25 * l_edge_acc + 0.10 * l_kpt_acc + l_prior + 0.06 * l_vol
        l_total.backward()
        optimizer.step()

        _clamp_delta_l2(shape_opt, shape_init, max_norm=2.5)
        _clamp_delta_l2(scale_opt, scale_init, max_norm=1.5)
        _clamp_delta_l2(expr_opt, expr_init, max_norm=0.5)
        for d_pose in delta_pose_opts:
            _clamp_delta_l2(d_pose, zero_pose, max_norm=0.35)
        loss_history.append(float(l_total.item()))

    updated_cameras = list(init_cameras)
    view_poses: List[torch.Tensor] = [
        _extract_body_pose_tensor((_resolve_raw_record(v, idx, raw_results).get("output") or {}).get("body_pose_params"), apose_dev, torch.device("cpu"))
        for idx, v in enumerate(segmented_views)
    ]
    with torch.no_grad():
        for payload, t_vec, rot_vec, d_pose in zip(view_payloads, cam_t_opts, cam_rot_opts, delta_pose_opts):
            r_final = (_axis_angle_to_matrix(rot_vec) @ payload["r_base"]).detach().cpu().numpy()
            t_final = t_vec.detach().cpu().numpy()
            updated_cameras[payload["list_idx"]] = (r_final, t_final, payload["focal_orig"])
            view_poses[payload["list_idx"]] = (payload["base_pose"] + d_pose).detach().cpu()

    if extra_outputs is not None:
        extra_outputs["view_poses"] = view_poses

    updated_fused = FusedSubjectParams(
        shape_params=shape_opt.detach().cpu(),
        scale_params=scale_opt.detach().cpu(),
        weights=fused.weights,
        distances=fused.distances,
        images_used=fused.images_used,
        images_total=fused.images_total,
        mean_confidence=fused.mean_confidence,
        expr_params=expr_opt.detach().cpu(),
    )
    return updated_fused, updated_cameras, loss_history, True


def _align_and_blend_native_view_verts(
    apose_scaled_verts: np.ndarray,
    raw_pred_verts: Any,
    azimuth_deg: float = 0.0,
    arm_vert_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Align a view's native pred_vertices into canonical A-pose scale/frame and blend articulated limbs."""
    va = np.asarray(apose_scaled_verts, dtype=np.float64)
    if raw_pred_verts is None or va.ndim != 2 or va.shape[1] != 3 or len(va) == 0:
        return np.asarray(apose_scaled_verts, dtype=np.float32)

    if isinstance(raw_pred_verts, torch.Tensor):
        vp = raw_pred_verts.detach().cpu().numpy().astype(np.float64)
    else:
        vp = np.asarray(raw_pred_verts, dtype=np.float64)
    if vp.ndim == 3 and vp.shape[0] == 1:
        vp = vp[0]
    if vp.shape != va.shape or not np.all(np.isfinite(vp)):
        return np.asarray(apose_scaled_verts, dtype=np.float32)

    y_min_a = float(va[:, 1].min())
    y_max_a = float(va[:, 1].max())
    h_a = max(y_max_a - y_min_a, 1e-4)
    y_rel_a = (va[:, 1] - y_min_a) / h_a
    x_ref_a = max(float(np.percentile(np.abs(va[:, 0]), 90)), 1e-4)

    keep_non_arm: Optional[np.ndarray] = None
    if arm_vert_mask is not None:
        mask_arr = np.asarray(arm_vert_mask, dtype=bool).reshape(-1)
        if mask_arr.shape == (len(va),) and not np.all(mask_arr) and np.any(mask_arr):
            keep_non_arm = mask_arr

    # Fit rigid alignment strictly on rigid torso/pelvis (excluding arms, head, and lower legs).
    if keep_non_arm is not None:
        core_mask = keep_non_arm & (y_rel_a >= 0.42) & (y_rel_a <= 0.86)
    else:
        core_mask = (
            (y_rel_a >= 0.42)
            & (y_rel_a <= 0.86)
            & (np.abs(va[:, 0]) <= min(0.16 * h_a, 0.62 * x_ref_a))
        )
    if int(np.sum(core_mask)) < 6:
        core_mask = (y_rel_a >= 0.30) & (y_rel_a <= 0.90) & (np.abs(va[:, 0]) <= 0.62 * x_ref_a)
    if int(np.sum(core_mask)) < 6:
        core_mask = np.ones(len(va), dtype=bool)
    core_idx = np.where(core_mask)[0]

    r_cv_pos = _rotation_y_and_cv_flip(float(azimuth_deg)).astype(np.float64)

    # Exclude negative-azimuth aliases so side/back views never align 180 degrees reversed.
    cand_mats = (
        np.eye(3, dtype=np.float64),
        r_cv_pos,
    )

    mu_a0 = va[core_idx].mean(axis=0)
    rms_a0 = float(np.sqrt(np.mean(np.sum((va[core_idx] - mu_a0) ** 2, axis=1))))
    if rms_a0 < 1e-6:
        return np.asarray(apose_scaled_verts, dtype=np.float32)

    best_mapped: Optional[np.ndarray] = None
    best_err = float("inf")
    cos_limit = math.cos(math.radians(15.0))
    cos_reject = math.cos(math.radians(25.0))

    for mat in cand_mats:
        vc0 = vp @ mat
        mu_c0 = vc0[core_idx].mean(axis=0)
        rms_c0 = float(np.sqrt(np.mean(np.sum((vc0[core_idx] - mu_c0) ** 2, axis=1))))
        if rms_c0 < 1e-6:
            continue
        s0 = rms_a0 / rms_c0
        v_init = s0 * (vc0 - mu_c0) + mu_a0

        res0 = np.linalg.norm(v_init[core_idx] - va[core_idx], axis=1)
        k_keep = max(4, int(round(0.68 * len(core_idx))))
        inliers = core_idx[np.argsort(res0)[:k_keep]]

        mu_c = vc0[inliers].mean(axis=0)
        mu_a = va[inliers].mean(axis=0)
        xc = vc0[inliers] - mu_c
        xa = va[inliers] - mu_a
        u_svd, s_svd, vt_svd = np.linalg.svd(xc.T @ xa)
        r_corr = u_svd @ vt_svd
        if np.linalg.det(r_corr) < 0.0:
            vt_svd[-1, :] *= -1.0
            r_corr = u_svd @ vt_svd

        cos_ang = 0.5 * (float(np.trace(r_corr)) - 1.0)
        if cos_ang < cos_reject:
            # Large rotational mismatch indicates flipped coordinate orientation; reject candidate.
            continue
        if cos_ang < cos_limit:
            r_corr = np.eye(3, dtype=np.float64)
            s_fit = s0
        else:
            denom = float(np.sum(xc * xc))
            s_fit = float(np.sum(s_svd)) / max(denom, 1e-12) if denom > 1e-12 else s0
            s_fit = float(np.clip(s_fit, 0.94 * s0, 1.06 * s0))

        v_mapped = s_fit * ((vc0 - mu_c) @ r_corr) + mu_a
        err = float(np.median(np.linalg.norm(v_mapped[inliers] - va[inliers], axis=1)))
        if err < best_err:
            best_err = err
            best_mapped = v_mapped

    if best_mapped is None or best_err > 0.08 * h_a:
        return np.asarray(apose_scaled_verts, dtype=np.float32)

    diffs = np.linalg.norm(best_mapped - va, axis=1)
    if float(np.max(diffs)) <= 0.012:
        return np.asarray(apose_scaled_verts, dtype=np.float32)

    w_limb = np.clip((diffs - 0.008) / 0.017, 0.0, 1.0)
    head_keep = (y_rel_a > 0.86) & (np.abs(va[:, 0]) <= 0.16 * h_a)
    lower_leg_keep = y_rel_a < 0.42
    w_limb[head_keep | lower_leg_keep] = 0.0
    if keep_non_arm is not None:
        w_limb[keep_non_arm] = 0.0
    else:
        torso_center_keep = (y_rel_a <= 0.86) & (np.abs(va[:, 0]) <= 0.14 * h_a)
        w_limb[torso_center_keep] = 0.0

    blended = (1.0 - w_limb[:, None]) * va + w_limb[:, None] * best_mapped
    if float(blended[:, 1].min()) < y_min_a - 0.005:
        blended[:, 1] = np.maximum(blended[:, 1], y_min_a)
    return blended.astype(np.float32)


def _evaluate_view_posed_meshes(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    fused: FusedSubjectParams,
    apose_params: torch.Tensor,
    apose_scaled_verts: np.ndarray,
    view_poses: Sequence[torch.Tensor],
    height_cm: float,
    raw_results: Optional[Sequence[Dict[str, Any]]] = None,
    segmented_views: Optional[Sequence[SegmentedView]] = None,
) -> List[np.ndarray]:
    """Evaluate posed per-view meshes grounded and scaled identically to the canonical A-pose mesh."""
    apose_cpu = apose_params.detach().cpu()
    needs_posed = any(not torch.allclose(p.detach().cpu(), apose_cpu, atol=1e-6) for p in view_poses)

    out_verts: List[np.ndarray] = []
    if not needs_posed:
        out_verts = [apose_scaled_verts for _ in view_poses]
    else:
        canon_verts_raw, _ = evaluate_mhr_pose(
            mhr_model, head_assets, fused.shape_params, fused.scale_params, apose_cpu, expr_t=fused.expr_params
        )
        min_y = float(canon_verts_raw[:, 1].min())
        raw_height_m = max(float(canon_verts_raw[:, 1].max()) - min_y, 1e-4)
        scale_factor = (float(height_cm) / 100.0) / raw_height_m

        for p in view_poses:
            p_cpu = p.detach().cpu()
            if torch.allclose(p_cpu, apose_cpu, atol=1e-6):
                out_verts.append(apose_scaled_verts)
                continue
            v_raw, _ = evaluate_mhr_pose(
                mhr_model, head_assets, fused.shape_params, fused.scale_params, p_cpu, expr_t=fused.expr_params
            )
            v_scaled = v_raw.copy()
            v_scaled[:, 1] -= min_y
            v_scaled *= scale_factor
            out_verts.append(v_scaled)

    if raw_results:
        raw_list = list(raw_results)
        arm_vert_mask = build_arm_vertex_mask(mhr_model, len(apose_scaled_verts))
        for idx in range(len(out_verts)):
            v_cur = out_verts[idx]
            if float(np.max(np.abs(v_cur - apose_scaled_verts))) > 1e-3:
                continue
            sv = segmented_views[idx] if (segmented_views is not None and idx < len(segmented_views)) else None
            if sv is not None:
                rec = _resolve_raw_record(sv, idx, raw_list)
                az_deg = float(getattr(sv, "azimuth_deg", 0.0))
            elif idx < len(raw_list) and isinstance(raw_list[idx], dict):
                rec = raw_list[idx]
                az_deg = float(rec.get("azimuth_deg", 0.0))
            else:
                continue
            out_dict = rec.get("output") if isinstance(rec.get("output"), dict) else rec
            raw_pred_v = (
                out_dict.get("pred_vertices")
                if isinstance(out_dict, dict) and "pred_vertices" in out_dict
                else rec.get("pred_vertices")
            )
            if raw_pred_v is not None:
                out_verts[idx] = _align_and_blend_native_view_verts(
                    apose_scaled_verts,
                    raw_pred_v,
                    azimuth_deg=az_deg,
                    arm_vert_mask=arm_vert_mask,
                )

    return out_verts


def _rodrigues_torch(rot_vec: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues rotation matrix for a single (3,) axis-angle vector."""
    theta = torch.norm(rot_vec) + 1e-8
    k = rot_vec / theta
    zero = torch.zeros((), device=rot_vec.device, dtype=rot_vec.dtype)
    skew = torch.stack(
        [
            torch.stack([zero, -k[2], k[1]]),
            torch.stack([k[2], zero, -k[0]]),
            torch.stack([-k[1], k[0], zero]),
        ]
    )
    eye = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype)
    return eye + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


def _contour_projection_loss_for_views(
    verts_list_t: Sequence[torch.Tensor],
    view_targets: Sequence[Dict[str, Any]],
    rot_mats_t: Sequence[torch.Tensor],
    t_vecs_t: Sequence[torch.Tensor],
    v_sub_idx: np.ndarray,
) -> torch.Tensor:
    """Compute differentiable distance-transform + boundary-chamfer silhouette loss across views."""
    total_loss = torch.zeros((), device=verts_list_t[0].device, dtype=verts_list_t[0].dtype)
    for idx_v, tgt in enumerate(view_targets):
        v_cam = verts_list_t[idx_v][v_sub_idx] @ rot_mats_t[idx_v].T + t_vecs_t[idx_v][None, :]
        z = torch.clamp(v_cam[:, 2], min=0.25)
        f_ds = tgt["focal_ds"]
        dw, dh = tgt["dw"], tgt["dh"]
        u_px = f_ds * (v_cam[:, 0] / z) + (dw / 2.0)
        v_px = f_ds * (v_cam[:, 1] / z) + (dh / 2.0)
        uv_px = torch.stack([u_px, v_px], dim=1)

        # Push projected vertices inside the target silhouette via normalized outer distance field.
        gx = 2.0 * (u_px / max(dw - 1, 1)) - 1.0
        gy = 2.0 * (v_px / max(dh - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        dt_samp = F.grid_sample(tgt["dt_out"], grid, mode="bilinear", padding_mode="border", align_corners=True)
        oob = (torch.relu(torch.abs(gx) - 1.0) ** 2 + torch.relu(torch.abs(gy) - 1.0) ** 2).mean()
        loss_inside = (dt_samp ** 2).mean() + 0.5 * oob

        # Pull projected outer vertices outward to cover target silhouette boundary points.
        b_uv = tgt["boundary_uv"]
        if len(b_uv) > 0:
            dists_sq = torch.cdist(b_uv, uv_px) ** 2
            loss_cover = dists_sq.min(dim=1).values.mean() / (tgt["diag"] ** 2)
        else:
            loss_cover = torch.zeros((), device=u_px.device, dtype=u_px.dtype)

        total_loss = total_loss + tgt["weight"] * (1.2 * loss_inside + 1.0 * loss_cover)
    return total_loss


def _adam_step_tensors(
    params: Sequence[torch.Tensor],
    m_states: List[torch.Tensor],
    v_states: List[torch.Tensor],
    step_idx: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Lightweight pure-PyTorch Adam update without importing torch._dynamo."""
    bc1 = 1.0 - (beta1 ** step_idx)
    bc2 = 1.0 - (beta2 ** step_idx)
    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            g = p.grad
            m_states[i].mul_(beta1).add_(g, alpha=1.0 - beta1)
            v_states[i].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            m_hat = m_states[i] / bc1
            v_hat = v_states[i] / bc2
            p.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)
            p.grad = None


def _run_contour_fallback_optimization(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    fused: FusedSubjectParams,
    segmented_views: List[SegmentedView],
    init_cameras: List[Tuple[np.ndarray, np.ndarray, float]],
    height_cm: float,
    weight_kg: float,
    sex: str,
    steps: int,
    apose_params: torch.Tensor,
    init_view_verts: Optional[List[np.ndarray]] = None,
) -> Tuple[FusedSubjectParams, List[Tuple[np.ndarray, np.ndarray, float]], List[float], bool]:
    """Optimize multi-view cameras and MHR parameters via differentiable silhouette/contour alignment."""
    view_weights = _prepare_view_weights(segmented_views, fused)
    active_indices = [i for i, w in enumerate(view_weights) if w > 0.0]
    if not active_indices or int(steps) <= 0:
        return fused, init_cameras, [], False

    faces_raw = head_assets["faces"]
    faces_np = (
        faces_raw.detach().cpu().numpy()
        if isinstance(faces_raw, torch.Tensor)
        else np.asarray(faces_raw, dtype=np.int64)
    )
    dev = head_assets["scale_mean"].device
    verts_0, _, _, _ = build_scaled_apose_mesh(
        mhr_model, head_assets, fused, float(height_cm), apose_params=apose_params
    )
    if len(verts_0) == 0:
        return fused, init_cameras, [], False

    best_iou, _ = _evaluate_multiview_silhouette_iou(
        verts_0, faces_np, init_cameras, segmented_views, view_verts_list=init_view_verts
    )
    best_fused = fused
    best_cameras = [
        (np.asarray(r, dtype=np.float32).copy(), np.asarray(t, dtype=np.float32).copy(), float(f))
        for r, t, f in init_cameras
    ]

    # Precompute compact distance-transform and boundary-contour targets per active view.
    view_targets: List[Dict[str, Any]] = []
    for idx in active_indices:
        sv = segmented_views[idx]
        raw_mask = sv.body_mask if np.any(sv.body_mask) else sv.person_mask
        h0, w0 = raw_mask.shape[:2]
        ds_scale = min(1.0, 128.0 / float(max(h0, w0)))
        dh = max(24, int(round(h0 * ds_scale)))
        dw = max(24, int(round(w0 * ds_scale)))
        mask_ds = (
            cv2.resize(raw_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
        )
        diag = max(math.hypot(dh, dw), 1.0)
        dt_np = cv2.distanceTransform((~mask_ds).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32) / diag
        dt_t = torch.from_numpy(dt_np).to(device=dev, dtype=torch.float32).view(1, 1, dh, dw)

        eroded = cv2.erode(mask_ds.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
        edge_ys, edge_xs = np.where(mask_ds & (~eroded))
        if len(edge_xs) > 96:
            pick = np.linspace(0, len(edge_xs) - 1, 96, dtype=np.int64)
            edge_xs, edge_ys = edge_xs[pick], edge_ys[pick]
        b_uv = (
            torch.from_numpy(np.stack([edge_xs, edge_ys], axis=1).astype(np.float32)).to(dev)
            if len(edge_xs) > 0
            else torch.empty((0, 2), device=dev, dtype=torch.float32)
        )
        _, _, focal = init_cameras[idx]
        view_targets.append(
            {
                "view_idx": idx,
                "weight": float(view_weights[idx]),
                "dh": dh,
                "dw": dw,
                "diag": float(diag),
                "focal_ds": float(focal) * (float(dw) / max(float(w0), 1.0)),
                "dt_out": dt_t,
                "boundary_uv": b_uv,
            }
        )

    v_sub_idx = np.linspace(0, len(verts_0) - 1, min(len(verts_0), 640), dtype=np.int64)
    base_v_list_t = [
        torch.from_numpy(
            (init_view_verts[t["view_idx"]] if init_view_verts else verts_0).astype(np.float32)
        ).to(dev)
        for t in view_targets
    ]
    r0_list = [torch.from_numpy(np.asarray(best_cameras[t["view_idx"]][0], dtype=np.float32)).to(dev) for t in view_targets]
    t0_list = [torch.from_numpy(np.asarray(best_cameras[t["view_idx"]][1], dtype=np.float32)).to(dev) for t in view_targets]

    # Stage A: Differentiable camera translation + bounded rotation refinement via pure-PyTorch Adam.
    n_act = len(view_targets)
    dt_vars = torch.zeros((n_act, 3), device=dev, dtype=torch.float32, requires_grad=True)
    drot_vars = torch.zeros((n_act, 3), device=dev, dtype=torch.float32, requires_grad=True)
    m_cam = [torch.zeros_like(dt_vars), torch.zeros_like(drot_vars)]
    v_cam = [torch.zeros_like(dt_vars), torch.zeros_like(drot_vars)]

    loss_history: List[float] = []
    cam_steps = max(4, min(int(steps), 18))
    best_cam_loss = float("inf")
    best_dt_snap = dt_vars.detach().clone()
    best_drot_snap = drot_vars.detach().clone()

    for s_idx in range(1, cam_steps + 1):
        r_cands = [r0_list[i] @ _rodrigues_torch(drot_vars[i]) for i in range(n_act)]
        t_cands = [t0_list[i] + dt_vars[i] for i in range(n_act)]
        loss_c = _contour_projection_loss_for_views(base_v_list_t, view_targets, r_cands, t_cands, v_sub_idx)
        loss_reg = 0.08 * (drot_vars ** 2).mean() + 0.02 * (dt_vars ** 2).mean()
        total_c = loss_c + loss_reg
        total_c.backward()
        _adam_step_tensors([dt_vars, drot_vars], m_cam, v_cam, s_idx, lr=0.025)
        with torch.no_grad():
            rot_norms = torch.norm(drot_vars, dim=1, keepdim=True).clamp(min=1e-8)
            max_rad = math.radians(12.0)
            scale_clamp = torch.clamp(max_rad / rot_norms, max=1.0)
            drot_vars.mul_(scale_clamp)
        l_val = float(total_c.detach().item())
        loss_history.append(l_val)
        if l_val < best_cam_loss:
            best_cam_loss = l_val
            best_dt_snap = dt_vars.detach().clone()
            best_drot_snap = drot_vars.detach().clone()

    cand_cameras = [
        (np.asarray(r, dtype=np.float32).copy(), np.asarray(t, dtype=np.float32).copy(), float(f))
        for r, t, f in best_cameras
    ]
    with torch.no_grad():
        for i, tgt in enumerate(view_targets):
            v_idx = tgt["view_idx"]
            r_new = (r0_list[i] @ _rodrigues_torch(best_drot_snap[i])).cpu().numpy().astype(np.float32)
            t_new = (t0_list[i] + best_dt_snap[i]).cpu().numpy().astype(np.float32)
            cand_cameras[v_idx] = (r_new, t_new, best_cameras[v_idx][2])

    cand_iou, _ = _evaluate_multiview_silhouette_iou(
        verts_0, faces_np, cand_cameras, segmented_views, view_verts_list=init_view_verts
    )
    if cand_iou > best_iou + 1e-4:
        best_iou = cand_iou
        best_cameras = cand_cameras

    # Stage B: Shape & scale refinement when steps >= 10 and silhouette overlap has clear room to improve.
    if int(steps) >= 10 and best_iou < 0.92:
        has_grad = False
        try:
            sh_p = fused.shape_params.detach().to(dev).clone().requires_grad_(True)
            sc_p = fused.scale_params.detach().to(dev).clone().requires_grad_(True)
            scale_vec_p = head_assets["scale_mean"][None, :] + sc_p @ head_assets["scale_comps"]
            full_pose_p = torch.cat([torch.zeros(1, 6, device=dev), apose_params[:, :130].to(dev)], dim=1)
            expr_p = torch.zeros((1, 72), device=dev, dtype=scale_vec_p.dtype)
            if fused.expr_params is not None:
                expr_p = fused.expr_params.detach().to(dev, dtype=scale_vec_p.dtype).view(1, 72)
            v_p, _ = mhr_model(sh_p, torch.cat([full_pose_p, scale_vec_p], dim=1), expr_p)
            has_grad = bool(getattr(v_p, "requires_grad", False))
        except Exception:
            has_grad = False

        r_fixed = [
            torch.from_numpy(np.asarray(best_cameras[t["view_idx"]][0], dtype=np.float32)).to(dev)
            for t in view_targets
        ]
        t_fixed = [
            torch.from_numpy(np.asarray(best_cameras[t["view_idx"]][1], dtype=np.float32)).to(dev)
            for t in view_targets
        ]

        if has_grad:
            sh_opt = fused.shape_params.detach().to(dev).clone().requires_grad_(True)
            sc_opt = fused.scale_params.detach().to(dev).clone().requires_grad_(True)
            sh_ref = fused.shape_params.detach().to(dev)
            sc_ref = fused.scale_params.detach().to(dev)
            expr_fix = torch.zeros((1, 72), device=dev, dtype=head_assets["scale_mean"].dtype)
            if fused.expr_params is not None:
                expr_fix = fused.expr_params.detach().to(dev, dtype=expr_fix.dtype).view(1, 72)
            full_pose_fix = torch.cat([torch.zeros(1, 6, device=dev), apose_params[:, :130].to(dev)], dim=1)
            m_sh = [torch.zeros_like(sh_opt), torch.zeros_like(sc_opt)]
            v_sh = [torch.zeros_like(sh_opt), torch.zeros_like(sc_opt)]
            shape_steps = min(int(steps), 14)
            best_sh_snap = sh_opt.detach().clone()
            best_sc_snap = sc_opt.detach().clone()
            best_s_loss = float("inf")
            grad_failed = False
            for s_idx in range(1, shape_steps + 1):
                try:
                    scale_vec = head_assets["scale_mean"][None, :] + sc_opt @ head_assets["scale_comps"]
                    v_raw, _ = mhr_model(sh_opt, torch.cat([full_pose_fix, scale_vec], dim=1), expr_fix)
                    v_m = v_raw[0] / 100.0
                    v_min_y = v_m[:, 1].min()
                    v_h = torch.clamp(v_m[:, 1].max() - v_min_y, min=1e-4)
                    s_fac = (float(height_cm) / 100.0) / v_h
                    v_scaled = (
                        v_m - torch.stack([torch.zeros_like(v_min_y), v_min_y, torch.zeros_like(v_min_y)])
                    ) * s_fac
                    loss_sil = _contour_projection_loss_for_views(
                        [v_scaled for _ in view_targets], view_targets, r_fixed, t_fixed, v_sub_idx
                    )
                    loss_prior = 0.02 * (((sh_opt - sh_ref) ** 2).mean() + ((sc_opt - sc_ref) ** 2).mean())
                    total_s = loss_sil + loss_prior
                    total_s.backward()
                except RuntimeError:
                    grad_failed = True
                    break
                _adam_step_tensors([sh_opt, sc_opt], m_sh, v_sh, s_idx, lr=0.03)
                l_val = float(total_s.detach().item())
                loss_history.append(l_val)
                if l_val < best_s_loss:
                    best_s_loss = l_val
                    best_sh_snap = sh_opt.detach().clone()
                    best_sc_snap = sc_opt.detach().clone()

            if not grad_failed:
                cand_fused = FusedSubjectParams(
                    shape_params=best_sh_snap.cpu(),
                    scale_params=best_sc_snap.cpu(),
                    weights=fused.weights,
                    distances=fused.distances,
                    images_used=fused.images_used,
                    images_total=fused.images_total,
                    mean_confidence=fused.mean_confidence,
                    expr_params=fused.expr_params,
                )
                v_cand, _, _, _ = build_scaled_apose_mesh(
                    mhr_model, head_assets, cand_fused, float(height_cm), apose_params=apose_params
                )
                iou_s, _ = _evaluate_multiview_silhouette_iou(v_cand, faces_np, best_cameras, segmented_views)
                if iou_s > best_iou + 0.005:
                    best_iou = iou_s
                    best_fused = cand_fused
            else:
                has_grad = False

        if not has_grad:
            # Non-differentiable coordinate probe on leading shape/scale modes when IoU < 0.92.
            cur_sh = fused.shape_params.detach().cpu().clone()
            cur_sc = fused.scale_params.detach().cpu().clone()
            n_modes = min(4, cur_sh.shape[1], cur_sc.shape[1])
            for m_idx in range(n_modes):
                for is_scale in (True, False):
                    for delta in (-0.25, 0.25):
                        sh_t = cur_sh.clone()
                        sc_t = cur_sc.clone()
                        if is_scale:
                            sc_t[0, m_idx] += delta
                        else:
                            sh_t[0, m_idx] += delta
                        cand_f = FusedSubjectParams(
                            shape_params=sh_t,
                            scale_params=sc_t,
                            weights=fused.weights,
                            distances=fused.distances,
                            images_used=fused.images_used,
                            images_total=fused.images_total,
                            mean_confidence=fused.mean_confidence,
                            expr_params=fused.expr_params,
                        )
                        v_c, _, _, _ = build_scaled_apose_mesh(
                            mhr_model, head_assets, cand_f, float(height_cm), apose_params=apose_params
                        )
                        iou_c, _ = _evaluate_multiview_silhouette_iou(v_c, faces_np, best_cameras, segmented_views)
                        if iou_c > best_iou + 0.005:
                            best_iou = iou_c
                            best_fused = cand_f
                            cur_sh, cur_sc = sh_t, sc_t

    return best_fused, best_cameras, loss_history, True


def fit_mhr_multiview(
    mhr_model: Any,
    head_assets: Dict[str, torch.Tensor],
    fused: FusedSubjectParams,
    segmented_views: List[SegmentedView],
    raw_results: List[Dict[str, Any]],
    height_cm: float,
    weight_kg: float,
    sex: str,
    steps: int = 50,
    prefer_pytorch3d: bool = True,
) -> Tuple[FusedSubjectParams, ProportionAudit, Dict[str, Any]]:
    """Optimize MHR parameters against hair-excluded silhouettes and audit weight consistency."""
    verify_license_compliance()
    if height_cm <= 0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")

    faces_raw = head_assets["faces"]
    faces_np = (
        faces_raw.detach().cpu().numpy()
        if isinstance(faces_raw, torch.Tensor)
        else np.asarray(faces_raw, dtype=np.int64)
    )

    apose_init = solve_straight_apose(
        mhr_model, head_assets, fused.shape_params, fused.scale_params, expr_t=fused.expr_params
    )
    verts_init, joints_init, _, raw_height_cm = build_scaled_apose_mesh(
        mhr_model, head_assets, fused, float(height_cm), apose_params=apose_init
    )

    # Evaluate view-specific posed meshes/joints for camera initialization when body_pose_params is provided.
    init_view_poses: List[torch.Tensor] = [
        _extract_body_pose_tensor(
            (_resolve_raw_record(view, idx, raw_results).get("output") or {}).get("body_pose_params"),
            apose_init,
            torch.device("cpu"),
        )
        for idx, view in enumerate(segmented_views)
    ]
    has_raw_pose = any(not torch.allclose(p, apose_init.detach().cpu(), atol=1e-6) for p in init_view_poses)
    canon_init_min_y: Optional[float] = None
    canon_init_scale: Optional[float] = None
    if has_raw_pose:
        canon_v_raw, _ = evaluate_mhr_pose(
            mhr_model, head_assets, fused.shape_params, fused.scale_params, apose_init, expr_t=fused.expr_params
        )
        canon_init_min_y = float(canon_v_raw[:, 1].min())
        canon_init_scale = (float(height_cm) / 100.0) / max(float(canon_v_raw[:, 1].max()) - canon_init_min_y, 1e-4)

    init_cameras: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for idx, view in enumerate(segmented_views):
        rec = _resolve_raw_record(view, idx, raw_results)
        out = rec.get("output") or {}
        if has_raw_pose and not torch.allclose(init_view_poses[idx], apose_init.detach().cpu(), atol=1e-6):
            v_v0, j_v0 = evaluate_mhr_pose(
                mhr_model,
                head_assets,
                fused.shape_params,
                fused.scale_params,
                init_view_poses[idx],
                expr_t=fused.expr_params,
            )
            v_v0 = v_v0.copy()
            j_v0 = j_v0.copy()
            v_v0[:, 1] -= float(canon_init_min_y)
            j_v0[:, 1] -= float(canon_init_min_y)
            v_v0 *= float(canon_init_scale)
            j_v0 *= float(canon_init_scale)
        else:
            v_v0, j_v0 = verts_init, joints_init
        init_cameras.append(_solve_initial_view_camera(v_v0, j_v0, view, out))

    updated_fused = fused
    final_cameras = init_cameras
    loss_history: List[float] = []
    used_pytorch3d = False
    used_contour_fallback = False
    extra_outputs: Dict[str, Any] = {}

    if prefer_pytorch3d and segmented_views:
        updated_fused, final_cameras, loss_history, used_pytorch3d = _run_pytorch3d_optimization(
            mhr_model=mhr_model,
            head_assets=head_assets,
            fused=fused,
            segmented_views=segmented_views,
            raw_results=raw_results,
            init_cameras=init_cameras,
            height_cm=float(height_cm),
            weight_kg=float(weight_kg or 0.0),
            sex=sex,
            steps=int(steps),
            apose_params=apose_init,
            extra_outputs=extra_outputs,
        )
        if not used_pytorch3d and int(steps) > 0:
            init_posed_for_opt: Optional[List[np.ndarray]] = None
            if has_raw_pose:
                init_posed_for_opt = _evaluate_view_posed_meshes(
                    mhr_model=mhr_model,
                    head_assets=head_assets,
                    fused=fused,
                    apose_params=apose_init,
                    apose_scaled_verts=verts_init,
                    view_poses=init_view_poses,
                    height_cm=float(height_cm),
                    raw_results=raw_results,
                    segmented_views=segmented_views,
                )
            updated_fused, final_cameras, loss_history, used_contour_fallback = (
                _run_contour_fallback_optimization(
                    mhr_model=mhr_model,
                    head_assets=head_assets,
                    fused=fused,
                    segmented_views=segmented_views,
                    init_cameras=init_cameras,
                    height_cm=float(height_cm),
                    weight_kg=float(weight_kg or 0.0),
                    sex=sex,
                    steps=int(steps),
                    apose_params=apose_init,
                    init_view_verts=init_posed_for_opt,
                )
            )

    params_changed = (
        not torch.allclose(updated_fused.shape_params.detach().cpu(), fused.shape_params.detach().cpu(), atol=1e-6)
        or not torch.allclose(updated_fused.scale_params.detach().cpu(), fused.scale_params.detach().cpu(), atol=1e-6)
    )
    if used_pytorch3d or (used_contour_fallback and params_changed):
        apose_final = solve_straight_apose(
            mhr_model,
            head_assets,
            updated_fused.shape_params,
            updated_fused.scale_params,
            expr_t=updated_fused.expr_params,
        )
        verts_final, _, _, raw_height_cm = build_scaled_apose_mesh(
            mhr_model, head_assets, updated_fused, float(height_cm), apose_params=apose_final
        )
    else:
        apose_final = apose_init
        verts_final = verts_init

    final_view_poses: List[torch.Tensor] = extra_outputs.get("view_poses", init_view_poses)
    if len(final_view_poses) != len(segmented_views):
        final_view_poses = init_view_poses
    view_posed_verts = _evaluate_view_posed_meshes(
        mhr_model=mhr_model,
        head_assets=head_assets,
        fused=updated_fused,
        apose_params=apose_final,
        apose_scaled_verts=verts_final,
        view_poses=final_view_poses,
        height_cm=float(height_cm),
        raw_results=raw_results,
        segmented_views=segmented_views,
    )

    mhr_iou, per_view_ious = _evaluate_multiview_silhouette_iou(
        verts_a=verts_final,
        faces_np=faces_np,
        cameras=final_cameras,
        segmented_views=segmented_views,
        view_verts_list=view_posed_verts,
    )
    audit = evaluate_proportion_audit(
        verts_a=verts_final,
        faces=faces_np,
        sex=sex,
        declared_weight_kg=float(weight_kg or 0.0),
        mhr_silhouette_iou=mhr_iou,
    )
    volume_liters = float(compute_closed_mesh_volume_liters(verts_final, faces_np))
    debug_info: Dict[str, Any] = {
        "used_pytorch3d": used_pytorch3d,
        "used_contour_fallback": used_contour_fallback,
        "steps_run": len(loss_history),
        "loss_history": loss_history,
        "per_view_ious": per_view_ious,
        "volume_liters": volume_liters,
        "estimated_weight_kg": float(audit.estimated_weight_kg),
        "raw_height_cm": float(raw_height_cm),
        "warnings": list(audit.warnings),
        "apose_params": apose_final.detach().cpu(),
        "final_cameras": [
            (np.asarray(r, dtype=np.float32).copy(), np.asarray(t, dtype=np.float32).copy(), float(f))
            for r, t, f in final_cameras
        ],
        "view_posed_verts": view_posed_verts,
        "view_poses": [p.detach().cpu() for p in final_view_poses],
    }
    return updated_fused, audit, debug_info
