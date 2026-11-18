"""Unit tests for metric multi-view MHR fitting, volume/weight estimation, and proportion_audit."""

import math
import unittest
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from bodyscan_pipeline.config import SEX_BODY_DENSITY_KG_PER_L


def _make_closed_box_mesh(extents: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Build a closed, outward-oriented triangle box centered at origin with given (dx, dy, dz) meters."""
    hx, hy, hz = [e * 0.5 for e in extents]
    verts = np.array(
        [
            [-hx, -hy, -hz],  # 0: -X -Y -Z
            [+hx, -hy, -hz],  # 1: +X -Y -Z
            [+hx, +hy, -hz],  # 2: +X +Y -Z
            [-hx, +hy, -hz],  # 3: -X +Y -Z
            [-hx, -hy, +hz],  # 4: -X -Y +Z
            [+hx, -hy, +hz],  # 5: +X -Y +Z
            [+hx, +hy, +hz],  # 6: +X +Y +Z
            [-hx, +hy, +hz],  # 7: -X +Y +Z
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # -Z face
            [4, 5, 6], [4, 6, 7],  # +Z face
            [0, 1, 5], [0, 5, 4],  # -Y face
            [3, 7, 6], [3, 6, 2],  # +Y face
            [0, 4, 7], [0, 7, 3],  # -X face
            [1, 2, 6], [1, 6, 5],  # +X face
        ],
        dtype=np.int64,
    )
    return verts, faces


def _make_uv_sphere_mesh(n_lat: int = 20, n_lon: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Create a watertight unit-radius closed triangle sphere centered at the origin."""
    verts = [[0.0, 1.0, 0.0]]  # top pole idx 0
    for i in range(1, n_lat):
        phi = math.pi * i / n_lat
        y = math.cos(phi)
        r = math.sin(phi)
        for j in range(n_lon):
            theta = 2.0 * math.pi * j / n_lon
            verts.append([r * math.cos(theta), y, r * math.sin(theta)])
    bot_idx = len(verts)
    verts.append([0.0, -1.0, 0.0])  # bottom pole

    faces = []
    # Top cap
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([0, 1 + jn, 1 + j])
    # Middle bands
    for i in range(n_lat - 2):
        row0 = 1 + i * n_lon
        row1 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([row0 + j, row0 + jn, row1 + jn])
            faces.append([row0 + j, row1 + jn, row1 + j])
    # Bottom cap
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([bot_idx, last_row + j, last_row + jn])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


class _DummyLBS:
    """Minimal LBS skinning stub for build_arm_vertex_mask."""

    def __init__(self, num_verts: int) -> None:
        self.vert_indices_flattened = torch.arange(num_verts, dtype=torch.int64)
        self.skin_indices_flattened = torch.zeros(num_verts, dtype=torch.int64)
        self.skin_weights_flattened = torch.ones(num_verts, dtype=torch.float32)


class _DummyCharacter:
    def __init__(self, num_verts: int) -> None:
        self.linear_blend_skinning = _DummyLBS(num_verts)


class _FakeMHRModel:
    """Lightweight differentiable fake MHR model producing a watertight humanoid-scaled capsule/ellipsoid."""

    def __init__(self, base_verts_m: np.ndarray) -> None:
        self.base_verts_m = torch.tensor(base_verts_m, dtype=torch.float32)
        self.character_torch = _DummyCharacter(len(base_verts_m))

    def __call__(
        self, shape_t: torch.Tensor, model_params: torch.Tensor, expr_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scale body widths/height subtly from shape_params[0, :3], scale_vec, and expr_t[0, 0]
        # Base ellipsoid spans y in [-0.85, 0.85] -> 1.70m, rx=0.22m, rz=0.14m (in cm below)
        sx = 0.22 * (1.0 + 0.12 * shape_t[:, 0:1])
        sy = 0.85 * (1.0 + 0.05 * model_params[:, 136:137])
        sz = 0.14 * (1.0 + 0.12 * shape_t[:, 1:2]) + 0.01 * expr_t[:, 0:1]

        vx = self.base_verts_m[:, 0:1].T * sx
        vy = self.base_verts_m[:, 1:2].T * sy
        vz = self.base_verts_m[:, 2:3].T * sz
        verts_m = torch.stack([vx, vy, vz], dim=-1)  # (B, V, 3)

        # Shift up so canonical feet sit near y = 0, crown at y ~ 1.70m
        verts_m = verts_m + torch.tensor([[[0.0, 0.85, 0.0]]], dtype=verts_m.dtype, device=verts_m.device)
        verts_cm = verts_m * 100.0

        b = shape_t.shape[0]
        skel_cm = torch.zeros((b, 127, 3), dtype=verts_cm.dtype, device=verts_cm.device)
        # Straight arms so solve_straight_apose converges cleanly
        skel_cm[:, 39] = torch.tensor([-20.0, 140.0, 0.0])
        skel_cm[:, 40] = torch.tensor([-35.0, 115.0, 0.0])
        skel_cm[:, 41] = torch.tensor([-50.0, 90.0, 0.0])
        skel_cm[:, 75] = torch.tensor([20.0, 140.0, 0.0])
        skel_cm[:, 76] = torch.tensor([35.0, 115.0, 0.0])
        skel_cm[:, 77] = torch.tensor([50.0, 90.0, 0.0])
        skel_cm[:, 110] = torch.tensor([0.0, 142.0, 0.0])
        skel_cm[:, 113] = torch.tensor([0.0, 152.0, 0.0])
        return verts_cm, skel_cm


def _render_soft_silhouette(
    verts_3d: torch.Tensor,
    azimuth_deg: float,
    img_size: int = 64,
    sigma_px: float = 2.0,
) -> torch.Tensor:
    """Pure-PyTorch differentiable soft silhouette renderer from 3D vertices."""
    rad = math.radians(azimuth_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot_y = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=verts_3d.dtype, device=verts_3d.device)
    v_rot = verts_3d @ rot_y.T
    # Map x in [-0.6, 0.6], y in [-1.0, 1.0] to pixel space [0, img_size - 1]
    u = (v_rot[:, 0] / 0.6) * (img_size * 0.42) + (img_size * 0.5)
    v = (-v_rot[:, 1] / 1.0) * (img_size * 0.42) + (img_size * 0.5)

    ys = torch.arange(img_size, dtype=verts_3d.dtype, device=verts_3d.device)
    xs = torch.arange(img_size, dtype=verts_3d.dtype, device=verts_3d.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dx = gx.unsqueeze(-1) - u.view(1, 1, -1)
    dy = gy.unsqueeze(-1) - v.view(1, 1, -1)
    d2 = dx * dx + dy * dy
    density = torch.exp(-d2 / (sigma_px * sigma_px)).sum(dim=-1)
    return 1.0 - torch.exp(-2.2 * density)


def _project_hard_mask(
    verts_m: np.ndarray,
    faces: np.ndarray,
    azimuth_deg: float,
    focal_length: float,
    cam_t: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Project closed triangle mesh to binary OpenCV silhouette mask at given azimuth."""
    rad = math.radians(azimuth_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    r_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    verts_cam = (verts_m @ (r_flip @ rot_y).T) + cam_t.reshape(1, 3)
    z = np.maximum(verts_cam[:, 2], 0.05)
    u = focal_length * (verts_cam[:, 0] / z) + width / 2.0
    v = focal_length * (verts_cam[:, 1] / z) + height / 2.0
    pts_2d = np.stack([u, v], axis=-1)

    mask = np.zeros((height, width), dtype=np.uint8)
    tris = np.round(pts_2d[faces]).astype(np.int32)
    for tri in tris:
        cv2.fillConvexPoly(mask, tri, 1)
    return mask > 0


class TestFitting(unittest.TestCase):
    """Test closed mesh volume, sex density estimation, soft silhouette+edge loss, and proportion_audit."""

    def test_closed_mesh_volume_liters_and_sex_weight_estimation(self) -> None:
        from bodyscan_pipeline.fitting import compute_closed_mesh_volume_liters, estimate_weight_kg

        verts_np, faces_np = _make_closed_box_mesh((0.40, 0.25, 1.00))
        # Shift off-origin to confirm translation invariance
        verts_shifted = verts_np + np.array([1.2, 0.85, -2.5], dtype=np.float64)
        vol_np = compute_closed_mesh_volume_liters(verts_shifted, faces_np)
        self.assertAlmostEqual(vol_np, 100.0, delta=0.05)

        # Reversed winding must also produce exact positive volume
        vol_rev = compute_closed_mesh_volume_liters(verts_shifted, faces_np[:, ::-1])
        self.assertAlmostEqual(vol_rev, 100.0, delta=0.05)

        # Differentiable PyTorch tensor volume check
        verts_t = torch.tensor(verts_shifted, dtype=torch.float32, requires_grad=True)
        vol_t = compute_closed_mesh_volume_liters(verts_t, faces_np)
        self.assertIsInstance(vol_t, torch.Tensor)
        self.assertAlmostEqual(float(vol_t.item()), 100.0, delta=0.05)
        vol_t.backward()
        self.assertIsNotNone(verts_t.grad)
        self.assertTrue(torch.isfinite(verts_t.grad).all())
        self.assertGreater(float(verts_t.grad.abs().sum().item()), 0.0)

        # Analytical ellipsoid check (radii 0.20m, 0.85m, 0.14m)
        sphere_v, sphere_f = _make_uv_sphere_mesh(n_lat=64, n_lon=96)
        ell_v = sphere_v * np.array([0.20, 0.85, 0.14], dtype=np.float32)
        expected_ell_l = (4.0 / 3.0) * math.pi * 0.20 * 0.85 * 0.14 * 1000.0
        vol_ell = compute_closed_mesh_volume_liters(ell_v, sphere_f)
        self.assertAlmostEqual(vol_ell, expected_ell_l, delta=0.15)

        # Sex-specific body density weight estimation (male 1.045 vs female 1.028 kg/L)
        w_male = estimate_weight_kg(verts_shifted, faces_np, "male")
        w_female = estimate_weight_kg(verts_shifted, faces_np, "female")
        self.assertAlmostEqual(w_male, 100.0 * SEX_BODY_DENSITY_KG_PER_L["male"], delta=0.05)
        self.assertAlmostEqual(w_female, 100.0 * SEX_BODY_DENSITY_KG_PER_L["female"], delta=0.05)
        self.assertGreater(w_male, w_female)

    def test_soft_silhouette_and_edge_loss_backpropagates_shape_scale(self) -> None:
        from bodyscan_pipeline.fitting import compute_soft_silhouette_and_edge_loss, estimate_weight_kg

        base_v_np, base_f_np = _make_uv_sphere_mesh(n_lat=14, n_lon=22)
        base_v = torch.tensor(base_v_np, dtype=torch.float32)
        gt_scale = torch.tensor([0.28, 0.82, 0.16], dtype=torch.float32)

        with torch.no_grad():
            gt_verts = base_v * gt_scale
            tgt_front = (_render_soft_silhouette(gt_verts, 0.0) > 0.5).float()
            tgt_side = (_render_soft_silhouette(gt_verts, 90.0) > 0.5).float()
            tgt_weight_kg = float(estimate_weight_kg(gt_verts, base_f_np, "male").item())

        # Start from distorted thin/deep proportions and recover target via Adam
        sx = torch.tensor(0.15, dtype=torch.float32, requires_grad=True)
        sz = torch.tensor(0.30, dtype=torch.float32, requires_grad=True)
        sy = torch.tensor(0.82, dtype=torch.float32)
        opt = torch.optim.Adam([sx, sz], lr=0.010, betas=(0.65, 0.999))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=2e-4)

        losses = []
        for _ in range(30):
            opt.zero_grad()
            cur_verts = base_v * torch.stack([sx, sy, sz])
            ren_front = _render_soft_silhouette(cur_verts, 0.0)
            ren_side = _render_soft_silhouette(cur_verts, 90.0)

            l_sil0, l_edge0 = compute_soft_silhouette_and_edge_loss(ren_front, tgt_front)
            l_sil1, l_edge1 = compute_soft_silhouette_and_edge_loss(ren_side, tgt_side)
            est_w = estimate_weight_kg(cur_verts, base_f_np, "male")
            l_vol = ((est_w - tgt_weight_kg) / tgt_weight_kg) ** 2

            l_total = 0.5 * (l_sil0 + l_sil1) + 0.25 * (l_edge0 + l_edge1) + 0.05 * l_vol
            l_total.backward()
            opt.step()
            sched.step()
            losses.append(float(l_total.item()))

        self.assertLess(losses[-1], losses[0] * 0.20)
        for idx in range(len(losses) - 1):
            self.assertLessEqual(losses[idx + 1], losses[idx] + 2e-3)
        window_means = [float(np.mean(losses[i : i + 5])) for i in range(0, len(losses), 5)]
        for k in range(len(window_means) - 1):
            self.assertLess(window_means[k + 1], window_means[k])

        with torch.no_grad():
            final_verts = base_v * torch.stack([sx, sy, sz])
            iou_vals = []
            for az, tgt in ((0.0, tgt_front), (90.0, tgt_side)):
                pred_b = _render_soft_silhouette(final_verts, az) > 0.5
                tgt_b = tgt > 0.5
                inter = float((pred_b & tgt_b).sum().item())
                union = float((pred_b | tgt_b).sum().item())
                iou_vals.append(inter / max(union, 1.0))

        self.assertGreater(min(iou_vals), 0.90)
        self.assertAlmostEqual(float(sx.item()), float(gt_scale[0].item()), delta=0.025)
        self.assertAlmostEqual(float(sz.item()), float(gt_scale[2].item()), delta=0.025)

    def test_fit_mhr_weight_zero_and_contradictory_mass_audit(self) -> None:
        from bodyscan_pipeline.fitting import (
            ProportionAudit,
            estimate_weight_kg,
            evaluate_proportion_audit,
            fit_mhr_multiview,
        )
        from bodyscan_pipeline.fusion import FusedSubjectParams, build_scaled_apose_mesh, evaluate_mhr_pose
        from bodyscan_pipeline.segmentation import SegmentedView

        verts_box, faces_box = _make_closed_box_mesh((0.40, 0.25, 1.00))
        # Exact expected male weight for 100.0 L box = 104.5 kg
        exact_w = float(estimate_weight_kg(verts_box, faces_box, "male"))
        self.assertAlmostEqual(exact_w, 104.5, delta=0.05)

        # Case A: declared_weight_kg == 0.0 -> "estimated_only", weight_diff_pct is None
        audit_a = evaluate_proportion_audit(
            verts_a=verts_box,
            faces=faces_box,
            sex="male",
            declared_weight_kg=0.0,
            mhr_silhouette_iou=0.88,
        )
        self.assertIsInstance(audit_a, ProportionAudit)
        self.assertEqual(audit_a.status, "estimated_only")
        self.assertIsNone(audit_a.weight_diff_pct)
        self.assertEqual(audit_a.warnings, [])

        # Case B: declared_weight_kg within 5% -> "matched", warnings == []
        audit_b = evaluate_proportion_audit(
            verts_a=verts_box,
            faces=faces_box,
            sex="male",
            declared_weight_kg=exact_w * 0.96,  # ~4.17% diff
            mhr_silhouette_iou=0.88,
        )
        self.assertEqual(audit_b.status, "matched")
        self.assertEqual(audit_b.warnings, [])
        self.assertLess(audit_b.weight_diff_pct, 8.0)

        # Case C: declared_weight_kg between 8% and 12% -> status stays "matched" + informational note
        audit_c = evaluate_proportion_audit(
            verts_a=verts_box,
            faces=faces_box,
            sex="male",
            declared_weight_kg=exact_w / 1.10,  # 10.0% diff
            mhr_silhouette_iou=0.88,
        )
        self.assertEqual(audit_c.status, "matched")
        self.assertEqual(len(audit_c.warnings), 1)
        self.assertAlmostEqual(audit_c.weight_diff_pct, 10.0, delta=0.2)

        # Case D: declared_weight_kg differs by 35% -> "mismatch_warning"
        audit_d = evaluate_proportion_audit(
            verts_a=verts_box,
            faces=faces_box,
            sex="male",
            declared_weight_kg=exact_w * 1.35,
            mhr_silhouette_iou=0.88,
        )
        self.assertEqual(audit_d.status, "mismatch_warning")
        self.assertGreaterEqual(len(audit_d.warnings), 1)

        # Case E: poor silhouette overlap (0.68 < 0.78) -> "mismatch_warning" even if weight matches
        audit_e = evaluate_proportion_audit(
            verts_a=verts_box,
            faces=faces_box,
            sex="male",
            declared_weight_kg=exact_w,
            mhr_silhouette_iou=0.68,
            outer_silhouette_iou=0.71,
        )
        self.assertEqual(audit_e.status, "mismatch_warning")
        self.assertGreaterEqual(len(audit_e.warnings), 1)
        audit_dict = audit_e.to_dict()
        self.assertEqual(audit_dict["status"], "mismatch_warning")
        self.assertEqual(audit_dict["outer_silhouette_iou"], 0.71)

        # End-to-end fit_mhr_multiview + fusion expr_params compatibility on CPU
        sphere_v, sphere_f = _make_uv_sphere_mesh(n_lat=14, n_lon=22)
        mhr_model = _FakeMHRModel(sphere_v)
        head_assets: Dict[str, torch.Tensor] = {
            "scale_mean": torch.zeros(68, dtype=torch.float32),
            "scale_comps": torch.zeros((28, 68), dtype=torch.float32),
            "faces": torch.tensor(sphere_f, dtype=torch.int64),
        }
        fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45), dtype=torch.float32),
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.full(4, 0.25, dtype=np.float64),
            distances=np.zeros(4, dtype=np.float64),
            images_used=4,
            images_total=4,
            mean_confidence=0.95,
            expr_params=torch.zeros((1, 72), dtype=torch.float32),
        )
        # Verify evaluate_mhr_pose & build_scaled_apose_mesh accept expr_params cleanly
        verts_eval, _ = evaluate_mhr_pose(
            mhr_model, head_assets, fused.shape_params, fused.scale_params, torch.zeros(1, 130), expr_t=fused.expr_params
        )
        self.assertEqual(verts_eval.shape, sphere_v.shape)
        verts_a, _, _, _ = build_scaled_apose_mesh(mhr_model, head_assets, fused, height_cm=170.0)

        h_img, w_img = 160, 120
        focal = 280.0
        cam_t = np.array([0.0, 0.85, 2.8], dtype=np.float32)
        seg_views = []
        raw_results = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            mask = _project_hard_mask(verts_a, sphere_f, az, focal, cam_t, h_img, w_img)
            empty = np.zeros_like(mask)
            seg_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=mask,
                    body_mask=mask,
                    hair_mask=empty,
                    face_skin_mask=empty,
                )
            )
            raw_results.append(
                {
                    "image_name": f"{label}.jpg",
                    "output": {
                        "pred_cam_t": cam_t.copy(),
                        "focal_length": np.array([focal], dtype=np.float32),
                    },
                }
            )

        # weight_kg == 0.0 -> estimated_only with clean silhouette IoU (also tests prefer_pytorch3d=True fallback)
        fitted_zero, audit_zero, debug_zero = fit_mhr_multiview(
            mhr_model=mhr_model,
            head_assets=head_assets,
            fused=fused,
            segmented_views=seg_views,
            raw_results=raw_results,
            height_cm=170.0,
            weight_kg=0.0,
            sex="male",
            steps=10,
            prefer_pytorch3d=True,
        )
        self.assertEqual(audit_zero.status, "estimated_only")
        self.assertIsNone(audit_zero.weight_diff_pct)
        self.assertGreater(audit_zero.mhr_silhouette_iou, 0.85)
        self.assertIn("volume_liters", debug_zero)

        # Contradictory 240 kg weight must trigger mismatch_warning without distorting mesh mass
        _, audit_heavy, _ = fit_mhr_multiview(
            mhr_model=mhr_model,
            head_assets=head_assets,
            fused=fused,
            segmented_views=seg_views,
            raw_results=raw_results,
            height_cm=170.0,
            weight_kg=240.0,
            sex="male",
            steps=10,
            prefer_pytorch3d=False,
        )
        self.assertEqual(audit_heavy.status, "mismatch_warning")
        self.assertGreaterEqual(len(audit_heavy.warnings), 1)
        self.assertAlmostEqual(audit_heavy.estimated_weight_kg, audit_zero.estimated_weight_kg, delta=1.0)
        self.assertEqual(fitted_zero.images_used, 4)

        # Ensure body_mask (hair-excluded) is used rather than person_mask with tall hair bun
        views_with_bun = []
        for sv in seg_views:
            hair_bun = np.zeros_like(sv.person_mask)
            hair_bun[:18, 40:80] = True
            views_with_bun.append(
                SegmentedView(
                    view_index=sv.view_index,
                    azimuth_deg=sv.azimuth_deg,
                    view_label=sv.view_label,
                    person_mask=sv.body_mask | hair_bun,
                    body_mask=sv.body_mask,
                    hair_mask=hair_bun,
                    face_skin_mask=sv.face_skin_mask,
                )
            )
        _, audit_bun, _ = fit_mhr_multiview(
            mhr_model=mhr_model,
            head_assets=head_assets,
            fused=fused,
            segmented_views=views_with_bun,
            raw_results=raw_results,
            height_cm=170.0,
            weight_kg=0.0,
            sex="male",
            steps=10,
            prefer_pytorch3d=True,
        )
        self.assertAlmostEqual(audit_bun.mhr_silhouette_iou, audit_zero.mhr_silhouette_iou, delta=1e-5)

        # License guard check
        import sys

        sys.modules["smplx"] = type(sys)("smplx")
        try:
            with self.assertRaises(RuntimeError):
                fit_mhr_multiview(
                    mhr_model=mhr_model,
                    head_assets=head_assets,
                    fused=fused,
                    segmented_views=seg_views,
                    raw_results=raw_results,
                    height_cm=170.0,
                    weight_kg=0.0,
                    sex="male",
                    prefer_pytorch3d=False,
                )
        finally:
            sys.modules.pop("smplx", None)

    def test_pytorch3d_optimization_path_and_dominant_silhouette_clamp(self) -> None:
        import sys
        import types
        from bodyscan_pipeline.fitting import fit_mhr_multiview
        from bodyscan_pipeline.fusion import FusedSubjectParams, build_scaled_apose_mesh
        from bodyscan_pipeline.segmentation import SegmentedView

        sphere_v, sphere_f = _make_uv_sphere_mesh(n_lat=12, n_lon=18)
        mhr_model = _FakeMHRModel(sphere_v)
        head_assets: Dict[str, torch.Tensor] = {
            "scale_mean": torch.zeros(68, dtype=torch.float32),
            "scale_comps": torch.zeros((28, 68), dtype=torch.float32),
            "faces": torch.tensor(sphere_f, dtype=torch.int64),
        }

        gt_fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45), dtype=torch.float32),
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.full(2, 0.5, dtype=np.float64),
            distances=np.zeros(2, dtype=np.float64),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )
        gt_verts, _, _, _ = build_scaled_apose_mesh(mhr_model, head_assets, gt_fused, height_cm=170.0)

        h_img, w_img = 96, 80
        focal = 180.0
        cam_t = np.array([0.0, 0.85, 2.8], dtype=np.float32)
        seg_views = []
        raw_results = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0))):
            mask = _project_hard_mask(gt_verts, sphere_f, az, focal, cam_t, h_img, w_img)
            empty = np.zeros_like(mask)
            seg_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=mask,
                    body_mask=mask,
                    hair_mask=empty,
                    face_skin_mask=empty,
                )
            )
            raw_results.append({"image_name": f"{label}.jpg", "output": {"focal_length": focal}})

        # Install lightweight differentiable pure-PyTorch stub for pytorch3d
        p3d_mod = types.ModuleType("pytorch3d")
        p3d_ren = types.ModuleType("pytorch3d.renderer")
        p3d_str = types.ModuleType("pytorch3d.structures")
        p3d_utl = types.ModuleType("pytorch3d.utils")

        class _StubMeshes:
            def __init__(self, verts, faces) -> None:
                self.verts = verts[0]
                self.faces = faces[0]

        class _StubRasterSettings:
            def __init__(self, image_size, blur_radius=0.0, faces_per_pixel=1) -> None:
                self.image_size = image_size

        class _StubRasterizer:
            def __init__(self, raster_settings) -> None:
                self.raster_settings = raster_settings

        class _StubSoftSilhouetteShader:
            pass

        class _StubMeshRenderer:
            def __init__(self, rasterizer, shader) -> None:
                self.h, self.w = rasterizer.raster_settings.image_size

            def __call__(self, meshes, cameras):
                r_mat, t_vec, k_mat = cameras
                v_cam = (meshes.verts @ r_mat[0].T) + t_vec[0].unsqueeze(0)
                z = torch.clamp(v_cam[:, 2], min=0.1)
                fx, fy = k_mat[0, 0, 0], k_mat[0, 1, 1]
                cx, cy = k_mat[0, 0, 2], k_mat[0, 1, 2]
                u = fx * (v_cam[:, 0] / z) + cx
                v = fy * (v_cam[:, 1] / z) + cy
                ys = torch.arange(self.h, dtype=v_cam.dtype, device=v_cam.device)
                xs = torch.arange(self.w, dtype=v_cam.dtype, device=v_cam.device)
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                d2 = (gx.unsqueeze(-1) - u.view(1, 1, -1)) ** 2 + (gy.unsqueeze(-1) - v.view(1, 1, -1)) ** 2
                density = torch.exp(-d2 / (3.0 * 3.0)).sum(dim=-1)
                alpha = 1.0 - torch.exp(-2.0 * density)
                rgba = torch.zeros((1, self.h, self.w, 4), dtype=v_cam.dtype, device=v_cam.device)
                rgba[0, :, :, 3] = alpha
                return rgba

        def _stub_cameras_from_opencv(r_mat, t_vec, cam_k, im_size):
            return (r_mat, t_vec, cam_k)

        p3d_str.Meshes = _StubMeshes
        p3d_ren.RasterizationSettings = _StubRasterSettings
        p3d_ren.MeshRasterizer = _StubRasterizer
        p3d_ren.SoftSilhouetteShader = _StubSoftSilhouetteShader
        p3d_ren.MeshRenderer = _StubMeshRenderer
        p3d_utl.cameras_from_opencv_projection = _stub_cameras_from_opencv

        init_shape = torch.zeros((1, 45), dtype=torch.float32)
        init_shape[0, 0] = -2.5
        init_shape[0, 1] = 2.5
        init_fused = FusedSubjectParams(
            shape_params=init_shape,
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.full(2, 0.5, dtype=np.float64),
            distances=np.zeros(2, dtype=np.float64),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )

        saved = {k: sys.modules.get(k) for k in ("pytorch3d", "pytorch3d.renderer", "pytorch3d.structures", "pytorch3d.utils")}
        sys.modules["pytorch3d"] = p3d_mod
        sys.modules["pytorch3d.renderer"] = p3d_ren
        sys.modules["pytorch3d.structures"] = p3d_str
        sys.modules["pytorch3d.utils"] = p3d_utl
        try:
            opt_fused, audit_fit, debug_fit = fit_mhr_multiview(
                mhr_model=mhr_model,
                head_assets=head_assets,
                fused=init_fused,
                segmented_views=seg_views,
                raw_results=raw_results,
                height_cm=170.0,
                weight_kg=250.0,  # Contradictory declared weight
                sex="male",
                steps=15,
                prefer_pytorch3d=True,
            )
        finally:
            for k, old in saved.items():
                if old is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old

        self.assertTrue(debug_fit["used_pytorch3d"])
        self.assertEqual(debug_fit["steps_run"], 15)
        self.assertLess(debug_fit["loss_history"][-1], debug_fit["loss_history"][0])
        self.assertIsNotNone(opt_fused.expr_params)
        self.assertEqual(audit_fit.status, "mismatch_warning")
        # Dominant L_sil + L2 clamp ensure contradictory 250 kg never distorts estimated mass near 250 kg
        self.assertLess(audit_fit.estimated_weight_kg, 140.0)
        self.assertLessEqual(float(torch.linalg.norm(opt_fused.shape_params - init_shape).item()), 2.5 + 1e-4)

    def test_solve_straight_apose_cache_reuses_result(self) -> None:
        from bodyscan_pipeline.fusion import FusedSubjectParams, build_scaled_apose_mesh, solve_straight_apose

        sphere_v, sphere_f = _make_uv_sphere_mesh(n_lat=8, n_lon=12)

        class _CountingFakeMHR(_FakeMHRModel):
            def __init__(self, verts: np.ndarray) -> None:
                super().__init__(verts)
                self.calls = 0

            def __call__(self, shape_t: torch.Tensor, model_params: torch.Tensor, expr_t: torch.Tensor):
                self.calls += 1
                return super().__call__(shape_t, model_params, expr_t)

        counting_mhr = _CountingFakeMHR(sphere_v)
        head_assets: Dict[str, torch.Tensor] = {
            "scale_mean": torch.zeros(68, dtype=torch.float32),
            "scale_comps": torch.zeros((28, 68), dtype=torch.float32),
            "faces": torch.tensor(sphere_f, dtype=torch.int64),
        }
        shape_t = torch.zeros((1, 45), dtype=torch.float32)
        scale_t = torch.ones((1, 28), dtype=torch.float32)
        expr_t = torch.zeros((1, 72), dtype=torch.float32)

        apose_1 = solve_straight_apose(counting_mhr, head_assets, shape_t, scale_t, expr_t=expr_t)
        first_calls = counting_mhr.calls
        self.assertGreaterEqual(first_calls, 2)
        self.assertLess(first_calls, 100)

        apose_2 = solve_straight_apose(counting_mhr, head_assets, shape_t, scale_t, expr_t=expr_t)
        self.assertEqual(counting_mhr.calls, first_calls)
        self.assertTrue(torch.allclose(apose_1, apose_2))

        fused = FusedSubjectParams(
            shape_params=shape_t,
            scale_params=scale_t,
            weights=np.array([1.0], dtype=np.float64),
            distances=np.array([0.0], dtype=np.float64),
            images_used=1,
            images_total=1,
            mean_confidence=1.0,
            expr_params=expr_t,
        )
        # Reuses cached solve_straight_apose + runs only 1 forward pass in evaluate_mhr_pose
        build_scaled_apose_mesh(counting_mhr, head_assets, fused, height_cm=170.0)
        self.assertEqual(counting_mhr.calls, first_calls + 1)

        # Explicitly passing apose_params also skips solving
        build_scaled_apose_mesh(counting_mhr, head_assets, fused, height_cm=170.0, apose_params=apose_1)
        self.assertEqual(counting_mhr.calls, first_calls + 2)

        # Different shape invalidates cache key and solves cleanly
        shape_other = shape_t.clone()
        shape_other[0, 0] = 0.5
        solve_straight_apose(counting_mhr, head_assets, shape_other, scale_t, expr_t=expr_t)
        self.assertEqual(counting_mhr.calls, 2 * first_calls + 2)

        # Explicit use_cache=False bypasses cache
        solve_straight_apose(counting_mhr, head_assets, shape_t, scale_t, expr_t=expr_t, use_cache=False)
        self.assertEqual(counting_mhr.calls, 3 * first_calls + 2)

    def test_mhr70_semantic_joint_mapping_and_camera_solve(self) -> None:
        from bodyscan_pipeline.fitting import (
            _pair_obs_kpts_to_skel_indices,
            _rotation_y_and_cv_flip,
            _solve_initial_view_camera,
        )
        from bodyscan_pipeline.segmentation import SegmentedView

        obs_idx, skel_idx = _pair_obs_kpts_to_skel_indices(70, 127)
        pair_dict = dict(zip(obs_idx.tolist(), skel_idx.tolist()))
        self.assertEqual(pair_dict[5], 75)   # left_shoulder
        self.assertEqual(pair_dict[6], 39)   # right_shoulder
        self.assertEqual(pair_dict[9], 2)    # left_hip
        self.assertEqual(pair_dict[10], 18)  # right_hip
        self.assertEqual(pair_dict[13], 4)   # left_ankle
        self.assertEqual(pair_dict[14], 20)  # right_ankle
        self.assertEqual(pair_dict[41], 42)  # right_wrist
        self.assertEqual(pair_dict[62], 78)  # left_wrist
        self.assertEqual(pair_dict[69], 110) # neck

        obs_128, skel_128 = _pair_obs_kpts_to_skel_indices(128, 127)
        self.assertEqual(len(obs_128), 127)
        self.assertTrue(np.array_equal(obs_128, skel_128))

        # Put valid anatomical coords at MHR-127 joints and poison MHR-127 indices [5..14, 41, 62, 69]
        joints_127 = np.full((127, 3), 9.0, dtype=np.float32)
        true_skel = {
            124: [0.03, 1.56, 0.08],
            122: [-0.03, 1.56, 0.08],
            75: [0.20, 1.40, 0.00],
            39: [-0.20, 1.40, 0.00],
            76: [0.28, 1.15, 0.02],
            40: [-0.28, 1.15, 0.02],
            2: [0.10, 0.90, 0.00],
            18: [-0.10, 0.90, 0.00],
            3: [0.11, 0.48, 0.01],
            19: [-0.11, 0.48, 0.01],
            4: [0.12, 0.06, 0.02],
            20: [-0.12, 0.06, 0.02],
            42: [-0.36, 0.90, 0.05],
            78: [0.36, 0.90, 0.05],
            110: [0.00, 1.44, 0.00],
        }
        for j_id, coord in true_skel.items():
            joints_127[j_id] = np.array(coord, dtype=np.float32)

        img_h, img_w = 240, 180
        focal = 260.0
        r_true = _rotation_y_and_cv_flip(0.0)
        t_true = np.array([0.18, 0.78, 3.10], dtype=np.float32)

        kpts_70 = np.full((70, 3), -1.0, dtype=np.float32)
        kpts_70[:, 2] = 0.0
        for mhr70_i, mhr127_j in pair_dict.items():
            p_cam = (joints_127[mhr127_j] @ r_true.T) + t_true
            u = focal * (p_cam[0] / p_cam[2]) + img_w / 2.0
            v = focal * (p_cam[1] / p_cam[2]) + img_h / 2.0
            kpts_70[mhr70_i] = [u, v, 0.95]

        empty_mask = np.zeros((img_h, img_w), dtype=bool)
        empty_mask[40:200, 50:130] = True
        view = SegmentedView(
            view_index=0,
            azimuth_deg=0.0,
            view_label="front",
            person_mask=empty_mask,
            body_mask=empty_mask,
            hair_mask=np.zeros_like(empty_mask),
            face_skin_mask=np.zeros_like(empty_mask),
        )
        verts_dummy = np.array([[0.0, 0.0, 0.0], [0.0, 1.70, 0.0]], dtype=np.float32)
        r_est, t_est, f_est = _solve_initial_view_camera(
            verts_dummy,
            joints_127,
            view,
            {"pred_keypoints_2d": kpts_70, "focal_length": focal},
        )
        self.assertEqual(f_est, focal)
        self.assertTrue(np.allclose(r_est, r_true, atol=1e-5))
        self.assertTrue(np.allclose(t_est, t_true, atol=1e-3), f"Expected {t_true}, got {t_est}")

    def test_pytorch3d_per_view_pose_and_delta_residual(self) -> None:
        import sys
        import types
        from bodyscan_pipeline.fitting import fit_mhr_multiview
        from bodyscan_pipeline.fusion import FusedSubjectParams, build_scaled_apose_mesh
        from bodyscan_pipeline.segmentation import SegmentedView

        sphere_v, sphere_f = _make_uv_sphere_mesh(n_lat=12, n_lon=18)

        class _PoseAwareFakeMHR(_FakeMHRModel):
            def __call__(self, shape_t: torch.Tensor, model_params: torch.Tensor, expr_t: torch.Tensor):
                verts_cm, skel_cm = super().__call__(shape_t, model_params, expr_t)
                # model_params[:, 6+27] tilts upper torso/arm vertices laterally (+X);
                # model_params[:, 6+37] tilts along +Z for side-view sensitivity.
                p_x = model_params[:, 33:34].unsqueeze(-1)
                p_z = model_params[:, 43:44].unsqueeze(-1)
                up_mask = torch.clamp(self.base_verts_m[:, 1:2].unsqueeze(0), min=0.0, max=1.0)
                dx_cm = 45.0 * p_x * up_mask
                dz_cm = 45.0 * p_z * up_mask
                shift_cm = torch.cat([dx_cm, torch.zeros_like(dx_cm), dz_cm], dim=-1)
                return verts_cm + shift_cm, skel_cm

        mhr_model = _PoseAwareFakeMHR(sphere_v)
        head_assets: Dict[str, torch.Tensor] = {
            "scale_mean": torch.zeros(68, dtype=torch.float32),
            "scale_comps": torch.zeros((28, 68), dtype=torch.float32),
            "faces": torch.tensor(sphere_f, dtype=torch.int64),
        }

        gt_fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45), dtype=torch.float32),
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.full(2, 0.5, dtype=np.float64),
            distances=np.zeros(2, dtype=np.float64),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )

        pose_true_0 = torch.zeros((1, 130), dtype=torch.float32)
        pose_true_0[0, 27] = 0.24
        pose_true_1 = torch.zeros((1, 130), dtype=torch.float32)
        pose_true_1[0, 37] = -0.22

        verts_gt0, _, _, _ = build_scaled_apose_mesh(mhr_model, head_assets, gt_fused, 170.0, apose_params=pose_true_0)
        verts_gt1, _, _, _ = build_scaled_apose_mesh(mhr_model, head_assets, gt_fused, 170.0, apose_params=pose_true_1)

        h_img, w_img = 96, 80
        focal = 180.0
        cam_t = np.array([0.0, 0.85, 2.8], dtype=np.float32)

        pose_pred_0 = np.zeros(130, dtype=np.float32)
        pose_pred_0[27] = 0.04  # Slightly off from +0.24
        pose_pred_1 = np.zeros(130, dtype=np.float32)
        pose_pred_1[37] = -0.02  # Slightly off from -0.22

        seg_views = []
        raw_results = []
        for idx, (label, az, v_gt, p_pred) in enumerate(
            (("front", 0.0, verts_gt0, pose_pred_0), ("right", 90.0, verts_gt1, pose_pred_1))
        ):
            mask = _project_hard_mask(v_gt, sphere_f, az, focal, cam_t, h_img, w_img)
            empty = np.zeros_like(mask)
            seg_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=mask,
                    body_mask=mask,
                    hair_mask=empty,
                    face_skin_mask=empty,
                )
            )
            raw_results.append(
                {
                    "image_name": f"{label}.jpg",
                    "output": {"focal_length": focal, "body_pose_params": p_pred},
                }
            )

        p3d_mod = types.ModuleType("pytorch3d")
        p3d_ren = types.ModuleType("pytorch3d.renderer")
        p3d_str = types.ModuleType("pytorch3d.structures")
        p3d_utl = types.ModuleType("pytorch3d.utils")

        class _StubMeshes:
            def __init__(self, verts, faces) -> None:
                self.verts = verts[0]
                self.faces = faces[0]

        class _StubRasterSettings:
            def __init__(self, image_size, blur_radius=0.0, faces_per_pixel=1) -> None:
                self.image_size = image_size

        class _StubRasterizer:
            def __init__(self, raster_settings) -> None:
                self.raster_settings = raster_settings

        class _StubSoftSilhouetteShader:
            pass

        class _StubMeshRenderer:
            def __init__(self, rasterizer, shader) -> None:
                self.h, self.w = rasterizer.raster_settings.image_size

            def __call__(self, meshes, cameras):
                r_mat, t_vec, k_mat = cameras
                v_cam = (meshes.verts @ r_mat[0].T) + t_vec[0].unsqueeze(0)
                z = torch.clamp(v_cam[:, 2], min=0.1)
                fx, fy = k_mat[0, 0, 0], k_mat[0, 1, 1]
                cx, cy = k_mat[0, 0, 2], k_mat[0, 1, 2]
                u = fx * (v_cam[:, 0] / z) + cx
                v = fy * (v_cam[:, 1] / z) + cy
                ys = torch.arange(self.h, dtype=v_cam.dtype, device=v_cam.device)
                xs = torch.arange(self.w, dtype=v_cam.dtype, device=v_cam.device)
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                d2 = (gx.unsqueeze(-1) - u.view(1, 1, -1)) ** 2 + (gy.unsqueeze(-1) - v.view(1, 1, -1)) ** 2
                density = torch.exp(-d2 / (3.0 * 3.0)).sum(dim=-1)
                alpha = 1.0 - torch.exp(-2.0 * density)
                rgba = torch.zeros((1, self.h, self.w, 4), dtype=v_cam.dtype, device=v_cam.device)
                rgba[0, :, :, 3] = alpha
                return rgba

        p3d_str.Meshes = _StubMeshes
        p3d_ren.RasterizationSettings = _StubRasterSettings
        p3d_ren.MeshRasterizer = _StubRasterizer
        p3d_ren.SoftSilhouetteShader = _StubSoftSilhouetteShader
        p3d_ren.MeshRenderer = _StubMeshRenderer
        p3d_utl.cameras_from_opencv_projection = lambda r_mat, t_vec, cam_k, im_size: (r_mat, t_vec, cam_k)

        init_shape = torch.zeros((1, 45), dtype=torch.float32)
        init_shape[0, 0] = -0.8
        init_fused = FusedSubjectParams(
            shape_params=init_shape,
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.full(2, 0.5, dtype=np.float64),
            distances=np.zeros(2, dtype=np.float64),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )

        saved = {k: sys.modules.get(k) for k in ("pytorch3d", "pytorch3d.renderer", "pytorch3d.structures", "pytorch3d.utils")}
        sys.modules["pytorch3d"] = p3d_mod
        sys.modules["pytorch3d.renderer"] = p3d_ren
        sys.modules["pytorch3d.structures"] = p3d_str
        sys.modules["pytorch3d.utils"] = p3d_utl
        try:
            _, _, debug_fit = fit_mhr_multiview(
                mhr_model=mhr_model,
                head_assets=head_assets,
                fused=init_fused,
                segmented_views=seg_views,
                raw_results=raw_results,
                height_cm=170.0,
                weight_kg=0.0,
                sex="male",
                steps=20,
                prefer_pytorch3d=True,
            )
        finally:
            for k, old in saved.items():
                if old is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old

        self.assertTrue(debug_fit["used_pytorch3d"])
        self.assertLess(debug_fit["loss_history"][-1], debug_fit["loss_history"][0])
        for key in ("apose_params", "final_cameras", "view_posed_verts", "view_poses"):
            self.assertIn(key, debug_fit)
        self.assertEqual(len(debug_fit["view_posed_verts"]), 2)
        self.assertGreater(float(debug_fit["view_poses"][0][0, 27]), 0.04)
        self.assertLess(float(debug_fit["view_poses"][1][0, 37]), -0.02)
        self.assertLessEqual(
            float(torch.linalg.norm(debug_fit["view_poses"][0].reshape(-1) - torch.from_numpy(pose_pred_0))),
            0.35 + 1e-4,
        )

    def test_solve_initial_view_camera_bounded_pnp_recovers_tilted_camera(self) -> None:
        import cv2
        import math
        from bodyscan_pipeline.fitting import (
            _rotation_y_and_cv_flip,
            _solve_initial_view_camera,
        )
        from bodyscan_pipeline.segmentation import SegmentedView

        verts_a, faces_a = _make_uv_sphere_mesh(n_lat=12, n_lon=16)
        verts_a = verts_a.copy()
        verts_a[:, 0] *= 0.24
        verts_a[:, 1] = verts_a[:, 1] * 0.78 + 0.85
        verts_a[:, 2] *= 0.16

        # Build rich non-planar 3D skeleton joints across head, shoulders, hips, knees, ankles, torso
        joints_a = np.zeros((128, 3), dtype=np.float32)
        template_pts = [
            (0.0, 1.56, 0.08),
            (-0.04, 1.58, 0.09),
            (0.04, 1.58, 0.09),
            (-0.18, 1.36, 0.02),
            (0.18, 1.36, -0.02),
            (-0.24, 1.12, 0.04),
            (0.24, 1.12, -0.03),
            (-0.12, 0.92, 0.03),
            (0.12, 0.92, -0.03),
            (-0.10, 0.52, 0.05),
            (0.10, 0.52, -0.04),
            (-0.09, 0.14, -0.02),
            (0.09, 0.14, 0.04),
            (0.0, 1.18, 0.10),
        ]
        for j in range(128):
            pt = template_pts[j % len(template_pts)]
            joints_a[j] = [
                pt[0] + 0.01 * math.sin(j),
                pt[1] + 0.01 * math.cos(j),
                pt[2] + 0.015 * math.sin(2 * j),
            ]

        r_base = _rotation_y_and_cv_flip(0.0)
        # Apply ~12 deg pitch + ~-8 deg roll relative to base canonical camera
        delta_rot, _ = cv2.Rodrigues(np.array([math.radians(12.0), math.radians(5.0), math.radians(-8.0)], dtype=np.float64))
        r_true = (delta_rot @ r_base.astype(np.float64)).astype(np.float32)
        t_true = np.array([0.04, 0.86, 2.75], dtype=np.float32)
        focal = 320.0
        img_h, img_w = 240, 180

        # Project 2D keypoints and silhouette using true tilted extrinsics
        j_cam = joints_a @ r_true.T + t_true[None, :]
        u_j = focal * (j_cam[:, 0] / j_cam[:, 2]) + img_w / 2.0
        v_j = focal * (j_cam[:, 1] / j_cam[:, 2]) + img_h / 2.0
        kpts_2d = np.stack([u_j, v_j, np.ones(len(joints_a), dtype=np.float32)], axis=1).astype(np.float32)

        v_cam = verts_a @ r_true.T + t_true[None, :]
        uv_v = np.stack(
            [
                focal * (v_cam[:, 0] / v_cam[:, 2]) + img_w / 2.0,
                focal * (v_cam[:, 1] / v_cam[:, 2]) + img_h / 2.0,
            ],
            axis=1,
        )
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for tri in np.round(uv_v[faces_a]).astype(np.int32):
            cv2.fillConvexPoly(mask, tri, 1)
        body_mask = mask > 0

        view = SegmentedView(
            view_index=0,
            azimuth_deg=0.0,
            view_label="front",
            person_mask=body_mask,
            body_mask=body_mask,
            hair_mask=np.zeros_like(body_mask),
            face_skin_mask=np.zeros_like(body_mask),
        )
        out = {
            "pred_keypoints_2d": kpts_2d,
            "pred_cam_t": t_true,
            "focal_length": np.array([focal], dtype=np.float32),
        }
        r_est, t_est, f_est = _solve_initial_view_camera(verts_a, joints_a, view, out)
        self.assertAlmostEqual(float(f_est), focal, delta=1e-3)
        cos_theta = float(np.clip((np.trace(r_est.T @ r_true) - 1.0) * 0.5, -1.0, 1.0))
        ang_err_deg = math.degrees(math.acos(cos_theta))
        self.assertLess(ang_err_deg, 3.0)
        self.assertLess(float(np.linalg.norm(t_est - t_true)), 0.08)

    def test_evaluate_view_posed_meshes_blends_native_pred_vertices_when_apose_fallback(self) -> None:
        from bodyscan_pipeline.fitting import _evaluate_view_posed_meshes, _rotation_y_and_cv_flip
        from bodyscan_pipeline.fusion import FusedSubjectParams

        torso_pts = []
        for y in np.linspace(0.20, 1.65, 8):
            for th in np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False):
                torso_pts.append([0.14 * math.cos(th), float(y), 0.10 * math.sin(th)])
        r_arm_apose = [[float(x), 1.32 - 0.10 * (x - 0.25), 0.0] for x in np.linspace(0.28, 0.65, 8)]
        l_arm_apose = [[-float(x), 1.32 - 0.10 * (x - 0.25), 0.0] for x in np.linspace(0.28, 0.65, 8)]
        apose_verts = np.asarray(torso_pts + r_arm_apose + l_arm_apose, dtype=np.float32)

        # Native side-view pred_vertices with arms hanging down at the sides + camera rigid offset & scale
        r_arm_hanging = [[0.20, float(1.32 - 0.55 * t), 0.04] for t in np.linspace(0.1, 1.0, 8)]
        l_arm_hanging = [[-0.20, float(1.32 - 0.55 * t), 0.04] for t in np.linspace(0.1, 1.0, 8)]
        side_canon = np.asarray(torso_pts + r_arm_hanging + l_arm_hanging, dtype=np.float64)
        r_view = _rotation_y_and_cv_flip(90.0).astype(np.float64)
        native_side_verts = ((side_canon * 0.92) @ r_view.T + np.array([[0.05, -0.40, 2.30]])).astype(np.float32)

        class _ConstAPoseMHR:
            def __call__(self, shape_t: torch.Tensor, model_params: torch.Tensor, expr_t: torch.Tensor):
                # Returns fixed A-pose regardless of pose params (100x scale -> cm)
                v = torch.from_numpy(apose_verts * 100.0).unsqueeze(0)
                j = torch.zeros((1, 127, 3), dtype=torch.float32)
                return v, j

        fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45), dtype=torch.float32),
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.array([0.5, 0.5], dtype=np.float32),
            distances=np.array([0.0, 0.0], dtype=np.float32),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )
        apose_p = torch.zeros((1, 204), dtype=torch.float32)
        view_poses = [apose_p.clone(), apose_p.clone()]
        raw_results = [
            {"azimuth_deg": 0.0, "output": {}},
            {"azimuth_deg": 90.0, "output": {"pred_vertices": native_side_verts}},
        ]

        out_verts = _evaluate_view_posed_meshes(
            mhr_model=_ConstAPoseMHR(),
            head_assets={},
            fused=fused,
            apose_params=apose_p,
            apose_scaled_verts=apose_verts,
            view_poses=view_poses,
            height_cm=170.0,
            raw_results=raw_results,
        )
        self.assertEqual(len(out_verts), 2)
        self.assertTrue(np.allclose(out_verts[0], apose_verts, atol=1e-5))
        n_torso = len(torso_pts)
        # Torso remains aligned while hanging arm bends cleanly down instead of staying in A-pose
        torso_err = float(np.mean(np.linalg.norm(out_verts[1][:n_torso] - apose_verts[:n_torso], axis=1)))
        self.assertLess(torso_err, 0.015)
        arm_tip_y = float(out_verts[1][n_torso + 7, 1])
        self.assertLess(arm_tip_y, 0.90)

    def test_align_and_blend_native_view_verts_preserves_legs_height_and_rejects_180_flip(self) -> None:
        from bodyscan_pipeline.fitting import _align_and_blend_native_view_verts, _rotation_y_and_cv_flip

        # Realistic human geometry: feet at y=0, legs [0..0.80], torso [0.75..1.45], head [1.45..1.72], A-pose arms
        verts_list = []
        is_non_arm = []
        # Legs + pelvis + torso + head
        for y in np.linspace(0.0, 1.72, 24):
            r = 0.08 if y < 0.80 else (0.14 if y < 1.45 else 0.09)
            for th in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                verts_list.append([r * math.cos(th), float(y), 0.75 * r * math.sin(th)])
                is_non_arm.append(True)
        # Raised A-pose arms
        for t in np.linspace(0.0, 1.0, 10):
            verts_list.append([float(0.24 + 0.36 * t), float(1.36 - 0.16 * t), 0.0])
            is_non_arm.append(False)
            verts_list.append([-float(0.24 + 0.36 * t), float(1.36 - 0.16 * t), 0.0])
            is_non_arm.append(False)

        apose_v = np.asarray(verts_list, dtype=np.float32)
        non_arm_mask = np.asarray(is_non_arm, dtype=bool)
        n_non_arm = int(np.sum(non_arm_mask))

        # Native mesh: arms hanging down 38 deg, noisy lower legs (-0.08m offset), 1.05x scale, rotated to 90 deg view
        canon_native = apose_v.copy().astype(np.float64)
        canon_native[:n_non_arm, 1] = np.where(
            canon_native[:n_non_arm, 1] < 0.70, canon_native[:n_non_arm, 1] - 0.08, canon_native[:n_non_arm, 1]
        )
        for i in range(n_non_arm, len(canon_native)):
            sign_x = 1.0 if canon_native[i, 0] > 0.0 else -1.0
            canon_native[i] = [sign_x * 0.22, 1.36 - 0.52 * ((i - n_non_arm) // 2) / 9.0, 0.03]

        r_side = _rotation_y_and_cv_flip(90.0).astype(np.float64)
        vp_side = ((canon_native * 1.05) @ r_side.T + np.array([[0.03, -0.25, 2.4]])).astype(np.float32)

        blended = _align_and_blend_native_view_verts(
            apose_v, vp_side, azimuth_deg=90.0, arm_vert_mask=non_arm_mask
        )
        # Feet grounded at y=0, total height preserved, torso and legs untouched!
        self.assertGreaterEqual(float(blended[:, 1].min()), -0.005)
        self.assertAlmostEqual(float(blended[:, 1].max() - blended[:, 1].min()), 1.72, delta=0.015)
        non_arm_err = float(np.max(np.linalg.norm(blended[non_arm_mask] - apose_v[non_arm_mask], axis=1)))
        self.assertLess(non_arm_err, 1e-5)
        self.assertLess(float(blended[n_non_arm + 18, 1]), 0.95)  # Hanging arm blended down

        # 180-deg reversed native mesh must be rejected (no corrupted blend)
        r_180 = np.diag([-1.0, 1.0, -1.0])
        vp_flipped = (canon_native @ r_180 @ r_side.T).astype(np.float32)
        blended_flip = _align_and_blend_native_view_verts(
            apose_v, vp_flipped, azimuth_deg=90.0, arm_vert_mask=non_arm_mask
        )
        self.assertTrue(np.allclose(blended_flip, apose_v, atol=1e-5))

    def test_fuse_multiview_params_rejects_scale_outliers_and_fuses_expr(self) -> None:
        from bodyscan_pipeline.fusion import fuse_multiview_params

        base_shape = np.zeros(45, dtype=np.float32)
        base_scale = np.ones(28, dtype=np.float32)
        base_expr = np.full(72, 0.25, dtype=np.float32)
        raw_results = [
            {"output": {"shape_params": base_shape.copy(), "scale_params": base_scale.copy(), "expr_params": base_expr.copy()}},
            {"output": {"shape_params": base_shape + 0.02, "scale_params": base_scale + 0.01, "expr_params": base_expr.copy()}},
            {"output": {"shape_params": base_shape - 0.02, "scale_params": base_scale - 0.01, "expr_params": base_expr.copy()}},
            # View 3 has IDENTICAL shape_params (shape_dist==0) but huge scale_params & expr_params corruption!
            {"output": {"shape_params": base_shape.copy(), "scale_params": base_scale + 4.0, "expr_params": np.full(72, 9.0, dtype=np.float32)}},
        ]
        fused = fuse_multiview_params(raw_results)
        self.assertEqual(fused.images_used, 3)
        self.assertEqual(fused.images_total, 4)
        self.assertAlmostEqual(float(fused.weights[3]), 0.0, delta=1e-6)
        self.assertIsNotNone(fused.expr_params)
        assert fused.expr_params is not None
        self.assertEqual(tuple(fused.expr_params.shape), (1, 72))
        self.assertTrue(torch.allclose(fused.expr_params, torch.full((1, 72), 0.25), atol=1e-4))
        self.assertTrue(torch.allclose(fused.scale_params, torch.ones((1, 28)), atol=0.05))

    def test_contour_fallback_optimization_and_bent_apose_and_occluded_side_pnp(self) -> None:
        from bodyscan_pipeline.fitting import _solve_initial_view_camera, fit_mhr_multiview
        from bodyscan_pipeline.fusion import FusedSubjectParams, build_scaled_apose_mesh, solve_straight_apose
        from bodyscan_pipeline.segmentation import SegmentedView

        # 1) Non-PyTorch3D contour fallback improves IoU over perturbed initialization
        sphere_v, sphere_f = _make_uv_sphere_mesh(n_lat=12, n_lon=18)

        class _ResponsiveMHR(_FakeMHRModel):
            def __call__(self, shape_t: torch.Tensor, model_params: torch.Tensor, expr_t: torch.Tensor):
                # Stretch width slightly by shape_t[:, 0] in a differentiable way
                v_base = (self.base_verts_m * 100.0).to(device=shape_t.device, dtype=torch.float32)
                stretch_x = torch.clamp(1.0 + 0.35 * shape_t[0, 0], min=0.4, max=2.0)
                v_mod = torch.stack([v_base[:, 0] * stretch_x, v_base[:, 1], v_base[:, 2]], dim=1)
                j = torch.zeros((1, 127, 3), dtype=torch.float32, device=shape_t.device)
                return v_mod.unsqueeze(0), j

        mhr = _ResponsiveMHR(sphere_v)
        head_assets = {
            "scale_mean": torch.zeros(68, dtype=torch.float32),
            "scale_comps": torch.zeros((28, 68), dtype=torch.float32),
            "faces": torch.tensor(sphere_f, dtype=torch.int64),
        }
        gt_fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45), dtype=torch.float32),
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.full(2, 0.5, dtype=np.float64),
            distances=np.zeros(2, dtype=np.float64),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )
        gt_fused.shape_params[0, 0] = 0.8
        gt_verts, _, _, _ = build_scaled_apose_mesh(mhr, head_assets, gt_fused, height_cm=170.0)
        h_img, w_img, focal = 110, 90, 200.0
        cam_t = np.array([0.08, 0.82, 2.7], dtype=np.float32)
        seg_views = []
        raw_results = []
        for idx, (lbl, az) in enumerate((("front", 0.0), ("right", 90.0))):
            m = _project_hard_mask(gt_verts, sphere_f, az, focal, cam_t, h_img, w_img)
            empty = np.zeros_like(m)
            seg_views.append(SegmentedView(idx, az, lbl, m, m, empty, empty))
            raw_results.append({"image_name": f"{lbl}.jpg", "output": {"focal_length": focal}})

        init_fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45), dtype=torch.float32),
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.full(2, 0.5, dtype=np.float64),
            distances=np.zeros(2, dtype=np.float64),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )
        init_fused.shape_params[0, 0] = -0.6  # perturbed narrow initial shape
        _, audit_no_opt, debug_no_opt = fit_mhr_multiview(
            mhr, head_assets, init_fused, seg_views, raw_results, 170.0, 0.0, "male", steps=14, prefer_pytorch3d=False
        )
        _, audit_fb, debug_fb = fit_mhr_multiview(
            mhr, head_assets, init_fused, seg_views, raw_results, 170.0, 0.0, "male", steps=14, prefer_pytorch3d=True
        )
        self.assertFalse(debug_fb["used_pytorch3d"])
        self.assertTrue(debug_fb["used_contour_fallback"])
        self.assertGreater(debug_fb["steps_run"], 0)
        self.assertGreater(audit_fb.mhr_silhouette_iou, audit_no_opt.mhr_silhouette_iou)

        # 2) Fast solve_straight_apose on bent 45-degree arms converges to < 3 deg in < 90 calls
        class _BentElbowMHR:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, shape_t: torch.Tensor, model_params: torch.Tensor, expr_t: torch.Tensor):
                self.calls += 1
                j = torch.zeros((1, 127, 3), dtype=torch.float32)
                p = model_params[0, 6:136]
                # Straighten right arm via idx=29, left arm via idx=39 (target ~ -0.7854 rad = -45 deg)
                j[0, 39] = torch.tensor([20.0, 140.0, 0.0])
                j[0, 40] = torch.tensor([45.0, 140.0, 0.0])
                ang_r = 0.7854 + float(p[29].item())
                j[0, 41] = j[0, 40] + torch.tensor([25.0 * math.cos(ang_r), -25.0 * math.sin(ang_r), 0.0])
                j[0, 75] = torch.tensor([-20.0, 140.0, 0.0])
                j[0, 76] = torch.tensor([-45.0, 140.0, 0.0])
                ang_l = 0.7854 - float(p[39].item())
                j[0, 77] = j[0, 76] + torch.tensor([-25.0 * math.cos(ang_l), -25.0 * math.sin(ang_l), 0.0])
                v = torch.tensor([[0.0, 0.0, 0.0], [0.0, 170.0, 0.0]], dtype=torch.float32).unsqueeze(0)
                return v, j

        bent_mhr = _BentElbowMHR()
        apose_sol = solve_straight_apose(bent_mhr, head_assets, gt_fused.shape_params, gt_fused.scale_params)
        self.assertLess(bent_mhr.calls, 90)
        self.assertAlmostEqual(float(apose_sol[0, 29].item()), -0.7854, delta=0.05)
        self.assertAlmostEqual(float(apose_sol[0, 39].item()), 0.7854, delta=0.05)

        # 3) Side-view PnP stays accurate even with occluded corrupted wrist keypoints (41, 62)
        joints_a = np.zeros((127, 3), dtype=np.float32)
        joints_a[124] = [-0.03, 1.62, 0.07]
        joints_a[122] = [0.03, 1.62, 0.07]
        joints_a[110] = [0.0, 1.46, 0.01]
        joints_a[75] = [-0.18, 1.38, 0.0]
        joints_a[39] = [0.18, 1.38, 0.0]
        joints_a[2] = [-0.10, 0.88, 0.0]
        joints_a[18] = [0.10, 0.88, 0.0]
        joints_a[42] = [0.35, 0.90, 0.0]
        joints_a[78] = [-0.35, 0.90, 0.0]
        from bodyscan_pipeline.config import MHR70_TO_MHR127_JOINT_MAP
        from bodyscan_pipeline.fitting import _rotation_y_and_cv_flip

        r_true = _rotation_y_and_cv_flip(90.0)
        t_true = np.array([0.05, 0.80, 2.75], dtype=np.float32)
        kpts_70 = np.zeros((70, 3), dtype=np.float32)
        for oi, sj in MHR70_TO_MHR127_JOINT_MAP:
            pt_cam = r_true @ joints_a[sj] + t_true
            u = focal * (pt_cam[0] / pt_cam[2]) + (w_img / 2.0)
            v = focal * (pt_cam[1] / pt_cam[2]) + (h_img / 2.0)
            kpts_70[oi] = [u, v, 0.9]
        # Corrupt occluded wrists heavily
        kpts_70[41] = [w_img - 2.0, 15.0, 0.9]
        kpts_70[62] = [2.0, h_img - 5.0, 0.9]
        sv_side = seg_views[1]
        _, t_side_est, _ = _solve_initial_view_camera(gt_verts, joints_a, sv_side, {
            "pred_keypoints_2d": kpts_70, "focal_length": np.array([focal], dtype=np.float32)
        })
        self.assertLess(float(np.linalg.norm(t_side_est - t_true)), 0.10)


if __name__ == "__main__":
    unittest.main()
