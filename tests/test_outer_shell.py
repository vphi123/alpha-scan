"""Unit tests for Stage 3 dual-layer clothed/outer body shell reconstruction."""

import dataclasses
import math
import sys
import unittest

import cv2
import numpy as np

from bodyscan_pipeline.config import INNER_BARRIER_M, NECK_BASE_JOINT, NECK_HEAD_JOINT
from bodyscan_pipeline.head_hair import HighResHeadOutput, reconstruct_highres_head_and_hair
from bodyscan_pipeline.segmentation import SegmentedView
from bodyscan_pipeline.texturing import render_view_depth_buffer


def _rotation_y_and_cv_flip(azimuth_deg: float) -> np.ndarray:
    """Return 3x3 rotation mapping canonical MHR (+Y up, +Z front) at azimuth_deg to OpenCV camera."""
    rad = math.radians(float(azimuth_deg))
    c, s = math.cos(rad), math.sin(rad)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    r_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    return (r_flip @ rot_y).astype(np.float32)


def _edge_degree_map(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Count how many triangles share each undirected edge."""
    counts: dict[tuple[int, int], int] = {}
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            counts[e] = counts.get(e, 0) + 1
    return counts


def _build_synthetic_mannequin(
    n_lon: int = 28,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a closed watertight humanoid mannequin and return (verts, faces, joints, ring_ys, ring_rads)."""
    ring_specs = [
        (0.02, 0.115),  # 0: feet
        (0.32, 0.125),  # 1: calves
        (0.62, 0.135),  # 2: thighs
        (0.86, 0.155),  # 3: hips
        (1.02, 0.145),  # 4: lower waist
        (1.16, 0.150),  # 5: chest
        (1.28, 0.155),  # 6: upper chest
        (1.34, 0.145),  # 7: shoulders base
        (1.38, 0.065),  # 8: NECK_BASE_JOINT collar ring
        (1.44, 0.060),  # 9: mid-neck
        (1.49, 0.078),  # 10: NECK_HEAD_JOINT / chin base
        (1.54, 0.105),  # 11: jaw
        (1.60, 0.110),  # 12: mid-face
        (1.65, 0.100),  # 13: brow
        (1.69, 0.072),  # 14: crown
    ]

    verts_list: list[list[float]] = [[0.0, 0.0, 0.0]]
    thetas = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    for y_val, rad in ring_specs:
        for th in thetas:
            verts_list.append([float(rad * np.cos(th)), float(y_val), float(rad * np.sin(th))])
    top_pole_idx = len(verts_list)
    verts_list.append([0.0, 1.72, 0.0])

    def ring_idx(r: int, k: int) -> int:
        return 1 + r * n_lon + (k % n_lon)

    faces_list: list[list[int]] = []
    for k in range(n_lon):
        faces_list.append([0, ring_idx(0, k), ring_idx(0, k + 1)])
    for r in range(len(ring_specs) - 1):
        for k in range(n_lon):
            b0 = ring_idx(r, k)
            b1 = ring_idx(r, k + 1)
            t0 = ring_idx(r + 1, k)
            t1 = ring_idx(r + 1, k + 1)
            faces_list.append([b0, t0, t1])
            faces_list.append([b0, t1, b1])
    last_r = len(ring_specs) - 1
    for k in range(n_lon):
        faces_list.append([top_pole_idx, ring_idx(last_r, k + 1), ring_idx(last_r, k)])

    verts = np.asarray(verts_list, dtype=np.float32)
    faces = np.asarray(faces_list, dtype=np.int64)

    joints = np.zeros((127, 3), dtype=np.float32)
    joints[0] = [0.0, 1.60, 0.11]
    joints[2] = [0.08, 0.86, 0.0]
    joints[18] = [-0.08, 0.86, 0.0]
    joints[39] = [-0.20, 1.34, 0.0]
    joints[75] = [0.20, 1.34, 0.0]
    joints[NECK_BASE_JOINT] = [0.0, 1.38, 0.0]
    joints[NECK_HEAD_JOINT] = [0.0, 1.49, 0.0]

    ring_ys = np.asarray([0.0] + [r[0] for r in ring_specs] + [1.72], dtype=np.float32)
    ring_rads = np.asarray([0.0] + [r[1] for r in ring_specs] + [0.0], dtype=np.float32)
    return verts, faces, joints, ring_ys, ring_rads


def _build_initial_head_output(
    verts: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
) -> HighResHeadOutput:
    """Build realistic HighResHeadOutput with face skin + hair volume above NECK_BASE_JOINT."""
    img_h, img_w = 240, 200
    px_per_m = 400.0
    cx = img_w / 2.0
    y_crown_m = 1.72
    y_crown_px = 30.0
    cy_skull = y_crown_px + (y_crown_m - 1.60) * px_per_m
    r_px = int(round(0.11 * px_per_m))
    y_neck_px = int(round(y_crown_px + (y_crown_m - 1.38) * px_per_m))

    views: list[SegmentedView] = []
    for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
        pmask = np.zeros((img_h, img_w), dtype=bool)
        hmask = np.zeros((img_h, img_w), dtype=bool)
        fmask = np.zeros((img_h, img_w), dtype=bool)
        cv2.circle(pmask.view(np.uint8), (int(cx), int(cy_skull)), r_px, 1, -1)
        cv2.rectangle(pmask.view(np.uint8), (int(cx - 24), int(cy_skull + 20)), (int(cx + 24), img_h - 10), 1, -1)
        cv2.ellipse(hmask.view(np.uint8), (int(cx), int(cy_skull - 24)), (r_px, 24), 0, 0, 360, 1, -1)
        if label == "front":
            cv2.ellipse(fmask.view(np.uint8), (int(cx), int(cy_skull + 4)), (r_px - 6, r_px - 8), 0, 0, 360, 1, -1)
        elif label in ("right", "left"):
            dir_s = 1.0 if label == "right" else -1.0
            sx0 = int(cx) if dir_s > 0 else int(cx - r_px - 2)
            sx1 = int(cx + r_px + 2) if dir_s > 0 else int(cx)
            cv2.rectangle(fmask.view(np.uint8), (max(0, sx0), int(cy_skull - 16)), (min(img_w - 1, sx1), y_neck_px - 4), 1, -1)
        pmask |= hmask
        fmask &= pmask
        hmask &= ~fmask
        bmask = pmask & (~hmask)
        views.append(
            SegmentedView(
                view_index=idx,
                azimuth_deg=az,
                view_label=label,
                person_mask=pmask,
                body_mask=bmask,
                hair_mask=hmask,
                face_skin_mask=fmask,
                nose_profile_curve=None,
                face_landmarks_ram=None,
                collar_hair_overlap=False,
                warnings=[],
            )
        )
    return reconstruct_highres_head_and_hair(verts, faces, joints, views, raw_results=[])


class TestOuterShell(unittest.TestCase):
    """Test metric depth affine alignment and dual-layer outer body shell invariants."""

    def test_align_metric_depth_map_recovers_exact_scale_and_shift(self) -> None:
        from bodyscan_pipeline.outer_shell import align_metric_depth_map

        rng = np.random.default_rng(42)
        h, w = 120, 96
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        valid_mask = ((xx - 48.0) / 24.0) ** 2 + ((yy - 60.0) / 42.0) ** 2 <= 1.0

        true_depth = np.full((h, w), -1.0, dtype=np.float32)
        rad_norm = np.sqrt(np.clip(((xx - 48.0) / 24.0) ** 2 + ((yy - 60.0) / 42.0) ** 2, 0.0, 1.0))
        true_depth[valid_mask] = 2.40 - 0.18 * np.sqrt(np.maximum(0.0, 1.0 - rad_norm[valid_mask] ** 2))

        raw_depth = np.full((h, w), np.nan, dtype=np.float32)
        noise = rng.normal(0.0, 0.0004, size=(h, w)).astype(np.float32)
        raw_depth[valid_mask] = 0.4 * true_depth[valid_mask] - 0.15 + noise[valid_mask]

        aligned = align_metric_depth_map(raw_depth, true_depth, valid_mask)
        self.assertEqual(aligned.shape, (h, w))
        self.assertTrue(np.isfinite(aligned[valid_mask]).all())
        mae = float(np.mean(np.abs(aligned[valid_mask] - true_depth[valid_mask])))
        self.assertLess(mae, 0.002)

        # Verify license compliance guard
        sys.modules["ultralytics"] = type(sys)("ultralytics")
        try:
            with self.assertRaises(RuntimeError):
                align_metric_depth_map(raw_depth, true_depth, valid_mask)
        finally:
            sys.modules.pop("ultralytics", None)

    def test_build_dual_layer_outer_shell_respects_inner_barrier_and_hull_cap(self) -> None:
        from bodyscan_pipeline.outer_shell import OuterShellOutput, build_dual_layer_outer_shell

        inner_verts, inner_faces, joints, ring_ys, ring_rads = _build_synthetic_mannequin(n_lon=28)
        inner_snapshot = inner_verts.copy()
        highres_head = _build_initial_head_output(inner_verts, inner_faces, joints)

        # Build synthetic ground-truth clothed shell by expanding torso/hips below collar cleanly
        gt_verts = highres_head.full_verts.copy()
        body_mask_v = ~highres_head.head_vert_mask
        neck_y = float(joints[NECK_BASE_JOINT, 1])

        # Collar vertices (shared with head faces) stay unmoved; torso below collar expands +2.0cm + chest ridge
        head_face_mask = np.any(highres_head.head_vert_mask[highres_head.full_faces], axis=1)
        head_side_v = np.unique(highres_head.full_faces[head_face_mask])
        collar_v = head_side_v[~highres_head.head_vert_mask[head_side_v]]
        is_collar = np.zeros(len(gt_verts), dtype=bool)
        is_collar[collar_v] = True

        movable = body_mask_v & (~is_collar) & (gt_verts[:, 1] < neck_y - 1e-4)
        vx = gt_verts[:, 0]
        vy = gt_verts[:, 1]
        vz = gt_verts[:, 2]
        rad_xz = np.maximum(np.hypot(vx, vz), 1e-6)
        ux = vx / rad_xz
        uz = vz / rad_xz

        # Smooth +2.0cm garment envelope over torso [0.75, 1.32] + +1.5cm front chest fold ridge
        torso_env = np.clip((vy - 0.65) / 0.15, 0.0, 1.0) * np.clip((1.35 - vy) / 0.08, 0.0, 1.0)
        cloth_offset = 0.020 * torso_env
        ridge_w = np.exp(-((vy - 1.15) / 0.08) ** 2) * np.exp(-((vx) / 0.06) ** 2) * np.clip(uz, 0.0, 1.0)
        cloth_offset += 0.015 * ridge_w

        gt_verts[movable, 0] += cloth_offset[movable] * ux[movable]
        gt_verts[movable, 2] += cloth_offset[movable] * uz[movable]

        # Render 4 cardinal pinhole views of clothed gt_verts + synthetic MoGe2 depth (with inward indent & spike stress test)
        img_h, img_w = 400, 220
        focal = 580.0
        cam_t = np.array([0.0, 0.86, 2.8], dtype=np.float32)

        segmented_views: list[SegmentedView] = []
        raw_results: list[dict] = []
        cameras: list[tuple[np.ndarray, np.ndarray, float]] = []

        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            r_v = _rotation_y_and_cv_flip(az)
            cameras.append((r_v, cam_t, focal))

            gt_cam = (gt_verts @ r_v.T) + cam_t.reshape(1, 3)
            zbuf_gt, _ = render_view_depth_buffer(
                gt_cam, highres_head.full_faces, focal, img_w, img_h, prefer_pytorch3d=False
            )
            person_mask = zbuf_gt > 0.0

            # Separate hair_mask above neck line using head/hair projection
            hair_cam = (highres_head.full_verts @ r_v.T) + cam_t.reshape(1, 3)
            hair_faces_only = highres_head.full_faces[np.all(highres_head.hair_vert_mask[highres_head.full_faces], axis=1)]
            zbuf_hair, _ = render_view_depth_buffer(
                hair_cam, hair_faces_only, focal, img_w, img_h, prefer_pytorch3d=False
            )
            hair_mask = (zbuf_hair > 0.0) & person_mask
            body_mask = person_mask & (~hair_mask)

            # Uncalibrated affine MoGe2 depth map: raw = 0.4 * true_z - 0.15
            raw_depth = np.full((img_h, img_w), np.nan, dtype=np.float32)
            valid_d = zbuf_gt > 0.0
            raw_depth[valid_d] = 0.4 * zbuf_gt[valid_d] - 0.15

            # Stress-test -INNER_BARRIER_M on Back view (push raw depth 1.2cm deeper inward at mid-back)
            if label == "back":
                cy_mid, cx_mid = int(img_h * 0.40), img_w // 2
                patch = np.zeros_like(valid_d)
                patch[cy_mid - 14 : cy_mid + 14, cx_mid - 12 : cx_mid + 12] = True
                patch &= valid_d
                raw_depth[patch] += 0.4 * 0.015  # deeper (+1.5cm farther from back camera)

            # Stress-test hull_cap on Right view (localized +5cm spike near torso edge)
            if label == "right":
                cy_mid = int(img_h * 0.40)
                patch_r = np.zeros_like(valid_d)
                patch_r[cy_mid - 10 : cy_mid + 10, :] = True
                patch_r &= valid_d
                raw_depth[patch_r] -= 0.4 * 0.04

            segmented_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=person_mask,
                    body_mask=body_mask,
                    hair_mask=hair_mask,
                    face_skin_mask=np.zeros_like(person_mask),
                    nose_profile_curve=None,
                    face_landmarks_ram=None,
                    collar_hair_overlap=False,
                    warnings=[],
                )
            )
            raw_results.append(
                {
                    "output": {
                        "pred_cam_t": cam_t.copy(),
                        "focal_length": focal,
                    },
                    "moge2_depth": raw_depth,
                }
            )

        out = build_dual_layer_outer_shell(
            inner_verts_a=inner_verts,
            inner_faces=inner_faces,
            joints_a=joints,
            highres_head=highres_head,
            segmented_views=segmented_views,
            raw_results=raw_results,
            prefer_pytorch3d=True,
        )

        self.assertIsInstance(out, OuterShellOutput)
        self.assertTrue(np.isfinite(out.outer_verts).all())
        self.assertTrue(np.array_equal(out.inner_mhr_verts, inner_snapshot))
        self.assertTrue(np.array_equal(out.inner_mhr_faces, inner_faces))

        # Watertight manifold check on combined outer shell
        edge_deg = _edge_degree_map(out.outer_faces)
        self.assertTrue(all(deg == 2 for deg in edge_deg.values()))

        # Head/hair/face masks preserved cleanly and mirror highres_head bit-exactly
        self.assertTrue(np.any(out.head_vert_mask))
        self.assertTrue(np.any(out.hair_vert_mask))
        self.assertTrue(np.any(out.face_skin_vert_mask))
        self.assertTrue(np.any(out.front_face_vert_mask))
        self.assertFalse(bool(np.any(out.hair_vert_mask & out.face_skin_vert_mask)))
        self.assertTrue(
            np.array_equal(
                out.outer_verts[out.head_vert_mask],
                highres_head.full_verts[highres_head.head_vert_mask],
            )
        )
        self.assertTrue(
            np.array_equal(
                out.outer_verts[out.hair_vert_mask],
                highres_head.full_verts[highres_head.hair_vert_mask],
            )
        )

        # Inner barrier check: outer torso vertices >= inner surface - INNER_BARRIER_M - 1e-4
        torso_chk = (~out.head_vert_mask) & (out.outer_verts[:, 1] >= 0.05) & (out.outer_verts[:, 1] <= neck_y - 1e-3)
        self.assertTrue(np.any(torso_chk))
        outer_torso = out.outer_verts[torso_chk]
        inner_r_at_y = np.interp(outer_torso[:, 1], ring_ys, ring_rads)
        outer_r = np.hypot(outer_torso[:, 0], outer_torso[:, 2])
        self.assertGreaterEqual(float(np.min(outer_r - inner_r_at_y)), -INNER_BARRIER_M - 1e-4)

        # Clothed expansion & chest ridge check: front chest expands well outside inner surface
        chest_ridge = torso_chk & (out.outer_verts[:, 1] > 1.05) & (out.outer_verts[:, 1] < 1.25) & (out.outer_verts[:, 2] > 0.10) & (np.abs(out.outer_verts[:, 0]) < 0.06)
        self.assertTrue(np.any(chest_ridge))
        chest_r_outer = np.hypot(out.outer_verts[chest_ridge, 0], out.outer_verts[chest_ridge, 2])
        chest_r_inner = np.interp(out.outer_verts[chest_ridge, 1], ring_ys, ring_rads)
        self.assertGreater(float(np.max(chest_r_outer - chest_r_inner)), 0.015)

        # Hull containment check: projected outer torso vertices stay inside dilated person_mask hull
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        torso_pts = out.outer_verts[~out.head_vert_mask]
        for sv, (r_v, t_v, f_v) in zip(segmented_views, cameras):
            dilated_hull = cv2.dilate(sv.person_mask.astype(np.uint8), dil_k) > 0
            pts_cam = (torso_pts @ r_v.T) + t_v.reshape(1, 3)
            u = np.round(f_v * (pts_cam[:, 0] / np.maximum(pts_cam[:, 2], 1e-4)) + img_w / 2.0).astype(np.int64)
            v = np.round(f_v * (pts_cam[:, 1] / np.maximum(pts_cam[:, 2], 1e-4)) + img_h / 2.0).astype(np.int64)
            in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            self.assertTrue(bool(np.all(in_bounds)))
            self.assertTrue(bool(np.all(dilated_hull[v[in_bounds], u[in_bounds]])))

        # Outer silhouette IoU > 0.80
        self.assertGreater(float(out.outer_silhouette_iou), 0.80)

        # S2 privacy: zero landmark fields on OuterShellOutput
        field_names = [f.name for f in dataclasses.fields(OuterShellOutput)]
        self.assertFalse(any("landmark" in name.lower() for name in field_names))

    def test_build_dual_layer_outer_shell_graceful_fallback_without_depth(self) -> None:
        from bodyscan_pipeline.outer_shell import OuterShellOutput, build_dual_layer_outer_shell

        inner_verts, inner_faces, joints, _, _ = _build_synthetic_mannequin(n_lon=20)
        highres_head = _build_initial_head_output(inner_verts, inner_faces, joints)

        # Fallback with zero segmented views
        out_empty = build_dual_layer_outer_shell(
            inner_verts_a=inner_verts,
            inner_faces=inner_faces,
            joints_a=joints,
            highres_head=highres_head,
            segmented_views=[],
            raw_results=[],
            prefer_pytorch3d=False,
        )
        self.assertIsInstance(out_empty, OuterShellOutput)
        self.assertTrue(np.isfinite(out_empty.outer_verts).all())
        self.assertTrue(np.array_equal(out_empty.inner_mhr_verts, inner_verts))
        self.assertEqual(float(out_empty.outer_silhouette_iou), 0.0)
        self.assertGreaterEqual(len(out_empty.warnings), 1)

        # Silhouette-only fallback with < 4 views and missing MoGe2 depth maps
        img_h, img_w, focal = 200, 120, 300.0
        cam_t = np.array([0.0, 0.86, 2.8], dtype=np.float32)
        sil_views = []
        sil_raw = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0))):
            r_v = _rotation_y_and_cv_flip(az)
            cam_pts = (highres_head.full_verts @ r_v.T) + cam_t.reshape(1, 3)
            zbuf, _ = render_view_depth_buffer(
                cam_pts, highres_head.full_faces, focal, img_w, img_h, prefer_pytorch3d=False
            )
            pmask = zbuf > 0.0
            sil_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=pmask,
                    body_mask=pmask,
                    hair_mask=np.zeros_like(pmask),
                    face_skin_mask=np.zeros_like(pmask),
                    nose_profile_curve=None,
                    face_landmarks_ram=None,
                    collar_hair_overlap=False,
                    warnings=[],
                )
            )
            sil_raw.append({"output": {"pred_cam_t": cam_t.copy(), "focal_length": focal}})

        out_sil = build_dual_layer_outer_shell(
            inner_verts_a=inner_verts,
            inner_faces=inner_faces,
            joints_a=joints,
            highres_head=highres_head,
            segmented_views=sil_views,
            raw_results=sil_raw,
            prefer_pytorch3d=False,
        )
        self.assertTrue(np.isfinite(out_sil.outer_verts).all())
        self.assertTrue(np.array_equal(out_sil.inner_mhr_verts, inner_verts))
        self.assertGreater(float(out_sil.outer_silhouette_iou), 0.85)
        self.assertTrue(any("MoGe2" in w for w in out_sil.warnings))
        self.assertTrue(any("< 4 cardinal views" in w for w in out_sil.warnings))

        sys.modules["nvdiffrast"] = type(sys)("nvdiffrast")
        try:
            with self.assertRaises(RuntimeError):
                build_dual_layer_outer_shell(
                    inner_verts_a=inner_verts,
                    inner_faces=inner_faces,
                    joints_a=joints,
                    highres_head=highres_head,
                    segmented_views=[],
                    raw_results=[],
                    prefer_pytorch3d=False,
                )
        finally:
            sys.modules.pop("nvdiffrast", None)

    def test_pytorch3d_oom_calls_cuda_empty_cache_and_falls_back(self) -> None:
        import types
        from unittest.mock import patch
        import torch
        from bodyscan_pipeline.outer_shell import build_dual_layer_outer_shell

        inner_verts, inner_faces, joints, _, _ = _build_synthetic_mannequin(n_lon=14)
        highres_head = _build_initial_head_output(inner_verts, inner_faces, joints)

        img_h, img_w, focal = 160, 120, 260.0
        cam_t = np.array([0.0, 0.86, 2.8], dtype=np.float32)
        r_v = _rotation_y_and_cv_flip(0.0)
        cam_pts = (highres_head.full_verts @ r_v.T) + cam_t.reshape(1, 3)
        zbuf, _ = render_view_depth_buffer(
            cam_pts, highres_head.full_faces, focal, img_w, img_h, prefer_pytorch3d=False
        )
        pmask = zbuf > 0.0
        views = [
            SegmentedView(
                view_index=0,
                azimuth_deg=0.0,
                view_label="front",
                person_mask=pmask,
                body_mask=pmask,
                hair_mask=np.zeros_like(pmask),
                face_skin_mask=np.zeros_like(pmask),
            )
        ]
        raw = [{"output": {"pred_cam_t": cam_t.copy(), "focal_length": focal}}]

        p3d_mod = types.ModuleType("pytorch3d")
        p3d_ren = types.ModuleType("pytorch3d.renderer")
        p3d_str = types.ModuleType("pytorch3d.structures")
        p3d_utl = types.ModuleType("pytorch3d.utils")

        class _StubMeshes:
            def __init__(self, verts, faces) -> None:
                self.verts = verts[0]

        class _StubRasterSettings:
            def __init__(self, image_size, blur_radius=0.0, faces_per_pixel=1) -> None:
                self.image_size = image_size
                self.faces_per_pixel = faces_per_pixel

        class _StubRasterizer:
            def __init__(self, raster_settings, cameras=None) -> None:
                self.raster_settings = raster_settings
                self.cameras = cameras

            def __call__(self, meshes):
                raise RuntimeError("stub rasterizer fallback")

        class _StubSoftSilhouetteShader:
            pass

        class _OOMMeshRenderer:
            def __init__(self, rasterizer, shader) -> None:
                pass

            def __call__(self, meshes, cameras):
                raise RuntimeError("CUDA out of memory")

        p3d_str.Meshes = _StubMeshes
        p3d_ren.RasterizationSettings = _StubRasterSettings
        p3d_ren.MeshRasterizer = _StubRasterizer
        p3d_ren.SoftSilhouetteShader = _StubSoftSilhouetteShader
        p3d_ren.MeshRenderer = _OOMMeshRenderer
        p3d_utl.cameras_from_opencv_projection = lambda r_mat, t_vec, cam_k, im_size: (r_mat, t_vec, cam_k)

        saved = {k: sys.modules.get(k) for k in ("pytorch3d", "pytorch3d.renderer", "pytorch3d.structures", "pytorch3d.utils")}
        sys.modules["pytorch3d"] = p3d_mod
        sys.modules["pytorch3d.renderer"] = p3d_ren
        sys.modules["pytorch3d.structures"] = p3d_str
        sys.modules["pytorch3d.utils"] = p3d_utl
        try:
            with patch.object(torch.cuda, "empty_cache") as mock_empty_cache:
                out = build_dual_layer_outer_shell(
                    inner_verts_a=inner_verts,
                    inner_faces=inner_faces,
                    joints_a=joints,
                    highres_head=highres_head,
                    segmented_views=views,
                    raw_results=raw,
                    prefer_pytorch3d=True,
                )
        finally:
            for k, old in saved.items():
                if old is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old

        mock_empty_cache.assert_called()
        self.assertTrue(np.isfinite(out.outer_verts).all())
        self.assertTrue(np.array_equal(out.inner_mhr_verts, inner_verts))
        self.assertTrue(any("PyTorch3D outer-shell optimization failed" in w for w in out.warnings))

    def test_pytorch3d_rebalanced_raster_path_runs_cleanly(self) -> None:
        import types
        import torch
        from bodyscan_pipeline.outer_shell import build_dual_layer_outer_shell

        inner_verts, inner_faces, joints, _, _ = _build_synthetic_mannequin(n_lon=14)
        highres_head = _build_initial_head_output(inner_verts, inner_faces, joints)

        img_h, img_w, focal = 96, 72, 180.0
        cam_t = np.array([0.0, 0.86, 2.8], dtype=np.float32)
        views = []
        raw = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            r_v = _rotation_y_and_cv_flip(az)
            cam_pts = (highres_head.full_verts @ r_v.T) + cam_t.reshape(1, 3)
            zbuf, _ = render_view_depth_buffer(
                cam_pts, highres_head.full_faces, focal, img_w, img_h, prefer_pytorch3d=False
            )
            pmask = zbuf > 0.0
            views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=pmask,
                    body_mask=pmask,
                    hair_mask=np.zeros_like(pmask),
                    face_skin_mask=np.zeros_like(pmask),
                )
            )
            raw.append({"output": {"pred_cam_t": cam_t.copy(), "focal_length": focal}})

        seen_fpp = []
        p3d_mod = types.ModuleType("pytorch3d")
        p3d_ren = types.ModuleType("pytorch3d.renderer")
        p3d_str = types.ModuleType("pytorch3d.structures")
        p3d_utl = types.ModuleType("pytorch3d.utils")

        class _StubMeshes:
            def __init__(self, verts, faces) -> None:
                self.verts = verts[0]

        class _StubRasterSettings:
            def __init__(self, image_size, blur_radius=0.0, faces_per_pixel=1) -> None:
                self.image_size = image_size
                seen_fpp.append(faces_per_pixel)

        class _StubRasterizer:
            def __init__(self, raster_settings, cameras=None) -> None:
                self.raster_settings = raster_settings
                self.cameras = cameras

            def __call__(self, meshes):
                raise RuntimeError("stub rasterizer fallback")

        class _StubSoftSilhouetteShader:
            pass

        class _StubMeshRenderer:
            def __init__(self, rasterizer, shader) -> None:
                self.h, self.w = rasterizer.raster_settings.image_size

            def __call__(self, meshes, cameras):
                r_mat, t_vec, k_mat = cameras
                v_cam = (meshes.verts[::4] @ r_mat[0].T) + t_vec[0].unsqueeze(0)
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

        saved = {k: sys.modules.get(k) for k in ("pytorch3d", "pytorch3d.renderer", "pytorch3d.structures", "pytorch3d.utils")}
        sys.modules["pytorch3d"] = p3d_mod
        sys.modules["pytorch3d.renderer"] = p3d_ren
        sys.modules["pytorch3d.structures"] = p3d_str
        sys.modules["pytorch3d.utils"] = p3d_utl
        try:
            out = build_dual_layer_outer_shell(
                inner_verts_a=inner_verts,
                inner_faces=inner_faces,
                joints_a=joints,
                highres_head=highres_head,
                segmented_views=views,
                raw_results=raw,
                prefer_pytorch3d=True,
            )
        finally:
            for k, old in saved.items():
                if old is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old

        self.assertIn(8, seen_fpp)
        self.assertNotIn(20, seen_fpp)
        self.assertGreater(float(out.outer_silhouette_iou), 0.80)
        self.assertFalse(any("PyTorch3D outer-shell optimization failed" in w for w in out.warnings))

    def test_outer_shell_uses_fitted_cameras_and_posed_verts_with_collar_and_cliff_clamp(self) -> None:
        import math
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.outer_shell import build_dual_layer_outer_shell

        # Build dense watertight body mesh aligned to _build_initial_head_output coordinates (feet=0, crown=1.72)
        n_lat, n_lon = 56, 32
        pts = [[0.0, 1.72, 0.0]]
        for i in range(1, n_lat):
            phi = math.pi * i / n_lat
            y = 0.86 * (1.0 + math.cos(phi))
            yn = y / 1.72
            if yn >= 0.84:
                rx, rz = 0.11, 0.11
            elif yn >= 0.78:
                rx, rz = 0.06, 0.06
            elif yn >= 0.45:
                rx, rz = 0.18, 0.13
            else:
                rx, rz = 0.11, 0.09
            for j in range(n_lon):
                th = 2.0 * math.pi * j / n_lon
                pts.append([rx * math.cos(th), y, rz * math.sin(th)])
        bot = len(pts)
        pts.append([0.0, 0.0, 0.0])
        faces = []
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([0, 1 + jn, 1 + j])
        for i in range(n_lat - 2):
            r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                faces.append([r0 + j, r0 + jn, r1 + jn])
                faces.append([r0 + j, r1 + jn, r1 + j])
        last_r = 1 + (n_lat - 2) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([bot, last_r + j, last_r + jn])

        inner_verts = np.asarray(pts, dtype=np.float32)
        inner_faces = np.asarray(faces, dtype=np.int64)
        joints = np.zeros((128, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.38, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.50, 0.0]

        head_out = _build_initial_head_output(inner_verts, inner_faces, joints)

        img_h, img_w = 180, 140
        focal = 360.0
        cam_t = np.array([0.0, 0.86, 2.70], dtype=np.float32)
        r_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

        fitted_cameras = []
        fitted_view_verts = []
        seg_views = []
        raw_results = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            rad = math.radians(az)
            c, s = math.cos(rad), math.sin(rad)
            rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
            r_cam = (r_flip @ rot_y).astype(np.float32)
            # Simulate slight view pose shift (subtle lean) in fitted_view_verts
            posed_v = inner_verts.copy()
            posed_v[:, 0] += 0.015 * np.sin(math.radians(az + 30.0))
            fitted_cameras.append((r_cam, cam_t.copy(), focal))
            fitted_view_verts.append(posed_v)

            v_cam = posed_v @ r_cam.T + cam_t[None, :]
            z = np.maximum(v_cam[:, 2], 0.1)
            u = focal * (v_cam[:, 0] / z) + img_w / 2.0
            v = focal * (v_cam[:, 1] / z) + img_h / 2.0
            uv = np.stack([u, v], axis=1)
            # Target silhouette with a loose sleeve bulge (+4 px) that should stay clamped <= 0.060 m
            mask_u8 = np.zeros((img_h, img_w), dtype=np.uint8)
            for tri in np.round(uv[inner_faces]).astype(np.int32):
                cv2.fillConvexPoly(mask_u8, tri, 1)
            mask_dil = cv2.dilate(mask_u8, np.ones((7, 7), np.uint8), iterations=1) > 0
            seg_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=mask_dil,
                    body_mask=mask_dil,
                    hair_mask=np.zeros_like(mask_dil),
                    face_skin_mask=np.zeros_like(mask_dil),
                )
            )
            # Raw depth map with extreme cliff spike in upper corner to test cliff guard
            raw_depth = np.full((img_h, img_w), 5.0, dtype=np.float32)
            raw_depth[mask_u8 > 0] = 2.55
            raw_results.append({"azimuth_deg": az, "focal_length": focal, "moge2_depth": raw_depth})

        out = build_dual_layer_outer_shell(
            inner_verts_a=inner_verts,
            inner_faces=inner_faces,
            joints_a=joints,
            highres_head=head_out,
            segmented_views=seg_views,
            raw_results=raw_results,
            prefer_pytorch3d=False,
            fitted_cameras=fitted_cameras,
            fitted_view_verts=fitted_view_verts,
        )
        from scipy.spatial import cKDTree

        tree_inner = cKDTree(inner_verts.astype(np.float64))
        body_v = out.outer_verts[~out.head_vert_mask].astype(np.float64)
        dists_to_inner, _ = tree_inner.query(body_v, workers=1)
        self.assertLessEqual(float(np.max(dists_to_inner)), 0.060 + 0.015 + 1e-3)
        self.assertGreater(float(out.outer_silhouette_iou), 0.75)

    def test_extremity_displacement_caps_and_edge_stretch_guard(self) -> None:
        from bodyscan_pipeline.outer_shell import (
            _compute_extremity_limits,
            _enforce_edge_stretch_guard,
        )

        verts, faces, joints, _, _ = _build_synthetic_mannequin()
        # Append extended right arm + wrist + hand vertices and corresponding joints (40=elbow, 41=wrist, 8=ankle)
        extra_v = np.array(
            [
                [0.57, 1.00, 0.0],   # wrist
                [0.68, 1.00, 0.0],   # finger tip
                [0.09, 0.10, 0.0],   # ankle
                [0.09, 0.01, 0.09],  # toe tip
            ],
            dtype=np.float32,
        )
        all_v = np.vstack([verts, extra_v])
        head_mask = all_v[:, 1] >= 1.48

        j_full = np.zeros((127, 3), dtype=np.float32)
        j_full[: len(joints)] = joints
        j_full[40] = [0.32, 1.20, 0.0]
        j_full[41] = [0.58, 1.00, 0.0]
        j_full[8] = [0.09, 0.10, 0.0]

        caps, smooth_boost, hand_foot_mask, wrist_ankle_mask = _compute_extremity_limits(
            all_v, joints_a=j_full, head_vert_mask=head_mask
        )
        idx_wrist, idx_finger, idx_ankle, idx_toe = range(len(verts), len(all_v))
        self.assertTrue(bool(hand_foot_mask[idx_finger]))
        self.assertTrue(bool(hand_foot_mask[idx_toe]))
        self.assertLessEqual(float(caps[idx_finger]), 0.004 + 1e-6)
        self.assertLessEqual(float(caps[idx_toe]), 0.004 + 1e-6)
        self.assertTrue(bool(wrist_ankle_mask[idx_wrist]))
        self.assertTrue(bool(wrist_ankle_mask[idx_ankle]))
        self.assertLessEqual(float(caps[idx_wrist]), 0.010 + 1e-6)
        self.assertLessEqual(float(caps[idx_ankle]), 0.010 + 1e-6)
        self.assertGreaterEqual(float(smooth_boost[idx_finger]), 4.0)

        # Edge stretch guard clamps spiky fan displacement to <= 1.35 * orig_edge_len
        fan_v = [[0.0, 0.0, 0.0]]
        for k in range(6):
            th = 2.0 * np.pi * k / 6.0
            fan_v.append([0.008 * np.cos(th), 0.008 * np.sin(th), 0.0])
        fan_v_arr = np.asarray(fan_v, dtype=np.float32)
        fan_f = np.asarray([[0, 1 + k, 1 + ((k + 1) % 6)] for k in range(6)], dtype=np.int64)
        fan_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(fan_v_arr), 1))
        d_spike = np.zeros(len(fan_v_arr), dtype=np.float32)
        d_spike[0] = 0.050

        d_clamped = _enforce_edge_stretch_guard(fan_v_arr, fan_f, fan_n, d_spike, max_ratio=1.35)
        out_v = fan_v_arr + d_clamped[:, None] * fan_n
        edges = np.vstack([fan_f[:, [0, 1]], fan_f[:, [1, 2]], fan_f[:, [2, 0]]])
        orig_len = np.linalg.norm(fan_v_arr[edges[:, 0]] - fan_v_arr[edges[:, 1]], axis=1)
        new_len = np.linalg.norm(out_v[edges[:, 0]] - out_v[edges[:, 1]], axis=1)
        self.assertLessEqual(float(np.max(new_len / np.maximum(orig_len, 1e-7))), 1.35 + 1e-5)

    def test_build_dual_layer_outer_shell_populates_undisplaced_verts_and_damps_extremities(self) -> None:
        from bodyscan_pipeline.outer_shell import build_dual_layer_outer_shell

        inner_verts, inner_faces, joints, ring_ys, ring_rads = _build_synthetic_mannequin()
        # Mark ankle joint on lower ring so foot verts (y <= 0.07) fall inside extremity damper
        joints[8] = [0.09, 0.10, 0.0]
        joints[14] = [-0.09, 0.10, 0.0]
        head_out = _build_initial_head_output(inner_verts, inner_faces, joints)

        img_h, img_w = 180, 120
        focal = 360.0
        cam_t = np.array([0.0, 0.86, 2.70], dtype=np.float32)
        seg_views = []
        raw_results = []
        ys = np.linspace(0.0, 1.72, 120)
        ys_u = ring_ys
        rs_u = ring_rads
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            r_cam = _rotation_y_and_cv_flip(az)
            mask = np.zeros((img_h, img_w), dtype=bool)
            for y in ys:
                r = float(np.interp(y, ys_u, rs_u)) + 0.050
                p0 = r_cam @ np.array([-r, y, 0.0], dtype=np.float32) + cam_t
                p1 = r_cam @ np.array([r, y, 0.0], dtype=np.float32) + cam_t
                u0 = int(round(focal * (p0[0] / p0[2]) + img_w / 2.0))
                u1 = int(round(focal * (p1[0] / p1[2]) + img_w / 2.0))
                v_px = int(round(focal * (p0[1] / p0[2]) + img_h / 2.0))
                if 0 <= v_px < img_h:
                    lo, hi = max(0, min(u0, u1)), min(img_w - 1, max(u0, u1))
                    mask[v_px, lo : hi + 1] = True
            seg_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=mask,
                    body_mask=mask,
                    hair_mask=np.zeros_like(mask),
                    face_skin_mask=np.zeros_like(mask),
                )
            )
            raw_results.append({"azimuth_deg": az, "focal_length": focal})

        out = build_dual_layer_outer_shell(
            inner_verts_a=inner_verts,
            inner_faces=inner_faces,
            joints_a=joints,
            highres_head=head_out,
            segmented_views=seg_views,
            raw_results=raw_results,
            prefer_pytorch3d=False,
        )
        self.assertIsNotNone(out.undisplaced_verts)
        self.assertEqual(out.undisplaced_verts.shape, out.outer_verts.shape)
        foot_mask = out.undisplaced_verts[:, 1] <= 0.05
        self.assertTrue(np.any(foot_mask))
        foot_disp = np.linalg.norm(out.outer_verts[foot_mask] - out.undisplaced_verts[foot_mask], axis=1)
        self.assertLessEqual(float(np.max(foot_disp)), 0.004 + 1e-4)

        edges = np.vstack([out.outer_faces[:, [0, 1]], out.outer_faces[:, [1, 2]], out.outer_faces[:, [2, 0]]])
        body_edge = (~out.head_vert_mask[edges[:, 0]]) & (~out.head_vert_mask[edges[:, 1]])
        orig_len = np.linalg.norm(
            out.undisplaced_verts[edges[body_edge, 0]] - out.undisplaced_verts[edges[body_edge, 1]], axis=1
        )
        new_len = np.linalg.norm(
            out.outer_verts[edges[body_edge, 0]] - out.outer_verts[edges[body_edge, 1]], axis=1
        )
        valid_e = orig_len > 1e-4
        self.assertLessEqual(float(np.max(new_len[valid_e] / orig_len[valid_e])), 1.35 + 1e-3)

    def test_compute_extremity_limits_clamps_mhr127_left_wrist_and_high_ankles(self) -> None:
        from bodyscan_pipeline.config import (
            MHR127_L_ANKLE,
            MHR127_L_ELBOW,
            MHR127_L_FOREARM_END,
            MHR127_L_WRIST,
            MHR127_R_ANKLE,
            MHR127_R_ELBOW,
            MHR127_R_FOREARM_END,
            MHR127_R_WRIST,
        )
        from bodyscan_pipeline.outer_shell import _compute_extremity_limits

        inner_verts, _, _, _, _ = _build_synthetic_mannequin()
        joints = np.zeros((127, 3), dtype=np.float32)
        joints[MHR127_R_ELBOW] = [-0.28, 1.15, 0.0]
        joints[MHR127_R_FOREARM_END] = [-0.46, 0.96, 0.0]
        joints[MHR127_R_WRIST] = [-0.49, 0.93, 0.0]
        joints[MHR127_L_ELBOW] = [0.28, 1.15, 0.0]
        joints[MHR127_L_FOREARM_END] = [0.46, 0.96, 0.0]
        joints[MHR127_L_WRIST] = [0.49, 0.93, 0.0]
        # Realistic human ankles at ~12% height (well above legacy 9% floor)
        joints[MHR127_L_ANKLE] = [0.10, 0.21, 0.0]
        joints[MHR127_R_ANKLE] = [-0.10, 0.21, 0.0]

        probe_verts = np.vstack(
            [
                inner_verts,
                np.array(
                    [
                        [0.54, 0.89, 0.0],    # left hand tip past wrist
                        [-0.54, 0.89, 0.0],   # right hand tip past wrist
                        [0.10, 0.20, 0.03],   # left ankle above 0.09*H
                        [-0.10, 0.20, 0.03],  # right ankle above 0.09*H
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        limits, _, _, _ = _compute_extremity_limits(probe_verts, joints)
        l_hand_lim, r_hand_lim, l_ank_lim, r_ank_lim = limits[-4:]
        self.assertLessEqual(float(l_hand_lim), 0.005)
        self.assertLessEqual(float(r_hand_lim), 0.005)
        self.assertLessEqual(float(l_ank_lim), 0.012)
        self.assertLessEqual(float(r_ank_lim), 0.012)

    def test_enforce_edge_stretch_guard_limits_1ring_displacement_cliff(self) -> None:
        from bodyscan_pipeline.outer_shell import _enforce_edge_stretch_guard

        # Center vertex has 18mm perpendicular spike while neighbors have 0mm (sqrt(20^2+18^2)/20 = 1.345 < 1.35)
        angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        ring = np.stack([0.020 * np.cos(angles), 0.020 * np.sin(angles), np.zeros(6)], axis=1)
        verts = np.vstack([np.array([[0.0, 0.0, 0.0]], dtype=np.float32), ring.astype(np.float32)])
        faces = np.asarray([[0, 1 + k, 1 + ((k + 1) % 6)] for k in range(6)], dtype=np.int64)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(verts), 1))
        d_spike = np.zeros(len(verts), dtype=np.float32)
        d_spike[0] = 0.018

        d_clamped = _enforce_edge_stretch_guard(verts, faces, normals, d_spike, max_ratio=1.35, max_grad=0.65)
        # 1-ring displacement gradient |d_0 - d_nbr| must be bounded by max_grad * orig_len (0.65 * 0.020 = 0.013)
        self.assertLessEqual(float(d_clamped[0] - np.min(d_clamped[1:])), 0.65 * 0.020 + 1e-4)

    def test_prepare_view_payloads_softly_downweights_mask_border_and_depth_step(self) -> None:
        from bodyscan_pipeline.outer_shell import _compute_oriented_outward_normals, _prepare_view_payloads

        inner_verts, inner_faces, joints, _, _ = _build_synthetic_mannequin()
        h, w, focal = 180, 120, 360.0
        cam_t = np.array([0.0, 0.86, 2.70], dtype=np.float32)
        r_cam = _rotation_y_and_cv_flip(0.0)
        pts_cam = (inner_verts @ r_cam.T) + cam_t.reshape(1, 3)
        u_p = focal * (pts_cam[:, 0] / pts_cam[:, 2]) + w / 2.0
        v_p = focal * (pts_cam[:, 1] / pts_cam[:, 2]) + h / 2.0
        u0, u1 = max(2, int(np.floor(u_p.min()))), min(w - 3, int(np.ceil(u_p.max())))
        v0, v1 = max(2, int(np.floor(v_p.min()))), min(h - 3, int(np.ceil(v_p.max())))
        mask = np.zeros((h, w), dtype=bool)
        mask[v0 : v1 + 1, u0 : u1 + 1] = True
        cx_mid = u0 + max(4, int(0.70 * (u1 - u0)))
        # Match realistic convex depth across torso so scale~1.0, with a sharp 0.50m cliff at cx_mid
        xs_grid = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
        cx_torso = 0.5 * float(u0 + u1)
        half_span = max(2.0, 0.5 * float(u1 - u0))
        depth = (2.55 + 0.15 * np.square((xs_grid - cx_torso) / half_span)).astype(np.float32)
        cy = (v0 + v1) // 2
        depth[max(0, cy - 8) : min(h, cy + 8), cx_mid : min(w, cx_mid + 4)] += 0.40

        sv = SegmentedView(
            view_index=0,
            azimuth_deg=0.0,
            view_label="front",
            person_mask=mask,
            body_mask=mask,
            hair_mask=np.zeros_like(mask),
            face_skin_mask=np.zeros_like(mask),
        )
        v_norms = _compute_oriented_outward_normals(inner_verts, inner_faces)
        mov_idx = np.arange(len(inner_verts), dtype=np.int64)
        payloads, _, _, _, _, has_depth = _prepare_view_payloads(
            inner_verts_a=inner_verts,
            inner_faces=inner_faces,
            joints_a=joints,
            base_verts=inner_verts,
            outer_faces=inner_faces,
            v_normals=v_norms,
            mov_idx=mov_idx,
            segmented_views=[sv],
            raw_results=[{"azimuth_deg": 0.0, "focal_length": focal, "depth_map": depth}],
            prefer_pytorch3d=False,
            fitted_cameras=[(r_cam, cam_t, focal)],
        )
        self.assertTrue(has_depth)
        dconf = payloads[0]["aligned_depth_valid"]
        self.assertIsNotNone(dconf)
        assert dconf is not None
        self.assertLess(float(dconf[cy, u0]), 0.15)
        self.assertLess(float(np.min(dconf[cy, cx_mid - 1 : cx_mid + 2])), 0.15)
        self.assertGreater(float(dconf[cy, u0 + 8 if u0 + 8 < cx_mid - 2 else (u0 + cx_mid) // 2]), 0.30)


if __name__ == "__main__":
    unittest.main()
