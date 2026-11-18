"""Unit tests for multi-view fusion, geometric slicing, loft profile, and JSON schema."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from bodyscan_pipeline.config import LOFT_PCT_MAX, LOFT_PCT_MIN, SCHEMA_VERSION
from bodyscan_pipeline.exporter import build_body_scan_json
from bodyscan_pipeline.fusion import FusedSubjectParams, fuse_multiview_params
from bodyscan_pipeline.measurements import (
    SimpleTriMesh,
    compute_loft_profile,
    extract_16_measurements,
    measure_limb,
    slice_loop_2d,
    validate_sanity,
)


def _make_cylinder_mesh(
    radius: float,
    height: float,
    sections: int = 32,
    axis: str = "z",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a closed cylinder mesh in pure NumPy along 'y' or 'z'."""
    cx, cy, cz = center
    half_h = 0.5 * height
    angles = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    c_arr, s_arr = np.cos(angles), np.sin(angles)
    pts = []
    if axis == "z":
        for z in (-half_h, half_h):
            for k in range(sections):
                pts.append([cx + radius * c_arr[k], cy + radius * s_arr[k], cz + z])
        pts.append([cx, cy, cz - half_h])
        pts.append([cx, cy, cz + half_h])
    else:
        for y in (-half_h, half_h):
            for k in range(sections):
                pts.append([cx + radius * c_arr[k], cy + y, cz + radius * s_arr[k]])
        pts.append([cx, cy - half_h, cz])
        pts.append([cx, cy + half_h, cz])
    bot_c = 2 * sections
    top_c = 2 * sections + 1
    faces = []
    for k in range(sections):
        kn = (k + 1) % sections
        faces.append([k, kn, sections + kn])
        faces.append([k, sections + kn, sections + k])
        faces.append([bot_c, kn, k])
        faces.append([top_c, sections + k, sections + kn])
    return np.asarray(pts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _make_box_mesh(
    extents: tuple[float, float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SimpleTriMesh:
    """Build a closed 12-triangle axis-aligned box in pure NumPy."""
    hx, hy, hz = 0.5 * extents[0], 0.5 * extents[1], 0.5 * extents[2]
    cx, cy, cz = center
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return SimpleTriMesh(vertices=verts, faces=faces)


def _concat_simple_meshes(parts: list[tuple[np.ndarray, np.ndarray]]) -> SimpleTriMesh:
    verts_list = []
    faces_list = []
    offset = 0
    for v, f in parts:
        verts_list.append(v)
        faces_list.append(f + offset)
        offset += len(v)
    return SimpleTriMesh(vertices=np.vstack(verts_list), faces=np.vstack(faces_list))


class TestBodyScanPipeline(unittest.TestCase):
    """Verify fusion, slicing accuracy, loft generation, and schema compliance."""

    def test_fuse_multiview_rejects_outlier(self) -> None:
        base_shape = np.zeros(45, dtype=np.float32)
        base_scale = np.ones(28, dtype=np.float32)
        raw = [
            {"output": {"shape_params": base_shape + 0.05, "scale_params": base_scale}},
            {"output": {"shape_params": base_shape + 4.50, "scale_params": base_scale * 9.0}},  # Outlier
            {"output": {"shape_params": base_shape - 0.02, "scale_params": base_scale}},
            {"output": {"shape_params": base_shape + 0.01, "scale_params": base_scale}},
        ]
        fused = fuse_multiview_params(raw)
        self.assertEqual(fused.images_used, 3)
        self.assertEqual(fused.images_total, 4)
        self.assertEqual(fused.weights[1], 0.0)
        self.assertTrue(np.allclose(fused.scale_params.numpy(), 1.0, atol=1e-5))

    def test_slice_loop_2d_cylinder_isolation(self) -> None:
        """Measure target cylinder circumference accurately even without trimesh installed."""
        radius_main = 0.08  # 8 cm -> expected circumference = 2 * pi * 8 = 50.265 cm
        cyl_main = _make_cylinder_mesh(radius=radius_main, height=0.5, sections=64, axis="z", center=(0.0, 0.0, 0.0))
        cyl_other = _make_cylinder_mesh(radius=0.05, height=0.5, sections=32, axis="z", center=(0.30, 0.0, 0.0))
        combined = _concat_simple_meshes([cyl_main, cyl_other])

        with patch("bodyscan_pipeline.measurements.trimesh", None):
            pts, circ_cm = slice_loop_2d(
                combined,
                origin=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                pick="closest",
            )
        self.assertIsNotNone(pts)
        expected_cm = 2.0 * math.pi * radius_main * 100.0
        self.assertAlmostEqual(circ_cm, expected_cm, delta=0.5)

    def test_loft_profile_count_and_dimensions(self) -> None:
        box = _make_box_mesh(extents=(0.32, 1.80, 0.22), center=(0.0, 0.90, 0.0))
        with patch("bodyscan_pipeline.measurements.trimesh", None):
            loft = compute_loft_profile(box, height_m=1.80)
        self.assertEqual(len(loft), LOFT_PCT_MAX - LOFT_PCT_MIN + 1)
        self.assertEqual(loft[0][0], 0.30)
        self.assertEqual(loft[-1][0], 0.86)
        self.assertAlmostEqual(loft[10][1], 0.32, places=3)
        self.assertAlmostEqual(loft[10][2], 0.22, places=3)

    def test_json_schema_and_sanity_checks(self) -> None:
        fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45)),
            scale_params=torch.ones((1, 28)),
            weights=np.array([0.33, 0.33, 0.34]),
            distances=np.array([0.1, 0.1, 0.1]),
            images_used=3,
            images_total=3,
            mean_confidence=0.333,
        )
        circs = {"neck": 32.0, "chest": 88.0, "waist": 68.0, "hip": 92.0}
        payload = build_body_scan_json(
            person_id="p01",
            tier="alpha_scan_plus",
            sex="female",
            height_cm=160.0,
            weight_kg=50.0,
            circumferences_cm=circs,
            loft=[[0.30, 0.25, 0.18]],
            fused=fused,
        )
        self.assertEqual(payload["schema"], SCHEMA_VERSION)
        self.assertEqual(payload["mesh"]["units"], "m")
        self.assertEqual(payload["mesh"]["pose"], "A")
        self.assertEqual(validate_sanity(circs, "female"), [])
        self.assertEqual(len(validate_sanity({"neck": 32.0, "chest": 105.0, "hip": 90.0}, "female")), 1)

    def test_measure_neck_perpendicular_filters_tiny_noise_fragments(self) -> None:
        from unittest.mock import patch
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.measurements import measure_neck_perpendicular

        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.38, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.50, 0.0]
        dummy_pts = np.array([[0.05, 0.0], [-0.05, 0.0], [0.0, 0.05]], dtype=np.float32)

        mixed_circs = iter([6.5, 34.5, 35.8, 36.2] + [37.0] * 10)
        with patch(
            "bodyscan_pipeline.measurements.slice_loop_2d",
            side_effect=lambda *args, **kwargs: (dummy_pts, next(mixed_circs)),
        ):
            _, neck_cm = measure_neck_perpendicular(object(), joints)
        self.assertAlmostEqual(neck_cm, 34.5, places=4)

        small_circs = iter([4.2, 5.1, 6.0] + [7.0] * 10)
        with patch(
            "bodyscan_pipeline.measurements.slice_loop_2d",
            side_effect=lambda *args, **kwargs: (dummy_pts, next(small_circs)),
        ):
            _, small_cm = measure_neck_perpendicular(object(), joints)
        self.assertAlmostEqual(small_cm, 4.2, places=4)

    def test_measure_limb_wrist_radius_gate_rejects_torso_and_allows_small_loops(self) -> None:
        from bodyscan_pipeline.config import LIMB_SPECS

        wrist_spec = next(s for s in LIMB_SPECS if s.name == "wrist_l")
        torso_part = _make_cylinder_mesh(radius=0.0953, height=0.90, sections=32, axis="y", center=(0.0, 0.85, 0.0))
        wrist_16 = _make_cylinder_mesh(radius=0.026, height=0.40, sections=16, axis="y", center=(0.25, 0.85, 0.0))
        combined_16 = _concat_simple_meshes([torso_part, wrist_16])

        joints = np.zeros((128, 3), dtype=np.float32)
        joints[wrist_spec.joint_start] = [0.25, 1.00, 0.0]
        joints[wrist_spec.joint_end] = [0.25, 0.72, 0.0]

        with patch("bodyscan_pipeline.measurements.trimesh", None):
            wrist_cm = measure_limb(
                combined_16,
                joints,
                wrist_spec.joint_start,
                wrist_spec.joint_end,
                wrist_spec.t_values,
                wrist_spec.mode,
                limb_name="wrist_l",
            )
        expected_cm = 2.0 * math.pi * 0.026 * 100.0
        self.assertAlmostEqual(wrist_cm, expected_cm, delta=1.0)
        self.assertLess(wrist_cm, 25.0)

        # Low-poly 3-vertex triangular wrist ring near torso still resolves cleanly (min_comp_len == 3)
        wrist_3 = _make_cylinder_mesh(radius=0.030, height=0.40, sections=3, axis="y", center=(0.25, 0.85, 0.0))
        combined_3 = _concat_simple_meshes([torso_part, wrist_3])
        with patch("bodyscan_pipeline.measurements.trimesh", None):
            wrist_tri_cm = measure_limb(
                combined_3,
                joints,
                wrist_spec.joint_start,
                wrist_spec.joint_end,
                wrist_spec.t_values,
                wrist_spec.mode,
                limb_name="wrist_l",
            )
        self.assertGreater(wrist_tri_cm, 8.0)
        self.assertLess(wrist_tri_cm, 25.0)

    def test_extract_16_measurements_numpy_fallback_without_trimesh(self) -> None:
        from bodyscan_pipeline.config import LIMB_SPECS, NECK_BASE_JOINT, NECK_HEAD_JOINT

        torso = _make_cylinder_mesh(radius=0.14, height=1.70, sections=24, axis="y", center=(0.0, 0.85, 0.0))
        arm_l = _make_cylinder_mesh(radius=0.032, height=0.60, sections=12, axis="y", center=(0.30, 1.00, 0.0))
        arm_r = _make_cylinder_mesh(radius=0.032, height=0.60, sections=12, axis="y", center=(-0.30, 1.00, 0.0))
        leg_l = _make_cylinder_mesh(radius=0.055, height=0.80, sections=12, axis="y", center=(0.15, 0.42, 0.0))
        leg_r = _make_cylinder_mesh(radius=0.055, height=0.80, sections=12, axis="y", center=(-0.15, 0.42, 0.0))
        full_mesh = _concat_simple_meshes([torso, arm_l, arm_r, leg_l, leg_r])
        torso_mesh = SimpleTriMesh(vertices=torso[0], faces=torso[1])

        joints = np.zeros((128, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.38, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.50, 0.0]
        for spec in LIMB_SPECS:
            is_left = spec.name.endswith("_l")
            x_arm = 0.30 if is_left else -0.30
            x_leg = 0.15 if is_left else -0.15
            if any(k in spec.name for k in ("bicep", "forearm", "wrist")):
                joints[spec.joint_start] = [x_arm, 1.22, 0.0]
                joints[spec.joint_end] = [x_arm, 0.82, 0.0]
            else:
                joints[spec.joint_start] = [x_leg, 0.78, 0.0]
                joints[spec.joint_end] = [x_leg, 0.10, 0.0]

        with patch("bodyscan_pipeline.measurements.trimesh", None):
            circs, _ = extract_16_measurements(full_mesh, torso_mesh, joints, 1.70, "female")

        self.assertEqual(len(circs), 16)
        self.assertGreater(circs["chest"], 70.0)
        self.assertLess(circs["wrist_l"], 28.0)
        self.assertLess(circs["wrist_r"], 28.0)

    def test_concave_cross_section_and_ordered_contour(self):
        # Extruded non-convex star/gear polygon (r = 0.16 +/- 0.05 * cos(4*theta))
        # Convex hull perimeter is ~103 cm; true concave arc length is ~132 cm.
        n_pts = 64
        thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
        radii = 0.16 + 0.05 * np.cos(4.0 * thetas)
        x = radii * np.cos(thetas)
        z = radii * np.sin(thetas)
        ring0 = np.stack([x, np.full(n_pts, 0.0), z], axis=-1)
        ring1 = np.stack([x, np.full(n_pts, 1.0), z], axis=-1)
        verts = np.vstack([ring0, ring1]).astype(np.float32)
        faces = []
        for i in range(n_pts):
            j = (i + 1) % n_pts
            faces.append([i, j, n_pts + j])
            faces.append([i, n_pts + j, n_pts + i])
        mesh = SimpleTriMesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64))

        loop, perim_cm = slice_loop_2d(mesh, origin=np.array([0.0, 0.5, 0.0]), normal=np.array([0.0, 1.0, 0.0]))
        self.assertIsNotNone(loop)
        analytical_cm = 100.0 * float(
            np.sum(np.linalg.norm(np.diff(np.vstack([ring0[:, [0, 2]], ring0[:1, [0, 2]]]), axis=0), axis=1))
        )
        # Perimeter must track concavity (> 115 cm) rather than collapsing to convex hull (~103 cm)
        self.assertGreater(perim_cm, 114.0)
        self.assertAlmostEqual(perim_cm, analytical_cm, delta=5.0)

        # Returned loop points must follow the closed contour monotonically without zig-zag chords
        edges = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
        self.assertLess(float(edges.max()), 0.06)

    def test_loft_profile_excludes_split_leg_gap(self):
        # Two separated cylindrical legs (x in [-0.14,-0.04] and [0.04,0.14], diameter=10cm each)
        # from y=0.0..0.85, joined under a single torso y=0.85..1.65 (width 32cm).
        def _cyl(cx: float, rx: float, rz: float, y0: float, y1: float, n_seg: int = 20):
            th = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
            vx = cx + rx * np.cos(th)
            vz = rz * np.sin(th)
            r0 = np.stack([vx, np.full(n_seg, y0), vz], axis=-1)
            r1 = np.stack([vx, np.full(n_seg, y1), vz], axis=-1)
            vv = np.vstack([r0, r1])
            ff = []
            for i in range(n_seg):
                j = (i + 1) % n_seg
                ff.append([i, j, n_seg + j])
                ff.append([i, n_seg + j, n_seg + i])
            return vv, np.asarray(ff, dtype=np.int64)

        lv, lf = _cyl(-0.09, 0.05, 0.05, 0.0, 0.85)
        rv, rf = _cyl(0.09, 0.05, 0.05, 0.0, 0.85)
        tv, tf = _cyl(0.0, 0.16, 0.10, 0.85, 1.65)
        all_v = np.vstack([lv, rv, tv]).astype(np.float32)
        all_f = np.vstack([lf, rf + len(lv), tf + len(lv) + len(rv)]).astype(np.int64)
        mesh = SimpleTriMesh(vertices=all_v, faces=all_f)

        loft = compute_loft_profile(mesh, height_m=1.65)
        # loft rows are [height_frac, width_m, depth_m].
        # In the split-leg zone (0.30..0.45 H), width is sum of leg widths (~0.20 m),
        # NOT 0.28 m (which would include the 0.08 m air gap between legs).
        leg_rows = [r for r in loft if 0.30 <= r[0] <= 0.45]
        self.assertTrue(len(leg_rows) >= 5)
        for row in leg_rows:
            self.assertLess(row[1], 0.22)
            self.assertGreater(row[1], 0.16)

    def test_slice_loop_2d_max_radius_uses_2d_centroid_distance(self):
        # Cylinder centered at x=0.18 with radius=0.09 -> outer vertices reach r=0.27 > max_radius_m=0.22,
        # while 2D centroid distance is 0.18 <= 0.22 so the whole circle must remain unclipped.
        th = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        vx = 0.18 + 0.09 * np.cos(th)
        vz = 0.09 * np.sin(th)
        verts = np.vstack(
            [np.stack([vx, np.zeros_like(vx), vz], axis=-1), np.stack([vx, np.ones_like(vx), vz], axis=-1)]
        ).astype(np.float32)
        faces = []
        n = len(th)
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])
        mesh = SimpleTriMesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64))

        loop, perim_cm = slice_loop_2d(
            mesh,
            origin=np.array([0.0, 0.5, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            max_radius_m=0.22,
        )
        self.assertIsNotNone(loop)
        self.assertAlmostEqual(perim_cm, 200.0 * np.pi * 0.09, delta=1.5)

    def test_slice_loop_2d_collinear_degenerate_returns_empty_safely(self):
        verts = np.array(
            [
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [-0.2, 1.0, 0.0],
                [0.2, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64)
        mesh = SimpleTriMesh(vertices=verts, faces=faces)

        loop, perim_cm = slice_loop_2d(mesh, origin=np.array([0.0, 0.5, 0.0]), normal=np.array([0.0, 1.0, 0.0]))
        self.assertIsNone(loop)
        self.assertEqual(perim_cm, 0.0)

    def test_scan_service_numpy_slicer_delegates_with_parity_and_neck_fallback(self):
        from bodyscan_pipeline import scan_service
        from bodyscan_pipeline.config import LIMB_SPECS, NECK_BASE_JOINT, NECK_HEAD_JOINT

        torso = _make_cylinder_mesh(radius=0.14, height=1.70, sections=24, axis="y", center=(0.0, 0.85, 0.0))
        arm_l = _make_cylinder_mesh(radius=0.032, height=0.60, sections=12, axis="y", center=(0.30, 1.00, 0.0))
        arm_r = _make_cylinder_mesh(radius=0.032, height=0.60, sections=12, axis="y", center=(-0.30, 1.00, 0.0))
        leg_l = _make_cylinder_mesh(radius=0.055, height=0.80, sections=12, axis="y", center=(0.15, 0.42, 0.0))
        leg_r = _make_cylinder_mesh(radius=0.055, height=0.80, sections=12, axis="y", center=(-0.15, 0.42, 0.0))
        full_mesh = _concat_simple_meshes([torso, arm_l, arm_r, leg_l, leg_r])
        torso_mesh = SimpleTriMesh(vertices=torso[0], faces=torso[1])

        joints = np.zeros((128, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.38, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.50, 0.0]
        for spec in LIMB_SPECS:
            is_left = spec.name.endswith("_l")
            x_arm = 0.30 if is_left else -0.30
            x_leg = 0.15 if is_left else -0.15
            if any(k in spec.name for k in ("bicep", "forearm", "wrist")):
                joints[spec.joint_start] = [x_arm, 1.22, 0.0]
                joints[spec.joint_end] = [x_arm, 0.82, 0.0]
            else:
                joints[spec.joint_start] = [x_leg, 0.78, 0.0]
                joints[spec.joint_end] = [x_leg, 0.10, 0.0]

        loop_m, perim_m = slice_loop_2d(
            full_mesh, origin=joints[NECK_BASE_JOINT], normal=np.array([0.0, 1.0, 0.0]), max_radius_m=0.35
        )
        loop_np, perim_np = scan_service._slice_loop_2d_numpy(
            full_mesh.vertices,
            full_mesh.faces,
            origin=joints[NECK_BASE_JOINT],
            normal=np.array([0.0, 1.0, 0.0]),
            max_radius_m=0.35,
        )
        self.assertAlmostEqual(perim_np, perim_m, places=5)
        self.assertIsNotNone(loop_np)
        self.assertEqual(loop_np.shape, loop_m.shape)

        # Tiny skinny neck region where raw neck <= 15 cm triggers neck fallback parity
        verts_skinny = full_mesh.vertices.copy()
        neck_band = (verts_skinny[:, 1] >= 1.36) & (verts_skinny[:, 1] <= 1.52) & (np.abs(verts_skinny[:, 0]) < 0.2)
        verts_skinny[neck_band, 0] *= 0.12
        verts_skinny[neck_band, 2] *= 0.12
        skinny_full = SimpleTriMesh(vertices=verts_skinny, faces=full_mesh.faces)
        torso_skinny_v = torso[0].copy()
        t_band = (torso_skinny_v[:, 1] >= 1.36) & (torso_skinny_v[:, 1] <= 1.52)
        torso_skinny_v[t_band, 0] *= 0.12
        torso_skinny_v[t_band, 2] *= 0.12
        skinny_torso = SimpleTriMesh(vertices=torso_skinny_v, faces=torso[1])

        with patch("bodyscan_pipeline.measurements.trimesh", None):
            circ_m, dbg_m = extract_16_measurements(skinny_full, skinny_torso, joints, 1.70, "female")
            loft_m = compute_loft_profile(skinny_torso, 1.70)
        keep_torso_mask = np.zeros(len(verts_skinny), dtype=bool)
        keep_torso_mask[: len(torso[0])] = True
        circ_np, dbg_np, loft_np = scan_service._measure_circumferences_and_loft_numpy(
            verts_skinny, full_mesh.faces, keep_torso_mask, joints, 1.70, "female"
        )
        self.assertEqual(circ_np["neck"], circ_m["neck"])
        self.assertEqual(len(loft_np), len(loft_m))
        self.assertEqual(set(dbg_np.keys()), set(dbg_m.keys()))
        self.assertGreaterEqual(circ_np["neck"], 25.0)


if __name__ == "__main__":
    unittest.main()
