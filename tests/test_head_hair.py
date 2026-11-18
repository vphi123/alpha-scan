"""Unit tests for Stage 2 high-res head subdivision, nose/depth sculpting, and 4-view hair shell."""

import dataclasses
import sys
import unittest

import cv2
import numpy as np

from bodyscan_pipeline.config import MHR_HEAD_SUBDIV_STEPS, NECK_BASE_JOINT, NECK_HEAD_JOINT
from bodyscan_pipeline.segmentation import SegmentedView


def _build_synthetic_humanoid_mesh(n_lon: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a closed watertight manifold humanoid mesh (+Y up, +Z front, +X left) with 127 joints."""
    ring_specs = [
        (0.02, 0.12),   # 0: feet ring
        (0.35, 0.13),   # 1: knees
        (0.75, 0.14),   # 2: thighs
        (1.00, 0.16),   # 3: hips
        (1.22, 0.14),   # 4: torso mid
        (1.34, 0.15),   # 5: upper chest / shoulders base
        (1.38, 0.065),  # 6: NECK_BASE_JOINT collar ring
        (1.44, 0.060),  # 7: mid-neck
        (1.49, 0.078),  # 8: NECK_HEAD_JOINT / chin base
        (1.54, 0.105),  # 9: lower face / jaw
        (1.60, 0.110),  # 10: mid-face / nose height (radius 0.110m)
        (1.65, 0.100),  # 11: brow / forehead
        (1.69, 0.072),  # 12: upper crown
    ]

    verts_list: list[list[float]] = []
    # Vertex 0: bottom pole
    verts_list.append([0.0, 0.0, 0.0])
    thetas = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    for y_val, rad in ring_specs:
        for th in thetas:
            verts_list.append([float(rad * np.cos(th)), float(y_val), float(rad * np.sin(th))])
    # Last vertex: top crown pole
    top_pole_idx = len(verts_list)
    verts_list.append([0.0, 1.72, 0.0])

    faces_list: list[list[int]] = []

    def ring_idx(r: int, k: int) -> int:
        return 1 + r * n_lon + (k % n_lon)

    # Bottom cap (outward normal -Y)
    for k in range(n_lon):
        faces_list.append([0, ring_idx(0, k), ring_idx(0, k + 1)])

    # Side quads between consecutive rings (outward normal radial)
    for r in range(len(ring_specs) - 1):
        for k in range(n_lon):
            b0 = ring_idx(r, k)
            b1 = ring_idx(r, k + 1)
            t0 = ring_idx(r + 1, k)
            t1 = ring_idx(r + 1, k + 1)
            faces_list.append([b0, t0, t1])
            faces_list.append([b0, t1, b1])

    # Top cap (outward normal +Y)
    last_r = len(ring_specs) - 1
    for k in range(n_lon):
        faces_list.append([top_pole_idx, ring_idx(last_r, k + 1), ring_idx(last_r, k)])

    verts = np.asarray(verts_list, dtype=np.float32)
    faces = np.asarray(faces_list, dtype=np.int64)

    joints = np.zeros((127, 3), dtype=np.float32)
    joints[0] = [0.0, 1.60, 0.11]                # Nose tip reference
    joints[2] = [0.08, 0.90, 0.0]                # Left hip
    joints[18] = [-0.08, 0.90, 0.0]              # Right hip
    joints[39] = [-0.20, 1.34, 0.0]              # Right shoulder
    joints[75] = [0.20, 1.34, 0.0]               # Left shoulder
    joints[NECK_BASE_JOINT] = [0.0, 1.38, 0.0]   # Neck base collar
    joints[NECK_HEAD_JOINT] = [0.0, 1.49, 0.0]   # Neck head
    return verts, faces, joints


def _edge_degree_map(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Count how many triangles share each undirected edge."""
    counts: dict[tuple[int, int], int] = {}
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            counts[e] = counts.get(e, 0) + 1
    return counts


def _make_profile_view(
    view_label: str,
    azimuth_deg: float,
    view_index: int,
    nose_bump_m: float = 0.0,
    bun_extra_m: float = 0.0,
    img_h: int = 240,
    img_w: int = 200,
    px_per_m: float = 400.0,
) -> SegmentedView:
    """Create a synthetic orthographic side/front/back SegmentedView matching canonical head geometry."""
    # Skull center at (0, 1.60, 0), radius 0.11m, top crown 1.72m -> mapped to image center x=100, crown y_px=30
    cx = img_w / 2.0
    y_crown_m = 1.72
    y_crown_px = 30.0
    cy_skull = y_crown_px + (y_crown_m - 1.60) * px_per_m  # y_px = 78.0
    r_px = int(round(0.11 * px_per_m))                     # 44 px

    person_mask = np.zeros((img_h, img_w), dtype=bool)
    hair_mask = np.zeros((img_h, img_w), dtype=bool)
    face_skin_mask = np.zeros((img_h, img_w), dtype=bool)

    # Cranium circle (radius 0.11m -> 44 px) + neck/upper torso column below
    cv2.circle(person_mask.view(np.uint8), (int(cx), int(cy_skull)), r_px, 1, -1)
    y_neck_px = int(round(y_crown_px + (y_crown_m - 1.38) * px_per_m))  # 166
    cv2.rectangle(person_mask.view(np.uint8), (int(cx - 24), int(cy_skull + 20)), (int(cx + 24), img_h - 10), 1, -1)

    if view_label in ("right", "left"):
        dir_sign = 1.0 if view_label == "right" else -1.0
        # Face skin covers anterior half below hairline down to chin
        skin_x0 = int(cx) if dir_sign > 0 else int(cx - r_px - 2)
        skin_x1 = int(cx + r_px + 2) if dir_sign > 0 else int(cx)
        cv2.rectangle(
            face_skin_mask.view(np.uint8),
            (max(0, skin_x0), int(cy_skull - 16)),
            (min(img_w - 1, skin_x1), y_neck_px - 4),
            1,
            -1,
        )
        face_skin_mask &= person_mask

        # Nose bump added on anterior silhouette around y_m=1.60 (y_px=78)
        if nose_bump_m > 0.0:
            bump_px = nose_bump_m * px_per_m
            tip_x = int(round(cx + dir_sign * (0.11 * px_per_m + bump_px)))
            base_x = int(round(cx + dir_sign * (0.11 * px_per_m * 0.92)))
            pts = np.array(
                [
                    [base_x, int(cy_skull - 14)],
                    [tip_x, int(cy_skull)],
                    [base_x, int(cy_skull + 10)],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(person_mask.view(np.uint8), [pts], 1)
            cv2.fillPoly(face_skin_mask.view(np.uint8), [pts], 1)

        # Optional tall back/top bun envelope strictly above neck_y_px
        if bun_extra_m > 0.0:
            bun_r = int(round(0.045 * px_per_m))
            # Top-back center at (-0.08m in Z, +0.08m above skull center) -> +bun_extra_m
            bun_dx = -dir_sign * (0.08 * px_per_m + 0.7 * bun_extra_m * px_per_m)
            bun_dy = -(0.07 * px_per_m + 0.7 * bun_extra_m * px_per_m)
            bx = int(round(cx + bun_dx))
            by = int(round(cy_skull + bun_dy))
            cv2.circle(person_mask.view(np.uint8), (bx, by), bun_r, 1, -1)
            cv2.circle(hair_mask.view(np.uint8), (bx, by), bun_r, 1, -1)
            # Crown hair band on top/back above hairline
            cv2.ellipse(
                hair_mask.view(np.uint8),
                (int(round(cx - dir_sign * 8)), int(round(cy_skull - 22))),
                (r_px - 2, 22),
                0,
                0,
                360,
                1,
                -1,
            )
            person_mask |= hair_mask

        # Extract clean nose_profile_curve across face band [y_crown_px+16 .. y_neck_px-12]
        y_start = int(cy_skull - 24)
        y_end = int(cy_skull + 36)
        curve_pts: list[tuple[float, float]] = []
        for y in range(y_start, y_end + 1):
            xs = np.where(person_mask[y])[0]
            if len(xs) > 0:
                x_ant = float(xs.max()) if view_label == "right" else float(xs.min())
                curve_pts.append((x_ant, float(y)))
        nose_curve = np.asarray(curve_pts, dtype=np.float32)
    else:
        nose_curve = None
        if view_label == "front":
            cv2.ellipse(
                face_skin_mask.view(np.uint8),
                (int(cx), int(cy_skull + 4)),
                (r_px - 6, r_px - 8),
                0,
                0,
                360,
                1,
                -1,
            )
            face_skin_mask &= person_mask
        if bun_extra_m > 0.0:
            # Crown / back hair silhouette for front & back views
            crown_r = int(round((0.11 + 0.6 * bun_extra_m) * px_per_m))
            cv2.ellipse(
                hair_mask.view(np.uint8),
                (int(cx), int(cy_skull - 24)),
                (r_px, crown_r // 2),
                0,
                0,
                360,
                1,
                -1,
            )
            if view_label == "back":
                cv2.circle(hair_mask.view(np.uint8), (int(cx), int(cy_skull - 10)), r_px + 4, 1, -1)
            person_mask |= hair_mask

    hair_mask &= ~face_skin_mask
    body_mask = person_mask & (~hair_mask)
    return SegmentedView(
        view_index=view_index,
        azimuth_deg=azimuth_deg,
        view_label=view_label,
        person_mask=person_mask,
        body_mask=body_mask,
        hair_mask=hair_mask,
        face_skin_mask=face_skin_mask,
        nose_profile_curve=nose_curve,
        face_landmarks_ram=None,
        collar_hair_overlap=False,
        warnings=[],
    )


class TestHeadHair(unittest.TestCase):
    """Test Stage 2 head subdivision, nose/depth sculpting, hair shell, and S2/3a/3b guarantees."""

    def test_partition_and_subdivide_head_preserves_exact_collar_ring(self) -> None:
        from bodyscan_pipeline.head_hair import HeadPartition, partition_and_subdivide_head

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        part = partition_and_subdivide_head(verts, faces, joints, mhr_model=None, subdiv_steps=MHR_HEAD_SUBDIV_STEPS)
        self.assertIsInstance(part, HeadPartition)

        # Head vertex count increases meaningfully (> 2x)
        self.assertGreater(len(part.subdiv_verts), 2 * len(part.head_vert_idx))
        self.assertGreater(len(part.collar_vert_idx), 10)
        self.assertEqual(len(part.collar_subdiv_idx), len(part.collar_vert_idx))

        # Collar coordinates remain bit-identical to original verts_a[collar_vert_idx]
        self.assertTrue(
            np.array_equal(part.subdiv_verts[part.collar_subdiv_idx], verts[part.collar_vert_idx])
        )

        # Subdivided head mesh has manifold degree-2 internal edges (zero T-junctions)
        head_edge_deg = _edge_degree_map(part.subdiv_faces)
        collar_set = set(part.collar_subdiv_idx.tolist())
        non_collar_boundary = [
            e for e, deg in head_edge_deg.items() if deg != 2 and not (e[0] in collar_set and e[1] in collar_set)
        ]
        self.assertEqual(non_collar_boundary, [])

        # Combined body + subdivided head is completely watertight manifold (every edge degree == 2)
        body_faces = faces[part.body_face_mask]
        head_faces_global = part.subdiv_faces.copy()
        n_orig = len(verts)
        # Map subdiv local indices back to a combined vertex space where original collar indices merge
        local_to_global = np.arange(len(part.subdiv_verts), dtype=np.int64) + n_orig
        local_to_global[part.collar_subdiv_idx] = part.collar_vert_idx
        combined_faces = np.vstack([body_faces, local_to_global[head_faces_global]])
        combined_deg = _edge_degree_map(combined_faces)
        self.assertTrue(all(deg == 2 for deg in combined_deg.values()))

        # Collar ring distance starts at 0 at collar and increases cleanly upward
        self.assertTrue(np.all(part.subdiv_collar_ring_dist[part.collar_subdiv_idx] == 0))
        self.assertTrue(np.all(part.subdiv_collar_ring_dist >= 0))
        self.assertGreaterEqual(int(part.subdiv_collar_ring_dist.max()), 5)
        self.assertEqual(part.cranium_verts.shape, part.subdiv_verts.shape)

    def test_fit_nose_profile_and_depth_matches_side_silhouette_with_zero_collar_drift(self) -> None:
        from bodyscan_pipeline.head_hair import fit_nose_profile_and_depth, partition_and_subdivide_head

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        part = partition_and_subdivide_head(verts, faces, joints, subdiv_steps=1)

        views_flat = [
            _make_profile_view("right", 90.0, 0, nose_bump_m=0.0),
            _make_profile_view("left", 270.0, 1, nose_bump_m=0.0),
        ]
        views_nose = [
            _make_profile_view("right", 90.0, 0, nose_bump_m=0.025),
            _make_profile_view("left", 270.0, 1, nose_bump_m=0.025),
        ]

        sculpted_flat = fit_nose_profile_and_depth(
            part.subdiv_verts,
            part.subdiv_faces,
            part.subdiv_collar_ring_dist,
            views_flat,
            moge2_depths=None,
        )
        sculpted_nose = fit_nose_profile_and_depth(
            part.subdiv_verts,
            part.subdiv_faces,
            part.subdiv_collar_ring_dist,
            views_nose,
            moge2_depths=None,
        )

        self.assertEqual(sculpted_nose.shape, part.subdiv_verts.shape)
        self.assertTrue(np.isfinite(sculpted_nose).all())

        # Collar ring_dist == 0 keeps exact 0.0 displacement
        collar_mask = part.subdiv_collar_ring_dist == 0
        self.assertTrue(np.array_equal(sculpted_nose[collar_mask], part.subdiv_verts[collar_mask]))
        self.assertTrue(np.array_equal(sculpted_flat[collar_mask], part.subdiv_verts[collar_mask]))

        # Anterior nose tip zone (around y=1.60, |x|<0.03, z>0.08) protrudes strongly toward +2.5cm bump
        nose_roi = (
            (np.abs(part.subdiv_verts[:, 1] - 1.60) <= 0.03)
            & (np.abs(part.subdiv_verts[:, 0]) <= 0.03)
            & (part.subdiv_verts[:, 2] >= 0.08)
        )
        self.assertTrue(np.any(nose_roi))
        peak_z_nose = float(sculpted_nose[nose_roi, 2].max())
        peak_z_flat = float(sculpted_flat[nose_roi, 2].max())
        self.assertGreater(peak_z_nose - peak_z_flat, 0.015)
        self.assertLess(abs((peak_z_nose - peak_z_flat) - 0.025), 0.012)

        # MoGe2 depth relief path with NaN outside foreground still yields finite relief and 0 collar drift
        depth_r = np.full(views_nose[0].person_mask.shape, np.nan, dtype=np.float32)
        depth_r[views_nose[0].face_skin_mask] = 2.0
        depth_r[73:83, 120:145] = 1.99  # slightly closer cheek/nose relief
        sculpted_depth = fit_nose_profile_and_depth(
            part.subdiv_verts,
            part.subdiv_faces,
            part.subdiv_collar_ring_dist,
            views_nose,
            moge2_depths=[depth_r, None],
        )
        self.assertTrue(np.isfinite(sculpted_depth).all())
        self.assertTrue(np.array_equal(sculpted_depth[collar_mask], part.subdiv_verts[collar_mask]))

    def test_hair_shell_captures_bun_contour_without_skull_penetration(self) -> None:
        from bodyscan_pipeline.head_hair import build_hair_volume_shell, partition_and_subdivide_head

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        part = partition_and_subdivide_head(verts, faces, joints, subdiv_steps=1)

        views_bun = [
            _make_profile_view("front", 0.0, 0, bun_extra_m=0.06),
            _make_profile_view("right", 90.0, 1, bun_extra_m=0.06),
            _make_profile_view("back", 180.0, 2, bun_extra_m=0.06),
            _make_profile_view("left", 270.0, 3, bun_extra_m=0.06),
        ]
        hair_verts, hair_faces = build_hair_volume_shell(
            part.cranium_verts,
            part.subdiv_faces,
            joints,
            views_bun,
            raw_results=[],
        )
        self.assertGreater(len(hair_verts), 20)
        self.assertGreater(len(hair_faces), 20)
        self.assertTrue(np.isfinite(hair_verts).all())

        # Closed watertight manifold hair shell
        hair_deg = _edge_degree_map(hair_faces)
        self.assertTrue(all(deg == 2 for deg in hair_deg.values()))

        # Radial distance from skull centroid for every hair shell vertex >= cranium reference radius - 1e-5
        skull_center = np.array([0.0, 1.60, 0.0], dtype=np.float32)
        upper_cran = part.cranium_verts[part.cranium_verts[:, 1] >= float(joints[NECK_HEAD_JOINT, 1]) + 0.04]
        cran_radii = np.linalg.norm(upper_cran - skull_center, axis=1)
        cran_ref_radius = float(np.min(cran_radii))
        hair_radii = np.linalg.norm(hair_verts - skull_center, axis=1)
        self.assertGreaterEqual(float(hair_radii.min()), cran_ref_radius - 1e-5)

        # Top/back hair shell vertices expand outward toward the tall bun contour
        back_top_hair = (hair_verts[:, 1] > 1.63) & (hair_verts[:, 2] < -0.02)
        self.assertTrue(np.any(back_top_hair))
        self.assertGreater(float(hair_radii[back_top_hair].max()), cran_ref_radius + 0.03)

        # Hair shell stays strictly above NECK_BASE_JOINT=110 (2 gon)
        self.assertGreaterEqual(float(hair_verts[:, 1].min()), float(joints[NECK_BASE_JOINT, 1]) - 1e-5)

    def test_reconstruct_highres_head_and_hair_stitch_and_disjoint_skin_vs_hair_masks(self) -> None:
        from bodyscan_pipeline.head_hair import HighResHeadOutput, reconstruct_highres_head_and_hair

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        views = [
            _make_profile_view("front", 0.0, 0, nose_bump_m=0.02, bun_extra_m=0.05),
            _make_profile_view("right", 90.0, 1, nose_bump_m=0.02, bun_extra_m=0.05),
            _make_profile_view("back", 180.0, 2, nose_bump_m=0.02, bun_extra_m=0.05),
            _make_profile_view("left", 270.0, 3, nose_bump_m=0.02, bun_extra_m=0.05),
        ]
        out = reconstruct_highres_head_and_hair(verts, faces, joints, views, raw_results=[])
        self.assertIsInstance(out, HighResHeadOutput)
        self.assertGreater(len(out.full_verts), len(verts))
        self.assertTrue(np.isfinite(out.full_verts).all())

        # Watertight manifold merged mesh
        full_deg = _edge_degree_map(out.full_faces)
        self.assertTrue(all(deg == 2 for deg in full_deg.values()))

        # Body vertices strictly below NECK_BASE_JOINT=110 remain bit-exact to inner verts_a
        neck_y = float(joints[NECK_BASE_JOINT, 1])
        orig_body_verts = verts[verts[:, 1] < neck_y - 1e-5]
        out_body_verts = out.full_verts[out.full_verts[:, 1] < neck_y - 1e-5]
        self.assertEqual(len(out_body_verts), len(orig_body_verts))
        self.assertTrue(np.array_equal(out_body_verts, orig_body_verts))

        # 3b disjointness: hair_vert_mask vs face_skin_vert_mask non-empty and strictly disjoint
        self.assertTrue(np.any(out.head_vert_mask))
        self.assertTrue(np.any(out.hair_vert_mask))
        self.assertTrue(np.any(out.face_skin_vert_mask))
        self.assertTrue(np.any(out.front_face_vert_mask))
        self.assertFalse(bool(np.any(out.hair_vert_mask & out.face_skin_vert_mask)))
        self.assertTrue(bool(np.all(out.head_vert_mask[out.hair_vert_mask])))
        self.assertGreaterEqual(
            float(out.full_verts[out.hair_vert_mask, 1].min()),
            neck_y - 1e-5,
        )

        # Full-body silhouette (wide torso/hips below neck) must still recover skull scale without torso bleed
        full_body_views = [
            _make_profile_view(lbl, az, i, nose_bump_m=0.02, bun_extra_m=0.05, img_h=560, img_w=240)
            for i, (lbl, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0)))
        ]
        for sv in full_body_views:
            cv2.rectangle(sv.person_mask.view(np.uint8), (50, 175), (190, 520), 1, -1)
            sv.body_mask = sv.person_mask & (~sv.hair_mask)
        out_fb = reconstruct_highres_head_and_hair(verts, faces, joints, full_body_views, raw_results=[])
        self.assertTrue(np.isfinite(out_fb.full_verts).all())
        fb_nose_roi = (out_fb.full_verts[:, 1] > 1.56) & (out_fb.full_verts[:, 1] < 1.64) & (out_fb.full_verts[:, 2] > 0.08)
        self.assertGreater(float(out_fb.full_verts[fb_nose_roi, 2].max()), 0.12)

    def test_head_hair_graceful_fallback_and_zero_landmark_persistence(self) -> None:
        from bodyscan_pipeline.head_hair import (
            HeadPartition,
            HighResHeadOutput,
            align_face_landmarks_3d,
            partition_and_subdivide_head,
            reconstruct_highres_head_and_hair,
        )

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=20)
        views_no_cues = [
            _make_profile_view("front", 0.0, 0),
            _make_profile_view("right", 90.0, 1),
            _make_profile_view("back", 180.0, 2),
            _make_profile_view("left", 270.0, 3),
        ]
        raw_no_depth = [{"output": {}, "moge2_depth": None} for _ in views_no_cues]

        out = reconstruct_highres_head_and_hair(verts, faces, joints, views_no_cues, raw_results=raw_no_depth)
        self.assertTrue(np.isfinite(out.full_verts).all())
        self.assertIsInstance(out.warnings, list)
        self.assertGreaterEqual(len(out.warnings), 1)

        # S2 privacy rule: zero landmark fields stored on HeadPartition or HighResHeadOutput
        for dc in (HeadPartition, HighResHeadOutput):
            field_names = [f.name for f in dataclasses.fields(dc)]
            self.assertFalse(any("landmark" in name.lower() for name in field_names))
        self.assertFalse(any("landmark" in k.lower() for k in vars(out).keys()))

        # Even when ephemeral face_landmarks_ram is provided in RAM, output keeps zero landmarks
        part = partition_and_subdivide_head(verts, faces, joints)
        views_with_lm = [
            _make_profile_view("front", 0.0, 0),
            _make_profile_view("right", 90.0, 1),
        ]
        fake_lm = np.zeros((478, 3), dtype=np.float32)
        fake_lm[:, 0] = 0.50
        fake_lm[:, 1] = 0.50
        fake_lm[1] = [0.50, 0.50, -0.08]    # nose tip protruding toward camera
        fake_lm[6] = [0.50, 0.44, -0.04]    # nose bridge
        fake_lm[33] = [0.40, 0.42, 0.0]     # right eye outer
        fake_lm[263] = [0.60, 0.42, 0.0]    # left eye outer
        fake_lm[152] = [0.50, 0.72, 0.0]    # chin
        fake_lm[10] = [0.50, 0.25, 0.0]     # forehead
        fake_lm[61] = [0.44, 0.60, 0.0]     # right mouth
        fake_lm[291] = [0.56, 0.60, 0.0]    # left mouth
        fake_lm[234] = [0.34, 0.50, 0.03]   # right cheek
        fake_lm[454] = [0.66, 0.50, 0.03]   # left cheek
        views_with_lm[0].face_landmarks_ram = fake_lm
        delta, skin_mask = align_face_landmarks_3d(part.subdiv_verts, joints, views_with_lm, [])
        self.assertEqual(delta.shape, part.subdiv_verts.shape)
        self.assertTrue(np.isfinite(delta).all())
        self.assertTrue(np.any(skin_mask))
        # Verify vertical orientation: nose tip (mid-face) receives positive +Z > 0.005 m and > chin
        mid_nose_roi = (
            skin_mask
            & (part.subdiv_verts[:, 1] >= 1.54)
            & (part.subdiv_verts[:, 1] <= 1.61)
            & (np.abs(part.subdiv_verts[:, 0]) <= 0.03)
            & (part.subdiv_verts[:, 2] >= 0.08)
        )
        chin_roi = skin_mask & (part.subdiv_verts[:, 1] <= float(joints[NECK_HEAD_JOINT, 1]) + 0.01)
        self.assertTrue(np.any(mid_nose_roi))
        self.assertTrue(np.any(chin_roi))
        self.assertGreater(float(delta[mid_nose_roi, 2].max()), 0.005)
        self.assertGreater(float(delta[mid_nose_roi, 2].max()), float(np.max(delta[chin_roi, 2])) + 0.003)

        out_lm = reconstruct_highres_head_and_hair(verts, faces, joints, views_with_lm, raw_results=[])
        self.assertTrue(np.isfinite(out_lm.full_verts).all())
        self.assertFalse(any("landmark" in k.lower() for k in vars(out_lm).keys()))

        # License compliance check
        sys.modules["ultralytics"] = type(sys)("ultralytics")
        try:
            with self.assertRaises(RuntimeError):
                reconstruct_highres_head_and_hair(verts, faces, joints, views_no_cues, raw_results=[])
        finally:
            sys.modules.pop("ultralytics", None)

    def test_moge2_depth_resizes_mismatched_shape_and_handles_nans(self) -> None:
        from bodyscan_pipeline.head_hair import (
            _apply_moge2_facial_relief,
            _estimate_head_frames,
            partition_and_subdivide_head,
        )

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=20)
        part = partition_and_subdivide_head(verts, faces, joints)

        views = [_make_profile_view("front", 0.0, 0, img_h=120, img_w=120, px_per_m=200.0)]
        # Make front skin mask cover anterior face region
        views[0].face_skin_mask = views[0].person_mask.copy()
        geom, frames = _estimate_head_frames(views, part.subdiv_verts, joints_a=joints)

        # Half-resolution (60x60) MoGe2 depth map with outer border NaNs and nose-center curvature
        yy_d, xx_d = np.mgrid[0:60, 0:60].astype(np.float32)
        nose_gauss = np.exp(-(((yy_d - 27.0) ** 2 + (xx_d - 30.0) ** 2) / (2.5 ** 2)))
        d_low = (1.80 - 0.025 * nose_gauss).astype(np.float32)
        d_low[:4, :] = np.nan
        relief = _apply_moge2_facial_relief(
            part.subdiv_verts, part.subdiv_faces, views, [d_low], frames, geom
        )
        self.assertEqual(relief.shape, part.subdiv_verts.shape)
        self.assertTrue(np.isfinite(relief).all())
        self.assertGreater(float(np.abs(relief).max()), 1e-4)

    def test_align_face_landmarks_3d_oblique_view_projects_along_u_axis(self) -> None:
        from bodyscan_pipeline.head_hair import align_face_landmarks_3d, partition_and_subdivide_head

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        part = partition_and_subdivide_head(verts, faces, joints)

        oblique_view = _make_profile_view("oblique", 45.0, 0)
        fake_lm = np.zeros((478, 3), dtype=np.float32)
        fake_lm[:, :2] = 0.50
        fake_lm[1] = [0.52, 0.50, -0.07]     # nose tip
        fake_lm[6] = [0.51, 0.44, -0.04]     # nose bridge
        fake_lm[33] = [0.40, 0.42, 0.0]      # right eye outer
        fake_lm[263] = [0.60, 0.42, 0.0]     # left eye outer
        fake_lm[152] = [0.50, 0.72, 0.0]     # chin
        fake_lm[10] = [0.50, 0.25, 0.0]      # forehead
        fake_lm[61] = [0.44, 0.60, 0.0]      # right mouth
        fake_lm[291] = [0.56, 0.60, 0.0]     # left mouth
        fake_lm[234] = [0.34, 0.50, 0.03]    # right cheek
        fake_lm[454] = [0.66, 0.50, 0.03]    # left cheek
        oblique_view.face_landmarks_ram = fake_lm

        delta, skin_mask = align_face_landmarks_3d(part.subdiv_verts, joints, [oblique_view], [])
        self.assertEqual(delta.shape, part.subdiv_verts.shape)
        self.assertTrue(np.isfinite(delta).all())
        self.assertTrue(np.any(skin_mask))

        # Project along oblique cam_dir=[-sin(45), 0, cos(45)] rather than unrotated +Z
        az_rad = np.deg2rad(45.0)
        cam_dir = np.array([-np.sin(az_rad), 0.0, np.cos(az_rad)], dtype=np.float32)
        proj_cam = delta[skin_mask] @ cam_dir
        self.assertGreater(float(proj_cam.max()), 0.004)
        # Crown / scalp vertices above hairline must never receive skin mask or TPS landmark edits
        y_crown = float(part.subdiv_verts[:, 1].max())
        scalp_roi = part.subdiv_verts[:, 1] > (y_crown - 0.04)
        self.assertTrue(np.any(scalp_roi))
        self.assertFalse(bool(np.any(skin_mask & scalp_roi)))
        self.assertEqual(float(np.abs(delta[scalp_roi]).max()), 0.0)

    def test_sample_hair_envelope_stops_at_air_gap_and_shell_resists_corrupted_flood_mask(self) -> None:
        from bodyscan_pipeline.head_hair import (
            _sample_hair_envelope_radius_px,
            build_hair_volume_shell,
            partition_and_subdivide_head,
        )

        h_mask = np.zeros((200, 200), dtype=bool)
        # True thin hair band from r=20..28 px, then a 16 px air void, then a distant wall blob
        h_mask[96:105, 120:129] = True
        h_mask[94:107, 145:178] = True
        r_hit = _sample_hair_envelope_radius_px(
            hair_mask=h_mask,
            cx_px=100.0,
            cy_px=100.0,
            du=1.0,
            dv=0.0,
            r_min_px=10.0,
            r_max_px=80.0,
            y_neck_px=165.0,
        )
        self.assertIsNotNone(r_hit)
        assert r_hit is not None
        self.assertLessEqual(float(r_hit), 31.0)

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        head_part = partition_and_subdivide_head(verts, faces, joints, subdiv_steps=1)
        img_h, img_w = 240, 200
        px_per_m = 400.0
        cx = img_w / 2.0
        cy_skull = 30.0 + (1.72 - 1.60) * px_per_m
        r_px = int(round(0.11 * px_per_m))
        clean_views = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            pm = np.zeros((img_h, img_w), dtype=bool)
            hm = np.zeros((img_h, img_w), dtype=bool)
            fm = np.zeros((img_h, img_w), dtype=bool)
            cv2.circle(pm.view(np.uint8), (int(cx), int(cy_skull)), r_px, 1, -1)
            cv2.rectangle(pm.view(np.uint8), (int(cx - 24), int(cy_skull + 20)), (int(cx + 24), img_h - 10), 1, -1)
            cv2.ellipse(hm.view(np.uint8), (int(cx), int(cy_skull - 22)), (r_px + 4, 24), 0, 0, 360, 1, -1)
            if label == "front":
                cv2.ellipse(fm.view(np.uint8), (int(cx), int(cy_skull + 4)), (r_px - 6, r_px - 8), 0, 0, 360, 1, -1)
            if idx == 2:
                # Corrupt one view with a giant rectangular wall flood box in hair_mask
                hm[5:135, 15:185] = True
            pm |= hm
            hm &= ~fm
            clean_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=pm,
                    body_mask=pm & (~hm),
                    hair_mask=hm,
                    face_skin_mask=fm & pm,
                    nose_profile_curve=None,
                    face_landmarks_ram=None,
                    collar_hair_overlap=False,
                    warnings=[],
                )
            )
        hair_v, hair_f = build_hair_volume_shell(
            cranium_verts=head_part.cranium_verts,
            cranium_faces=head_part.subdiv_faces,
            joints_a=joints,
            segmented_views=clean_views,
            raw_results=[],
        )
        self.assertGreater(len(hair_v), 0)
        self.assertGreater(len(hair_f), 0)
        sc = np.array([0.0, 1.60, 0.0], dtype=np.float32)
        base_r = float(np.linalg.norm(head_part.cranium_verts - sc[None, :], axis=1).max())
        hair_r = float(np.linalg.norm(hair_v - sc[None, :], axis=1).max())
        # Smooth anatomical crown/bun cap keeps shell well below boxy wall-flood explosion
        self.assertLessEqual(hair_r - base_r, 0.080)

    def test_highres_head_output_exposes_undisplaced_verts_and_body_only_idx(self) -> None:
        from bodyscan_pipeline.head_hair import HighResHeadOutput, reconstruct_highres_head_and_hair

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        views = [
            _make_profile_view("front", 0.0, 0, nose_bump_m=0.025, bun_extra_m=0.05),
            _make_profile_view("right", 90.0, 1, nose_bump_m=0.025, bun_extra_m=0.05),
            _make_profile_view("back", 180.0, 2, nose_bump_m=0.025, bun_extra_m=0.05),
            _make_profile_view("left", 270.0, 3, nose_bump_m=0.025, bun_extra_m=0.05),
        ]
        out = reconstruct_highres_head_and_hair(verts, faces, joints, views, raw_results=[])
        self.assertIsNotNone(out.undisplaced_verts)
        self.assertIsNotNone(out.body_only_idx)
        assert out.undisplaced_verts is not None
        assert out.body_only_idx is not None
        self.assertEqual(out.undisplaced_verts.shape, out.full_verts.shape)
        self.assertTrue(np.isfinite(out.undisplaced_verts).all())

        n_body = len(out.body_only_idx)
        self.assertGreater(n_body, 0)
        np.testing.assert_allclose(out.undisplaced_verts[:n_body], verts[out.body_only_idx], atol=1e-6)
        np.testing.assert_allclose(out.full_verts[:n_body], verts[out.body_only_idx], atol=1e-6)
        # Sculpted head/hair vertices differ from undisplaced base where nose/bun displacement is applied
        self.assertGreater(float(np.linalg.norm(out.full_verts - out.undisplaced_verts, axis=1).max()), 0.015)

        # Backwards-compatible default construction without optional fields
        legacy_out = HighResHeadOutput(
            full_verts=out.full_verts,
            full_faces=out.full_faces,
            head_vert_mask=out.head_vert_mask,
            hair_vert_mask=out.hair_vert_mask,
            face_skin_vert_mask=out.face_skin_vert_mask,
            front_face_vert_mask=out.front_face_vert_mask,
            warnings=[],
        )
        self.assertIsNone(legacy_out.undisplaced_verts)
        self.assertIsNone(legacy_out.body_only_idx)

    def test_build_hair_volume_shell_damps_excessive_area_and_bounds_inflation(self) -> None:
        from bodyscan_pipeline.head_hair import build_hair_volume_shell, partition_and_subdivide_head

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        part = partition_and_subdivide_head(verts, faces, joints, subdiv_steps=1)

        img_h, img_w = 240, 200
        px_per_m = 400.0
        cx = img_w / 2.0
        cy_skull = 30.0 + (1.72 - 1.60) * px_per_m
        r_px = int(round(0.105 * px_per_m))

        flooded_views = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            pm = np.zeros((img_h, img_w), dtype=bool)
            hm = np.zeros((img_h, img_w), dtype=bool)
            cv2.circle(pm.view(np.uint8), (int(cx), int(cy_skull)), r_px, 1, -1)
            # Rectangular 3.0x+ skull area flood box in every view
            hm[10:150, 25:175] = True
            pm |= hm
            flooded_views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=pm,
                    body_mask=pm & (~hm),
                    hair_mask=hm,
                    face_skin_mask=np.zeros((img_h, img_w), dtype=bool),
                    nose_profile_curve=None,
                    face_landmarks_ram=None,
                    collar_hair_overlap=False,
                    warnings=[],
                )
            )

        hair_v, _ = build_hair_volume_shell(
            cranium_verts=part.cranium_verts,
            cranium_faces=part.subdiv_faces,
            joints_a=joints,
            segmented_views=flooded_views,
            raw_results=[],
        )
        sc = np.array([0.0, 1.60, 0.0], dtype=np.float32)
        base_r = float(np.linalg.norm(part.cranium_verts - sc[None, :], axis=1).max())
        hair_r = float(np.linalg.norm(hair_v - sc[None, :], axis=1).max())
        # Both view_quality damping (0.15) and natural max_dr_per_vert (<=0.042) prevent blocky helmets
        self.assertLessEqual(hair_r - base_r, 0.045)

    def test_compute_vertex_normals_zero_area_uses_up_fallback(self) -> None:
        from bodyscan_pipeline.head_hair import _compute_vertex_normals

        verts = np.zeros((3, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        norms = _compute_vertex_normals(verts, faces)
        np.testing.assert_allclose(norms, np.tile([0.0, 1.0, 0.0], (3, 1)), atol=1e-6)

    def test_estimate_head_frames_uses_raw_cam_focal_when_hair_shrinks_body_mask(self) -> None:
        from bodyscan_pipeline.head_hair import _estimate_head_frames, partition_and_subdivide_head

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=20)
        part = partition_and_subdivide_head(verts, faces, joints, subdiv_steps=0)

        img_h, img_w = 320, 240
        pm = np.zeros((img_h, img_w), dtype=bool)
        pm[40:300, 90:150] = True
        hm = np.zeros((img_h, img_w), dtype=bool)
        # Hair mask covers upper 200px of mask so body_mask is shrunk to lower legs only
        hm[40:240, 90:150] = True
        view = SegmentedView(
            view_index=0,
            azimuth_deg=0.0,
            view_label="front",
            person_mask=pm,
            body_mask=pm & (~hm),
            hair_mask=hm,
            face_skin_mask=np.zeros((img_h, img_w), dtype=bool),
        )
        raw_results = [
            {
                "azimuth_deg": 0.0,
                "focal_length": 1000.0,
                "output": {"pred_cam_t": [0.0, 0.0, 2.5]},
            }
        ]
        _, frames = _estimate_head_frames(
            segmented_views=[view],
            head_verts=part.cranium_verts,
            joints_a=joints,
            raw_results=raw_results,
        )
        self.assertEqual(len(frames), 1)
        self.assertIsNotNone(frames[0])
        assert frames[0] is not None
        self.assertAlmostEqual(float(frames[0]["px_per_m"]), 400.0, delta=5.0)

    def test_build_hair_volume_shell_damps_medium_halo_and_caps_single_view_temple_spike(self) -> None:
        from bodyscan_pipeline.head_hair import build_hair_volume_shell, partition_and_subdivide_head

        verts, faces, joints = _build_synthetic_humanoid_mesh(n_lon=24)
        part = partition_and_subdivide_head(verts, faces, joints, subdiv_steps=1)

        img_h, img_w = 240, 200
        px_per_m = 400.0
        cx = img_w / 2.0
        cy_skull = 50.0 + (1.72 - 1.60) * px_per_m
        r_px = int(round(0.105 * px_per_m))

        views = []
        raw_results = []
        for idx, (label, az) in enumerate((("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))):
            pm = np.zeros((img_h, img_w), dtype=bool)
            hm = np.zeros((img_h, img_w), dtype=bool)
            cv2.circle(pm.view(np.uint8), (int(cx), int(cy_skull)), r_px, 1, -1)
            if idx == 0:
                # Single-view 35mm temple wing protrusion (~2.0x skull area) uncorroborated in other views
                cv2.circle(hm.view(np.uint8), (int(cx), int(cy_skull)), int(round(r_px * 1.42)), 1, -1)
            else:
                cv2.circle(hm.view(np.uint8), (int(cx), int(cy_skull)), r_px + 2, 1, -1)
            pm |= hm
            views.append(
                SegmentedView(
                    view_index=idx,
                    azimuth_deg=az,
                    view_label=label,
                    person_mask=pm,
                    body_mask=pm & (~hm),
                    hair_mask=hm,
                    face_skin_mask=np.zeros((img_h, img_w), dtype=bool),
                )
            )
            raw_results.append(
                {
                    "azimuth_deg": az,
                    "focal_length": 1000.0,
                    "output": {"pred_cam_t": [0.0, 0.0, 2.5]},
                }
            )

        hair_v, _ = build_hair_volume_shell(
            cranium_verts=part.cranium_verts,
            cranium_faces=part.subdiv_faces,
            joints_a=joints,
            segmented_views=views,
            raw_results=raw_results,
        )
        base_max_x = float(np.abs(part.cranium_verts[:, 0]).max())
        hair_max_x = float(np.abs(hair_v[:, 0]).max())
        # Uncorroborated single-view temple spike is strictly capped (<= 15mm + smoothing margin)
        self.assertLessEqual(hair_max_x - base_max_x, 0.017)


if __name__ == "__main__":
    unittest.main()
