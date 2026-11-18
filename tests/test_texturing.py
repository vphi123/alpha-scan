"""Unit tests for Step 3 UV unwrapping, multi-view texture baking, and privacy tiers."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch


class TestStep3Texturing(unittest.TestCase):
    """Test Step 3 UV parameterization, projection, baking, inpainting, and blurring."""

    def test_step3_config_constants_and_license(self) -> None:
        from bodyscan_pipeline import config

        self.assertEqual(config.DEFAULT_TEXTURE_SIZE, 2048)
        self.assertGreaterEqual(config.UV_GUTTER_DILATE_PX, 4)
        self.assertGreater(config.TEXTURE_MIN_VIEW_COS, 0.0)
        self.assertLess(config.TEXTURE_MIN_VIEW_COS, 0.5)
        self.assertIn("nvdiffrast", config.FORBIDDEN_MODULES)
        self.assertIn("ultralytics", config.FORBIDDEN_MODULES)
        self.assertIn("smplx", config.FORBIDDEN_MODULES)

    def test_build_head_face_vertex_mask_geometric(self) -> None:
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.texturing import build_head_face_vertex_mask

        # Synthetic humanoid vertex set from y=0 (feet) to y=1.70 (crown)
        verts = np.array(
            [
                [0.0, 0.50, 0.12],   # leg front (False)
                [0.0, 1.15, 0.14],   # chest front (False)
                [0.0, 1.40, 0.05],   # neck lower (False)
                [0.0, 1.56, 0.10],   # face nose/cheek (+Z) (True)
                [0.0, 1.62, -0.09],  # back of skull (-Z) (True)
                [0.0, 1.70, 0.00],   # crown (True)
            ],
            dtype=np.float32,
        )
        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.38, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.48, 0.0]

        head_mask, front_face_mask = build_head_face_vertex_mask(
            verts_a=verts,
            joints_a=joints,
            mhr_model=None,
        )
        self.assertEqual(head_mask.tolist(), [False, False, False, True, True, True])
        self.assertEqual(front_face_mask.tolist(), [False, False, False, True, False, False])

    def test_build_uv_layout_and_rasterize_opengl_orientation(self) -> None:
        from bodyscan_pipeline.texturing import build_uv_layout, rasterize_uv_atlas_map

        # Quad: v=1 at top edge (rows near 0), v=0 at bottom edge (rows near tex_size-1)
        verts = np.array(
            [
                [-1.0, -1.0, 0.0],  # 0: bottom-left (u=0, v=0)
                [1.0, -1.0, 0.0],   # 1: bottom-right (u=1, v=0)
                [1.0, 1.0, 0.0],    # 2: top-right (u=1, v=1)
                [-1.0, 1.0, 0.0],   # 3: top-left (u=0, v=1)
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        layout = build_uv_layout(verts=verts, faces=faces, mhr_uvs=uvs)
        self.assertEqual(layout.source, "mhr")
        self.assertEqual(layout.uv_faces.shape, (2, 3))
        self.assertEqual(len(layout.vmapping), 4)

        tri_map, bary_map = rasterize_uv_atlas_map(layout, tex_size=64)
        self.assertEqual(tri_map.shape, (64, 64))
        self.assertEqual(bary_map.shape, (64, 64, 3))
        valid = tri_map >= 0
        self.assertGreater(int(valid.sum()), 60 * 60)

        # Barycentric coordinates must sum to 1 and reconstruct pixel UV accurately
        bary_sums = bary_map[valid].sum(axis=-1)
        self.assertTrue(np.allclose(bary_sums, 1.0, atol=1e-4))

        # Top row pixel (row=4, col=4) must map to u near 0, v near 1 (GLB/OpenGL convention)
        f_top = int(tri_map[4, 4])
        uv_top = (bary_map[4, 4, :, None] * layout.uv[layout.uv_faces[f_top]]).sum(axis=0)
        self.assertLess(uv_top[0], 0.15)
        self.assertGreater(uv_top[1], 0.85)

        # Bottom row pixel (row=59, col=59) must map to u near 1, v near 0
        f_bot = int(tri_map[59, 59])
        uv_bot = (bary_map[59, 59, :, None] * layout.uv[layout.uv_faces[f_bot]]).sum(axis=0)
        self.assertGreater(uv_bot[0], 0.85)
        self.assertLess(uv_bot[1], 0.15)

    def test_extract_mhr_uvs_with_split_seams(self) -> None:
        from bodyscan_pipeline.texturing import build_uv_layout, extract_mhr_uvs

        class DummyMHR:
            uvs = np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.5]],
                dtype=np.float32,
            )
            uv_faces = np.array([[0, 1, 2], [4, 2, 3]], dtype=np.int64)

        verts = np.zeros((4, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        ext_uv, ext_uv_faces = extract_mhr_uvs(DummyMHR(), None, num_faces=2)
        self.assertIsNotNone(ext_uv)
        layout = build_uv_layout(verts, faces, mhr_uvs=ext_uv, mhr_uv_faces=ext_uv_faces)
        self.assertEqual(layout.source, "mhr")
        self.assertEqual(len(layout.vmapping), 5)
        self.assertEqual(int(layout.vmapping[4]), 0)

    def test_resolve_camera_sign_handles_flipped_y(self) -> None:
        from bodyscan_pipeline.texturing import project_points_pinhole, transform_vertices_to_camera

        k3d_cv = np.array(
            [
                [0.0, -0.5, 0.0],   # top in OpenCV (+Y down)
                [-0.2, 0.0, 0.05],  # left
                [0.2, 0.0, 0.05],   # right
                [-0.1, 0.6, 0.0],   # bottom-left
                [0.1, 0.6, 0.0],    # bottom-right
            ],
            dtype=np.float32,
        )
        cam_t_cv = np.array([0.05, -0.02, 3.0], dtype=np.float32)
        focal = 800.0
        w, h = 400, 600
        pts_cv = k3d_cv + cam_t_cv
        k2d_target = project_points_pinhole(pts_cv, focal_length=focal, img_width=w, img_height=h)

        # Case A: standard OpenCV input
        verts_cv_out = transform_vertices_to_camera(
            verts=k3d_cv,
            cam_t=cam_t_cv,
            focal_length=focal,
            img_width=w,
            img_height=h,
            kpts_3d=k3d_cv,
            kpts_2d=k2d_target,
        )
        proj_a = project_points_pinhole(verts_cv_out, focal, w, h)
        self.assertLess(float(np.median(np.linalg.norm(proj_a - k2d_target, axis=1))), 0.5)

        # Case B: MHR-style +Y up vertices & cam_t against OpenCV 2D keypoints
        k3d_yup = k3d_cv * np.array([1.0, -1.0, 1.0], dtype=np.float32)
        cam_t_yup = cam_t_cv * np.array([1.0, -1.0, 1.0], dtype=np.float32)
        verts_yup_out = transform_vertices_to_camera(
            verts=k3d_yup,
            cam_t=cam_t_yup,
            focal_length=focal,
            img_width=w,
            img_height=h,
            kpts_3d=k3d_yup,
            kpts_2d=k2d_target,
        )
        proj_b = project_points_pinhole(verts_yup_out, focal, w, h)
        self.assertLess(float(np.median(np.linalg.norm(proj_b - k2d_target, axis=1))), 0.5)

        # Case C: 3-column keypoints (x, y, conf) where invalid/off-frame detections carry conf=0
        k2d_with_conf = np.column_stack([k2d_target, np.ones(len(k2d_target), dtype=np.float32)])
        k2d_bad = np.vstack([k2d_with_conf, np.array([[0.0, 0.0, 0.0]] * 6, dtype=np.float32)])
        k3d_bad = np.vstack([k3d_yup, np.full((6, 3), 99.0, dtype=np.float32)])
        verts_conf_out = transform_vertices_to_camera(
            verts=k3d_yup,
            cam_t=cam_t_yup,
            focal_length=focal,
            img_width=w,
            img_height=h,
            kpts_3d=k3d_bad,
            kpts_2d=k2d_bad,
        )
        proj_c = project_points_pinhole(verts_conf_out, focal, w, h)
        self.assertLess(float(np.median(np.linalg.norm(proj_c - k2d_target, axis=1))), 0.5)

    def test_render_view_depth_buffer_occlusion(self) -> None:
        from bodyscan_pipeline.texturing import render_view_depth_buffer

        # Two overlapping quads facing camera at z=2.0 and z=2.5
        verts_cam = np.array(
            [
                [-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [0.2, 0.2, 2.0], [-0.2, 0.2, 2.0],  # front
                [-0.4, -0.4, 2.5], [0.4, -0.4, 2.5], [0.4, 0.4, 2.5], [-0.4, 0.4, 2.5],  # back
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
        zbuf, _ = render_view_depth_buffer(verts_cam, faces, focal_length=500.0, img_width=200, img_height=200)
        self.assertEqual(zbuf.shape, (200, 200))
        # Center occluded by z=2.0 quad; outer ring shows z=2.5 quad; corner is empty (<0)
        self.assertAlmostEqual(float(zbuf[100, 100]), 2.0, delta=0.02)
        self.assertAlmostEqual(float(zbuf[170, 100]), 2.5, delta=0.02)
        self.assertLess(float(zbuf[5, 5]), 0.0)

    def test_step3_multiview_bake_inpaint_closeup_and_blur(self) -> None:
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.texturing import ViewObservation, run_step3_texturing

        # Closed 3D box humanoid proxy: x in [-0.25, 0.25], y in [0.0, 1.6], z in [-0.15, 0.15]
        verts_a = np.array(
            [
                [-0.25, 0.0, 0.15], [0.25, 0.0, 0.15], [0.25, 1.6, 0.15], [-0.25, 1.6, 0.15],    # Front (+Z): 0,1,2,3
                [0.25, 0.0, -0.15], [-0.25, 0.0, -0.15], [-0.25, 1.6, -0.15], [0.25, 1.6, -0.15], # Back (-Z): 4,5,6,7
            ],
            dtype=np.float32,
        )
        # CCW outward winding from outside box
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],  # Front (+Z) - faces 0,1
                [4, 5, 6], [4, 6, 7],  # Back (-Z) - faces 2,3
                [5, 0, 3], [5, 3, 6],  # Left (-X) - faces 4,5 (unseen hole)
                [1, 4, 7], [1, 7, 2],  # Right (+X) - faces 6,7 (unseen hole)
                [3, 2, 7], [3, 7, 6],  # Top (+Y) - faces 8,9 (unseen hole)
                [5, 4, 1], [5, 1, 0],  # Bottom (-Y) - faces 10,11 (unseen hole)
            ],
            dtype=np.int64,
        )
        # Disjoint UV atlas patches inside [0.05, 0.95]^2 (OpenGL v=0 bottom, v=1 top)
        # Front occupies u in [0.05, 0.45], v in [0.10, 0.90]
        # Back occupies u in [0.55, 0.95], v in [0.10, 0.90]
        # Side patches squeeze at v in [0.02, 0.08]
        mhr_uvs = np.array(
            [
                [0.05, 0.10], [0.45, 0.10], [0.45, 0.90], [0.05, 0.90],  # 0..3 Front
                [0.55, 0.10], [0.95, 0.10], [0.95, 0.90], [0.55, 0.90],  # 4..7 Back
                [0.05, 0.02], [0.25, 0.02], [0.25, 0.06], [0.05, 0.06],  # 8..11 Side L (unused corner)
            ],
            dtype=np.float32,
        )
        uv_faces = faces.copy()
        uv_faces[4:] = np.array(
            [
                [8, 9, 10], [8, 10, 11],
                [8, 9, 10], [8, 10, 11],
                [8, 9, 10], [8, 10, 11],
                [8, 9, 10], [8, 10, 11],
            ],
            dtype=np.int64,
        )

        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.30, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.40, 0.0]

        # Build front photo (OpenCV camera looking at +Z torso shifted center to y=0.8, placed at z=2.5)
        w, h, focal = 300, 400, 450.0
        verts_front_cv = verts_a.copy()
        verts_front_cv[:, 1] = -(verts_a[:, 1] - 0.8)  # OpenCV +Y down
        verts_front_cv[:, 2] = -verts_a[:, 2]          # Front (+Z canonical) faces camera (-Z -> smaller Z)
        cam_t_front = np.array([0.0, 0.0, 2.5], dtype=np.float32)

        img_front = np.full((h, w, 3), 240, dtype=np.uint8)
        # Render torso red + head high-frequency stripe check pattern onto img_front
        # Top of front quad (y_a >= 1.4 -> y_cv <= -0.6 -> v_img ~ 450*(-0.6)/2.35 + 200 ~ 85)
        img_front[120:360, 80:220] = [210, 40, 40]  # red torso
        img_front[30:119, 80:220] = [230, 185, 155]  # skin head
        img_front[50:90:4, 110:190] = [20, 20, 20]    # high-frequency face detail stripes

        # Back view rotated 180 deg around Y so back (-Z canonical) faces camera
        verts_back_cv = verts_front_cv.copy()
        verts_back_cv[:, 0] = -verts_front_cv[:, 0]
        verts_back_cv[:, 2] = -verts_front_cv[:, 2]
        cam_t_back = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        img_back = np.full((h, w, 3), 240, dtype=np.uint8)
        img_back[30:360, 80:220] = [35, 90, 220]  # blue back

        obs_list = [
            ViewObservation(
                image_rgb=img_front,
                pred_vertices=verts_front_cv,
                pred_cam_t=cam_t_front,
                focal_length=focal,
                weight=1.0,
                view_name="front.jpg",
            ),
            ViewObservation(
                image_rgb=img_back,
                pred_vertices=verts_back_cv,
                pred_cam_t=cam_t_back,
                focal_length=focal,
                weight=1.0,
                view_name="back.jpg",
            ),
        ]

        out = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=obs_list,
            mhr_model=None,
            tex_size=256,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=uv_faces,
            prefer_pytorch3d=False,
        )
        self.assertEqual(out.tex_face_rgb.shape, (256, 256, 3))
        self.assertEqual(out.tex_blur_rgb.shape, (256, 256, 3))

        # Front torso sample (u=0.25 -> col=64, v=0.40 -> row=153) must be red
        torso_rgb = out.tex_face_rgb[153, 64]
        self.assertGreater(int(torso_rgb[0]), 160)
        self.assertLess(int(torso_rgb[2]), 90)

        # Back torso sample (u=0.75 -> col=191, v=0.40 -> row=153) must be blue
        back_rgb = out.tex_face_rgb[153, 191]
        self.assertGreater(int(back_rgb[2]), 160)
        self.assertLess(int(back_rgb[0]), 90)

        # Unseen side patch (row ~ 244, col ~ 38) must be inpainted non-zero
        side_rgb = out.tex_face_rgb[244, 38]
        self.assertGreater(int(side_rgb.sum()), 30)

        # Head blur: torso texels outside head_uv_mask remain identical, while head + adjacent gutter are smoothed
        body_texels = out.coverage_mask & (~out.head_uv_mask)
        self.assertTrue(np.any(body_texels))
        self.assertTrue(np.array_equal(out.tex_face_rgb[body_texels], out.tex_blur_rgb[body_texels]))
        self.assertTrue(np.array_equal(out.tex_face_rgb[120:, :], out.tex_blur_rgb[120:, :]))
        self.assertTrue(np.any(out.head_uv_mask))
        front_head_slice = (slice(26, 51), slice(30, 96))
        orig_grad = float(np.abs(np.diff(out.tex_face_rgb[front_head_slice].astype(np.float32), axis=0)).mean())
        blur_grad = float(np.abs(np.diff(out.tex_blur_rgb[front_head_slice].astype(np.float32), axis=0)).mean())
        self.assertLess(blur_grad, orig_grad * 0.35)

        # Closeup face view overrides head front face color cleanly without tinting torso
        img_closeup = img_front.copy()
        img_closeup[30:119, 80:220] = [60, 210, 90]  # bright green marker exclusively in closeup face view
        obs_closeup = obs_list + [
            ViewObservation(
                image_rgb=img_closeup,
                pred_vertices=verts_front_cv,
                pred_cam_t=cam_t_front,
                focal_length=focal,
                weight=1.0,
                is_closeup_face=True,
                view_name="closeup_face.jpg",
            )
        ]
        out_close = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=obs_closeup,
            mhr_model=None,
            tex_size=256,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=uv_faces,
            prefer_pytorch3d=False,
        )
        # Head texel near u=0.25, v=0.87 -> row=33, col=64 becomes green-dominant
        head_px = out_close.tex_face_rgb[33, 64]
        self.assertGreater(int(head_px[1]), int(head_px[0]))
        # Torso remains red
        self.assertGreater(int(out_close.tex_face_rgb[153, 64, 0]), 150)

    def test_build_view_observations_skips_outliers_and_json_metadata(self) -> None:
        from bodyscan_pipeline.exporter import build_body_scan_json
        from bodyscan_pipeline.fusion import FusedSubjectParams
        from bodyscan_pipeline.texturing import build_view_observations

        dummy_img = np.full((64, 48, 3), 120, dtype=np.uint8)
        raw = [
            {
                "image_name": "front.jpg",
                "image_rgb": dummy_img,
                "output": {
                    "pred_vertices": np.zeros((10, 3), dtype=np.float32),
                    "pred_cam_t": np.array([0.0, 0.0, 2.5], dtype=np.float32),
                    "focal_length": np.array([500.0], dtype=np.float32),
                },
            },
            {
                "image_name": "bad_outlier.jpg",
                "image_rgb": dummy_img,
                "output": {
                    "pred_vertices": np.zeros((10, 3), dtype=np.float32),
                    "pred_cam_t": np.array([0.0, 0.0, 2.5], dtype=np.float32),
                    "focal_length": 500.0,
                },
            },
        ]
        obs = build_view_observations(raw, weights=np.array([1.0, 0.0]))
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].view_name, "front.jpg")
        self.assertAlmostEqual(obs[0].focal_length, 500.0)

        fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45)),
            scale_params=torch.ones((1, 28)),
            weights=np.array([1.0, 0.0]),
            distances=np.array([0.0, 5.0]),
            images_used=1,
            images_total=2,
            mean_confidence=1.0,
        )
        payload = build_body_scan_json(
            person_id="p_tex",
            tier="alpha_scan_pro",
            sex="female",
            height_cm=162.0,
            weight_kg=50.0,
            circumferences_cm={"neck": 32.0},
            loft=[[0.3, 0.25, 0.18]],
            fused=fused,
            textures={
                "resolution": 2048,
                "uv_source": "xatlas",
                "levels": {
                    "1_plain": "scan.glb",
                    "2_blurred_face": "tex_blur.jpg",
                    "3_clear_face": "tex_face.jpg",
                },
            },
        )
        self.assertEqual(payload["textures"]["resolution"], 2048)
        self.assertEqual(payload["textures"]["levels"]["2_blurred_face"], "tex_blur.jpg")

    def test_secure_delete_files_writes_audit_log(self) -> None:
        from bodyscan_pipeline.exporter import secure_delete_files

        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "1.jpg")
            p2 = os.path.join(tmp, "2.jpg")
            for p in (p1, p2):
                with open(p, "wb") as f:
                    f.write(b"fakejpegdata")
            log_path = os.path.join(tmp, "deletion_audit.jsonl")

            entry = secure_delete_files([p1, p2], audit_log_path=log_path, actor="demo_ui")
            self.assertFalse(os.path.exists(p1))
            self.assertFalse(os.path.exists(p2))
            self.assertTrue(os.path.exists(log_path))
            self.assertEqual(entry["deleted_count"], 2)
            with open(log_path, "r", encoding="utf-8") as f:
                logged = json.loads(f.readline())
            self.assertEqual(logged["actor"], "demo_ui")
            self.assertEqual(logged["deleted_files"], ["1.jpg", "2.jpg"])

    def test_license_compliance_blocks_nvdiffrast(self) -> None:
        from bodyscan_pipeline.texturing import render_view_depth_buffer

        sys.modules["nvdiffrast"] = object()  # type: ignore[assignment]
        try:
            verts = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int64)
            with self.assertRaises(RuntimeError):
                render_view_depth_buffer(verts, faces, focal_length=500.0, img_width=100, img_height=100)
        finally:
            sys.modules.pop("nvdiffrast", None)

    def test_build_texture_metadata_respects_privacy_level(self) -> None:
        from bodyscan_pipeline.exporter import build_texture_metadata
        from bodyscan_pipeline.texturing import TexturedScanOutputs, UVLayout

        dummy_layout = UVLayout(
            uv=np.zeros((3, 2), dtype=np.float32),
            uv_faces=np.array([[0, 1, 2]], dtype=np.int64),
            vmapping=np.array([0, 1, 2], dtype=np.int64),
            orig_faces=np.array([[0, 1, 2]], dtype=np.int64),
            source="xatlas",
        )
        dummy_out = TexturedScanOutputs(
            uv_layout=dummy_layout,
            tex_face_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            tex_blur_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            head_uv_mask=np.ones((8, 8), dtype=bool),
            coverage_mask=np.ones((8, 8), dtype=bool),
            used_pytorch3d=False,
        )
        meta_lvl2 = build_texture_metadata(dummy_out, privacy_level=2)
        self.assertIn("2_blurred_face", meta_lvl2["levels"])
        self.assertNotIn("3_clear_face", meta_lvl2["levels"])
        meta_lvl3 = build_texture_metadata(dummy_out, privacy_level=3)
        self.assertIn("3_clear_face", meta_lvl3["levels"])

    def test_foreground_mask_rejects_background_wall_bleed(self) -> None:
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.texturing import ViewObservation, run_step3_texturing

        # Front/back closed torso box projecting onto col 80..220 in a 300x400 image
        verts_a = np.array(
            [
                [-0.25, 0.0, 0.15], [0.25, 0.0, 0.15], [0.25, 1.6, 0.15], [-0.25, 1.6, 0.15],
                [0.25, 0.0, -0.15], [-0.25, 0.0, -0.15], [-0.25, 1.6, -0.15], [0.25, 1.6, -0.15],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [5, 0, 3], [5, 3, 6],
                [1, 4, 7], [1, 7, 2],
                [3, 2, 7], [3, 7, 6],
                [5, 4, 1], [5, 1, 0],
            ],
            dtype=np.int64,
        )
        mhr_uvs = np.array(
            [
                [0.05, 0.10], [0.45, 0.10], [0.45, 0.90], [0.05, 0.90],
                [0.55, 0.10], [0.95, 0.10], [0.95, 0.90], [0.55, 0.90],
                [0.05, 0.02], [0.25, 0.02], [0.25, 0.06], [0.05, 0.06],
            ],
            dtype=np.float32,
        )
        uv_faces = faces.copy()
        uv_faces[4:] = np.array([[8, 9, 10], [8, 10, 11]] * 4, dtype=np.int64)

        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.30, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.40, 0.0]

        w, h, focal = 300, 400, 450.0
        verts_front_cv = verts_a.copy()
        verts_front_cv[:, 1] = -(verts_a[:, 1] - 0.8)
        verts_front_cv[:, 2] = -verts_a[:, 2]
        cam_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)

        # Torso projects onto col ~102..198; bright magenta wall overlaps outer ~30% of torso edges
        img_front = np.full((h, w, 3), [255, 0, 255], dtype=np.uint8)
        img_front[70:330, 116:184] = [205, 55, 45]
        fg_mask = np.zeros((h, w), dtype=bool)
        fg_mask[70:330, 116:184] = True

        obs = [
            ViewObservation(
                image_rgb=img_front,
                pred_vertices=verts_front_cv,
                pred_cam_t=cam_t,
                focal_length=focal,
                foreground_mask=fg_mask,
                mask_erode_px=2,
                view_name="front.jpg",
            )
        ]

        out = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=obs,
            tex_size=128,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=uv_faces,
            prefer_pytorch3d=False,
        )
        magenta_hits = (
            (out.tex_face_rgb[..., 0] > 180)
            & (out.tex_face_rgb[..., 1] < 80)
            & (out.tex_face_rgb[..., 2] > 180)
        )
        self.assertEqual(int(magenta_hits.sum()), 0)
        # Eroded inner torso UV center (u=0.25 -> col=32, v=0.50 -> row=64) keeps clean red RGB
        center_px = out.tex_face_rgb[64, 32]
        self.assertGreater(int(center_px[0]), 160)
        self.assertLess(int(center_px[2]), 80)
        # Outer torso edge UV (u=0.07 -> col=9, v=0.50 -> row=64) fell outside foreground_mask -> uncovered before inpaint
        self.assertFalse(bool(out.coverage_mask[64, 9]))

    def test_strict_no_invent_skips_telea_on_unobserved_facial_texels(self) -> None:
        from unittest import mock

        import cv2
        from bodyscan_pipeline.texturing import inpaint_and_dilate_texture

        tri_id_map = np.full((64, 64), -1, dtype=np.int32)
        tri_id_map[8:56, 8:56] = 0

        no_inpaint_mask = np.zeros((64, 64), dtype=bool)
        no_inpaint_mask[16:48, 16:48] = True

        accum_rgb = np.zeros((64, 64, 3), dtype=np.float32)
        accum_w = np.zeros((64, 64), dtype=np.float32)

        # Red marker left of nose/eye gap; blue marker right of gap; [16:48, 28:36] left unobserved
        accum_rgb[16:48, 16:28] = [230.0, 20.0, 20.0]
        accum_w[16:48, 16:28] = 1.0
        accum_rgb[16:48, 36:48] = [20.0, 20.0, 230.0]
        accum_w[16:48, 36:48] = 1.0

        # Non-face region has observed torso pixels and an unobserved crevice to fill
        accum_rgb[48:56, 8:32] = [190.0, 160.0, 140.0]
        accum_w[48:56, 8:32] = 1.0

        with mock.patch("cv2.inpaint", wraps=cv2.inpaint) as spy_inpaint:
            tex_strict = inpaint_and_dilate_texture(
                accum_rgb,
                accum_w,
                tri_id_map,
                dilate_px=4,
                strict_no_invent=True,
                no_inpaint_mask=no_inpaint_mask,
            )
            spy_inpaint.assert_not_called()

        # Protected facial crevice is smoothly filled from observed neighbor facial texels (zero [0,0,0] black holes, zero TELEA)
        gap_patch = tex_strict[16:48, 28:36]
        self.assertEqual(int((gap_patch.sum(axis=-1) == 0).sum()), 0)
        # Left side leans red, right side leans blue, and observed borders remain 100% exact
        self.assertGreater(float(gap_patch[:, :2, 0].mean()), float(gap_patch[:, :2, 2].mean()) + 30.0)
        self.assertGreater(float(gap_patch[:, -2:, 2].mean()), float(gap_patch[:, -2:, 0].mean()) + 30.0)
        np.testing.assert_array_equal(tex_strict[16:48, 16:28], np.broadcast_to(np.array([230, 20, 20], dtype=np.uint8), (32, 12, 3)))
        np.testing.assert_array_equal(tex_strict[16:48, 36:48], np.broadcast_to(np.array([20, 20, 230], dtype=np.uint8), (32, 12, 3)))
        # Non-face unobserved region is cleanly filled from observed non-face median/neighbor
        self.assertGreater(int(tex_strict[52, 45].sum()), 100)

        # Contrast with legacy strict_no_invent=False where TELEA fills the facial gap
        tex_legacy = inpaint_and_dilate_texture(
            accum_rgb,
            accum_w,
            tri_id_map,
            dilate_px=4,
            strict_no_invent=False,
        )
        self.assertGreater(int(tex_legacy[32, 31].sum()), 30)

    def test_level2_blur_isolates_face_skin_and_preserves_hair(self) -> None:
        from bodyscan_pipeline.texturing import generate_blurred_face_texture

        tri_id_map = np.full((128, 128), -1, dtype=np.int32)
        tri_id_map[8:120, 8:120] = 0
        head_uv_mask = tri_id_map >= 0

        hair_uv = np.zeros((128, 128), dtype=bool)
        hair_uv[8:56, 8:120] = True
        face_uv = np.zeros((128, 128), dtype=bool)
        face_uv[56:120, 8:120] = True

        tex_face = np.zeros((128, 128, 3), dtype=np.uint8)
        # High-frequency alternating dark/light hair stripes in hair_uv bordering face_uv at row 56
        for r in range(8, 56):
            tex_face[r, 8:120] = [235, 215, 190] if (r % 4 < 2) else [25, 20, 18]
        # High-frequency facial detail in face_uv
        for r in range(56, 120):
            for c in range(8, 120):
                tex_face[r, c] = [225, 170, 140] if ((r + c) % 4 < 2) else [35, 30, 30]

        tex_blur = generate_blurred_face_texture(
            tex_face,
            head_uv_mask,
            tri_id_map,
            blur_target_mask=face_uv,
            preserve_sharp_mask=hair_uv,
        )

        # Hair UV texels remain 100% bit-identical to unblurred tex_face (3b giữ rõ)
        self.assertTrue(np.array_equal(tex_blur[hair_uv], tex_face[hair_uv]))
        # Facial skin UV variance drops strongly in Level 2
        orig_std = float(np.std(tex_face[face_uv].astype(np.float32)))
        blur_std = float(np.std(tex_blur[face_uv].astype(np.float32)))
        self.assertLess(blur_std, orig_std * 0.35)

    def test_dual_material_head_body_texturing_outputs_coverage_stats(self) -> None:
        import struct

        from bodyscan_pipeline.config import HEAD_TEXTURE_SIZE, NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.exporter import (
            build_body_scan_json,
            build_texture_metadata,
            export_textured_scan_artifacts,
        )
        from bodyscan_pipeline.fusion import FusedSubjectParams
        from bodyscan_pipeline.texturing import ViewObservation, run_step3_texturing

        # Partitioned synthetic humanoid: 8-vert body box (y in [0.0, 1.25]) + 8-vert head box (y in [1.35, 1.70])
        # Head top (y=1.70) = hair; head bottom/front (y=1.35..1.55) = face_skin
        verts_a = np.array(
            [
                # Body: 0..7
                [-0.22, 0.0, 0.14], [0.22, 0.0, 0.14], [0.22, 1.25, 0.14], [-0.22, 1.25, 0.14],
                [0.22, 0.0, -0.14], [-0.22, 0.0, -0.14], [-0.22, 1.25, -0.14], [0.22, 1.25, -0.14],
                # Head lower/face ring (y=1.35): 8..11; Head upper/hair crown ring (y=1.70): 12..15
                [-0.11, 1.35, 0.12], [0.11, 1.35, 0.12], [0.11, 1.35, -0.12], [-0.11, 1.35, -0.12],
                [-0.11, 1.70, 0.12], [0.11, 1.70, 0.12], [0.11, 1.70, -0.12], [-0.11, 1.70, -0.12],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                # Body box (faces 0..11)
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [5, 0, 3], [5, 3, 6],
                [1, 4, 7], [1, 7, 2],
                [3, 2, 7], [3, 7, 6],
                [5, 4, 1], [5, 1, 0],
                # Head box (faces 12..23)
                [8, 9, 13], [8, 13, 12],     # Front (+Z)
                [10, 11, 15], [10, 15, 14],  # Back (-Z)
                [11, 8, 12], [11, 12, 15],   # Left (-X)
                [9, 10, 14], [9, 14, 13],    # Right (+X)
                [12, 13, 14], [12, 14, 15],  # Top crown (+Y)
                [11, 10, 9], [11, 9, 8],     # Bottom jaw/chin (-Y)
            ],
            dtype=np.int64,
        )
        # Disjoint UV coords per body and head primitives (v=0 bottom, v=1 top)
        mhr_uvs = np.array(
            [
                [0.05, 0.10], [0.45, 0.10], [0.45, 0.90], [0.05, 0.90],  # 0..3 body front
                [0.55, 0.10], [0.95, 0.10], [0.95, 0.90], [0.55, 0.90],  # 4..7 body back
                [0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45],  # 8..11 body side
                [0.10, 0.10], [0.45, 0.10], [0.55, 0.10], [0.90, 0.10],  # 12..15 head lower (face)
                [0.10, 0.90], [0.45, 0.90], [0.55, 0.90], [0.90, 0.90],  # 16..19 head upper (hair)
                [0.10, 0.50], [0.45, 0.50], [0.55, 0.50], [0.90, 0.50],  # 20..23 head mid/cap
            ],
            dtype=np.float32,
        )
        uv_faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [8, 9, 10], [8, 10, 11],
                [8, 9, 10], [8, 10, 11],
                [8, 9, 10], [8, 10, 11],
                [8, 9, 10], [8, 10, 11],
                # Head UV faces
                [12, 13, 17], [12, 17, 16],  # front
                [14, 15, 19], [14, 19, 18],  # back
                [20, 21, 17], [20, 17, 16],  # left
                [22, 23, 19], [22, 19, 18],  # right
                [16, 17, 19], [16, 19, 18],  # top hair cap
                [12, 13, 14], [12, 14, 15],  # bottom
            ],
            dtype=np.int64,
        )

        # Subdivide head vertical faces so hair (y>=1.55) and face skin (y<=1.50) sit on distinct triangles
        # Actually use vertex masks directly: crown 12..15 are hair, lower front 8..9 are face_skin, back 10..11 are hair
        head_vert_mask = np.zeros(len(verts_a), dtype=bool)
        head_vert_mask[8:] = True
        hair_vert_mask = np.zeros(len(verts_a), dtype=bool)
        hair_vert_mask[12:] = True
        hair_vert_mask[10:12] = True
        face_skin_vert_mask = np.zeros(len(verts_a), dtype=bool)
        face_skin_vert_mask[8:10] = True

        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.28, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.35, 0.0]

        w, h, focal = 300, 400, 450.0
        verts_front_cv = verts_a.copy()
        verts_front_cv[:, 1] = -(verts_a[:, 1] - 0.85)
        verts_front_cv[:, 2] = -verts_a[:, 2]
        verts_back_cv = verts_front_cv.copy()
        verts_back_cv[:, 0] = -verts_front_cv[:, 0]
        verts_back_cv[:, 2] = -verts_front_cv[:, 2]
        cam_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)

        img_front = np.full((h, w, 3), 240, dtype=np.uint8)
        img_front[110:360, 80:220] = [190, 45, 45]  # torso
        img_front[35:105, 110:190] = [225, 180, 150]  # face skin
        for r in range(35, 105, 2):
            img_front[r, 110:190] = [25, 25, 25]  # hair/face high-freq detail
        fg_front = np.zeros((h, w), dtype=bool)
        fg_front[30:365, 75:225] = True

        img_back = np.full((h, w, 3), 240, dtype=np.uint8)
        img_back[35:360, 80:220] = [40, 85, 210]
        fg_back = fg_front.copy()

        from types import SimpleNamespace

        from bodyscan_pipeline.texturing import build_view_observations

        obs = build_view_observations(
            raw_results=[
                {
                    "image_name": "front.jpg",
                    "image_rgb": img_front,
                    "output": {"pred_vertices": verts_front_cv, "pred_cam_t": cam_t, "focal_length": focal},
                },
                {
                    "image_name": "back.jpg",
                    "image_rgb": img_back,
                    "output": {"pred_vertices": verts_back_cv, "pred_cam_t": cam_t, "focal_length": focal},
                },
            ],
            segmented_views=[
                SimpleNamespace(view_index=0, person_mask=fg_front, hair_mask=None, face_skin_mask=None),
                SimpleNamespace(view_index=1, person_mask=fg_back, hair_mask=None, face_skin_mask=None),
            ],
        )
        self.assertEqual(len(obs), 2)
        self.assertIsNotNone(obs[0].foreground_mask)

        out = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=obs,
            tex_size=128,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=uv_faces,
            prefer_pytorch3d=False,
            head_vert_mask=head_vert_mask,
            hair_vert_mask=hair_vert_mask,
            face_skin_vert_mask=face_skin_vert_mask,
            dual_material=True,
        )

        self.assertIsNotNone(out.head_uv_layout)
        self.assertIsNotNone(out.head_tex_face_rgb)
        self.assertIsNotNone(out.head_tex_blur_rgb)
        self.assertEqual(out.head_tex_face_rgb.shape, (HEAD_TEXTURE_SIZE, HEAD_TEXTURE_SIZE, 3))
        self.assertIsNotNone(out.face_blur_uv_mask)
        self.assertIsNotNone(out.hair_uv_mask)
        self.assertIsNotNone(out.head_coverage_mask)
        self.assertTrue(np.any(out.hair_uv_mask))
        self.assertTrue(np.any(out.face_blur_uv_mask))
        self.assertFalse(bool(np.any(out.hair_uv_mask & out.face_blur_uv_mask)))
        self.assertTrue(np.array_equal(out.head_tex_blur_rgb[out.hair_uv_mask], out.head_tex_face_rgb[out.hair_uv_mask]))
        obs_face = out.head_coverage_mask & out.face_blur_uv_mask
        self.assertTrue(np.any(obs_face))
        face_orig_std = float(np.std(out.head_tex_face_rgb[obs_face].astype(np.float32)))
        face_blur_std = float(np.std(out.head_tex_blur_rgb[obs_face].astype(np.float32)))
        self.assertLess(face_blur_std, face_orig_std * 0.50)
        self.assertGreater(out.coverage_pct, 0.0)
        self.assertGreater(out.head_coverage_pct, 0.0)

        meta = build_texture_metadata(out, privacy_level=3)
        self.assertEqual(meta["materials"], ["body", "head"])
        self.assertGreater(meta["coverage_pct"], 0.0)
        self.assertGreater(meta["head_coverage_pct"], 0.0)

        fused = FusedSubjectParams(
            shape_params=torch.zeros((1, 45)),
            scale_params=torch.ones((1, 28)),
            weights=np.array([0.5, 0.5]),
            distances=np.array([0.0, 0.0]),
            images_used=2,
            images_total=2,
            mean_confidence=1.0,
        )
        body_json = build_body_scan_json(
            person_id="p_dual",
            tier="alpha_scan_pro",
            sex="female",
            height_cm=170.0,
            weight_kg=55.0,
            circumferences_cm={"neck": 32.0},
            loft=[[0.3, 0.25, 0.18]],
            fused=fused,
            textures=meta,
            core_mesh_filename="scan_core.glb",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = export_textured_scan_artifacts(
                verts_a=verts_a,
                body_json=body_json,
                output_dir=tmp,
                tex_outputs=out,
                export_preview_glb=True,
                privacy_level=3,
                core_verts=verts_a[:8],
                core_faces=faces[:12],
                core_mesh_filename="scan_core.glb",
            )
            self.assertIn("glb", artifacts)
            self.assertIn("glb_preview_blur", artifacts)
            self.assertIn("glb_preview_face", artifacts)
            self.assertIn("core_glb", artifacts)
            self.assertTrue(os.path.exists(artifacts["core_glb"]))

            with open(artifacts["glb_preview_face"], "rb") as fp:
                glb_raw = fp.read()
            magic, version, total_len = struct.unpack_from("<III", glb_raw, 0)
            self.assertEqual(magic, 0x46546C67)
            self.assertEqual(version, 2)
            self.assertEqual(total_len, len(glb_raw))
            j_len, _ = struct.unpack_from("<II", glb_raw, 12)
            gltf = json.loads(glb_raw[20 : 20 + j_len].decode("utf-8").rstrip(" \x00"))
            self.assertEqual(len(gltf["meshes"][0]["primitives"]), 2)
            self.assertEqual([m["name"] for m in gltf["materials"]], ["body", "head"])
            self.assertEqual(len(gltf["images"]), 2)

            with open(artifacts["core_glb"], "rb") as fp:
                core_raw = fp.read()
            c_magic, c_version, c_len = struct.unpack_from("<III", core_raw, 0)
            self.assertEqual(c_magic, 0x46546C67)
            self.assertEqual(c_version, 2)
            self.assertEqual(c_len, len(core_raw))

    def test_transform_vertices_to_camera_handles_y_z_negation(self) -> None:
        from bodyscan_pipeline.texturing import project_points_pinhole, transform_vertices_to_camera

        kpts_3d = np.array(
            [
                [-0.20, -0.45, -0.05],
                [0.20, -0.45, 0.02],
                [0.00, 0.15, -0.04],
                [-0.15, 0.55, 0.05],
                [0.18, 0.50, -0.08],
            ],
            dtype=np.float32,
        )
        cam_t = np.array([0.05, -0.03, 3.20], dtype=np.float32)
        focal, img_w, img_h = 360.0, 240, 320
        true_cam_k = (kpts_3d * np.array([1.0, -1.0, -1.0], dtype=np.float32)) + cam_t.reshape(1, 3)
        kpts_2d = project_points_pinhole(true_cam_k, focal, img_w, img_h)

        verts = np.array([[0.10, -0.25, -0.10], [-0.15, 0.40, 0.05]], dtype=np.float32)
        expected_verts = (verts * np.array([1.0, -1.0, -1.0], dtype=np.float32)) + cam_t.reshape(1, 3)
        recovered = transform_vertices_to_camera(
            verts=verts,
            cam_t=cam_t,
            focal_length=focal,
            img_width=img_w,
            img_height=img_h,
            kpts_3d=kpts_3d,
            kpts_2d=kpts_2d,
        )
        self.assertTrue(np.allclose(recovered, expected_verts, atol=1e-4))

    def test_build_view_observations_fitted_cameras_and_view_verts(self) -> None:
        from types import SimpleNamespace
        from bodyscan_pipeline.texturing import (
            build_head_face_vertex_mask,
            build_view_observations,
            run_step3_texturing,
        )

        # Build simple front/back closed sphere mesh centered around y=0.85m
        n_lat, n_lon = 10, 14
        pts = [[0.0, 1.70, 0.0]]
        for i in range(1, n_lat):
            phi = np.pi * i / n_lat
            y = 0.85 + 0.85 * np.cos(phi)
            rad = 0.24 * np.sin(phi)
            for j in range(n_lon):
                th = 2.0 * np.pi * j / n_lon
                pts.append([rad * np.cos(th), y, rad * np.sin(th)])
        bot_idx = len(pts)
        pts.append([0.0, 0.0, 0.0])
        verts_a = np.asarray(pts, dtype=np.float32)

        faces_l = []
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces_l.append([0, 1 + jn, 1 + j])
        for i in range(n_lat - 2):
            r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                faces_l.append([r0 + j, r0 + jn, r1 + jn])
                faces_l.append([r0 + j, r1 + jn, r1 + j])
        rb = 1 + (n_lat - 2) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces_l.append([rb + j, rb + jn, bot_idx])
        faces = np.asarray(faces_l, dtype=np.int64)

        joints = np.zeros((127, 3), dtype=np.float32)
        joints[110] = [0.0, 1.38, 0.0]
        joints[113] = [0.0, 1.50, 0.0]
        uvs = np.stack([(verts_a[:, 0] + 0.3) / 0.6, verts_a[:, 1] / 1.70], axis=-1).astype(np.float32)

        head_mask, skin_mask = build_head_face_vertex_mask(verts_a, joints)
        hair_mask = head_mask & (~skin_mask)
        r_front = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        r_back = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        t_cam = np.array([0.0, 0.85, 2.80], dtype=np.float32)
        focal = 360.0
        fitted_cameras = [(r_front, t_cam, focal), (r_back, t_cam, focal)]
        fitted_view_verts = [verts_a.copy(), verts_a.copy()]

        h, w = 240, 180
        img_f = np.full((h, w, 3), 235, dtype=np.uint8)
        img_f[35:205, 50:130] = [190, 75, 55]
        img_f[35:75, 65:115] = [30, 30, 30]
        img_b = np.full((h, w, 3), 235, dtype=np.uint8)
        img_b[35:205, 50:130] = [45, 95, 195]
        fg = np.zeros((h, w), dtype=bool)
        fg[30:210, 45:135] = True

        obs = build_view_observations(
            raw_results=[
                {
                    "image_name": "front.jpg",
                    "image_rgb": img_f,
                    "output": {"pred_vertices": np.zeros((2, 3), dtype=np.float32), "pred_cam_t": np.zeros(3), "focal_length": 100.0},
                },
                {
                    "image_name": "back.jpg",
                    "image_rgb": img_b,
                    "output": {"pred_vertices": np.zeros((2, 3), dtype=np.float32), "pred_cam_t": np.zeros(3), "focal_length": 100.0},
                },
            ],
            segmented_views=[
                SimpleNamespace(view_index=0, person_mask=fg, hair_mask=None, face_skin_mask=None),
                SimpleNamespace(view_index=1, person_mask=fg, hair_mask=None, face_skin_mask=None),
            ],
            fitted_cameras=fitted_cameras,
            fitted_view_verts=fitted_view_verts,
        )
        self.assertEqual(len(obs), 2)
        self.assertTrue(obs[0].calibrated_camera)
        self.assertEqual(obs[0].focal_length, focal)
        self.assertTrue(np.allclose(obs[0].pred_cam_t, t_cam))
        self.assertTrue(np.allclose(obs[0].pred_vertices, verts_a @ r_front.T))
        self.assertTrue(np.allclose(obs[1].pred_vertices, verts_a @ r_back.T))

        tex = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=obs,
            tex_size=96,
            mhr_uvs=uvs,
            mhr_uv_faces=faces.copy(),
            prefer_pytorch3d=False,
            head_vert_mask=head_mask,
            hair_vert_mask=hair_mask,
            face_skin_vert_mask=skin_mask,
            dual_material=True,
        )
        self.assertGreater(tex.coverage_pct, 0.0)
        self.assertTrue(np.array_equal(tex.head_tex_blur_rgb[tex.hair_uv_mask], tex.head_tex_face_rgb[tex.hair_uv_mask]))

    def test_strict_no_invent_bimodal_body_island_preserves_skin_pocket_color(self) -> None:
        from unittest.mock import patch
        import cv2
        from bodyscan_pipeline.texturing import inpaint_and_dilate_texture

        h = w = 128
        tri_id_map = np.full((h, w), -1, dtype=np.int32)
        tri_id_map[8:120, 8:120] = 0  # single connected UV island

        accum_rgb = np.zeros((h, w, 3), dtype=np.float32)
        accum_w = np.zeros((h, w), dtype=np.float32)
        # 75% dark black leggings in lower rows, 25% warm orange skin in top rows
        accum_rgb[8:42, 8:120] = np.array([205.0, 140.0, 100.0], dtype=np.float32)
        accum_w[8:42, 8:120] = 1.0
        accum_rgb[42:120, 8:120] = np.array([15.0, 15.0, 18.0], dtype=np.float32)
        accum_w[42:120, 8:120] = 1.0

        # Carve an interior 16x16 cavity inside the warm skin zone (>4 px deep) + protected facial hole
        accum_rgb[16:32, 52:68] = 0.0
        accum_w[16:32, 52:68] = 0.0
        protected = np.zeros((h, w), dtype=bool)
        accum_rgb[12:20, 16:24] = 0.0
        accum_w[12:20, 16:24] = 0.0
        protected[12:20, 16:24] = True

        with patch("cv2.inpaint", side_effect=AssertionError("cv2.inpaint must stay disabled in strict_no_invent")):
            out = inpaint_and_dilate_texture(
                accum_rgb=accum_rgb,
                accum_w=accum_w,
                tri_id_map=tri_id_map,
                dilate_px=2,
                strict_no_invent=True,
                no_inpaint_mask=protected,
            )

        # Protected hole surrounded by warm skin fills smoothly from observed neighbors (zero [0,0,0] black hole)
        self.assertGreater(float(out[12:20, 16:24, 0].mean()), 160.0)
        self.assertEqual(int((out[12:20, 16:24].sum(axis=-1) == 0).sum()), 0)
        # Cavity surrounded by warm skin must stay warm skin, not global black-pants median
        self.assertGreater(float(out[18:30, 54:66, 0].mean()), 160.0)
        self.assertLess(float(out[18:30, 54:66, 2].mean()), 130.0)

    def test_resolution_adaptive_distance_feather_rejects_5px_gym_halo(self) -> None:
        import cv2
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.texturing import ViewObservation, run_step3_texturing

        verts_a = np.array(
            [
                [-0.25, 0.0, 0.15], [0.25, 0.0, 0.15], [0.25, 1.6, 0.15], [-0.25, 1.6, 0.15],
                [0.25, 0.0, -0.15], [-0.25, 0.0, -0.15], [-0.25, 1.6, -0.15], [0.25, 1.6, -0.15],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [5, 0, 3], [5, 3, 6],
                [1, 4, 7], [1, 7, 2],
                [3, 2, 7], [3, 7, 6],
                [5, 4, 1], [5, 1, 0],
            ],
            dtype=np.int64,
        )
        mhr_uvs = np.array(
            [
                [0.05, 0.10], [0.45, 0.10], [0.45, 0.90], [0.05, 0.90],
                [0.55, 0.10], [0.95, 0.10], [0.95, 0.90], [0.55, 0.90],
                [0.05, 0.02], [0.25, 0.02], [0.25, 0.06], [0.05, 0.06],
            ],
            dtype=np.float32,
        )
        uv_faces = faces.copy()
        uv_faces[4:] = np.array([[8, 9, 10], [8, 10, 11]] * 4, dtype=np.int64)
        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.30, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.40, 0.0]

        img_h, img_w, focal = 1200, 800, 1350.0
        theta = np.deg2rad(65.0)
        rot_y = np.array(
            [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]],
            dtype=np.float32,
        )
        verts_rot = (verts_a - np.array([0.0, 0.8, 0.0], dtype=np.float32)) @ rot_y.T
        verts_rot_cv = verts_rot.copy()
        verts_rot_cv[:, 1] = -verts_rot[:, 1]
        verts_rot_cv[:, 2] = -verts_rot[:, 2]
        cam_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)

        u_v = focal * verts_rot_cv[:, 0] / (verts_rot_cv[:, 2] + 2.5) + img_w / 2.0
        v_v = focal * verts_rot_cv[:, 1] / (verts_rot_cv[:, 2] + 2.5) + img_h / 2.0
        u0, u1 = int(np.ceil(u_v.min())), int(np.floor(u_v.max()))
        v0, v1 = int(np.ceil(v_v.min())), int(np.floor(v_v.max()))

        img_gym = np.full((img_h, img_w, 3), [245, 245, 245], dtype=np.uint8)
        img_gym[850:1150, :] = [20, 110, 230]
        img_gym[v0 + 5 : v1 - 5, u0 + 5 : u1 - 5] = [190, 60, 50]
        fg_halo5 = np.zeros((img_h, img_w), dtype=bool)
        fg_halo5[max(0, v0) : min(img_h, v1), max(0, u0) : min(img_w, u1)] = True
        fg_halo5 = cv2.dilate(fg_halo5.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

        out = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=[
                ViewObservation(
                    image_rgb=img_gym,
                    pred_vertices=verts_rot_cv,
                    pred_cam_t=cam_t,
                    focal_length=focal,
                    foreground_mask=fg_halo5,
                    view_name="right.jpg",
                )
            ],
            tex_size=256,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=uv_faces,
            prefer_pytorch3d=False,
        )
        cov = out.coverage_mask
        self.assertTrue(np.any(cov))
        white_bleed = cov & (out.tex_face_rgb[..., 0] > 215) & (out.tex_face_rgb[..., 1] > 215) & (out.tex_face_rgb[..., 2] > 215)
        blue_bleed = cov & (out.tex_face_rgb[..., 2] > 180) & (out.tex_face_rgb[..., 0] < 80)
        bleed_frac = float(int(np.sum(white_bleed | blue_bleed))) / float(max(1, int(cov.sum())))
        self.assertLess(bleed_frac, 0.005)

    def test_side_view_hanging_arm_occlusion_guard_prevents_torso_skin_bleed(self) -> None:
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.texturing import build_view_observations, run_step3_texturing

        verts_a = np.array(
            [
                [-0.22, 0.0, 0.14], [0.22, 0.0, 0.14], [0.22, 1.50, 0.14], [-0.22, 1.50, 0.14],
                [0.22, 0.0, -0.14], [-0.22, 0.0, -0.14], [-0.22, 1.50, -0.14], [0.22, 1.50, -0.14],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],     # front
                [4, 5, 6], [4, 6, 7],     # back
                [5, 0, 3], [5, 3, 6],     # left (-X)
                [1, 4, 7], [1, 7, 2],     # right (+X)
                [3, 2, 7], [3, 7, 6],
                [5, 4, 1], [5, 1, 0],
            ],
            dtype=np.int64,
        )
        mhr_uvs = np.array(
            [
                [0.05, 0.55], [0.45, 0.55], [0.45, 0.95], [0.05, 0.95],
                [0.55, 0.55], [0.95, 0.55], [0.95, 0.95], [0.55, 0.95],
                [0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45],
            ],
            dtype=np.float32,
        )
        uv_faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [8, 9, 10], [8, 10, 11],
                [8, 9, 10], [8, 10, 11],
                [0, 1, 2], [0, 2, 3],
                [0, 1, 2], [0, 2, 3],
            ],
            dtype=np.int64,
        )
        joints = np.zeros((127, 3), dtype=np.float32)
        joints[75] = [0.22, 1.22, 0.0]
        joints[39] = [-0.22, 1.22, 0.0]
        joints[2] = [0.12, 0.72, 0.0]
        joints[18] = [-0.12, 0.72, 0.0]
        joints[NECK_BASE_JOINT] = [0.0, 1.25, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.35, 0.0]

        h, w, focal = 400, 300, 450.0
        cam_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        verts_side = verts_a - np.array([0.0, 0.75, 0.0], dtype=np.float32)
        # Rotate +90 deg so right (+X) face faces camera directly
        r_right = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
        verts_side_cv = verts_side.copy()
        verts_side_cv[:, 1] = -verts_side[:, 1]
        verts_side_cv[:, 2] = -verts_side[:, 2]

        img_side = np.full((h, w, 3), 240, dtype=np.uint8)
        img_side[60:340, 105:195] = [20, 20, 25]  # dark crop top on side torso
        # Bright orange hanging arm crossing vertically through middle of torso
        img_side[105:275, 136:164] = [225, 145, 95]

        # Build MHR-70 keypoints indicating side-view hanging arm over torso (sh=(5,6), el=(7,8), wr=(62,41), hip=(9,10))
        k70 = np.zeros((70, 3), dtype=np.float32)
        k70[:, 2] = 1.0
        k70[5] = [150.0, 95.0, 1.0]
        k70[6] = [152.0, 95.0, 1.0]
        k70[7] = [150.0, 180.0, 1.0]
        k70[8] = [152.0, 180.0, 1.0]
        k70[62] = [150.0, 265.0, 1.0]
        k70[41] = [152.0, 265.0, 1.0]
        k70[9] = [148.0, 255.0, 1.0]
        k70[10] = [154.0, 255.0, 1.0]

        obs = build_view_observations(
            raw_results=[
                {
                    "image_name": "02_right.jpg",
                    "image_rgb": img_side,
                    "output": {
                        "pred_vertices": verts_side_cv,
                        "pred_cam_t": cam_t,
                        "focal_length": focal,
                        "pred_keypoints_2d": k70,
                    },
                }
            ],
            fitted_cameras=[(r_right, cam_t, focal)],
            fitted_view_verts=[verts_side],
        )
        self.assertEqual(len(obs), 1)
        self.assertIsNotNone(obs[0].arm_occlusion_mask)

        out = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=obs,
            tex_size=128,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=uv_faces,
            prefer_pytorch3d=False,
        )
        # Side torso UV center (u=0.25 -> col=32, v=0.25 -> row=96) keeps dark crop top, NOT orange arm skin
        side_torso_rgb = out.tex_face_rgb[96, 32]
        self.assertLess(int(side_torso_rgb[0]), 60)
        self.assertLess(int(side_torso_rgb[1]), 60)

    def test_harmonic_inpaint_avoids_64px_block_steps_across_color_gradient(self) -> None:
        from bodyscan_pipeline.texturing import inpaint_and_dilate_texture

        h = w = 160
        tri_id_map = np.full((h, w), -1, dtype=np.int32)
        tri_id_map[8:152, 8:152] = 0
        accum_rgb = np.zeros((h, w, 3), dtype=np.float32)
        accum_w = np.zeros((h, w), dtype=np.float32)
        # Observed top/bottom borders: bright orange top vs black pants bottom, wide 96px hole in middle
        accum_rgb[8:28, 8:152] = [235.0, 130.0, 60.0]
        accum_w[8:28, 8:152] = 1.0
        accum_rgb[124:152, 8:152] = [18.0, 18.0, 22.0]
        accum_w[124:152, 8:152] = 1.0

        out = inpaint_and_dilate_texture(
            accum_rgb=accum_rgb,
            accum_w=accum_w,
            tri_id_map=tri_id_map,
            dilate_px=2,
            strict_no_invent=True,
        )
        # Inside the interpolated gap [32:120, 20:140], adjacent rows must transition smoothly without discrete tile steps
        interior = out[32:120, 20:140, 0].astype(np.float32)
        max_row_step = float(np.max(np.abs(np.diff(interior, axis=0))))
        self.assertLess(max_row_step, 20.0)

    def test_grazing_silhouette_visibility_retains_valid_oblique_surface(self) -> None:
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.texturing import ViewObservation, run_step3_texturing

        # Cylinder-like octagonal prism rotated so side panel sits at ~64 deg grazing angle (cos ~ 0.438)
        verts_a = np.array(
            [
                [-0.20, 0.0, 0.14], [0.20, 0.0, 0.14], [0.20, 1.5, 0.14], [-0.20, 1.5, 0.14],
                [0.20, 0.0, -0.14], [-0.20, 0.0, -0.14], [-0.20, 1.5, -0.14], [0.20, 1.5, -0.14],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [5, 0, 3], [5, 3, 6],
                [1, 4, 7], [1, 7, 2],
                [3, 2, 7], [3, 7, 6],
                [5, 4, 1], [5, 1, 0],
            ],
            dtype=np.int64,
        )
        mhr_uvs = np.array(
            [
                [0.05, 0.55], [0.45, 0.55], [0.45, 0.95], [0.05, 0.95],
                [0.55, 0.55], [0.95, 0.55], [0.95, 0.95], [0.55, 0.95],
            ],
            dtype=np.float32,
        )
        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.30, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.40, 0.0]

        h, w, focal = 320, 240, 500.0
        theta = np.deg2rad(62.0)  # cos(62) ~ 0.469 (> TEXTURE_MIN_VIEW_COS=0.38)
        r_y = np.array(
            [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]],
            dtype=np.float32,
        )
        verts_rot = (verts_a - np.array([0.0, 0.75, 0.0], dtype=np.float32)) @ r_y.T
        verts_cv = verts_rot.copy()
        verts_cv[:, 1] = -verts_rot[:, 1]
        verts_cv[:, 2] = -verts_rot[:, 2]
        cam_t = np.array([0.0, 0.0, 2.4], dtype=np.float32)

        img = np.full((h, w, 3), [35, 165, 95], dtype=np.uint8)
        out = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=[
                ViewObservation(
                    image_rgb=img,
                    pred_vertices=verts_cv,
                    pred_cam_t=cam_t,
                    focal_length=focal,
                    foreground_mask=np.ones((h, w), dtype=bool),
                    mask_erode_px=0,
                    view_name="front.jpg",
                )
            ],
            tex_size=128,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=faces,
            prefer_pytorch3d=False,
        )
        # Front face (cols 6..58, rows 6..58) is at ~62 deg grazing angle and must still be covered
        self.assertGreater(float(out.coverage_mask[16:48, 16:48].mean()), 0.5)

    def test_narrow_ankle_pose_drift_rejects_blue_banner_background_bleed(self) -> None:
        from bodyscan_pipeline.config import NECK_BASE_JOINT, NECK_HEAD_JOINT
        from bodyscan_pipeline.texturing import ViewObservation, run_step3_texturing

        # Narrow ankle/shin strip (6cm wide) shifted slightly right (+1.5cm) relative to mask
        verts_a = np.array(
            [
                [-0.03, 0.0, 0.04], [0.03, 0.0, 0.04], [0.03, 1.4, 0.04], [-0.03, 1.4, 0.04],
                [0.03, 0.0, -0.04], [-0.03, 0.0, -0.04], [-0.03, 1.4, -0.04], [0.03, 1.4, -0.04],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [5, 0, 3], [5, 3, 6], [1, 4, 7], [1, 7, 2], [3, 2, 7], [3, 7, 6], [5, 4, 1], [5, 1, 0]],
            dtype=np.int64,
        )
        mhr_uvs = np.array(
            [
                [0.10, 0.10], [0.45, 0.10], [0.45, 0.90], [0.10, 0.90],
                [0.55, 0.10], [0.90, 0.10], [0.90, 0.90], [0.55, 0.90],
            ],
            dtype=np.float32,
        )
        joints = np.zeros((127, 3), dtype=np.float32)
        joints[NECK_BASE_JOINT] = [0.0, 1.25, 0.0]
        joints[NECK_HEAD_JOINT] = [0.0, 1.35, 0.0]

        h, w, focal = 600, 400, 900.0
        cam_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        verts_cv = verts_a.copy()
        verts_cv[:, 1] = -(verts_a[:, 1] - 0.7)
        verts_cv[:, 2] = -verts_a[:, 2]
        # Shift mesh 1.5cm right into overlapping mask halo covered by blue banner
        verts_cv[:, 0] += 0.015

        img = np.full((h, w, 3), [25, 105, 225], dtype=np.uint8)
        # True shin sits around x=189..211; slightly oversized mask overlaps blue banner on x=211..218
        img[80:520, 188:211] = [195, 145, 115]
        fg_mask = np.zeros((h, w), dtype=bool)
        fg_mask[75:525, 185:219] = True

        out = run_step3_texturing(
            verts_a=verts_a,
            faces=faces,
            joints_a=joints,
            observations=[
                ViewObservation(
                    image_rgb=img,
                    pred_vertices=verts_cv,
                    pred_cam_t=cam_t,
                    focal_length=focal,
                    foreground_mask=fg_mask,
                    mask_erode_px=1,
                    view_name="front.jpg",
                )
            ],
            tex_size=128,
            mhr_uvs=mhr_uvs,
            mhr_uv_faces=faces,
            prefer_pytorch3d=False,
        )
        blue_hits = out.coverage_mask & (out.tex_face_rgb[..., 2] > 180) & (out.tex_face_rgb[..., 0] < 80)
        self.assertEqual(int(blue_hits.sum()), 0)

    def test_exporter_degenerate_normals_444_jpeg_and_single_sided_glb(self) -> None:
        import io
        import json
        import struct
        import tempfile
        from pathlib import Path
        from PIL import Image
        from bodyscan_pipeline.exporter import (
            _compute_smooth_vertex_normals,
            _encode_rgb_jpeg_bytes,
            export_textured_scan_artifacts,
        )
        from bodyscan_pipeline.texturing import TexturedScanOutputs, UVLayout

        deg_n = _compute_smooth_vertex_normals(np.zeros((3, 3), dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int64))
        np.testing.assert_allclose(deg_n, np.tile([0.0, 1.0, 0.0], (3, 1)), atol=1e-6)

        # Verify JPEG 4:4:4 subsampling preserves sharp red/blue edge cleanly
        stripe = np.zeros((32, 32, 3), dtype=np.uint8)
        stripe[:, :16] = [230, 15, 15]
        stripe[:, 16:] = [15, 15, 230]
        jpg_bytes = _encode_rgb_jpeg_bytes(stripe)
        with Image.open(io.BytesIO(jpg_bytes)) as im:
            self.assertEqual(im.format, "JPEG")
            # Pillow JpegImageFile stores subsampling in im.info or quantization/layer; 4:4:4 keeps sharp edge chroma
            arr = np.asarray(im)
        self.assertGreater(float(arr[:, 14, 0].mean()), 190.0)
        self.assertGreater(float(arr[:, 17, 2].mean()), 190.0)

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        layout = UVLayout(uv=uv, uv_faces=faces, vmapping=np.arange(3, dtype=np.int64), orig_faces=faces, source="mhr")
        tex_img = np.full((64, 64, 3), [180, 120, 90], dtype=np.uint8)
        tex_out = TexturedScanOutputs(
            uv_layout=layout,
            tex_face_rgb=tex_img,
            tex_blur_rgb=tex_img,
            head_uv_mask=np.zeros((64, 64), dtype=bool),
            coverage_mask=np.ones((64, 64), dtype=bool),
            used_pytorch3d=False,
            coverage_pct=100.0,
        )
        body_json = {"mesh": {"file": "scan.glb", "core_file": "scan_core.glb"}}
        with tempfile.TemporaryDirectory() as tmp:
            arts = export_textured_scan_artifacts(
                verts_a=verts,
                body_json=body_json,
                output_dir=tmp,
                tex_outputs=tex_out,
                export_preview_glb=True,
                privacy_level=3,
                core_verts=verts,
                core_faces=faces,
            )
            for key in ("glb", "core_glb", "glb_preview_blur", "glb_preview_face"):
                raw = Path(arts[key]).read_bytes()
                chunk_len = struct.unpack_from("<I", raw, 12)[0]
                gltf = json.loads(raw[20 : 20 + chunk_len].decode("utf-8"))
                for mat in gltf.get("materials", []):
                    self.assertFalse(mat.get("doubleSided", True), msg=f"{key} had doubleSided=True")


if __name__ == "__main__":
    unittest.main()
