"""Unit tests for shared scan_service core, pure binary .glb writer, MockScanEngine, and demo_ui wrapper."""

import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def _make_sample_image(path: str, color=(200, 140, 110), size=(64, 96), fmt="JPEG") -> str:
    img = Image.new("RGB", size, color)
    img.save(path, format=fmt)
    return path


def _parse_glb_chunks(glb_bytes: bytes):
    magic, version, total_len = struct.unpack_from("<III", glb_bytes, 0)
    offset = 12
    j_len, j_type = struct.unpack_from("<II", glb_bytes, offset)
    offset += 8
    gltf_json = json.loads(glb_bytes[offset : offset + j_len].decode("utf-8").rstrip(" \x00"))
    offset += j_len
    bin_bytes = b""
    b_type = None
    if offset < len(glb_bytes):
        b_len, b_type = struct.unpack_from("<II", glb_bytes, offset)
        offset += 8
        bin_bytes = glb_bytes[offset : offset + b_len]
    return magic, version, total_len, j_len, j_type, gltf_json, b_type, bin_bytes


class TestScanService(unittest.TestCase):
    """Verify ANATOMICAL_ORDER, glTF 2.0 .glb writer, MockScanEngine privacy/cleanup, and demo_ui integration."""

    def test_anatomical_constants_16_keys(self) -> None:
        from bodyscan_pipeline.scan_service import (
            ANATOMICAL_ORDER,
            BODY_LABELS_EN,
            BODY_LABELS_VI,
            PRIVACY_CHOICES,
            PRIVACY_LEVEL_MAP,
        )

        self.assertEqual(len(ANATOMICAL_ORDER), 16)
        self.assertEqual(set(ANATOMICAL_ORDER), set(BODY_LABELS_VI.keys()))
        self.assertEqual(set(ANATOMICAL_ORDER), set(BODY_LABELS_EN.keys()))
        self.assertEqual(len(PRIVACY_CHOICES), 3)
        self.assertEqual(set(PRIVACY_LEVEL_MAP.values()), {1, 2, 3})

    def test_write_minimal_glb_geometry_and_embedded_jpeg(self) -> None:
        from bodyscan_pipeline.scan_service import write_minimal_glb

        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (80, 160, 210)).save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            out_glb = os.path.join(tmp, "textured.glb")
            write_minimal_glb(verts, faces, normals, uvs, jpeg_bytes, out_glb)
            raw = Path(out_glb).read_bytes()

        magic, version, total_len, j_len, j_type, gltf, b_type, bin_bytes = _parse_glb_chunks(raw)
        self.assertEqual(magic, 0x46546C67)
        self.assertEqual(version, 2)
        self.assertEqual(total_len, len(raw))
        self.assertEqual(j_type, 0x4E4F534A)
        self.assertEqual(j_len % 4, 0)
        self.assertEqual(b_type, 0x004E4942)
        self.assertEqual(len(bin_bytes) % 4, 0)

        pos_acc = gltf["accessors"][0]
        self.assertEqual(pos_acc["type"], "VEC3")
        self.assertEqual(pos_acc["count"], 3)
        self.assertAlmostEqual(pos_acc["max"][1], 1.5, places=4)
        self.assertIn("images", gltf)
        self.assertEqual(gltf["images"][0]["mimeType"], "image/jpeg")
        bv_img = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
        recovered = bin_bytes[bv_img["byteOffset"] : bv_img["byteOffset"] + bv_img["byteLength"]]
        self.assertEqual(recovered, jpeg_bytes)

    def test_mock_engine_privacy_levels_and_upload_shredding(self) -> None:
        from bodyscan_pipeline.config import LOFT_PCT_MAX, LOFT_PCT_MIN, SCHEMA_VERSION
        from bodyscan_pipeline.scan_service import ANATOMICAL_ORDER, MockScanEngine, ScanRequest

        engine = MockScanEngine(step_delay_sec=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            for level, expected_viewer_key in [(1, "glb"), (2, "glb_preview_blur"), (3, "glb_preview_face")]:
                with self.subTest(privacy_level=level):
                    up_dir = os.path.join(tmp, f"up_{level}")
                    out_dir = os.path.join(tmp, f"out_{level}")
                    os.makedirs(up_dir, exist_ok=True)
                    photos = [
                        _make_sample_image(os.path.join(up_dir, f"angle_{i}.jpg"))
                        for i in range(4)
                    ]
                    face_path = (
                        _make_sample_image(os.path.join(up_dir, "face.jpg"), color=(220, 180, 160))
                        if level == 3
                        else None
                    )
                    events = []
                    res = engine.process(
                        ScanRequest(
                            image_paths=photos,
                            face_image_path=face_path,
                            height_cm=172.0,
                            sex="male",
                            weight_kg=68.0,
                            person_id="pt_test",
                            privacy_level=level,
                            consent_granted=True,
                            output_dir=out_dir,
                            actor="unit_test",
                        ),
                        progress_cb=lambda pct, step: events.append((pct, step)),
                    )

                    self.assertEqual(res.mode, "mock")
                    self.assertEqual(res.body_json["schema"], SCHEMA_VERSION)
                    self.assertEqual(res.viewer_artifact_key, expected_viewer_key)
                    circs = res.body_json["circumferences_cm"]
                    self.assertEqual(list(circs.keys()), list(ANATOMICAL_ORDER))
                    for k, v in circs.items():
                        self.assertGreater(v, 10.0, f"{k} too small")
                    loft = res.body_json["shape_profile"]["loft"]
                    self.assertEqual(len(loft), LOFT_PCT_MAX - LOFT_PCT_MIN + 1)
                    self.assertTrue(os.path.exists(res.artifacts[expected_viewer_key]))

                    if level in (1, 2):
                        self.assertNotIn("tex_face", res.artifacts)
                        self.assertNotIn("glb_preview_face", res.artifacts)
                        self.assertFalse(os.path.exists(os.path.join(out_dir, "tex_face.jpg")))
                    if level == 1:
                        self.assertNotIn("tex_blur", res.artifacts)

                    # Raw photos must be shredded
                    for p in photos:
                        self.assertFalse(os.path.exists(p), f"Raw photo still exists: {p}")
                    if face_path:
                        self.assertFalse(os.path.exists(face_path))
                    self.assertTrue(os.path.exists(os.path.join(out_dir, "deletion_audit.jsonl")))

                    # Strictly increasing progress callbacks ending at 100
                    pcts = [p for p, _ in events]
                    self.assertEqual(pcts[-1], 100)
                    self.assertEqual(pcts, sorted(pcts))

    def test_mock_engine_shreds_uploads_even_on_corrupted_image(self) -> None:
        from bodyscan_pipeline.scan_service import MockScanEngine, ScanRequest

        engine = MockScanEngine(step_delay_sec=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = os.path.join(tmp, "corrupt.jpg")
            with open(bad_file, "wb") as f:
                f.write(b"NOT_AN_IMAGE_PAYLOAD_1234")
            out_dir = os.path.join(tmp, "out")
            with self.assertRaises(ValueError):
                engine.process(
                    ScanRequest(
                        image_paths=[bad_file],
                        height_cm=165.0,
                        sex="female",
                        privacy_level=2,
                        consent_granted=True,
                        output_dir=out_dir,
                    )
                )
            self.assertFalse(os.path.exists(bad_file))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "deletion_audit.jsonl")))

    def test_demo_ui_imports_without_gradio_and_shares_constants(self) -> None:
        from bodyscan_pipeline.demo_ui import (
            BODY_LABELS_VI,
            PRIVACY_CHOICES,
            PRIVACY_LEVEL_MAP,
            ScanDemoEngine,
        )
        from bodyscan_pipeline.scan_service import (
            ANATOMICAL_ORDER,
            BODY_LABELS_VI as SVC_LABELS_VI,
            PRIVACY_CHOICES as SVC_CHOICES,
            PRIVACY_LEVEL_MAP as SVC_MAP,
        )

        self.assertIs(PRIVACY_CHOICES, SVC_CHOICES)
        self.assertIs(PRIVACY_LEVEL_MAP, SVC_MAP)
        self.assertIs(BODY_LABELS_VI, SVC_LABELS_VI)
        self.assertEqual(len(ANATOMICAL_ORDER), 16)
        self.assertTrue(hasattr(ScanDemoEngine, "process"))

    def test_mock_mannequin_uv_upright_orientation(self) -> None:
        from bodyscan_pipeline.scan_service import (
            _build_mock_mannequin,
            _render_mock_texture,
        )

        verts, faces, _, uvs = _build_mock_mannequin(170.0, "female", 54.0)
        self.assertGreater(len(faces), 0)
        crown_idx = int(np.argmax(verts[:, 1]))
        ankle_idx = int(np.argmin(verts[:, 1]))
        # OpenGL/glTF raw buffer convention with Three.js tex.flipY=False:
        # row 0 (top of JPEG, crown/head) maps to v close to 0.0; lower body maps to larger v.
        self.assertLess(float(uvs[crown_idx, 1]), 0.12)
        self.assertGreater(float(uvs[ankle_idx, 1]), 0.35)

        # Sex-distinct silhouette proportions (male chest/shoulder wider than female at same height)
        m_verts, _, _, _ = _build_mock_mannequin(170.0, "male", 65.0)
        f_chest_band = verts[(verts[:, 1] > 1.25) & (verts[:, 1] < 1.38)]
        m_chest_band = m_verts[(m_verts[:, 1] > 1.25) & (m_verts[:, 1] < 1.38)]
        self.assertGreater(
            float(m_chest_band[:, 0].max() - m_chest_band[:, 0].min()),
            float(f_chest_band[:, 0].max() - f_chest_band[:, 0].min()),
        )

        # Privacy Level 2 blur_face=True blurs face skin band while keeping top hair crown crisp (3b giữ rõ)
        sharp_bytes = _render_mock_texture((210, 170, 145), (45, 70, 130), blur_face=False)
        blur_bytes = _render_mock_texture((210, 170, 145), (45, 70, 130), blur_face=True)
        im_sharp = np.asarray(Image.open(io.BytesIO(sharp_bytes)).convert("RGB"))
        im_blur = np.asarray(Image.open(io.BytesIO(blur_bytes)).convert("RGB"))
        # Top hair crown rows [:26] retain crisp strand detail (identical between L2 and L3)
        self.assertTrue(np.allclose(im_sharp[:26, :, :], im_blur[:26, :, :], atol=3))
        # Facial band below hairline [36:72] is strongly smoothed in Level 2
        self.assertGreater(
            float(np.abs(im_sharp[36:72].astype(np.float32) - im_blur[36:72].astype(np.float32)).mean()),
            4.0,
        )

    def test_write_multi_material_glb_two_primitives_and_dual_layer_core(self) -> None:
        from bodyscan_pipeline.scan_service import MockScanEngine, ScanRequest, write_multi_material_glb

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (200, 100, 50)).save(buf, format="JPEG")
        tex_body = buf.getvalue()
        buf2 = io.BytesIO()
        Image.new("RGB", (16, 16), (60, 180, 90)).save(buf2, format="JPEG")
        tex_head = buf2.getvalue()

        v_tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        f_tri = np.array([[0, 1, 2]], dtype=np.int32)
        n_tri = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        uv_tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            out_glb = os.path.join(tmp, "multi.glb")
            write_multi_material_glb(
                [
                    {"verts": v_tri, "faces": f_tri, "normals": n_tri, "uvs": uv_tri, "texture_jpeg_bytes": tex_body, "name": "body"},
                    {"verts": v_tri + 0.1, "faces": f_tri, "normals": n_tri, "uvs": uv_tri, "texture_jpeg_bytes": tex_head, "name": "head"},
                ],
                out_glb,
            )
            magic, version, total_len, _, _, gltf, _, bin_bytes = _parse_glb_chunks(Path(out_glb).read_bytes())
            self.assertEqual(magic, 0x46546C67)
            self.assertEqual(version, 2)
            self.assertEqual(total_len, Path(out_glb).stat().st_size)
            self.assertEqual(len(gltf["meshes"][0]["primitives"]), 2)
            self.assertEqual(len(gltf["materials"]), 2)
            self.assertEqual(len(gltf["images"]), 2)
            self.assertEqual(len(gltf["textures"]), 2)
            self.assertEqual(len(bin_bytes) % 4, 0)

            # MockScanEngine dual_layer produces scan_core.glb + proportion_audit + 2-primitive display GLB
            up = os.path.join(tmp, "up")
            out_d = os.path.join(tmp, "out")
            os.makedirs(up, exist_ok=True)
            photos = [_make_sample_image(os.path.join(up, f"p{i}.jpg")) for i in range(4)]
            res = MockScanEngine(0.0).process(
                ScanRequest(
                    image_paths=photos,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=54.0,
                    privacy_level=2,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=out_d,
                )
            )
            self.assertIn("core_glb", res.artifacts)
            self.assertTrue(os.path.exists(res.artifacts["core_glb"]))
            self.assertEqual(res.body_json["mesh"]["core_file"], "scan_core.glb")
            self.assertEqual(res.body_json["reconstruction_mode"], "dual_layer")
            self.assertEqual(res.body_json["effective_reconstruction_mode"], "dual_layer")
            self.assertEqual(res.body_json["proportion_audit"]["status"], "matched")
            _, _, _, _, _, disp_gltf, _, _ = _parse_glb_chunks(Path(res.artifacts[res.viewer_artifact_key]).read_bytes())
            self.assertEqual(len(disp_gltf["meshes"][0]["primitives"]), 2)
            self.assertEqual(len(disp_gltf["materials"]), 2)

    def test_mock_engine_reconstruction_mode_validation_and_audit(self) -> None:
        from bodyscan_pipeline.scan_service import (
            MockScanEngine,
            ScanRequest,
            _build_mock_mannequin_parts,
        )

        engine = MockScanEngine(0.0)
        with tempfile.TemporaryDirectory() as tmp:
            # Invalid reconstruction_mode rejected cleanly
            bad_img = _make_sample_image(os.path.join(tmp, "bad_mode.jpg"))
            with self.assertRaises(ValueError):
                engine.process(
                    ScanRequest(
                        image_paths=[bad_img],
                        height_cm=166.0,
                        sex="female",
                        reconstruction_mode="invalid_mode",
                        consent_granted=True,
                        output_dir=os.path.join(tmp, "out_bad"),
                    )
                )

            # <4 photos with dual_layer auto-downgrades to calibrated_mhr with warning & no core_glb
            up_2 = os.path.join(tmp, "up_2")
            os.makedirs(up_2, exist_ok=True)
            two_photos = [_make_sample_image(os.path.join(up_2, f"p{i}.jpg")) for i in range(2)]
            res_2 = engine.process(
                ScanRequest(
                    image_paths=two_photos,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=0.0,
                    privacy_level=2,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=os.path.join(tmp, "out_2"),
                )
            )
            self.assertEqual(res_2.reconstruction_mode, "dual_layer")
            self.assertEqual(res_2.effective_reconstruction_mode, "calibrated_mhr")
            self.assertEqual(res_2.body_json["effective_reconstruction_mode"], "calibrated_mhr")
            self.assertNotIn("core_glb", res_2.artifacts)
            self.assertTrue(any("calibrated_mhr" in w for w in res_2.warnings))
            self.assertEqual(res_2.proportion_audit["status"], "estimated_only")

            # 0.08 < |est - decl| / decl <= 0.12 keeps status="matched" with informational note
            # At 166cm female, est_w = 57.87 kg; weight_kg=52.5 gives diff_pct = 10.23%
            up_minor = os.path.join(tmp, "up_minor")
            os.makedirs(up_minor, exist_ok=True)
            minor_photos = [_make_sample_image(os.path.join(up_minor, f"p{i}.jpg")) for i in range(4)]
            res_minor = engine.process(
                ScanRequest(
                    image_paths=minor_photos,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=52.5,
                    privacy_level=2,
                    reconstruction_mode="hi_res_head",
                    consent_granted=True,
                    output_dir=os.path.join(tmp, "out_minor"),
                )
            )
            self.assertEqual(res_minor.effective_reconstruction_mode, "hi_res_head")
            self.assertEqual(res_minor.proportion_audit["status"], "matched")
            self.assertGreater(res_minor.proportion_audit["weight_diff_pct"], 8.0)
            self.assertLessEqual(res_minor.proportion_audit["weight_diff_pct"], 12.0)
            self.assertTrue(len(res_minor.proportion_audit["warnings"]) >= 1)
            self.assertTrue(len(res_minor.warnings) >= 1)
            self.assertNotIn("core_glb", res_minor.artifacts)
            self.assertNotIn("core_file", res_minor.body_json["mesh"])
            _, _, _, _, _, hr_gltf, _, _ = _parse_glb_chunks(Path(res_minor.artifacts[res_minor.viewer_artifact_key]).read_bytes())
            self.assertEqual(len(hr_gltf["meshes"][0]["primitives"]), 2)
            # hi_res_head includes 4-view hair cap/bun without outer torso garment offset; core omits both
            (hr_body_v, _, _, _), (hr_head_v, _, _, _) = _build_mock_mannequin_parts(
                166.0, "female", 54.0, outer_shell=False, include_hair=True
            )
            (dl_body_v, _, _, _), (dl_head_v, _, _, _) = _build_mock_mannequin_parts(
                166.0, "female", 54.0, outer_shell=True, include_hair=True
            )
            (core_body_v, _, _, _), (core_head_v, _, _, _) = _build_mock_mannequin_parts(
                166.0, "female", 54.0, outer_shell=False, include_hair=False
            )
            self.assertEqual(len(hr_head_v), len(dl_head_v))
            self.assertGreater(len(hr_head_v), len(core_head_v))
            # Outer torso garment offset widens dual_layer torso depth (Z) & waist/chest beyond hi_res_head & core
            self.assertGreater(
                float(dl_body_v[:, 2].max() - dl_body_v[:, 2].min()),
                float(hr_body_v[:, 2].max() - hr_body_v[:, 2].min()),
            )
            self.assertAlmostEqual(
                float(hr_body_v[:, 2].max() - hr_body_v[:, 2].min()),
                float(core_body_v[:, 2].max() - core_body_v[:, 2].min()),
                places=5,
            )

            # Contradictory weight (>12% off) flags mismatch_warning
            up_4 = os.path.join(tmp, "up_4")
            os.makedirs(up_4, exist_ok=True)
            four_photos = [_make_sample_image(os.path.join(up_4, f"p{i}.jpg")) for i in range(4)]
            res_mismatch = engine.process(
                ScanRequest(
                    image_paths=four_photos,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=105.0,
                    privacy_level=1,
                    reconstruction_mode="calibrated_mhr",
                    consent_granted=True,
                    output_dir=os.path.join(tmp, "out_mismatch"),
                )
            )
            self.assertEqual(res_mismatch.proportion_audit["status"], "mismatch_warning")
            self.assertTrue(len(res_mismatch.proportion_audit["warnings"]) >= 1)
            self.assertTrue(len(res_mismatch.warnings) >= 1)

    def test_gpu_engine_orchestrates_3tier_pipeline_with_fake_mhr_cpu(self) -> None:
        import cv2
        import math
        from unittest.mock import patch
        from PIL import Image
        import torch
        from bodyscan_pipeline.config import (
            DEFAULT_RECONSTRUCTION_MODE,
            HEAD_TEXTURE_SIZE,
            NECK_BASE_JOINT,
            NECK_HEAD_JOINT,
        )
        from bodyscan_pipeline.scan_service import GpuScanEngine, ScanRequest

        # Build watertight humanoid capsule mesh (units in meters; FakeMHR emits *100 cm)
        n_lat, n_lon = 18, 24
        pts_m = [[0.0, 1.66, 0.0]]
        for i in range(1, n_lat):
            phi = math.pi * i / n_lat
            y = 0.83 * (1.0 + math.cos(phi))
            yn = y / 1.66
            if yn >= 0.84:
                t_h = min(1.0, (yn - 0.84) / 0.16)
                rx = 0.056 + 0.034 * math.sin(math.pi * t_h)
                rz = 0.056 + 0.042 * math.sin(math.pi * t_h)
            elif yn >= 0.80:
                rx, rz = 0.055, 0.055  # neck (~34.5cm circ)
            elif yn >= 0.67:
                rx, rz = 0.160, 0.118  # chest
            elif yn >= 0.56:
                rx, rz = 0.128, 0.095  # waist
            elif yn >= 0.45:
                rx, rz = 0.170, 0.124  # hip
            else:
                rx = 0.062 + 0.070 * (yn / 0.45)
                rz = 0.056 + 0.052 * (yn / 0.45)
            for j in range(n_lon):
                theta = 2.0 * math.pi * j / n_lon
                pts_m.append([rx * math.cos(theta), y, rz * math.sin(theta)])
        bot_idx = len(pts_m)
        pts_m.append([0.0, 0.0, 0.0])

        faces_list = []
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces_list.append([0, 1 + jn, 1 + j])
        for i in range(n_lat - 2):
            r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                faces_list.append([r0 + j, r0 + jn, r1 + jn])
                faces_list.append([r0 + j, r1 + jn, r1 + j])
        last_r = 1 + (n_lat - 2) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces_list.append([bot_idx, last_r + j, last_r + jn])

        base_verts_m = torch.tensor(pts_m, dtype=torch.float32)
        faces_t = torch.tensor(faces_list, dtype=torch.int64)
        n_verts = len(pts_m)

        # Canonical skeleton joints in meters (scaled to cm inside FakeMHR)
        base_joints_m = torch.zeros(128, 3, dtype=torch.float32)
        base_joints_m[0] = torch.tensor([0.0, 1.54, 0.11])  # nose anterior +Z
        base_joints_m[2] = torch.tensor([0.09, 0.85, 0.0])   # L hip
        base_joints_m[18] = torch.tensor([-0.09, 0.85, 0.0]) # R hip
        base_joints_m[39] = torch.tensor([-0.14, 1.32, 0.0]) # R shoulder
        base_joints_m[75] = torch.tensor([0.14, 1.32, 0.0])  # L shoulder
        base_joints_m[NECK_BASE_JOINT] = torch.tensor([0.0, 1.36, 0.0])
        base_joints_m[NECK_HEAD_JOINT] = torch.tensor([0.0, 1.50, 0.0])
        # Fill remaining limb joints safely inside torso/leg envelope so circumference slices find geometry
        for jid in range(128):
            if jid not in (0, 2, 18, 39, 75, NECK_BASE_JOINT, NECK_HEAD_JOINT):
                frac_y = 0.25 + 0.60 * ((jid % 17) / 16.0)
                base_joints_m[jid] = torch.tensor([0.0, 1.66 * frac_y, 0.0])

        class _FakeLBS:
            def __init__(self) -> None:
                self.vert_indices_flattened = torch.arange(n_verts, dtype=torch.long)
                # Keep all vertices non-arm (joint 2 = torso) so torso_mesh stays closed and connected
                self.skin_indices_flattened = torch.full((n_verts,), 2, dtype=torch.long)
                self.skin_weights_flattened = torch.ones((n_verts,), dtype=torch.float32)

        class _FakeCharTorch:
            def __init__(self) -> None:
                self.linear_blend_skinning = _FakeLBS()

        class _FakeMHR:
            def __init__(self) -> None:
                self.character_torch = _FakeCharTorch()

            def __call__(self, shape_t, model_params, expr_vec):
                # Differentiable wrt shape_t and scale suffix so fit_mhr_multiview optimizes cleanly
                s0 = shape_t[:, :1].unsqueeze(-1)
                sc0 = model_params[:, 136:137].unsqueeze(-1)
                radial_gain = 1.0 + 0.03 * s0 + 0.03 * sc0
                vx = base_verts_m.unsqueeze(0)[..., 0:1] * radial_gain
                vy = base_verts_m.unsqueeze(0)[..., 1:2] * (1.0 + 0.01 * sc0)
                vz = base_verts_m.unsqueeze(0)[..., 2:3] * radial_gain
                verts_cm = torch.cat([vx, vy, vz], dim=-1) * 100.0
                joints_cm = base_joints_m.unsqueeze(0).expand(shape_t.shape[0], -1, -1) * 100.0
                return verts_cm, joints_cm

        fake_mhr = _FakeMHR()

        class _FakeHeadPose:
            def __init__(self) -> None:
                self.scale_mean = torch.zeros(28, dtype=torch.float32)
                self.scale_comps = torch.eye(28, dtype=torch.float32)
                self.faces = faces_t
                self.mhr = fake_mhr

        class _FakeSamModel:
            def __init__(self) -> None:
                self.head_pose = _FakeHeadPose()

        class _FakeFovEstimator:
            def infer_depth(self, rgb: np.ndarray) -> np.ndarray:
                h, w = rgb.shape[:2]
                return np.full((h, w), 2.5, dtype=np.float32)

        class _FakeEstimator:
            def __init__(self) -> None:
                self.fov_estimator = _FakeFovEstimator()
                self._angles_deg = [0.0, 90.0, 180.0, 270.0]

            def process_one_image(self, rgb: np.ndarray, bbox_thr: float = 0.5, use_mask: bool = False):
                h, w = rgb.shape[:2]
                # Read encoded cardinal index from lossless corner patch
                tag = int(round((240.0 - float(rgb[2:10, 2:10, 0].mean())) / 12.0)) % 4
                az = np.radians(self._angles_deg[tag])
                c, s = float(np.cos(az)), float(np.sin(az))
                rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
                kpts_cv = base_joints_m.numpy().copy()
                # Convert +Y up to OpenCV +Y down centered around 0.83
                kpts_cv[:, 1] = -(kpts_cv[:, 1] - 0.83)
                kpts_rot = kpts_cv @ rot_y.T
                verts_cv = base_verts_m.numpy().copy()
                verts_cv[:, 1] = -(verts_cv[:, 1] - 0.83)
                verts_cv[:, 2] = -verts_cv[:, 2]
                r_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float32) @ rot_y
                j_cam_2d = base_joints_m.numpy() @ r_cv.T + np.array([0.0, 0.83, 2.6], dtype=np.float32)
                k2_u = 480.0 * j_cam_2d[:, 0] / np.maximum(j_cam_2d[:, 2], 0.1) + (w * 0.5)
                k2_v = 480.0 * j_cam_2d[:, 1] / np.maximum(j_cam_2d[:, 2], 0.1) + (h * 0.5)
                return [
                    {
                        "bbox": np.array([w * 0.25, h * 0.08, w * 0.75, h * 0.92], dtype=np.float32),
                        "focal_length": 480.0,
                        "pred_cam_t": np.array([0.0, 0.0, 2.6], dtype=np.float32),
                        "shape_params": np.zeros(45, dtype=np.float32),
                        "scale_params": np.zeros(28, dtype=np.float32),
                        "body_pose_params": np.zeros(130, dtype=np.float32),
                        "expr_params": np.zeros(72, dtype=np.float32),
                        "pred_keypoints_3d": kpts_rot,
                        "pred_keypoints_2d": np.stack([k2_u, k2_v], axis=-1).astype(np.float32),
                        "pred_vertices": verts_cv @ rot_y.T,
                    }
                ]

        def _make_cardinal_studio_photo(path: str, view_idx: int) -> str:
            h, w = 384, 256
            canvas = np.full((h, w, 3), 240, dtype=np.uint8)
            az = math.radians([0.0, 90.0, 180.0, 270.0][view_idx % 4])
            c_az, s_az = math.cos(az), math.sin(az)
            rot_y = np.array([[c_az, 0.0, s_az], [0.0, 1.0, 0.0], [-s_az, 0.0, c_az]], dtype=np.float32)
            verts_cv = base_verts_m.numpy().copy()
            verts_cv[:, 1] = -(verts_cv[:, 1] - 0.83)
            verts_cv[:, 2] = -verts_cv[:, 2]
            v_cam = verts_cv @ rot_y.T
            z_cam = np.maximum(0.5, 2.6 + v_cam[:, 2])
            px = (480.0 * v_cam[:, 0] / z_cam + w * 0.5).astype(np.int32)
            py = (480.0 * v_cam[:, 1] / z_cam + h * 0.5).astype(np.int32)
            f_arr = np.asarray(faces_list, dtype=np.int64)
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            for tri in f_arr:
                poly = np.stack([px[tri], py[tri]], axis=-1).astype(np.int32)
                cv2.fillConvexPoly(mask_u8, poly, 1)
            y_neck_px = int(480.0 * (-(1.34 - 0.83)) / 2.6 + h * 0.5)
            y_hair_px = int(480.0 * (-(1.60 - 0.83)) / 2.6 + h * 0.5)
            rows_grid = np.arange(h)[:, None]
            cols_grid = np.arange(w)[None, :]
            body_sel = (mask_u8 > 0) & (rows_grid >= y_neck_px)
            above_neck = (mask_u8 > 0) & (rows_grid < y_neck_px)
            v_slot = view_idx % 4
            if v_slot == 2:
                # Back of head above the neck is hair, not facial skin checkerboard.
                face_sel = np.zeros((h, w), dtype=bool)
                hair_sel = above_neck
            elif v_slot == 1:
                face_sel = above_neck & (rows_grid >= y_hair_px) & (cols_grid <= w // 2)
                hair_sel = above_neck & (~face_sel)
            elif v_slot == 3:
                face_sel = above_neck & (rows_grid >= y_hair_px) & (cols_grid >= w // 2)
                hair_sel = above_neck & (~face_sel)
            else:
                face_sel = above_neck & (rows_grid >= y_hair_px)
                hair_sel = above_neck & (rows_grid < y_hair_px)
            canvas[body_sel] = (65, 185, 195)
            canvas[face_sel] = (212, 166, 140)
            # Subtle high-frequency skin-toned detail in face region so Level 2 blur vs Level 3 sharp is measurable
            face_detail = face_sel & ((rows_grid % 4 < 2) ^ (cols_grid % 4 < 2))
            canvas[face_detail] = (176, 118, 92)
            canvas[hair_sel] = (28, 18, 14)
            # Subtle dark hair strand variation so preserved Level 2 hair texture retains texture detail
            hair_detail = hair_sel & (cols_grid % 4 < 2)
            canvas[hair_detail] = (44, 30, 22)
            canvas[:12, :12] = (max(0, 240 - view_idx * 12), 240, 240)
            Image.fromarray(canvas).save(path, format="PNG")
            return path

        def _extract_glb_embedded_images_and_counts(glb_path: str):
            import io
            _, _, _, _, _, gltf_doc, _, bin_chunk = _parse_glb_chunks(Path(glb_path).read_bytes())
            prims = gltf_doc["meshes"][0]["primitives"]
            acc_counts = []
            unique_verts_per_prim = []
            for p in prims:
                pos_acc = gltf_doc["accessors"][p["attributes"]["POSITION"]]
                pos_bv = gltf_doc["bufferViews"][pos_acc["bufferView"]]
                pos_off = int(pos_bv.get("byteOffset", 0)) + int(pos_acc.get("byteOffset", 0))
                pos_len = int(pos_acc["count"]) * 12
                pos_arr = np.frombuffer(bin_chunk[pos_off : pos_off + pos_len], dtype=np.float32).reshape(-1, 3)
                unique_verts_per_prim.append(len(np.unique(pos_arr.round(5), axis=0)))
                acc_counts.append(
                    (
                        int(pos_acc["count"]),
                        int(gltf_doc["accessors"][p["indices"]]["count"]) // 3,
                    )
                )
            imgs = []
            for img_meta in gltf_doc.get("images", []):
                bv = gltf_doc["bufferViews"][img_meta["bufferView"]]
                off = int(bv.get("byteOffset", 0))
                raw_img = bin_chunk[off : off + int(bv["byteLength"])]
                with Image.open(io.BytesIO(raw_img)) as im:
                    imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            return gltf_doc, acc_counts, unique_verts_per_prim, imgs

        with tempfile.TemporaryDirectory() as tmp, patch(
            "bodyscan_pipeline.inference.load_sam3d_estimator",
            return_value=(_FakeSamModel(), _FakeEstimator()),
        ), patch(
            "bodyscan_pipeline.segmentation.extract_face_landmarks_ram",
            return_value=np.array([[0.50, 0.55, 0.0], [0.50, 0.78, 0.0], [0.35, 0.42, 0.0], [0.65, 0.42, 0.0]], dtype=np.float32),
        ):
            engine = GpuScanEngine(sam_repo_dir=tmp, checkpoint_path=tmp, mhr_path=tmp)

            # Case 1: dual_layer at privacy_level=1 still generates 3a detailed scan.glb + scan_core.glb
            up1 = os.path.join(tmp, "up_l1")
            out1 = os.path.join(tmp, "out_l1")
            os.makedirs(up1, exist_ok=True)
            photos1 = [_make_cardinal_studio_photo(os.path.join(up1, f"v{i}.png"), i) for i in range(4)]
            res1 = engine.process(
                ScanRequest(
                    image_paths=photos1,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=0.0,
                    privacy_level=1,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=out1,
                )
            )
            self.assertEqual(res1.effective_reconstruction_mode, "dual_layer", msg=f"warnings={res1.warnings}")
            self.assertEqual(res1.proportion_audit["status"], "estimated_only", msg=f"audit={res1.proportion_audit} warnings={res1.warnings}")
            self.assertIn("glb", res1.artifacts)
            self.assertIn("core_glb", res1.artifacts)
            self.assertTrue(os.path.exists(res1.artifacts["core_glb"]))
            self.assertNotIn("tex_blur", res1.artifacts)
            self.assertNotIn("tex_face", res1.artifacts)
            self.assertNotIn("face_landmarks", json.dumps(res1.body_json))
            for p in photos1:
                self.assertFalse(os.path.exists(p))

            # Case 2: dual_layer at privacy_level=2 exports scan_core.glb, Level 2 blur, and head material metadata
            up2 = os.path.join(tmp, "up_l2")
            out2 = os.path.join(tmp, "out_l2")
            os.makedirs(up2, exist_ok=True)
            photos2 = [_make_cardinal_studio_photo(os.path.join(up2, f"v{i}.png"), i) for i in range(4)]
            res2 = engine.process(
                ScanRequest(
                    image_paths=photos2,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=58.0,
                    privacy_level=2,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=out2,
                )
            )
            self.assertEqual(res2.effective_reconstruction_mode, "dual_layer")
            self.assertIn(res2.proportion_audit["status"], ("matched", "mismatch_warning"))
            self.assertIn("core_glb", res2.artifacts)
            self.assertIn("tex_blur", res2.artifacts, msg=f"warnings={res2.warnings}")
            self.assertIn("glb_preview_blur", res2.artifacts)
            self.assertNotIn("tex_face", res2.artifacts)
            self.assertGreater(res2.body_json["circumferences_cm"]["chest"], 40.0)
            self.assertEqual(res2.body_json["textures"]["materials"], ["body", "head"])
            self.assertEqual(res2.body_json["textures"]["head_resolution"], HEAD_TEXTURE_SIZE)
            self.assertNotIn("face_landmarks", json.dumps(res2.body_json))
            gltf1, counts1, uniq_v1, _ = _extract_glb_embedded_images_and_counts(res1.artifacts["glb"])
            _, counts2_geom, uniq_v2, _ = _extract_glb_embedded_images_and_counts(res2.artifacts["glb"])
            gltf2, counts2_disp, uniq_v2_disp, _ = _extract_glb_embedded_images_and_counts(res2.artifacts[res2.viewer_artifact_key])
            # 3a: Level 1 detailed scan.glb matches Level 2 dual_layer primitives and detailed 3D vertex/face topology
            self.assertEqual(len(counts1), 2)
            self.assertEqual([m["name"] for m in gltf1["materials"]], ["body", "head"])
            self.assertEqual([tri for _, tri in counts1], [tri for _, tri in counts2_geom])
            self.assertEqual(uniq_v1, uniq_v2)
            self.assertEqual(uniq_v1, uniq_v2_disp)
            self.assertGreater(sum(uniq_v1), len(pts_m))
            self.assertEqual(len(gltf2["meshes"][0]["primitives"]), 2)
            self.assertEqual(sum(tri for _, tri in counts1), sum(tri for _, tri in counts2_disp))

            # Case 3: hi_res_head at privacy_level=3 + closeup_face exports crisp Level 3 texture & MoGe2 depth relief
            up3 = os.path.join(tmp, "up_l3")
            out3 = os.path.join(tmp, "out_l3")
            os.makedirs(up3, exist_ok=True)
            photos3 = [_make_cardinal_studio_photo(os.path.join(up3, f"v{i}.png"), i) for i in range(4)]
            face3 = _make_cardinal_studio_photo(os.path.join(up3, "face.png"), 0)
            res3 = engine.process(
                ScanRequest(
                    image_paths=photos3,
                    face_image_path=face3,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=58.0,
                    privacy_level=3,
                    reconstruction_mode="hi_res_head",
                    consent_granted=True,
                    output_dir=out3,
                )
            )
            self.assertEqual(res3.effective_reconstruction_mode, "hi_res_head")
            self.assertNotIn("MoGe2", " ".join(res3.warnings))
            self.assertNotIn("core_glb", res3.artifacts)
            self.assertIn("tex_face", res3.artifacts)
            self.assertIn("glb_preview_blur", res3.artifacts)
            self.assertIn("glb_preview_face", res3.artifacts)
            self.assertFalse(os.path.exists(face3))
            self.assertNotIn("face_landmarks", json.dumps(res3.body_json))

            # 3b: Level 2 head texture blurs facial skin texels while preserving sharp hair texels vs Level 3
            _, _, _, imgs3_blur = _extract_glb_embedded_images_and_counts(res3.artifacts["glb_preview_blur"])
            _, _, _, imgs3_face = _extract_glb_embedded_images_and_counts(res3.artifacts["glb_preview_face"])
            self.assertEqual(len(imgs3_blur), 2)
            self.assertEqual(len(imgs3_face), 2)
            head_l2 = imgs3_blur[1].astype(np.float32)
            head_l3 = imgs3_face[1].astype(np.float32)
            diff_head = np.abs(head_l3 - head_l2).mean(axis=2)
            hair_px = (head_l3.max(axis=2) < 60.0) & (head_l3.min(axis=2) > 5.0)
            skin_px = (head_l3[:, :, 0] > 140.0) & ((head_l3[:, :, 0] - head_l3[:, :, 2]) > 35.0)
            self.assertTrue(np.any(hair_px), "Expected dark hair texels in head atlas")
            self.assertTrue(np.any(skin_px), "Expected warm facial skin texels in head atlas")
            self.assertLess(float(diff_head[hair_px].mean()), 2.0)
            self.assertGreater(float(diff_head[skin_px].max()), 5.0)
            skin_adj = skin_px[:, :-1] & skin_px[:, 1:]
            self.assertTrue(np.any(skin_adj), "Expected contiguous warm skin texels in head atlas")
            grad_skin_l3 = float(np.abs(head_l3[:, 1:] - head_l3[:, :-1]).mean(axis=2)[skin_adj].mean())
            grad_skin_l2 = float(np.abs(head_l2[:, 1:] - head_l2[:, :-1]).mean(axis=2)[skin_adj].mean())
            self.assertLess(grad_skin_l2, grad_skin_l3 * 0.7)

            # Case 3b-mode: explicit 4-view calibrated_mhr keeps single body primitive and omits core_glb
            up_cal = os.path.join(tmp, "up_cal")
            out_cal = os.path.join(tmp, "out_cal")
            os.makedirs(up_cal, exist_ok=True)
            photos_cal = [_make_cardinal_studio_photo(os.path.join(up_cal, f"v{i}.png"), i) for i in range(4)]
            res_cal = engine.process(
                ScanRequest(
                    image_paths=photos_cal,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=58.0,
                    privacy_level=2,
                    reconstruction_mode="calibrated_mhr",
                    consent_granted=True,
                    output_dir=out_cal,
                )
            )
            self.assertEqual(res_cal.effective_reconstruction_mode, "calibrated_mhr")
            self.assertNotIn("core_glb", res_cal.artifacts)
            gltf_cal, counts_cal, _, _ = _extract_glb_embedded_images_and_counts(res_cal.artifacts["glb"])
            self.assertEqual(len(gltf_cal["meshes"][0]["primitives"]), 1)
            self.assertEqual(sum(tri for _, tri in counts_cal), len(faces_list))

            # Case 4: <4 views auto-downgrades dual_layer to calibrated_mhr with warning
            up4 = os.path.join(tmp, "up_l4")
            out4 = os.path.join(tmp, "out_l4")
            os.makedirs(up4, exist_ok=True)
            photos4 = [_make_cardinal_studio_photo(os.path.join(up4, f"v{i}.png"), i) for i in range(2)]
            res4 = engine.process(
                ScanRequest(
                    image_paths=photos4,
                    height_cm=166.0,
                    sex="female",
                    weight_kg=58.0,
                    privacy_level=2,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=out4,
                )
            )
            self.assertEqual(res4.reconstruction_mode, "dual_layer")
            self.assertEqual(res4.effective_reconstruction_mode, "calibrated_mhr")
            self.assertTrue(any("calibrated_mhr" in w for w in res4.warnings))
            self.assertNotIn("core_glb", res4.artifacts)

            # Case 5: mid-pipeline exception still shreds uploaded photos and clears RAM-only landmarks/results in finally:
            from bodyscan_pipeline.segmentation import process_views_semantics as _real_pvs

            captured_sv_list = None
            captured_sv_refs = []
            captured_raw_list = None

            def _spy_process_views(raw_res, estimator=None):
                nonlocal captured_sv_list, captured_sv_refs, captured_raw_list
                captured_raw_list = raw_res
                captured_sv_list = _real_pvs(raw_res, estimator=estimator)
                captured_sv_refs = list(captured_sv_list)
                return captured_sv_list

            up_err = os.path.join(tmp, "up_err")
            out_err = os.path.join(tmp, "out_err")
            os.makedirs(up_err, exist_ok=True)
            photos_err = [_make_cardinal_studio_photo(os.path.join(up_err, f"v{i}.png"), i) for i in range(4)]
            with patch("bodyscan_pipeline.segmentation.process_views_semantics", side_effect=_spy_process_views), patch(
                "bodyscan_pipeline.fitting.fit_mhr_multiview",
                side_effect=RuntimeError("simulated mid-pipeline failure"),
            ):
                with self.assertRaises(RuntimeError):
                    engine.process(
                        ScanRequest(
                            image_paths=photos_err,
                            height_cm=166.0,
                            sex="female",
                            weight_kg=58.0,
                            privacy_level=2,
                            reconstruction_mode="dual_layer",
                            consent_granted=True,
                            output_dir=out_err,
                        )
                    )
            for p in photos_err:
                self.assertFalse(os.path.exists(p))
            self.assertTrue(os.path.exists(os.path.join(out_err, "deletion_audit.jsonl")))
            self.assertTrue(len(captured_sv_refs) == 4)
            for sv in captured_sv_refs:
                self.assertIsNone(sv.face_landmarks_ram)
                self.assertIsNone(sv.nose_profile_curve)
            self.assertEqual(len(captured_sv_list), 0)
            self.assertEqual(len(captured_raw_list), 0)

    def test_gpu_engine_wires_fitted_cameras_view_verts_cached_apose_and_fallback(self) -> None:
        from unittest.mock import patch
        import torch
        from bodyscan_pipeline.fitting import ProportionAudit
        from bodyscan_pipeline.fusion import FusedSubjectParams
        from bodyscan_pipeline.scan_service import (
            ANATOMICAL_ORDER,
            GpuScanEngine,
            ScanRequest,
            _transfer_detail_verts_to_view,
        )
        from bodyscan_pipeline.texturing import TexturedScanOutputs, UVLayout, ViewObservation

        # Verify _transfer_detail_verts_to_view stays translation-invariant across shifted/rotated view meshes
        base_v = np.array(
            [
                [0.0, 0.0, 0.2],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [-0.2, 0.0, 0.0],
                [0.0, -0.2, 0.0],
            ],
            dtype=np.float32,
        )
        base_f = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3]],
            dtype=np.int64,
        )
        detail_v = np.vstack([base_v, np.array([[0.0, 0.05, 0.25]], dtype=np.float32)])
        shift = np.array([[1.25, -0.83, 0.4]], dtype=np.float32)
        view_shifted = base_v + shift
        transferred = _transfer_detail_verts_to_view(base_v, base_f, detail_v, view_shifted)
        np.testing.assert_allclose(transferred[-1], np.array([0.0, 0.05, 0.25], dtype=np.float32) + shift[0], atol=1e-4)

        class _FakeHeadPose:
            scale_mean = torch.zeros(28, dtype=torch.float32)
            scale_comps = torch.eye(28, dtype=torch.float32)
            faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=torch.int64)
            mhr = object()

        class _FakeSam:
            head_pose = _FakeHeadPose()

        verts_canon = np.array(
            [[0.0, 0.0, 0.1], [0.15, 0.0, 0.0], [0.0, 1.68, 0.0], [-0.15, 0.0, 0.0], [0.0, 0.85, -0.1]],
            dtype=np.float32,
        )
        joints_canon = np.zeros((128, 3), dtype=np.float32)
        keep_torso = np.ones((5,), dtype=bool)
        fused_stub = FusedSubjectParams(
            shape_params=torch.zeros((1, 45), dtype=torch.float32),
            scale_params=torch.ones((1, 28), dtype=torch.float32),
            weights=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
            distances=np.zeros((4,), dtype=np.float32),
            images_used=4,
            images_total=4,
            mean_confidence=1.0,
        )
        audit_stub = ProportionAudit(
            declared_weight_kg=58.0,
            estimated_weight_kg=57.5,
            weight_diff_pct=0.86,
            mhr_silhouette_iou=0.92,
            outer_silhouette_iou=None,
            status="matched",
            warnings=[],
        )
        fake_circs = {k: 42.0 for k in ANATOMICAL_ORDER}
        fake_loft = [[0.5, 0.25, 0.18]] * 57
        uv_stub = UVLayout(
            uv=np.zeros((5, 2), dtype=np.float32),
            uv_faces=np.array([[0, 1, 2]], dtype=np.int64),
            vmapping=np.arange(5, dtype=np.int64),
            orig_faces=np.array([[0, 1, 2]], dtype=np.int64),
            source="mhr",
        )
        tex_out_stub = TexturedScanOutputs(
            uv_layout=uv_stub,
            tex_face_rgb=np.full((16, 16, 3), 180, dtype=np.uint8),
            tex_blur_rgb=np.full((16, 16, 3), 170, dtype=np.uint8),
            head_uv_mask=np.zeros((16, 16), dtype=bool),
            coverage_mask=np.ones((16, 16), dtype=bool),
            used_pytorch3d=False,
            coverage_pct=88.5,
        )

        sentinel_apose = torch.full((1, 130), 0.25, dtype=torch.float32)
        sentinel_cams = [(np.eye(3, dtype=np.float32), np.array([0.0, 0.84, 2.5], dtype=np.float32), 500.0)] * 4
        sentinel_view_verts = [verts_canon.copy() for _ in range(4)]

        with tempfile.TemporaryDirectory() as tmp, patch(
            "bodyscan_pipeline.inference.load_sam3d_estimator",
            return_value=(_FakeSam(), object()),
        ):
            engine = GpuScanEngine(sam_repo_dir=tmp, checkpoint_path=tmp, mhr_path=tmp)

            for case_name, debug_dict, expect_wired in (
                (
                    "populated_debug",
                    {
                        "apose_params": sentinel_apose,
                        "final_cameras": sentinel_cams,
                        "view_posed_verts": sentinel_view_verts,
                    },
                    True,
                ),
                ("empty_debug_fallback", {}, False),
            ):
                up_dir = os.path.join(tmp, f"up_{case_name}")
                out_dir = os.path.join(tmp, f"out_{case_name}")
                os.makedirs(up_dir, exist_ok=True)
                photos = [_make_sample_image(os.path.join(up_dir, f"p{i}.jpg")) for i in range(4)]

                captured = {"apose_kw": "UNSET", "obs_kwargs": {}, "strip_ret": None, "orig_raw": None}
                orig_raw = [{"view_idx": i, "image_path": photos[i], "output": {"pred_cam_t": [0, 0, 2.5]}} for i in range(4)]
                captured["orig_raw"] = orig_raw
                cleaned_sentinel = [{"cleaned_copy": True}]

                def _fake_strip(items):
                    captured["strip_ret"] = cleaned_sentinel
                    return cleaned_sentinel

                def _fake_build_apose(mhr_m, head_a, fused_arg, h_cm, **kw):
                    captured["apose_kw"] = kw.get("apose_params", "MISSING")
                    return verts_canon.copy(), joints_canon.copy(), keep_torso.copy(), 168.0

                def _fake_build_obs(raw_res, **kw):
                    captured["obs_kwargs"] = dict(kw)
                    return [
                        ViewObservation(
                            image_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
                            pred_vertices=verts_canon.copy(),
                            pred_cam_t=np.array([0.0, 0.84, 2.5], dtype=np.float32),
                            focal_length=500.0,
                            calibrated_camera=bool("fitted_cameras" in kw),
                        )
                    ]

                with patch(
                    "bodyscan_pipeline.inference.run_inference_on_images",
                    return_value=orig_raw,
                ), patch(
                    "bodyscan_pipeline.segmentation.process_views_semantics", return_value=[]
                ), patch("bodyscan_pipeline.segmentation.strip_ram_only_semantics", side_effect=_fake_strip), patch(
                    "bodyscan_pipeline.fusion.fuse_multiview_params", return_value=fused_stub
                ), patch(
                    "bodyscan_pipeline.fitting.fit_mhr_multiview", return_value=(fused_stub, audit_stub, debug_dict)
                ), patch(
                    "bodyscan_pipeline.fusion.build_scaled_apose_mesh", side_effect=_fake_build_apose
                ), patch(
                    "bodyscan_pipeline.measurements.measure_all_circumferences",
                    return_value=(fake_circs, {}),
                ), patch(
                    "bodyscan_pipeline.measurements.compute_loft_profile",
                    return_value=fake_loft,
                ), patch(
                    "bodyscan_pipeline.scan_service._measure_circumferences_and_loft_numpy",
                    return_value=(fake_circs, {}, fake_loft),
                ), patch(
                    "bodyscan_pipeline.measurements.validate_sanity", return_value=[]
                ), patch(
                    "bodyscan_pipeline.texturing.build_view_observations", side_effect=_fake_build_obs
                ), patch(
                    "bodyscan_pipeline.texturing.extract_mhr_uvs", return_value=(np.zeros((5, 2), dtype=np.float32), None)
                ), patch(
                    "bodyscan_pipeline.texturing.run_step3_texturing", return_value=tex_out_stub
                ):
                    res = engine.process(
                        ScanRequest(
                            image_paths=photos,
                            height_cm=168.0,
                            sex="female",
                            weight_kg=58.0,
                            privacy_level=2,
                            reconstruction_mode="calibrated_mhr",
                            consent_granted=True,
                            output_dir=out_dir,
                        )
                    )

                self.assertEqual(res.body_json["textures"]["views_used"], 4)
                self.assertEqual(len(orig_raw), 0)
                self.assertEqual(len(cleaned_sentinel), 0)
                if expect_wired:
                    self.assertIs(captured["apose_kw"], sentinel_apose)
                    self.assertIs(captured["obs_kwargs"].get("fitted_cameras"), sentinel_cams)
                    self.assertIs(captured["obs_kwargs"].get("fitted_view_verts"), sentinel_view_verts)
                else:
                    self.assertEqual(captured["apose_kw"], "MISSING")
                    self.assertNotIn("fitted_cameras", captured["obs_kwargs"])
                    self.assertNotIn("fitted_view_verts", captured["obs_kwargs"])

    def test_mock_engine_duplicate_cardinal_downgrade_and_schema_parity(self) -> None:
        from bodyscan_pipeline.exporter import build_texture_metadata
        from bodyscan_pipeline.scan_service import MockScanEngine, ScanRequest
        from bodyscan_pipeline.texturing import TexturedScanOutputs, UVLayout

        engine = MockScanEngine(0.0)
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Identical file path repeated 4x -> calibrated_mhr downgrade
            single = _make_sample_image(os.path.join(tmp, "only_one.jpg"))
            res_same_path = engine.process(
                ScanRequest(
                    image_paths=[single, single, single, single],
                    height_cm=168.0,
                    sex="female",
                    privacy_level=2,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=os.path.join(tmp, "out_same_path"),
                )
            )
            self.assertEqual(res_same_path.effective_reconstruction_mode, "calibrated_mhr")
            self.assertNotIn("core_glb", res_same_path.artifacts)

            # 2. 4 distinct paths holding identical non-uniform image bytes -> calibrated_mhr downgrade
            grad = np.tile(np.linspace(20, 230, 64, dtype=np.uint8)[None, :, None], (48, 1, 3))
            buf = io.BytesIO()
            Image.fromarray(grad).save(buf, format="PNG")
            nonuniform_bytes = buf.getvalue()
            dup_dir = os.path.join(tmp, "dup_nonuniform")
            os.makedirs(dup_dir, exist_ok=True)
            dup_paths = []
            for idx in range(4):
                p = os.path.join(dup_dir, f"view_{idx}.png")
                Path(p).write_bytes(nonuniform_bytes)
                dup_paths.append(p)
            res_dup_bytes = engine.process(
                ScanRequest(
                    image_paths=dup_paths,
                    height_cm=168.0,
                    sex="female",
                    privacy_level=2,
                    reconstruction_mode="hi_res_head",
                    consent_granted=True,
                    output_dir=os.path.join(tmp, "out_dup_bytes"),
                )
            )
            self.assertEqual(res_dup_bytes.effective_reconstruction_mode, "calibrated_mhr")

            # 3. Conflicting explicit cardinal filenames via source_filenames -> calibrated_mhr downgrade
            conf_dir = os.path.join(tmp, "conf_names")
            os.makedirs(conf_dir, exist_ok=True)
            conf_paths = [_make_sample_image(os.path.join(conf_dir, f"u{i}.jpg")) for i in range(4)]
            res_conflict = engine.process(
                ScanRequest(
                    image_paths=conf_paths,
                    source_filenames=["front_a.jpg", "truoc_b.jpg", "right.jpg", "back.jpg"],
                    height_cm=168.0,
                    sex="female",
                    privacy_level=2,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=os.path.join(tmp, "out_conflict"),
                )
            )
            self.assertEqual(res_conflict.effective_reconstruction_mode, "calibrated_mhr")

            # 4. 4 valid generic solid temp files preserve dual_layer & parity with exporter.build_texture_metadata
            ok_dir = os.path.join(tmp, "ok_generic")
            os.makedirs(ok_dir, exist_ok=True)
            ok_paths = [_make_sample_image(os.path.join(ok_dir, f"p{i}.jpg")) for i in range(4)]
            res_ok = engine.process(
                ScanRequest(
                    image_paths=ok_paths,
                    height_cm=168.0,
                    sex="female",
                    weight_kg=56.0,
                    privacy_level=3,
                    reconstruction_mode="dual_layer",
                    consent_granted=True,
                    output_dir=os.path.join(tmp, "out_ok"),
                )
            )
            self.assertEqual(res_ok.effective_reconstruction_mode, "dual_layer")
            self.assertEqual(res_ok.body_json["mesh"]["file"], "scan.glb")

            fake_uv = UVLayout(
                uv=np.zeros((3, 2), dtype=np.float32),
                uv_faces=np.array([[0, 1, 2]], dtype=np.int64),
                vmapping=np.arange(3, dtype=np.int64),
                orig_faces=np.array([[0, 1, 2]], dtype=np.int64),
                source="mhr",
            )
            fake_tex = TexturedScanOutputs(
                uv_layout=fake_uv,
                tex_face_rgb=np.zeros((512, 512, 3), dtype=np.uint8),
                tex_blur_rgb=np.zeros((512, 512, 3), dtype=np.uint8),
                head_uv_mask=np.zeros((512, 512), dtype=bool),
                coverage_mask=np.ones((512, 512), dtype=bool),
                used_pytorch3d=False,
                head_uv_layout=fake_uv,
                head_tex_face_rgb=np.zeros((512, 512, 3), dtype=np.uint8),
                coverage_pct=100.0,
                head_coverage_pct=100.0,
            )
            gpu_tex_meta = build_texture_metadata(fake_tex, privacy_level=3, views_used=4)
            self.assertEqual(set(res_ok.body_json["textures"].keys()), set(gpu_tex_meta.keys()))

    def test_run_scan_cli_texture_notice_without_enable_texture(self) -> None:
        import contextlib
        from unittest.mock import patch
        import torch
        from bodyscan_pipeline import run_scan
        from bodyscan_pipeline.fitting import ProportionAudit
        from bodyscan_pipeline.fusion import FusedSubjectParams
        from bodyscan_pipeline.scan_service import ANATOMICAL_ORDER

        with tempfile.TemporaryDirectory() as tmp:
            cache_pt = os.path.join(tmp, "step2_raw_outputs.pt")
            head_pt = os.path.join(tmp, "mhr_head_assets.pt")
            raw_list = [{"view_idx": 0, "image_path": "", "output": {"shape_params": np.zeros(45), "scale_params": np.zeros(28)}}]
            torch.save(raw_list, cache_pt)
            fake_head_assets = {
                "scale_mean": torch.zeros(28, dtype=torch.float32),
                "scale_comps": torch.eye(28, dtype=torch.float32),
                "faces": torch.tensor([[0, 1, 2]], dtype=torch.int64),
            }
            torch.save(fake_head_assets, head_pt)
            out_dir = os.path.join(tmp, "out")
            verts = np.array([[0.0, 0.0, 0.1], [0.1, 0.8, 0.0], [0.0, 1.65, 0.0]], dtype=np.float32)
            joints = np.zeros((128, 3), dtype=np.float32)
            keep_t = np.ones((3,), dtype=bool)
            fused = FusedSubjectParams(
                shape_params=torch.zeros((1, 45), dtype=torch.float32),
                scale_params=torch.ones((1, 28), dtype=torch.float32),
                weights=np.array([1.0], dtype=np.float32),
                distances=np.array([0.0], dtype=np.float32),
                images_used=1,
                images_total=1,
                mean_confidence=1.0,
            )
            audit = ProportionAudit(55.0, 55.0, 0.0, 0.9, None, "matched", [])
            circs = {k: 40.0 for k in ANATOMICAL_ORDER}
            loft = [[0.5, 0.25, 0.18]] * 57

            argv = [
                "run_scan.py",
                "--cached_outputs",
                cache_pt,
                "--head_assets",
                head_pt,
                "--height_cm",
                "165.0",
                "--sex",
                "female",
                "--output_dir",
                out_dir,
                "--privacy_level",
                "3",
                "--reconstruction_mode",
                "calibrated_mhr",
            ]
            buf = io.StringIO()
            with patch("sys.argv", argv), patch(
                "torch.jit.load", return_value=object()
            ), patch(
                "bodyscan_pipeline.run_scan.fuse_multiview_params", return_value=fused
            ), patch(
                "bodyscan_pipeline.run_scan.build_scaled_apose_mesh",
                return_value=(verts, joints, keep_t, 165.0),
            ), patch(
                "bodyscan_pipeline.run_scan.evaluate_proportion_audit", return_value=audit
            ), patch(
                "bodyscan_pipeline.run_scan._measure_circumferences_and_loft_numpy",
                return_value=(circs, {}, loft),
            ), patch(
                "bodyscan_pipeline.run_scan.validate_sanity", return_value=[]
            ), patch(
                "bodyscan_pipeline.run_scan.trimesh", None
            ), contextlib.redirect_stdout(buf):
                run_scan.main()

            stdout_text = buf.getvalue()
            self.assertIn("--enable_texture", stdout_text)
            body_json = json.loads(Path(os.path.join(out_dir, "body_scan.json")).read_text(encoding="utf-8"))
            self.assertIsNone(body_json.get("textures"))
            self.assertTrue(any("--enable_texture" in w for w in body_json.get("warnings", [])))

    def test_transfer_detail_verts_to_view_prevents_cross_part_binding_and_preserves_same_len_offsets(self) -> None:
        from bodyscan_pipeline.scan_service import _transfer_detail_verts_to_view

        def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
            v = np.array(
                [
                    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                ],
                dtype=np.float32,
            )
            f = np.array(
                [
                    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
                ],
                dtype=np.int64,
            )
            return v, f

        torso_v, torso_f = _box(-0.15, 0.15, 0.80, 1.40, -0.10, 0.10)
        arm_v, arm_f = _box(0.19, 0.55, 1.20, 1.34, -0.06, 0.06)
        base_verts_a = np.vstack([torso_v, arm_v])
        base_faces = np.vstack([torso_f, arm_f + len(torso_v)])

        # Re-posed view B drops the right arm downward by 0.50 m while torso stays static
        view_verts_b = base_verts_a.copy()
        view_verts_b[len(torso_v) :, 1] -= 0.50

        # Coat verts anchored on torso right wall (x=0.15, y=1.22..1.32), displaced outward (+0.048m to x=0.198,
        # spatially closer to A-pose arm x=0.19 than torso x=0.15!)
        undisp_coat = np.array(
            [
                [0.15, 1.22, -0.04],
                [0.15, 1.30, -0.04],
                [0.15, 1.22, 0.04],
                [0.15, 1.30, 0.04],
            ],
            dtype=np.float32,
        )
        disp_coat = undisp_coat.copy()
        disp_coat[:, 0] += 0.048
        detail_verts_a = np.vstack([base_verts_a, disp_coat])
        undisplaced_verts_a = np.vstack([base_verts_a, undisp_coat])

        transferred = _transfer_detail_verts_to_view(
            detail_verts_a,
            base_verts_a,
            view_verts_b,
            base_faces=base_faces,
            undisplaced_verts_a=undisplaced_verts_a,
        )
        coat_transferred = transferred[len(base_verts_a) :]
        # Without undisplaced_verts_a & Voronoi clamp, coat verts cross-bind to arm (y drops by -0.50 m).
        # With the fix, coat stays on torso side wall around y=1.22..1.30 and x ~ 0.198.
        self.assertGreater(float(np.min(coat_transferred[:, 1])), 1.15)
        self.assertLess(float(np.max(np.abs(coat_transferred - disp_coat))), 0.015)

        # Same vertex count preserves valid outward detail offset (+0.02 m) and clamps spikes (0.50 m -> <= 0.065 m)
        normals = np.zeros_like(base_verts_a)
        for tri in base_faces:
            e1 = base_verts_a[tri[1]] - base_verts_a[tri[0]]
            e2 = base_verts_a[tri[2]] - base_verts_a[tri[0]]
            fn = np.cross(e1, e2)
            for vi in tri:
                normals[vi] += fn
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        same_len_vd = base_verts_a + 0.02 * normals
        same_len_vd[0] = base_verts_a[0] + 0.50 * normals[0]
        same_transferred = _transfer_detail_verts_to_view(
            same_len_vd, base_verts_a, view_verts_b, base_faces=base_faces
        )
        preserved_off = float(np.linalg.norm(same_transferred[1] - view_verts_b[1]))
        clamped_off = float(np.linalg.norm(same_transferred[0] - view_verts_b[0]))
        self.assertAlmostEqual(preserved_off, 0.02, delta=1e-3)
        self.assertLessEqual(clamped_off, 0.065 + 1e-4)

    def test_slice_loop_2d_numpy_max_radius_and_backend_capabilities(self) -> None:
        import math
        from bodyscan_pipeline.scan_service import (
            MockScanEngine,
            _slice_loop_2d_numpy,
            get_backend_capabilities,
        )

        def _cyl_y(cx: float, radius: float, sections: int) -> tuple[np.ndarray, np.ndarray]:
            pts = []
            ang = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
            for y in (0.4, 1.2):
                for th in ang:
                    pts.append([cx + radius * math.cos(th), y, radius * math.sin(th)])
            faces = []
            for k in range(sections):
                kn = (k + 1) % sections
                faces.append([k, kn, sections + kn])
                faces.append([k, sections + kn, sections + k])
            return np.asarray(pts, dtype=np.float32), np.asarray(faces, dtype=np.int64)

        vt, ft = _cyl_y(0.0, 0.0953, 32)   # ~59.9 cm torso
        vw, fw = _cyl_y(0.25, 0.030, 3)    # sparse 3-vertex wrist loop
        verts = np.vstack([vt, vw])
        faces = np.vstack([ft, fw + len(vt)])
        _, circ_cm = _slice_loop_2d_numpy(
            verts,
            faces,
            origin=np.array([0.25, 0.85, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            pick="closest",
            max_radius_m=0.14,
        )
        self.assertGreater(circ_cm, 8.0)
        self.assertLess(circ_cm, 22.0)

        caps = get_backend_capabilities(MockScanEngine(0.0))
        self.assertEqual(caps["segmentation"], "mock")
        self.assertEqual(caps["rasterizer"], "mock")
        self.assertIn("gpu_name", caps)
        self.assertIn("pytorch3d_available", caps)
        self.assertIn("moge2_available", caps)

    def test_build_fallback_planar_uvs_chart_separation_and_truthful_uv_source(self):
        from unittest import mock
        from bodyscan_pipeline.scan_service import _build_fallback_planar_uvs
        from bodyscan_pipeline import texturing

        verts = np.array(
            [
                [-0.18, 0.80, 0.08], [0.18, 0.80, 0.08], [0.0, 1.30, 0.08],
                [-0.18, 0.80, -0.08], [0.18, 0.80, -0.08], [0.0, 1.30, -0.08],
                [-0.08, 1.50, 0.09], [0.08, 1.50, 0.09], [0.0, 1.65, 0.09],
                [-0.08, 1.50, -0.09], [0.08, 1.50, -0.09], [0.0, 1.65, -0.09],
                [-0.10, 1.68, 0.11], [0.10, 1.68, 0.11], [0.0, 1.78, 0.11],
                [-0.09, 1.66, 0.06], [0.09, 1.66, 0.06], [0.0, 1.75, 0.06],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [4, 3, 5],
                [6, 7, 8],
                [10, 9, 11],
                [12, 13, 14],
                [16, 15, 17],
            ],
            dtype=np.int64,
        )
        head_mask = np.zeros(len(verts), dtype=bool)
        head_mask[6:] = True
        hair_mask = np.zeros(len(verts), dtype=bool)
        hair_mask[12:] = True
        uvs, uv_faces = _build_fallback_planar_uvs(
            verts, faces, head_vert_mask=head_mask, hair_vert_mask=hair_mask
        )
        boxes = []
        for f_idx in range(len(faces)):
            tri_uv = uvs[uv_faces[f_idx]]
            boxes.append((
                float(tri_uv[:, 0].min()), float(tri_uv[:, 0].max()),
                float(tri_uv[:, 1].min()), float(tri_uv[:, 1].max()),
            ))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x0a, x1a, y0a, y1a = boxes[i]
                x0b, x1b, y0b, y1b = boxes[j]
                overlap_x = min(x1a, x1b) - max(x0a, x0b)
                overlap_y = min(y1a, y1b) - max(y0a, y0b)
                self.assertFalse(overlap_x > 1e-4 and overlap_y > 1e-4, f"Charts {i} and {j} overlap")

        import importlib.util
        from bodyscan_pipeline.scan_service import get_backend_capabilities

        with mock.patch.object(importlib.util, "find_spec", side_effect=lambda n: None if n == "xatlas" else importlib.machinery.ModuleSpec(n, None)):
            caps_no_xatlas = get_backend_capabilities(None, engine_mode="gpu")
            self.assertEqual(caps_no_xatlas["uv_unwrapper"], "planar_fallback")

    def test_transfer_detail_verts_uncalibrated_closeup_scale_and_long_hair(self):
        from bodyscan_pipeline.scan_service import _transfer_detail_verts_to_view

        # Outward-wound open cylinder (ring0 y=0, ring1 y=0.25) so normals point +radial
        theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ring0 = np.stack([0.12 * np.cos(theta), np.zeros_like(theta), 0.12 * np.sin(theta)], axis=-1)
        ring1 = np.stack([0.12 * np.cos(theta), np.full_like(theta, 0.25), 0.12 * np.sin(theta)], axis=-1)
        base_a = np.vstack([ring0, ring1]).astype(np.float32)
        faces_list = []
        n = len(theta)
        for i in range(n):
            j = (i + 1) % n
            faces_list.append([i, n + j, j])
            faces_list.append([i, n + i, n + j])
        faces = np.asarray(faces_list, dtype=np.int64)

        # Hair offset of +0.08 m outward radially on upper ring
        detail_a = base_a.copy()
        detail_a[n:, 0] *= 1.0 + (0.08 / 0.12)
        detail_a[n:, 2] *= 1.0 + (0.08 / 0.12)

        scale = 0.846
        view_b = (base_a * scale).astype(np.float32)
        transferred = _transfer_detail_verts_to_view(
            detail_verts_a=detail_a,
            base_verts_a=base_a,
            view_verts_b=view_b,
            base_faces=faces,
            undisplaced_verts_a=base_a,
        )
        rad_b = np.linalg.norm(view_b[n:, [0, 2]], axis=1)
        rad_t = np.linalg.norm(transferred[n:, [0, 2]], axis=1)
        disp_obs = float(np.mean(rad_t - rad_b))
        self.assertAlmostEqual(disp_obs, 0.08 * scale, delta=0.008)

    def test_write_multi_material_glb_sanitizes_normals_npot_and_double_sided(self):
        import io
        from PIL import Image
        from bodyscan_pipeline.scan_service import write_multi_material_glb

        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
        uvs = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def _jpeg_bytes(w: int, h: int, color) -> bytes:
            buf = io.BytesIO()
            Image.new("RGB", (w, h), color).save(buf, format="JPEG", quality=85)
            return buf.getvalue()

        npot_bytes = _jpeg_bytes(300, 300, (180, 150, 130))
        pot_bytes = _jpeg_bytes(256, 256, (120, 110, 100))

        with tempfile.TemporaryDirectory() as tmp:
            glb_path = os.path.join(tmp, "out.glb")
            write_multi_material_glb(
                [
                    {"name": "body", "verts": verts, "faces": faces, "uvs": uvs, "texture_jpeg_bytes": npot_bytes},
                    {"name": "head", "verts": verts, "faces": faces, "uvs": uvs, "texture_jpeg_bytes": pot_bytes},
                ],
                glb_path,
            )
            _, _, _, _, _, gltf, _, bin_chunk = _parse_glb_chunks(Path(glb_path).read_bytes())
            for mat in gltf["materials"]:
                self.assertFalse(mat.get("doubleSided", True))
            samplers = gltf["samplers"]
            self.assertEqual(samplers[0]["minFilter"], 9729)
            self.assertEqual(samplers[1]["minFilter"], 9987)
            for prim in gltf["meshes"][0]["primitives"]:
                n_acc = gltf["accessors"][prim["attributes"]["NORMAL"]]
                n_bv = gltf["bufferViews"][n_acc["bufferView"]]
                off = int(n_bv.get("byteOffset", 0)) + int(n_acc.get("byteOffset", 0))
                normals = np.frombuffer(bin_chunk[off : off + int(n_acc["count"]) * 12], dtype=np.float32).reshape(-1, 3)
                norms = np.linalg.norm(normals, axis=1)
                self.assertTrue(np.all(np.isfinite(normals)))
                self.assertTrue(np.allclose(norms, 1.0, atol=1e-4))
                self.assertTrue(np.allclose(normals[1], [0.0, 1.0, 0.0], atol=1e-4))

    def test_vietnamese_nfd_role_inference_and_explicit_slot_roles(self):
        import unicodedata
        from bodyscan_pipeline.scan_service import (
            MockScanEngine,
            ScanRequest,
            _infer_cardinal_role_from_name,
            _count_mock_unique_angles,
        )

        nfd_front = unicodedata.normalize("NFD", "ảnh_chính_diện_01.jpg")
        nfd_right = unicodedata.normalize("NFD", "góc_phải.png")
        nfd_left = unicodedata.normalize("NFD", "bên_trái.png")
        nfd_back = unicodedata.normalize("NFD", "mặt_sau.png")
        self.assertEqual(_infer_cardinal_role_from_name(nfd_front), "front")
        self.assertEqual(_infer_cardinal_role_from_name(nfd_right), "right")
        self.assertEqual(_infer_cardinal_role_from_name(nfd_left), "left")
        self.assertEqual(_infer_cardinal_role_from_name(nfd_back), "back")

        with tempfile.TemporaryDirectory() as tmp:
            p0 = _make_sample_image(os.path.join(tmp, "img_a.jpg"), (120, 120, 120))
            p1 = _make_sample_image(os.path.join(tmp, "img_b.jpg"), (120, 120, 120))
            # Duplicate same filename+bytes or identical explicit slot_roles collapse to 1 unique angle;
            # distinct explicit slot_roles expand to 2 unique angles.
            self.assertEqual(_count_mock_unique_angles([p0, p1], source_filenames=["img.jpg", "img.jpg"]), 1)
            self.assertEqual(_count_mock_unique_angles([p0, p1], slot_roles=["front", "front"]), 1)
            self.assertEqual(_count_mock_unique_angles([p0, p1], slot_roles=["front", "back"]), 2)
            eng = MockScanEngine(0.0)
            res = eng.process(
                ScanRequest(
                    image_paths=[p0, p1],
                    height_cm=166.0,
                    output_dir=os.path.join(tmp, "out"),
                    consent_granted=True,
                    slot_roles=["front", "right"],
                )
            )
            self.assertEqual(res.images_used, 2)
            self.assertEqual(res.body_json["textures"]["views_used"], 2)

    def test_demo_ui_consent_gate_temp_cleanup_and_file_filtering(self):
        import shutil
        import time
        from unittest import mock
        from bodyscan_pipeline import demo_ui

        with tempfile.TemporaryDirectory() as tmp:
            img_path = _make_sample_image(os.path.join(tmp, "front.jpg"), (128, 128, 128))
            fake_engine = mock.MagicMock()
            fake_engine._lock = mock.MagicMock()
            fake_engine.model = None
            fake_engine.estimator = None
            fake_engine.process.side_effect = RuntimeError("gpu failed mid-run")

            with mock.patch.object(demo_ui, "GpuScanEngine", return_value=fake_engine):
                demo_eng = demo_ui.ScanDemoEngine(".", "ckpt", "mhr")

            with self.assertRaises(ValueError):
                demo_eng.process([img_path], None, 166.0, 56.0, 2, "dual_layer", consent_granted=False)

            before_dirs = set(Path(tempfile.gettempdir()).glob("alphascan_demo_*"))
            with self.assertRaises(Exception):
                demo_eng.process([img_path], None, 166.0, 56.0, 2, "dual_layer", consent_granted=True)
            after_dirs = set(Path(tempfile.gettempdir()).glob("alphascan_demo_*"))
            self.assertEqual(after_dirs - before_dirs, set())

            old_dir = Path(tempfile.mkdtemp(prefix="alphascan_demo_oldtest_"))
            try:
                old_ts = time.time() - 7200.0
                os.utime(old_dir, (old_ts, old_ts))
                demo_ui._prune_stale_demo_dirs(max_age_sec=3600.0, keep_recent=0)
                self.assertFalse(old_dir.exists())
            finally:
                shutil.rmtree(old_dir, ignore_errors=True)

    def test_preserve_raised_arm_pose_and_height_scaling(self) -> None:
        from types import SimpleNamespace
        import torch
        from bodyscan_pipeline.fusion import FusedSubjectParams, build_scaled_apose_mesh, build_scaled_posed_mesh
        from bodyscan_pipeline.scan_service import _select_primary_view_pose

        # Build a minimal humanoid vertex set (feet=0.0m, hips=0.90m, head=1.70m, right wrist=0.85m in A-pose)
        base_verts_cm = torch.tensor(
            [
                [0.0, 0.0, 0.0],       # 0: feet
                [-10.0, 90.0, 0.0],    # 1: hip L
                [10.0, 90.0, 0.0],     # 2: hip R
                [0.0, 170.0, 0.0],     # 3: crown of head
                [35.0, 85.0, 0.0],     # 4: right wrist (moves up when pose[0, 10] > 0)
            ],
            dtype=torch.float32,
        )
        base_joints_cm = torch.zeros(128, 3, dtype=torch.float32)
        base_joints_cm[2] = torch.tensor([-10.0, 90.0, 0.0])
        base_joints_cm[18] = torch.tensor([10.0, 90.0, 0.0])
        base_joints_cm[42] = torch.tensor([35.0, 85.0, 0.0])  # right wrist

        class _FakeLBS:
            vert_indices_flattened = torch.arange(5, dtype=torch.long)
            skin_indices_flattened = torch.tensor([2, 2, 18, 2, 42], dtype=torch.long)
            skin_weights_flattened = torch.ones(5, dtype=torch.float32)

        class _PoseMHR:
            character_torch = SimpleNamespace(linear_blend_skinning=_FakeLBS())

            def __call__(self, shape_t, model_params, expr_vec):
                # body_pose_params start at index 6 in model_params
                arm_raise = float(model_params[0, 6 + 10].item())
                v = base_verts_cm.clone()
                j = base_joints_cm.clone()
                # Raising arm lifts right wrist by +115 cm (above 170 cm head -> 200 cm!)
                v[4, 1] += 115.0 * arm_raise
                j[42, 1] += 115.0 * arm_raise
                return v.unsqueeze(0), j.unsqueeze(0)

        mhr = _PoseMHR()
        head_assets = {
            "scale_mean": torch.zeros(68, dtype=torch.float32),
            "scale_comps": torch.zeros(28, 68, dtype=torch.float32),
        }
        fused = FusedSubjectParams(
            shape_params=torch.zeros(1, 45, dtype=torch.float32),
            scale_params=torch.zeros(1, 28, dtype=torch.float32),
            expr_params=torch.zeros(1, 72, dtype=torch.float32),
            weights=[1.0],
            distances=[0.0],
            images_used=1,
            images_total=1,
            mean_confidence=0.95,
        )
        apose = torch.zeros(1, 130, dtype=torch.float32)
        raised_pose = torch.zeros(1, 130, dtype=torch.float32)
        raised_pose[0, 10] = 1.0  # arm raised high

        verts_a, joints_a, _, raw_h_cm = build_scaled_apose_mesh(mhr, head_assets, fused, 170.0, apose_params=apose)
        self.assertAlmostEqual(raw_h_cm, 170.0, places=2)
        self.assertAlmostEqual(float(verts_a[3, 1]), 1.70, places=3)
        self.assertAlmostEqual(float(verts_a[4, 1]), 0.85, places=3)

        # Select primary view pose (must choose front view over side view)
        views = [
            SimpleNamespace(role="right", azimuth_deg=90.0),
            SimpleNamespace(role="front", azimuth_deg=0.0),
        ]
        chosen = _select_primary_view_pose(
            segmented_views=views,
            fit_debug={"view_poses": [apose, raised_pose]},
            raw_results=[],
            apose_params=apose,
        )
        self.assertIsNotNone(chosen)
        self.assertAlmostEqual(float(chosen[0, 10].item()), 1.0, places=4)

        # Build posed mesh: feet stay at y=0, head stays at 1.70m (NO shrinking!), wrist rises to 2.00m
        verts_p, joints_p = build_scaled_posed_mesh(
            mhr,
            head_assets,
            fused,
            chosen,
            height_cm=170.0,
            raw_apose_height_cm=raw_h_cm,
            apose_params=apose,
        )
        self.assertAlmostEqual(float(verts_p[:, 1].min()), 0.0, places=4)
        self.assertAlmostEqual(float(verts_p[3, 1]), 1.70, places=3)  # head height preserved
        self.assertGreater(float(verts_p[4, 1]), float(verts_p[3, 1]) + 0.20)  # raised wrist above head
        self.assertAlmostEqual(float(joints_p[42, 1]), 2.00, places=3)


if __name__ == "__main__":
    unittest.main()
