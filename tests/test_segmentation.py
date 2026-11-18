"""Unit tests for foreground segmentation, view azimuth inference, head semantics, and S2 RAM-only stripping."""

import os
import sys
import tempfile
import unittest
import warnings

import cv2
import numpy as np
from PIL import Image


def _make_synthetic_person_rgb(height: int = 180, width: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic studio RGB image, ground-truth mask, and bbox [x0, y0, x1, y1]."""
    rgb = np.full((height, width, 3), (225, 225, 230), dtype=np.uint8)
    gt_mask = np.zeros((height, width), dtype=bool)

    # Head
    cv2.circle(rgb, (60, 34), 13, (215, 175, 145), -1)
    cv2.circle(gt_mask.view(np.uint8), (60, 34), 13, 1, -1)
    # Torso / shirt
    cv2.rectangle(rgb, (42, 48), (78, 108), (40, 75, 130), -1)
    cv2.rectangle(gt_mask.view(np.uint8), (42, 48), (78, 108), 1, -1)
    # Legs / pants
    cv2.rectangle(rgb, (44, 109), (57, 160), (35, 45, 70), -1)
    cv2.rectangle(gt_mask.view(np.uint8), (44, 109), (57, 160), 1, -1)
    cv2.rectangle(rgb, (63, 109), (76, 160), (35, 45, 70), -1)
    cv2.rectangle(gt_mask.view(np.uint8), (63, 109), (76, 160), 1, -1)

    bbox = np.array([30.0, 12.0, 90.0, 168.0], dtype=np.float32)
    return rgb, gt_mask, bbox


def _rotate_y_mhr_to_opencv(pts_mhr: np.ndarray, deg: float) -> np.ndarray:
    """Rotate canonical MHR (+X left, +Y up, +Z front) by deg around +Y and convert to OpenCV (+Y down, -Z front)."""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    pts_rot = pts_mhr @ rot_y.T
    pts_cv = pts_rot.copy()
    pts_cv[:, 1] = -pts_rot[:, 1]
    pts_cv[:, 2] = -pts_rot[:, 2]
    pts_cv += np.array([0.0, 0.8, 2.6], dtype=np.float32)
    return pts_cv


def _build_canonical_mhr_keypoints() -> np.ndarray:
    """Construct canonical MHR-like 127-joint keypoints (+X left, +Y up, +Z front)."""
    kpts = np.zeros((127, 3), dtype=np.float32)
    kpts[0] = [0.0, 1.60, 0.12]      # Nose (+Z anterior)
    kpts[2] = [0.10, 0.90, 0.0]      # Left hip
    kpts[18] = [-0.10, 0.90, 0.0]    # Right hip
    kpts[39] = [-0.20, 1.40, 0.0]    # Right shoulder (-X in MHR)
    kpts[75] = [0.20, 1.40, 0.0]     # Left shoulder (+X in MHR)
    kpts[110] = [0.0, 1.42, 0.0]     # NECK_BASE_JOINT
    kpts[113] = [0.0, 1.50, 0.0]     # NECK_HEAD_JOINT
    return kpts


class TestSegmentation(unittest.TestCase):
    """Test foreground masking, azimuth sorting, head semantics, and RAM-only privacy stripping."""

    def test_segment_person_mask_erode_and_license_guard(self) -> None:
        from bodyscan_pipeline.segmentation import segment_person_mask

        rgb, gt_mask, bbox = _make_synthetic_person_rgb()
        mask_raw = segment_person_mask(rgb, bbox=bbox, human_segmentor=None, erode_px=0)
        mask_eroded = segment_person_mask(rgb, bbox=bbox, human_segmentor=None, erode_px=3)

        self.assertEqual(mask_raw.dtype, np.bool_)
        self.assertEqual(mask_raw.shape, rgb.shape[:2])
        inter = int(np.sum(mask_raw & gt_mask))
        union = int(np.sum(mask_raw | gt_mask))
        self.assertGreater(inter / max(union, 1), 0.85)
        self.assertFalse(bool(np.any(mask_raw[:8, :])))

        # Erosion strictly trims boundary pixels without inventing new pixels
        self.assertLess(int(mask_eroded.sum()), int(mask_raw.sum()))
        self.assertTrue(bool(np.all(mask_raw | (~mask_eroded))))

        # Custom human_segmentor hook (run_sam interface) works cleanly and applies erosion
        class DummySAMSegmentor:
            def run_sam(self, img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
                return gt_mask.astype(np.uint8)[None, ...]

        hook_eroded = segment_person_mask(rgb, bbox=bbox, human_segmentor=DummySAMSegmentor(), erode_px=2)
        self.assertLess(int(hook_eroded.sum()), int(gt_mask.sum()))
        self.assertTrue(bool(np.all(gt_mask | (~hook_eroded))))

        # Fallback without bbox also stays non-empty and bounded
        mask_no_bbox = segment_person_mask(rgb, bbox=None, human_segmentor=None, erode_px=0)
        self.assertGreater(int(mask_no_bbox.sum()), 500)

        # License guard must reject forbidden modules
        sys.modules["ultralytics"] = type(sys)("ultralytics")
        try:
            with self.assertRaises(RuntimeError):
                segment_person_mask(rgb, bbox=bbox)
        finally:
            sys.modules.pop("ultralytics", None)

    def test_infer_view_azimuth_handles_shuffled_cardinal_views(self) -> None:
        from bodyscan_pipeline.segmentation import (
            classify_view_azimuth,
            infer_view_azimuth,
            process_views_semantics,
        )

        kpts_mhr = _build_canonical_mhr_keypoints()
        shuffled_angles = [180.0, 270.0, 0.0, 90.0]
        expected_labels = ["back", "left", "front", "right"]

        inferred_az = []
        raw_results = []
        rgb, _, bbox = _make_synthetic_person_rgb()
        for idx, deg in enumerate(shuffled_angles):
            kpts_cv = _rotate_y_mhr_to_opencv(kpts_mhr, deg)
            az = infer_view_azimuth(kpts_cv)
            inferred_az.append(az)
            raw_results.append(
                {
                    "image_path": f"view_{idx}.jpg",
                    "image_name": f"view_{idx}.jpg",
                    "image_rgb": rgb.copy(),
                    "output": {
                        "bbox": bbox.copy(),
                        "pred_keypoints_3d": kpts_cv,
                    },
                }
            )

        for got, exp in zip(inferred_az, shuffled_angles):
            diff = abs(((got - exp + 180.0) % 360.0) - 180.0)
            self.assertLess(diff, 1.0)
        self.assertEqual([classify_view_azimuth(a) for a in inferred_az], expected_labels)

        # Verify canonical +Y up MHR keypoints, oblique 45 deg, and None fallback
        rad45 = np.deg2rad(45.0)
        rot45 = np.array([[np.cos(rad45), 0.0, np.sin(rad45)], [0.0, 1.0, 0.0], [-np.sin(rad45), 0.0, np.cos(rad45)]], dtype=np.float32)
        az_mhr45 = infer_view_azimuth(kpts_mhr @ rot45.T)
        self.assertAlmostEqual(az_mhr45, 45.0, delta=1.0)
        self.assertEqual(classify_view_azimuth(az_mhr45), "oblique")
        self.assertEqual(infer_view_azimuth(None, fallback_index=2, total_views=4), 180.0)

        # Verify process_views_semantics preserves order and populates SegmentedView labels
        views = process_views_semantics(raw_results)
        self.assertEqual(len(views), 4)
        self.assertEqual([v.view_index for v in views], [0, 1, 2, 3])
        self.assertEqual([v.view_label for v in views], expected_labels)

    def test_extract_head_semantics_hair_exclusion_and_nose_profile(self) -> None:
        from bodyscan_pipeline.segmentation import extract_head_semantics

        h, w = 220, 160
        neck_y_px = 112.0
        rgb = np.full((h, w, 3), (235, 235, 240), dtype=np.uint8)
        person_mask = np.zeros((h, w), dtype=bool)
        hair_gt = np.zeros((h, w), dtype=bool)

        # Torso below neck_y_px
        cv2.rectangle(rgb, (56, 113), (104, 200), (60, 110, 180), -1)
        cv2.rectangle(person_mask.view(np.uint8), (56, 113), (104, 200), 1, -1)

        # Face skin on right-facing profile (forehead baseline x=94, chin x=96)
        cv2.rectangle(rgb, (64, 42), (94, 112), (218, 176, 144), -1)
        cv2.rectangle(person_mask.view(np.uint8), (64, 42), (94, 112), 1, -1)
        # Prominent nose protrusion peaking at x=112 around y=68..76
        nose_pts = np.array([[94, 58], [112, 72], [94, 80]], dtype=np.int32)
        cv2.fillPoly(rgb, [nose_pts], (218, 176, 144))
        cv2.fillPoly(person_mask.view(np.uint8), [nose_pts], 1)

        # Upper/back hair bun strictly above neck_y_px (y=16..48, x=48..84)
        cv2.circle(rgb, (64, 30), 15, (32, 24, 20), -1)
        cv2.circle(person_mask.view(np.uint8), (64, 30), 15, 1, -1)
        cv2.circle(hair_gt.view(np.uint8), (64, 30), 15, 1, -1)

        sem = extract_head_semantics(
            rgb=rgb,
            person_mask=person_mask,
            neck_y_px=neck_y_px,
            view_label="right",
        )
        self.assertFalse(sem["collar_hair_overlap"])
        self.assertEqual(sem["warnings"], [])
        self.assertTrue(np.array_equal(sem["body_mask"], person_mask & (~sem["hair_mask"])))

        # Hair bun excluded from body_mask and matched by hair_mask
        hair_iou = np.sum(sem["hair_mask"] & hair_gt) / max(1, np.sum(sem["hair_mask"] | hair_gt))
        self.assertGreater(float(hair_iou), 0.75)
        self.assertFalse(bool(np.any(sem["body_mask"][16:28, 56:72])))

        # Nose profile curve captures prominent nose tip x > forehead baseline (94)
        curve = sem["nose_profile_curve"]
        self.assertIsNotNone(curve)
        self.assertEqual(curve.ndim, 2)
        self.assertEqual(curve.shape[1], 2)
        self.assertGreaterEqual(len(curve), 10)
        peak_x = float(curve[:, 0].max())
        forehead_x = float(np.median(curve[:4, 0]))
        self.assertGreaterEqual(peak_x, 108.0)
        self.assertGreater(peak_x - forehead_x, 10.0)

        # Left-profile mirrored horizontally captures nose tip min_x < forehead baseline
        sem_left = extract_head_semantics(
            rgb=rgb[:, ::-1, :].copy(),
            person_mask=person_mask[:, ::-1].copy(),
            neck_y_px=neck_y_px,
            view_label="left",
        )
        left_curve = sem_left["nose_profile_curve"]
        self.assertIsNotNone(left_curve)
        left_forehead_x = float(np.median(left_curve[:4, 0]))
        left_peak_x = float(left_curve[:, 0].min())
        self.assertGreater(left_forehead_x - left_peak_x, 10.0)

        # Front view yields nose_profile_curve=None
        sem_front = extract_head_semantics(rgb=rgb, person_mask=person_mask, neck_y_px=neck_y_px, view_label="front")
        self.assertIsNone(sem_front["nose_profile_curve"])

    def test_extract_head_semantics_collar_overlap_warning(self) -> None:
        from bodyscan_pipeline.segmentation import extract_head_semantics

        h, w = 220, 160
        neck_y_px = 105.0
        rgb = np.full((h, w, 3), (235, 235, 240), dtype=np.uint8)
        person_mask = np.zeros((h, w), dtype=bool)

        # Shirt / shoulders below neck_y_px
        cv2.rectangle(rgb, (45, 106), (115, 200), (65, 120, 185), -1)
        cv2.rectangle(person_mask.view(np.uint8), (45, 106), (115, 200), 1, -1)
        # Head skin
        cv2.circle(rgb, (80, 72), 24, (215, 175, 145), -1)
        cv2.circle(person_mask.view(np.uint8), (80, 72), 24, 1, -1)
        # Long dark hair draping from crown (y=28) well below neck_y_px=105 down to y=155 onto shoulders
        cv2.rectangle(rgb, (56, 28), (104, 60), (30, 22, 18), -1)
        cv2.rectangle(person_mask.view(np.uint8), (56, 28), (104, 60), 1, -1)
        cv2.rectangle(rgb, (52, 58), (64, 155), (30, 22, 18), -1)
        cv2.rectangle(person_mask.view(np.uint8), (52, 58), (64, 155), 1, -1)
        cv2.rectangle(rgb, (96, 58), (108, 155), (30, 22, 18), -1)
        cv2.rectangle(person_mask.view(np.uint8), (96, 58), (108, 155), 1, -1)

        sem = extract_head_semantics(
            rgb=rgb,
            person_mask=person_mask,
            neck_y_px=neck_y_px,
            view_label="front",
        )
        self.assertTrue(sem["collar_hair_overlap"])
        self.assertGreaterEqual(len(sem["warnings"]), 1)
        joined = " ".join(sem["warnings"]).lower()
        self.assertIn("collar", joined)
        self.assertIn("hair", joined)

    def test_strip_raw_images_removes_head_crops_and_landmarks(self) -> None:
        from bodyscan_pipeline.inference import run_inference_on_images, strip_raw_images_for_cache
        from bodyscan_pipeline.segmentation import (
            SegmentedView,
            extract_face_landmarks_ram,
            strip_ram_only_semantics,
        )

        # High-resolution synthetic portrait image (2400x1800)
        hi_img = np.full((2400, 1800, 3), (220, 220, 225), dtype=np.uint8)
        cv2.rectangle(hi_img, (500, 200), (1300, 2200), (50, 90, 150), -1)
        cv2.circle(hi_img, (900, 360), 140, (215, 175, 145), -1)

        class DummyEstimator:
            def process_one_image(self, rgb: np.ndarray, bbox_thr: float = 0.5, use_mask: bool = False):
                h_in, w_in = rgb.shape[:2]
                return [
                    {
                        "bbox": np.array([0.25 * w_in, 0.08 * h_in, 0.75 * w_in, 0.92 * h_in], dtype=np.float32),
                        "pred_keypoints_3d": _rotate_y_mhr_to_opencv(_build_canonical_mhr_keypoints(), 0.0),
                        "depth_map": np.full((h_in, w_in), 2.5, dtype=np.float32),
                    }
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "front.png")
            Image.fromarray(hi_img).save(img_path)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results = run_inference_on_images(
                    DummyEstimator(),
                    [img_path],
                    max_side=600,
                    keep_rgb=True,
                    keep_head_crop=True,
                    head_max_side=3200,
                    extract_depth=True,
                )
                lm = extract_face_landmarks_ram(results[0]["head_crop_rgb"])

            self.assertEqual(len(caught), 0)
            self.assertIsNone(lm)  # mediapipe not installed on Windows CPU -> clean None
            self.assertEqual(len(results), 1)
            rec = results[0]
            self.assertIn("image_rgb", rec)
            self.assertIn("head_crop_rgb", rec)
            self.assertIn("head_crop_bbox", rec)
            self.assertIn("moge2_depth", rec)
            # Head crop preserves resolution beyond inference max_side=600
            self.assertGreater(rec["head_crop_rgb"].shape[0], 350)

            rec["face_landmarks_ram"] = np.zeros((478, 3), dtype=np.float32)
            rec["segmented_view"] = SegmentedView(
                view_index=0,
                azimuth_deg=0.0,
                view_label="front",
                person_mask=np.ones((10, 10), dtype=bool),
                body_mask=np.ones((10, 10), dtype=bool),
                hair_mask=np.zeros((10, 10), dtype=bool),
                face_skin_mask=np.zeros((10, 10), dtype=bool),
                face_landmarks_ram=np.zeros((478, 3), dtype=np.float32),
            )

            for stripper in (strip_raw_images_for_cache, strip_ram_only_semantics):
                stripped = stripper([rec])
                self.assertEqual(len(stripped), 1)
                for forbidden_key in (
                    "image_rgb",
                    "head_crop_rgb",
                    "moge2_depth",
                    "face_landmarks_ram",
                    "segmented_view",
                ):
                    self.assertNotIn(forbidden_key, stripped[0])
                self.assertEqual(stripped[0]["image_name"], "front.png")
                self.assertIn("output", stripped[0])

            # Input record remains untouched in RAM
            self.assertIn("face_landmarks_ram", rec)

    def test_load_sam3d_estimator_wires_human_segmentor_and_use_mask_fallback(self) -> None:
        import types
        from pathlib import Path
        from unittest.mock import patch
        import torch

        import torchvision  # noqa: F401  Pre-import before patch.dict(sys.modules)
        from bodyscan_pipeline.inference import (
            _SAM2SegmentorAdapter,
            _TorchvisionHumanSegmentor,
            load_sam3d_estimator,
            run_inference_on_images,
        )
        from bodyscan_pipeline.segmentation import process_views_semantics, segment_person_mask

        rgb, gt_mask, bbox = _make_synthetic_person_rgb()

        class _FakeEstimatorObj:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.human_segmentor = kwargs.get("human_segmentor")
                self.fov_estimator = kwargs.get("fov_estimator")

        fake_sam_mod = types.ModuleType("sam_3d_body")
        fake_sam_mod.SAM3DBodyEstimator = _FakeEstimatorObj
        fake_sam_mod.load_sam_3d_body = lambda ckpt, device=None, mhr_path=None: ("fake_model", {"cfg": True})

        fake_det_mod = types.ModuleType("tools.build_detector")
        fake_det_mod.HumanDetector = lambda **kwargs: ("det", kwargs)

        fake_fov_mod = types.ModuleType("tools.build_fov_estimator")
        fake_fov_mod.FOVEstimator = lambda **kwargs: ("fov", kwargs)

        fake_tools_pkg = types.ModuleType("tools")

        class _MockToolSegmentor:
            def __init__(self, name="sam2", device=None, path="") -> None:
                self.name = name
                self.device = device
                self.path = path

            def run_sam(self, img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
                return gt_mask.astype(np.uint8)[None, ...]

        fake_build_seg_mod = types.ModuleType("tools.build_segmentor")
        fake_build_seg_mod.HumanSegmentor = _MockToolSegmentor

        base_mods = {
            "sam_3d_body": fake_sam_mod,
            "tools": fake_tools_pkg,
            "tools.build_detector": fake_det_mod,
            "tools.build_fov_estimator": fake_fov_mod,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) tools.build_segmentor and tools.build_sam paths wire human_segmentor cleanly
            with patch.dict(sys.modules, {**base_mods, "tools.build_segmentor": fake_build_seg_mod}):
                _, est = load_sam3d_estimator(tmpdir, "ckpt", "mhr", device=torch.device("cpu"))
                self.assertIsInstance(est.human_segmentor, _MockToolSegmentor)

                # License guard rejects flame / briaai / rmbg even if loaded
                for bad_mod in ("flame", "briaai", "rmbg"):
                    with patch.dict(sys.modules, {bad_mod: types.ModuleType(bad_mod)}):
                        with self.assertRaises(RuntimeError):
                            load_sam3d_estimator(tmpdir, "ckpt", "mhr", device=torch.device("cpu"))

            fake_build_sam_mod = types.ModuleType("tools.build_sam")
            fake_build_sam_mod.HumanSegmentor = _MockToolSegmentor
            with patch.dict(
                sys.modules,
                {**base_mods, "tools.build_segmentor": None, "tools.build_sam": fake_build_sam_mod},
            ):
                _, est_bs = load_sam3d_estimator(tmpdir, "ckpt", "mhr", device=torch.device("cpu"))
                self.assertIsInstance(est_bs.human_segmentor, _MockToolSegmentor)

            # 2) Meta sam2 fallback path wires SAM2ImagePredictor when checkpoint exists
            sam2_ckpt = Path(tmpdir) / "sam2.1_hiera_base_plus.pt"
            sam2_ckpt.write_bytes(b"stub-ckpt")
            captured_sam2_calls = {}

            class _FakeSAM2Predictor:
                def __init__(self, model) -> None:
                    self.model = model
                    self._last_img = None

                def set_image(self, img: np.ndarray) -> None:
                    self._last_img = img

                def predict(self, box=None, multimask_output=False):
                    h_in, w_in = self._last_img.shape[:2]
                    out_m = np.zeros((1, 1, h_in, w_in), dtype=np.float32)
                    out_m[0, 0] = gt_mask.astype(np.float32)
                    return out_m, np.array([0.99], dtype=np.float32), None

            fake_sam2_pkg = types.ModuleType("sam2")
            fake_sam2_build = types.ModuleType("sam2.build_sam")

            def _fake_build_sam2(cfg_name: str, ckpt: str, device: str = "cpu"):
                captured_sam2_calls["cfg"] = cfg_name
                captured_sam2_calls["ckpt"] = ckpt
                return "sam2_model_obj"

            fake_sam2_build.build_sam2 = _fake_build_sam2
            fake_sam2_pred = types.ModuleType("sam2.sam2_image_predictor")
            fake_sam2_pred.SAM2ImagePredictor = _FakeSAM2Predictor

            sam2_mods = {
                **base_mods,
                "sam2": fake_sam2_pkg,
                "sam2.build_sam": fake_sam2_build,
                "sam2.sam2_image_predictor": fake_sam2_pred,
            }
            with patch.dict(sys.modules, sam2_mods), patch.dict(
                os.environ, {"ALPHASCAN_SAM2_CHECKPOINT": str(sam2_ckpt)}
            ):
                # Block tools.build_segmentor / tools.build_sam so sam2 path runs
                with patch.dict(sys.modules, {"tools.build_segmentor": None, "tools.build_sam": None}):
                    _, est_sam2 = load_sam3d_estimator(tmpdir, "ckpt", "mhr", device=torch.device("cpu"))
            self.assertIsInstance(est_sam2.human_segmentor, _SAM2SegmentorAdapter)
            self.assertIsInstance(est_sam2.human_segmentor.predictor, _FakeSAM2Predictor)
            self.assertEqual(captured_sam2_calls["ckpt"], str(sam2_ckpt))
            self.assertIn("b+", captured_sam2_calls["cfg"])
            sam2_run_masks = est_sam2.human_segmentor.run_sam(rgb, bbox[None, :])
            self.assertEqual(sam2_run_masks.shape, (1, rgb.shape[0], rgb.shape[1]))
            self.assertEqual(sam2_run_masks.dtype, np.uint8)
            self.assertTrue(np.array_equal(sam2_run_masks[0] > 0, gt_mask))
            self.assertEqual(
                est_sam2.human_segmentor.run_sam(rgb, np.zeros((0, 4), dtype=np.float32)).shape,
                (0, rgb.shape[0], rgb.shape[1]),
            )
            sam2_mask = segment_person_mask(rgb, bbox=bbox, human_segmentor=est_sam2.human_segmentor)
            self.assertTrue(np.array_equal(sam2_mask, gt_mask))

            # 3) Missing SAM/SAM2 packages fall back to Torchvision segmentor, or None when torchvision is absent
            with patch.dict(
                sys.modules,
                {**base_mods, "tools.build_segmentor": None, "tools.build_sam": None, "sam2": None},
            ):
                with warnings.catch_warnings(record=True) as caught_tv:
                    warnings.simplefilter("always")
                    _, est_tv = load_sam3d_estimator(tmpdir, "ckpt", "mhr", device=torch.device("cpu"))
                self.assertIsInstance(est_tv.human_segmentor, _TorchvisionHumanSegmentor)
                self.assertTrue(hasattr(est_tv.human_segmentor, "run_sam"))
                self.assertGreaterEqual(len(caught_tv), 1)

            with patch.dict(
                sys.modules,
                {**base_mods, "tools.build_segmentor": None, "tools.build_sam": None, "sam2": None},
            ), patch("bodyscan_pipeline.inference.build_torchvision_human_segmentor", return_value=None):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    _, est_none = load_sam3d_estimator(tmpdir, "ckpt", "mhr", device=torch.device("cpu"))
                self.assertIsNone(est_none.human_segmentor)
                self.assertGreaterEqual(len(caught), 1)
                self.assertIn("human_segmentor", str(caught[0].message))

                # enable_human_segmentor=False emits zero warnings
                with warnings.catch_warnings(record=True) as caught_off:
                    warnings.simplefilter("always")
                    _, est_off = load_sam3d_estimator(
                        tmpdir, "ckpt", "mhr", device=torch.device("cpu"), enable_human_segmentor=False
                    )
                self.assertIsNone(est_off.human_segmentor)
                self.assertEqual(len(caught_off), 0)

            # 4) run_inference_on_images sets use_mask based on human_segmentor and falls back cleanly
            img_path = os.path.join(tmpdir, "person.png")
            Image.fromarray(rgb).save(img_path)

            seen_use_mask = []

            class _UseMaskEstimator:
                def __init__(self, seg) -> None:
                    self.human_segmentor = seg

                def process_one_image(self, img: np.ndarray, bbox_thr: float = 0.5, use_mask: bool = False):
                    seen_use_mask.append(use_mask)
                    return [
                        {
                            "bbox": bbox.copy(),
                            "pred_keypoints_3d": _rotate_y_mhr_to_opencv(_build_canonical_mhr_keypoints(), 0.0),
                            "mask": gt_mask.copy(),
                        }
                    ]

            res_on = run_inference_on_images(_UseMaskEstimator(object()), [img_path])
            res_off = run_inference_on_images(_UseMaskEstimator(None), [img_path])
            self.assertEqual(seen_use_mask, [True, False])
            self.assertEqual(len(res_on), 1)
            self.assertEqual(len(res_off), 1)

            # Precomputed output['mask'] is reused cleanly by process_views_semantics
            res_on[0]["image_rgb"] = rgb.copy()
            views = process_views_semantics(res_on, estimator=None)
            self.assertTrue(np.array_equal(views[0].person_mask, gt_mask))

            # Older stub process_one_image without use_mask kwarg succeeds
            class _LegacyStubEstimator:
                human_segmentor = object()

                def process_one_image(self, img: np.ndarray, bbox_thr: float = 0.5):
                    return [{"bbox": bbox.copy()}]

            res_legacy = run_inference_on_images(_LegacyStubEstimator(), [img_path])
            self.assertEqual(len(res_legacy), 1)

            # Raising when use_mask=True (empty mask) falls back cleanly to use_mask=False
            class _EmptyMaskRetryEstimator:
                human_segmentor = object()

                def __init__(self) -> None:
                    self.calls = []

                def process_one_image(self, img: np.ndarray, bbox_thr: float = 0.5, use_mask: bool = False):
                    self.calls.append(use_mask)
                    if use_mask:
                        raise RuntimeError("empty mask produced")
                    return [{"bbox": bbox.copy()}]

            retry_est = _EmptyMaskRetryEstimator()
            with warnings.catch_warnings(record=True) as retry_warns:
                warnings.simplefilter("always")
                res_retry = run_inference_on_images(retry_est, [img_path])
            self.assertEqual(retry_est.calls, [True, False])
            self.assertEqual(len(res_retry), 1)
            self.assertGreaterEqual(len(retry_warns), 1)

            # Raising AttributeError when use_mask=True (e.g. missing .run_sam) falls back cleanly
            class _MissingRunSamEstimator:
                human_segmentor = object()

                def __init__(self) -> None:
                    self.calls = []

                def process_one_image(self, img: np.ndarray, bbox_thr: float = 0.5, use_mask: bool = False):
                    self.calls.append(use_mask)
                    if use_mask:
                        raise AttributeError("'object' object has no attribute 'run_sam'")
                    return [{"bbox": bbox.copy()}]

            attr_est = _MissingRunSamEstimator()
            with warnings.catch_warnings(record=True) as attr_warns:
                warnings.simplefilter("always")
                res_attr = run_inference_on_images(attr_est, [img_path])
            self.assertEqual(attr_est.calls, [True, False])
            self.assertEqual(len(res_attr), 1)
            self.assertGreaterEqual(len(attr_warns), 1)

    def test_mediapipe_solutions_and_tasks_dual_support_with_s2_and_warning(self) -> None:
        import types
        from pathlib import Path
        from unittest.mock import patch

        from bodyscan_pipeline.segmentation import (
            _segment_with_mediapipe,
            extract_face_landmarks_ram,
            reset_mediapipe_warning_cache,
            segment_person_mask,
        )

        rgb, gt_mask, bbox = _make_synthetic_person_rgb()
        crop = rgb[10:70, 30:90].copy()

        class _FakeLandmark:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        fake_478 = [_FakeLandmark(0.5, 0.4, -0.05) for _ in range(478)]

        class _FakeSolSegmenter:
            def __init__(self, model_selection: int = 0) -> None:
                self.model_selection = model_selection

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def process(self, img: np.ndarray):
                return types.SimpleNamespace(segmentation_mask=gt_mask.astype(np.float32))

        class _FakeSolFaceMesh:
            def __init__(self, **kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def process(self, img: np.ndarray):
                return types.SimpleNamespace(multi_face_landmarks=[types.SimpleNamespace(landmark=fake_478)])

        fake_mp_solutions = types.ModuleType("mediapipe")
        fake_mp_solutions.solutions = types.SimpleNamespace(
            selfie_segmentation=types.SimpleNamespace(SelfieSegmentation=_FakeSolSegmenter),
            face_mesh=types.SimpleNamespace(FaceMesh=_FakeSolFaceMesh),
        )

        try:
            reset_mediapipe_warning_cache()
            with patch.dict(sys.modules, {"mediapipe": fake_mp_solutions}):
                seg_sol = _segment_with_mediapipe(rgb)
                lm_sol = extract_face_landmarks_ram(crop)
            self.assertIsNotNone(seg_sol)
            self.assertTrue(np.array_equal(seg_sol, gt_mask))
            self.assertIsNotNone(lm_sol)
            self.assertEqual(lm_sol.shape, (478, 3))
            self.assertEqual(lm_sol.dtype, np.float32)

            # mp.tasks fallback when mp.solutions is None (mediapipe >= 0.10.30)
            with tempfile.TemporaryDirectory() as tmpdir:
                seg_task_file = Path(tmpdir) / "selfie_segmenter.tflite"
                face_task_file = Path(tmpdir) / "face_landmarker.task"
                seg_task_file.write_bytes(b"seg-task")
                face_task_file.write_bytes(b"face-task")
                before_files = set(os.listdir(tmpdir))

                class _FakeMPImage:
                    def __init__(self, image_format=None, data=None) -> None:
                        self._data = data

                    def numpy_view(self) -> np.ndarray:
                        return self._data

                class _FakeBaseOptions:
                    def __init__(self, model_asset_path: str) -> None:
                        self.model_asset_path = model_asset_path

                class _FakeImageSegmenterOptions:
                    def __init__(self, **kwargs) -> None:
                        self.kwargs = kwargs

                class _FakeFaceLandmarkerOptions:
                    def __init__(self, **kwargs) -> None:
                        self.kwargs = kwargs

                class _FakeTaskSegmenter:
                    def __init__(self, options) -> None:
                        self.options = options

                    @classmethod
                    def create_from_options(cls, options):
                        return cls(options)

                    def segment(self, mp_img):
                        bg_prob = 1.0 - gt_mask.astype(np.float32)
                        fg_prob = gt_mask.astype(np.float32)
                        return types.SimpleNamespace(
                            confidence_masks=[_FakeMPImage(data=bg_prob), _FakeMPImage(data=fg_prob)],
                            category_mask=None,
                        )

                    def close(self) -> None:
                        pass

                class _FakeTaskFaceLandmarker:
                    def __init__(self, options) -> None:
                        self.options = options

                    @classmethod
                    def create_from_options(cls, options):
                        return cls(options)

                    def detect(self, mp_img):
                        return types.SimpleNamespace(face_landmarks=[fake_478])

                    def close(self) -> None:
                        pass

                fake_mp_tasks = types.ModuleType("mediapipe")
                fake_mp_tasks.solutions = None
                fake_mp_tasks.Image = _FakeMPImage
                fake_mp_tasks.ImageFormat = types.SimpleNamespace(SRGB=1)
                fake_mp_tasks.tasks = types.SimpleNamespace(
                    python=types.SimpleNamespace(
                        BaseOptions=_FakeBaseOptions,
                        vision=types.SimpleNamespace(
                            RunningMode=types.SimpleNamespace(IMAGE="IMAGE"),
                            ImageSegmenterOptions=_FakeImageSegmenterOptions,
                            ImageSegmenter=_FakeTaskSegmenter,
                            FaceLandmarkerOptions=_FakeFaceLandmarkerOptions,
                            FaceLandmarker=_FakeTaskFaceLandmarker,
                        ),
                    )
                )

                reset_mediapipe_warning_cache()
                with patch.dict(sys.modules, {"mediapipe": fake_mp_tasks}), patch.dict(
                    os.environ,
                    {
                        "ALPHASCAN_MP_SEGMENTER_TASK": str(seg_task_file),
                        "ALPHASCAN_MP_FACE_LANDMARKER_TASK": str(face_task_file),
                    },
                ):
                    with warnings.catch_warnings(record=True) as task_warns:
                        warnings.simplefilter("always")
                        seg_task = _segment_with_mediapipe(rgb)
                        lm_task = extract_face_landmarks_ram(crop)
                self.assertEqual(len(task_warns), 0)
                self.assertIsNotNone(seg_task)
                self.assertTrue(np.array_equal(seg_task, gt_mask))
                self.assertIsNotNone(lm_task)
                self.assertEqual(lm_task.shape, (478, 3))
                # S2 strict: landmarks never written to disk
                self.assertEqual(set(os.listdir(tmpdir)), before_files)

                # Explicit task_model_path also works when env var is unset
                with patch.dict(sys.modules, {"mediapipe": fake_mp_tasks}), patch.dict(
                    os.environ, {"ALPHASCAN_MP_FACE_LANDMARKER_TASK": ""}, clear=False
                ):
                    lm_param = extract_face_landmarks_ram(crop, task_model_path=str(face_task_file))
                self.assertIsNotNone(lm_param)
                self.assertEqual(lm_param.shape, (478, 3))

                # category_mask-only ImageSegmenter also works cleanly
                class _CatOnlyTaskSegmenter(_FakeTaskSegmenter):
                    def segment(self, mp_img):
                        return types.SimpleNamespace(
                            confidence_masks=None,
                            category_mask=_FakeMPImage(data=gt_mask.astype(np.uint8)),
                        )

                fake_mp_tasks.tasks.python.vision.ImageSegmenter = _CatOnlyTaskSegmenter
                with patch.dict(sys.modules, {"mediapipe": fake_mp_tasks}), patch.dict(
                    os.environ, {"ALPHASCAN_MP_SEGMENTER_TASK": str(seg_task_file)}
                ):
                    seg_cat = _segment_with_mediapipe(rgb)
                self.assertIsNotNone(seg_cat)
                self.assertTrue(np.array_equal(seg_cat, gt_mask))

                # Missing .task files warn ONCE per fallback kind and cleanly fall back to GrabCut
                reset_mediapipe_warning_cache()
                with patch.dict(sys.modules, {"mediapipe": fake_mp_tasks}), patch.dict(
                    os.environ,
                    {"ALPHASCAN_MP_SEGMENTER_TASK": "", "ALPHASCAN_MP_FACE_LANDMARKER_TASK": ""},
                ):
                    with warnings.catch_warnings(record=True) as miss_warns:
                        warnings.simplefilter("always")
                        mask_fb_1 = segment_person_mask(rgb, bbox=bbox)
                        mask_fb_2 = segment_person_mask(rgb, bbox=bbox)
                        lm_fb_1 = extract_face_landmarks_ram(crop)
                        lm_fb_2 = extract_face_landmarks_ram(crop)
                self.assertGreater(int(mask_fb_1.sum()), 500)
                self.assertGreater(int(mask_fb_2.sum()), 500)
                self.assertIsNone(lm_fb_1)
                self.assertIsNone(lm_fb_2)
                self.assertEqual(len(miss_warns), 2)

                # Protobuf import error on Colab emits diagnostic warning once instead of failing silently
                reset_mediapipe_warning_cache()
                real_import = __import__

                def _proto_fail_import(name, globals=None, locals=None, fromlist=(), level=0):
                    if name == "mediapipe":
                        raise ImportError("cannot import name 'builder' from 'google.protobuf.internal'")
                    return real_import(name, globals, locals, fromlist, level)

                with patch("builtins.__import__", side_effect=_proto_fail_import):
                    with warnings.catch_warnings(record=True) as proto_warns:
                        warnings.simplefilter("always")
                        self.assertIsNone(_segment_with_mediapipe(rgb))
                        self.assertIsNone(extract_face_landmarks_ram(crop))
                self.assertEqual(len(proto_warns), 2)

            # Verify requirements.txt pins mediapipe<0.11 and documents optional packages safely
            req_text = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(encoding="utf-8")
            self.assertIn("mediapipe>=0.10.14,<0.11", req_text)
            for opt_name in ("sam_3d_body", "detectron2", "moge", "pytorch3d", "trimesh", "xatlas"):
                self.assertIn(opt_name, req_text)
            non_comment = "\n".join(ln for ln in req_text.splitlines() if not ln.strip().startswith("#"))
            for bad in ("ultralytics", "smplx", "nvdiffrast", "flame", "briaai", "rmbg"):
                self.assertNotIn(bad, non_comment.lower())
        finally:
            reset_mediapipe_warning_cache()

    def test_mhr70_vs_mhr127_keypoint_semantic_dispatch(self) -> None:
        from bodyscan_pipeline.segmentation import (
            _estimate_neck_y_from_output,
            _estimate_spine_up,
            _resolve_lr_indices,
            classify_view_azimuth,
            infer_view_azimuth,
        )

        # MHR-70 / COCO-WholeBody skeleton (70 joints): shoulders (5, 6), hips (11, 12), nose (0).
        # Deliberately set eye (2) & heel (18) to extreme off-axis points to prove (2, 18) are never used.
        kpts70 = np.zeros((70, 3), dtype=np.float32)
        kpts70[0] = [0.0, 1.60, 0.12]       # Nose (+Z anterior)
        kpts70[5] = [0.20, 1.40, 0.0]       # Left shoulder
        kpts70[6] = [-0.20, 1.40, 0.0]      # Right shoulder
        kpts70[11] = [0.10, 0.90, 0.0]      # Left hip
        kpts70[12] = [-0.10, 0.90, 0.0]     # Right hip
        kpts70[2] = [3.50, 5.00, 3.50]      # Right eye (corrupted/skewed decoy for MHR-127 hip idx 2)
        kpts70[18] = [-3.50, -4.00, -3.50]  # Right heel (decoy for MHR-127 hip idx 18)

        self.assertEqual(_resolve_lr_indices(kpts70, (75, 39)), (5, 6))
        sh_mid70 = 0.5 * (kpts70[5] + kpts70[6])
        spine70 = _estimate_spine_up(kpts70, sh_mid70, None)
        np.testing.assert_allclose(spine70, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-5)

        for deg, label in zip((0.0, 90.0, 180.0, 270.0), ("front", "right", "back", "left")):
            kpts70_cv = _rotate_y_mhr_to_opencv(kpts70, deg)
            az = infer_view_azimuth(kpts70_cv)
            diff = abs(((az - deg + 180.0) % 360.0) - 180.0)
            self.assertLess(diff, 1.0)
            self.assertEqual(classify_view_azimuth(az), label)

        # Contrast with MHR-127 skeleton: still uses (75, 39) and (2, 18)
        kpts127 = _build_canonical_mhr_keypoints()
        self.assertEqual(_resolve_lr_indices(kpts127, (75, 39)), (75, 39))
        sh_mid127 = 0.5 * (kpts127[75] + kpts127[39])
        spine127 = _estimate_spine_up(kpts127, sh_mid127, None)
        np.testing.assert_allclose(spine127, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-5)

        # Neck Y estimation on MHR-70 2D & 3D keypoints
        k2d_70 = np.zeros((70, 2), dtype=np.float32)
        k2d_70[5] = [40.0, 52.0]
        k2d_70[6] = [80.0, 54.0]
        mask_dummy = np.ones((180, 120), dtype=bool)
        self.assertAlmostEqual(_estimate_neck_y_from_output({"pred_keypoints_2d": k2d_70}, mask_dummy, None), 53.0, places=3)
        neck_3d_y = _estimate_neck_y_from_output(
            {"pred_keypoints_3d": kpts70, "pred_cam_t": np.array([0.0, 0.0, 2.5]), "focal_length": 400.0},
            mask_dummy,
            None,
        )
        self.assertIsNotNone(neck_3d_y)

    def test_moge2_single_pass_cache_and_warning(self) -> None:
        from bodyscan_pipeline.inference import (
            _extract_moge2_depth,
            clear_moge2_depth_cache,
            run_inference_on_images,
        )

        rgb = np.full((64, 48, 3), 120, dtype=np.uint8)
        depth_expected = np.full((64, 48), 2.25, dtype=np.float32)

        class _CountingFovEst:
            def __init__(self) -> None:
                self.calls = 0

            def infer_depth(self, img: np.ndarray) -> np.ndarray:
                self.calls += 1
                return np.full(img.shape[:2], 3.5, dtype=np.float32)

        class _EstHolder:
            def __init__(self, fov_estimator) -> None:
                self.fov_estimator = fov_estimator

        try:
            clear_moge2_depth_cache()
            fov_cnt = _CountingFovEst()
            holder = _EstHolder(fov_cnt)

            # 1) Inspects best_output keys ('moge2_depth', 'depth_map', 'depth') first without calling fov_est
            for k in ("moge2_depth", "depth_map", "depth"):
                got = _extract_moge2_depth(holder, rgb, {k: depth_expected})
                self.assertIsNotNone(got)
                np.testing.assert_allclose(got, depth_expected)
            self.assertEqual(fov_cnt.calls, 0)

            # 2) Caches per (id(fov_est), id(rgb)) across repeated calls
            d1 = _extract_moge2_depth(holder, rgb, {})
            d2 = _extract_moge2_depth(holder, rgb, {})
            self.assertEqual(fov_cnt.calls, 1)
            np.testing.assert_allclose(d1, d2)
            rgb2 = np.full((64, 48, 3), 200, dtype=np.uint8)
            _ = _extract_moge2_depth(holder, rgb2, {})
            self.assertEqual(fov_cnt.calls, 2)

            # 3) Single-pass capture inside run_inference_on_images avoids calling inner.infer twice
            class _InnerMoGe:
                def __init__(self) -> None:
                    self.infer_calls = 0

                def infer(self, img_t):
                    self.infer_calls += 1
                    h_t, w_t = img_t.shape[-2:]
                    return {"depth": np.full((1, h_t, w_t), 2.75, dtype=np.float32)}

            class _OuterFov:
                def __init__(self) -> None:
                    self.fov_estimator = _InnerMoGe()

                def get_cam_intrinsics(self, img: np.ndarray) -> float:
                    import torch

                    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                    _ = self.fov_estimator.infer(t)
                    return 480.0

            class _SinglePassEstimator:
                def __init__(self) -> None:
                    self.fov_estimator = _OuterFov()
                    self.human_segmentor = None

                def process_one_image(self, img: np.ndarray, bbox_thr: float = 0.5, use_mask: bool = False):
                    focal = self.fov_estimator.get_cam_intrinsics(img)
                    return [{"bbox": np.array([5.0, 5.0, 40.0, 60.0], dtype=np.float32), "focal_length": focal}]

            with tempfile.TemporaryDirectory() as tmpdir:
                p = os.path.join(tmpdir, "v0.png")
                Image.fromarray(rgb).save(p)
                sp_est = _SinglePassEstimator()
                res = run_inference_on_images(sp_est, [p], extract_depth=True)
                self.assertEqual(sp_est.fov_estimator.fov_estimator.infer_calls, 1)
                self.assertIn("moge2_depth", res[0])
                np.testing.assert_allclose(res[0]["moge2_depth"], 2.75)

            # 3b) Inner MoGe module with empty parameters() generator defaults to CPU without StopIteration
            class _EmptyParamsInnerMoGe:
                def parameters(self):
                    return iter(())

                def infer(self, img_t):
                    h_t, w_t = img_t.shape[-2:]
                    return {"depth": np.full((1, h_t, w_t), 3.25, dtype=np.float32)}

            class _EmptyParamsOuterFov:
                def __init__(self) -> None:
                    self.fov_estimator = _EmptyParamsInnerMoGe()

            empty_param_depth = _extract_moge2_depth(_EstHolder(_EmptyParamsOuterFov()), rgb, {})
            self.assertIsNotNone(empty_param_depth)
            np.testing.assert_allclose(empty_param_depth, 3.25)

            # 4) Emits warnings.warn instead of silently swallowing errors
            class _FailingFovEst:
                def infer_depth(self, img: np.ndarray):
                    raise RuntimeError("cuda oom in moge2")

            with warnings.catch_warnings(record=True) as err_warns:
                warnings.simplefilter("always")
                failed = _extract_moge2_depth(_EstHolder(_FailingFovEst()), rgb, {})
            self.assertIsNone(failed)
            self.assertGreaterEqual(len(err_warns), 1)
            self.assertIn("MoGe2", str(err_warns[0].message))
        finally:
            clear_moge2_depth_cache()

    def test_mhr_prior_guided_grabcut_rejects_gym_wall_banner_and_detached_blobs(self) -> None:
        from unittest.mock import patch
        import cv2
        import numpy as np
        from bodyscan_pipeline.segmentation import (
            _project_mhr_person_prior,
            _prune_person_mask_components,
            segment_person_mask,
        )

        img_h, img_w = 220, 160
        rgb = np.full((img_h, img_w, 3), 238, dtype=np.uint8)
        # Distant blue gym banner inside a loose detection bbox that a bare GrabCut could absorb
        rgb[155:205, 22:142] = np.array([25, 95, 210], dtype=np.uint8)
        # True foreground person: head + crop-top torso + dark leggings
        true_person = np.zeros((img_h, img_w), dtype=bool)
        cv2.circle(true_person.view(np.uint8), (80, 42), 16, 1, -1)
        cv2.circle(rgb, (80, 42), 16, (205, 145, 110), -1)
        cv2.rectangle(true_person.view(np.uint8), (58, 58), (102, 118), 1, -1)
        cv2.rectangle(rgb, (58, 58), (102, 118), (20, 20, 24), -1)
        cv2.rectangle(true_person.view(np.uint8), (60, 118), (76, 196), 1, -1)
        cv2.rectangle(true_person.view(np.uint8), (84, 118), (100, 196), 1, -1)
        cv2.rectangle(rgb, (60, 118), (76, 196), (18, 18, 22), -1)
        cv2.rectangle(rgb, (84, 118), (100, 196), (18, 18, 22), -1)

        # Construct Y-up MHR vertices (y in [0.05..1.65]) plus matching 2D keypoints (y_px head < ankles)
        ys_u, xs_u = np.mgrid[0.10:1.62:24j, -0.18:0.18:8j]
        verts_yup = np.stack([xs_u.ravel(), ys_u.ravel(), np.zeros_like(xs_u.ravel())], axis=1).astype(np.float32)
        focal = 360.0
        cam_t = np.array([0.0, 0.90, 3.0], dtype=np.float32)
        # With Y-up auto-flip: u = f*(x)/z + w/2, v = f*(-y + ty)/z + h/2
        u_pts = focal * verts_yup[:, 0] / cam_t[2] + img_w / 2.0
        v_pts = focal * (-verts_yup[:, 1] + cam_t[1]) / cam_t[2] + img_h / 2.0
        kpts_3d = np.array(
            [
                [0.0, 1.56, 0.0],
                [-0.14, 1.32, 0.0],
                [0.14, 1.32, 0.0],
                [-0.10, 0.88, 0.0],
                [0.10, 0.88, 0.0],
                [-0.08, 0.14, 0.0],
                [0.08, 0.14, 0.0],
            ],
            dtype=np.float32,
        )
        # Fill 17 COCO-style keypoints so _get_keypoint_semantics maps head/shoulders/hips cleanly
        k3_17 = np.zeros((17, 3), dtype=np.float32)
        k3_17[0] = kpts_3d[0]
        k3_17[5], k3_17[6] = kpts_3d[2], kpts_3d[1]
        k3_17[11], k3_17[12] = kpts_3d[4], kpts_3d[3]
        k3_17[15], k3_17[16] = kpts_3d[6], kpts_3d[5]
        k2_17 = np.stack(
            [
                focal * k3_17[:, 0] / cam_t[2] + img_w / 2.0,
                focal * (-k3_17[:, 1] + cam_t[1]) / cam_t[2] + img_h / 2.0,
                np.ones(17, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        raw_out = {
            "pred_vertices": verts_yup,
            "pred_keypoints_3d": k3_17,
            "pred_keypoints_2d": k2_17,
            "pred_cam_t": cam_t,
            "focal_length": np.array([focal], dtype=np.float32),
        }
        prior_core = _project_mhr_person_prior(raw_out, img_h, img_w)
        self.assertIsNotNone(prior_core)
        assert prior_core is not None
        self.assertGreater(int(prior_core[: img_h // 2].sum()), 20)
        self.assertGreater(int(np.sum(prior_core & true_person)), int(0.4 * np.sum(prior_core)))

        loose_bbox = np.array([15.0, 12.0, 148.0, 210.0], dtype=np.float32)
        with patch("bodyscan_pipeline.segmentation._segment_with_mediapipe", return_value=None):
            seg_prior = segment_person_mask(rgb, bbox=loose_bbox, prior_core=prior_core)

        inter = float(np.sum(seg_prior & true_person))
        union = float(np.sum(seg_prior | true_person))
        self.assertGreater(inter / max(1.0, union), 0.72)

        # Verify detached distant ceiling/wall island is pruned relative to prior_core
        noisy = seg_prior.copy()
        noisy[5:20, 130:155] = True
        pruned = _prune_person_mask_components(noisy, prior_core=prior_core)
        self.assertFalse(bool(np.any(pruned[5:20, 130:155])))
        self.assertTrue(np.array_equal(pruned & true_person, seg_prior & true_person))
        _ = (u_pts, v_pts)

    def test_extract_head_semantics_vetoes_white_wall_halo_and_bounds_shoulder_growth(self) -> None:
        from bodyscan_pipeline.segmentation import extract_head_semantics

        h, w = 320, 240
        neck_y_px = 108.0
        rgb = np.full((h, w, 3), 240, dtype=np.uint8)

        # Dark crop top + torso below neck, dark hair crown + draped hair locks, warm skin face
        cv2.rectangle(rgb, (85, 110), (155, 290), (25, 25, 30), -1)
        cv2.circle(rgb, (120, 54), 30, (38, 32, 30), -1)
        cv2.circle(rgb, (120, 78), 24, (38, 32, 30), -1)
        # Side draped hair extending slightly onto shoulders down to y=126
        cv2.rectangle(rgb, (92, 95), (104, 126), (38, 32, 30), -1)
        cv2.rectangle(rgb, (136, 95), (148, 126), (38, 32, 30), -1)
        cv2.ellipse(rgb, (120, 82), (18, 22), 0, 0, 360, (205, 150, 120), -1)

        pmask_clean = rgb[..., 0] < 230
        out_clean = extract_head_semantics(rgb, pmask_clean, neck_y_px=neck_y_px, view_label="front")
        n_clean = int(out_clean["hair_mask"].sum())
        self.assertGreater(n_clean, 1000)

        # Dilate person_mask by 4px into white gym wall [240, 240, 240]
        pmask_halo = cv2.dilate(pmask_clean.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        out_halo = extract_head_semantics(rgb, pmask_halo, neck_y_px=neck_y_px, view_label="front")
        hm_halo = out_halo["hair_mask"]
        n_halo = int(hm_halo.sum())
        white_leak = int(np.sum(hm_halo & (rgb[..., 0] > 220) & (rgb[..., 1] > 220) & (rgb[..., 2] > 220)))

        self.assertLess(float(white_leak) / max(1.0, float(n_halo)), 0.01)
        self.assertLessEqual(float(n_halo), 1.05 * float(n_clean))

        # Verify below-neck shoulder hair growth never exceeds 0.65 * hair_above
        hair_above_cnt = int(hm_halo[: int(round(neck_y_px)) + 1, :].sum())
        hair_below_cnt = int(hm_halo[int(round(neck_y_px)) + 1 :, :].sum())
        self.assertLessEqual(hair_below_cnt, int(np.ceil(0.65 * hair_above_cnt)) + 1)

    def test_multi_person_scoring_selects_primary_central_subject_over_bystanders(self) -> None:
        import torch
        from bodyscan_pipeline.inference import _TorchvisionHumanSegmentor, _select_best_person
        from bodyscan_pipeline.segmentation import _segment_with_human_segmentor

        h, w = 200, 300
        rgb = np.full((h, w, 3), 235, dtype=np.uint8)
        m_bystander_left = np.zeros((h, w), dtype=np.float32)
        m_bystander_left[20:170, 5:65] = 1.0
        m_center = np.zeros((h, w), dtype=np.float32)
        m_center[25:185, 115:185] = 1.0
        m_bystander_right = np.zeros((h, w), dtype=np.float32)
        m_bystander_right[15:190, 235:295] = 1.0

        # 1) Multi-instance Mask R-CNN prediction via _TorchvisionHumanSegmentor picks central subject
        class _FakeMaskRCNN:
            def __call__(self, tensors):
                return [
                    {
                        "labels": torch.tensor([1, 1, 1], dtype=torch.int64),
                        "scores": torch.tensor([0.96, 0.90, 0.95], dtype=torch.float32),
                        "boxes": torch.tensor(
                            [
                                [5.0, 20.0, 65.0, 170.0],
                                [115.0, 25.0, 185.0, 185.0],
                                [235.0, 15.0, 295.0, 190.0],
                            ],
                            dtype=torch.float32,
                        ),
                        "masks": torch.from_numpy(
                            np.stack([m_bystander_left, m_center, m_bystander_right], axis=0)[:, None]
                        ),
                    }
                ]

        tv_seg = _TorchvisionHumanSegmentor(_FakeMaskRCNN(), device=torch.device("cpu"))
        chosen_center = tv_seg.segment(rgb)
        self.assertIsNotNone(chosen_center)
        assert chosen_center is not None
        self.assertTrue(bool(np.any(chosen_center[:, 115:185])))
        self.assertFalse(bool(np.any(chosen_center[:, :80])))
        self.assertFalse(bool(np.any(chosen_center[:, 220:])))

        # 2) _segment_with_human_segmentor selects single primary SAM instance instead of unioning bystanders
        class _MultiInstanceSAM:
            def run_sam(self, img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
                return np.stack([m_bystander_left > 0.5, m_center > 0.5, m_bystander_right > 0.5], axis=0)

        sam_chosen = _segment_with_human_segmentor(rgb, bbox=None, human_segmentor=_MultiInstanceSAM())
        self.assertIsNotNone(sam_chosen)
        assert sam_chosen is not None
        self.assertTrue(bool(np.any(sam_chosen[:, 115:185])))
        self.assertFalse(bool(np.any(sam_chosen[:, :80])))
        self.assertFalse(bool(np.any(sam_chosen[:, 220:])))

        # 3) _select_best_person favors central subject over larger far-edge bystander
        picked = _select_best_person(
            [
                {"bbox": np.array([0.0, 10.0, 75.0, 195.0], dtype=np.float32), "score": 0.90},
                {"bbox": np.array([115.0, 25.0, 185.0, 185.0], dtype=np.float32), "score": 0.90},
            ],
            img_shape=(h, w),
        )
        self.assertAlmostEqual(float(picked["bbox"][0]), 115.0)

    def test_heterogeneous_gym_wall_hair_bg_modes_and_mhr70_neck_and_precomputed_prune(self) -> None:
        import torch
        from bodyscan_pipeline.inference import _TorchvisionHumanSegmentor
        from bodyscan_pipeline.segmentation import (
            _estimate_neck_y_from_output,
            extract_head_semantics,
            process_views_semantics,
        )

        # 1) Heterogeneous gym background (white wall + dark trim/poster + warm banner) with low-contrast hair
        h, w = 180, 140
        rgb = np.full((h, w, 3), 240, dtype=np.uint8)  # bright white wall
        rgb[:55, :60] = np.array([44, 42, 46], dtype=np.uint8)  # dark gym poster/trim near head
        rgb[:70, 82:] = np.array([210, 165, 130], dtype=np.uint8)  # warm wooden/banner panel

        person_mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.ogrid[:h, :w]
        head_ellipse = ((xx - 70) / 14.0) ** 2 + ((yy - 34) / 16.0) ** 2 <= 1.0
        torso_rect = (yy >= 48) & (yy < 165) & (xx >= 48) & (xx <= 92)
        # Include a small ring/halo leakage into dark poster region inside person_mask
        bg_halo_in_mask = (yy >= 16) & (yy <= 50) & (xx >= 50) & (xx <= 58)
        person_mask |= head_ellipse | torso_rect | bg_halo_in_mask

        # Paint face skin and dark hair + shirt
        rgb[torso_rect] = np.array([65, 110, 175], dtype=np.uint8)
        face_zone = ((xx - 70) / 9.5) ** 2 + ((yy - 36) / 10.5) ** 2 <= 1.0
        hair_zone = head_ellipse & (yy <= 32)
        rgb[face_zone] = np.array([215, 170, 140], dtype=np.uint8)
        rgb[hair_zone] = np.array([34, 30, 30], dtype=np.uint8)
        # Disconnected dark hair-like island inside person_mask halo on poster side
        person_mask[18:24, 32:40] = True
        rgb[18:24, 32:40] = np.array([34, 30, 30], dtype=np.uint8)

        sem = extract_head_semantics(
            rgb=rgb,
            person_mask=person_mask,
            neck_y_px=49.0,
            view_label="front",
            face_landmarks_ram=None,
            head_bbox=(54, 16, 86, 50),
        )
        hair_bg_leak = float(np.sum(sem["hair_mask"] & bg_halo_in_mask)) / float(max(np.sum(bg_halo_in_mask), 1))
        self.assertLess(hair_bg_leak, 0.15)
        self.assertFalse(bool(np.any(sem["hair_mask"][18:24, 32:40])))

        # 2) MHR-70 (len == 70) uses index 69 neck_y when valid & positive before shoulder fallback
        k2d_70 = np.zeros((70, 3), dtype=np.float32)
        k2d_70[5] = [55.0, 56.0, 0.9]
        k2d_70[6] = [85.0, 58.0, 0.9]
        k2d_70[69] = [70.0, 49.0, 0.95]
        neck_y_70 = _estimate_neck_y_from_output({"pred_keypoints_2d": k2d_70}, person_mask, None)
        self.assertIsNotNone(neck_y_70)
        self.assertAlmostEqual(float(neck_y_70), 49.0, delta=1e-4)

        # 3) process_views_semantics prunes detached blob on precomputed out["mask"] and records metadata
        polluted_mask = (head_ellipse | torso_rect).copy()
        polluted_mask[140:168, 5:35] = True  # detached poster/weight-rack blob far from core
        focal = 280.0
        cam_t = np.array([0.0, 0.85, 2.5], dtype=np.float32)
        xs_c = np.linspace(-0.20, 0.20, 12, dtype=np.float32)
        ys_c = np.linspace(0.05, 1.65, 24, dtype=np.float32)
        xv, yv = np.meshgrid(xs_c, ys_c)
        pts_m = np.stack([xv.reshape(-1), yv.reshape(-1), np.zeros(xv.size, dtype=np.float32)], axis=1)
        out_rec = {
            "mask": polluted_mask,
            "pred_vertices": pts_m,
            "pred_cam_t": cam_t,
            "focal_length": np.array([focal], dtype=np.float32),
            "pred_keypoints_2d": k2d_70,
        }
        views = process_views_semantics([{"image_rgb": rgb, "output": out_rec}])
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].mask_source, "precomputed")
        self.assertFalse(views[0].landmarks_active)
        self.assertFalse(bool(np.any(views[0].person_mask[140:168, 5:35])))

        # 4) _TorchvisionHumanSegmentor.run_sam returns (N, H, W) uint8 binary masks
        class _FakeTVModel:
            def __call__(self, tensors):
                m = np.zeros((h, w), dtype=np.float32)
                m[torso_rect] = 0.9
                return [
                    {
                        "labels": torch.tensor([1], dtype=torch.int64),
                        "scores": torch.tensor([0.92], dtype=torch.float32),
                        "boxes": torch.tensor([[48.0, 48.0, 92.0, 165.0]], dtype=torch.float32),
                        "masks": torch.from_numpy(m[None, None, ...]),
                    }
                ]

        tv_seg = _TorchvisionHumanSegmentor(_FakeTVModel(), device=torch.device("cpu"))
        sam_masks = tv_seg.run_sam(rgb, boxes=np.array([[46.0, 46.0, 94.0, 168.0]], dtype=np.float32))
        self.assertEqual(sam_masks.shape, (1, h, w))
        self.assertEqual(sam_masks.dtype, np.uint8)
        self.assertTrue(bool(np.any(sam_masks[0] > 0)))


if __name__ == "__main__":
    unittest.main()
