"""Foreground person segmentation, canonical view azimuth inference, and RAM-only head semantics."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from .config import MHR70_NECK_JOINT, NECK_BASE_JOINT, NECK_HEAD_JOINT
from .inference import (
    prepare_image_rgb,
    strip_raw_images_for_cache,
    verify_license_compliance,
)

CARDINAL_CENTERS: Tuple[Tuple[str, float], ...] = (
    ("front", 0.0),
    ("right", 90.0),
    ("back", 180.0),
    ("left", 270.0),
)

_MP_SEGMENTER_TASK_FILENAMES: Tuple[str, ...] = (
    "selfie_segmenter.tflite",
    "selfie_multiclass_256x256.tflite",
    "selfie_segmenter_landscape.tflite",
    "image_segmenter.task",
    "deeplab_v3.tflite",
)
_MP_FACE_LANDMARKER_TASK_FILENAMES: Tuple[str, ...] = (
    "face_landmarker.task",
    "face_landmarker_v2_with_blendshapes.task",
)
_MEDIAPIPE_WARNED: set[str] = set()


def reset_mediapipe_warning_cache() -> None:
    """Reset once-only MediaPipe diagnostic warning state (for tests)."""
    _MEDIAPIPE_WARNED.clear()


def _warn_mediapipe_once(kind: str, message: str) -> None:
    """Emit a diagnostic warning once per MediaPipe fallback kind."""
    if kind not in _MEDIAPIPE_WARNED:
        _MEDIAPIPE_WARNED.add(kind)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


@dataclass
class SegmentedView:
    """Per-view foreground/semantic masks, inferred orientation, and RAM-only facial cues."""

    view_index: int
    azimuth_deg: float
    view_label: str
    person_mask: np.ndarray
    body_mask: np.ndarray
    hair_mask: np.ndarray
    face_skin_mask: np.ndarray
    nose_profile_curve: Optional[np.ndarray] = None
    face_landmarks_ram: Optional[np.ndarray] = None
    collar_hair_overlap: bool = False
    warnings: List[str] = field(default_factory=list)
    mask_source: str = "unknown"
    landmarks_active: bool = False


def _select_primary_instance_mask(
    masks_3d: np.ndarray,
    height: int,
    width: int,
    scores: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None,
    prior_core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select the primary person mask from (N, H, W) candidates using score * area^0.35 * exp(-2.5 * center_dist)."""
    n_inst = len(masks_3d)
    if n_inst == 0:
        return np.zeros((height, width), dtype=bool)
    if n_inst == 1:
        m0 = masks_3d[0]
        if m0.shape != (height, width):
            m0 = cv2.resize(m0.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        return m0 > 0.5 if m0.dtype != np.bool_ else m0

    prior_bool: Optional[np.ndarray] = None
    if prior_core is not None:
        p_arr = np.asarray(prior_core, dtype=bool)
        if p_arr.shape == (height, width) and np.any(p_arr):
            prior_bool = p_arr

    has_tight_bbox = False
    bx0 = by0 = bx1 = by1 = b_area = bcx = bcy = 0.0
    if bbox is not None and len(bbox) >= 4:
        bx0, by0, bx1, by1 = [float(v) for v in bbox[:4]]
        b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        has_tight_bbox = b_area < 0.70 * float(max(1, height * width))

    best_mask = np.zeros((height, width), dtype=bool)
    best_val = -float("inf")
    for idx in range(n_inst):
        m_raw = masks_3d[idx]
        if m_raw.shape != (height, width):
            m_raw = cv2.resize(m_raw.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        m_bool = m_raw if m_raw.dtype == np.bool_ else (m_raw.astype(np.float32) > 0.5)
        ys, xs = np.where(m_bool)
        if len(ys) == 0:
            continue
        area = float(len(ys))
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        center_dist_norm = float(
            np.hypot(
                (cx - 0.5 * float(width)) / max(1.0, float(width)),
                (cy - 0.5 * float(height)) / max(1.0, float(height)),
            )
        )
        sc = float(scores[idx]) if (scores is not None and idx < len(scores)) else 1.0
        central_score = sc * (area ** 0.35) * float(np.exp(-2.5 * center_dist_norm))

        if prior_bool is not None:
            inter_p = float(np.sum(m_bool & prior_bool))
            union_p = float(max(1, np.sum(m_bool | prior_bool)))
            iou_p = inter_p / union_p
            val = iou_p * 100.0 + central_score * 1e-3 if iou_p > 0.02 else central_score * 1e-4
        elif has_tight_bbox:
            px0, py0, px1, py1 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
            ix = max(0.0, min(bx1, px1) - max(bx0, px0))
            iy = max(0.0, min(by1, py1) - max(by0, py0))
            inter = ix * iy
            p_box_area = max(1.0, (px1 - px0) * (py1 - py0))
            iou = inter / max(1.0, b_area + p_box_area - inter)
            b_dist = float(
                np.hypot((cx - bcx) / max(1.0, float(width)), (cy - bcy) / max(1.0, float(height)))
            )
            val = (0.75 * iou + 0.25 * sc) * float(np.exp(-1.5 * b_dist)) * 100.0 if iou > 0.05 else central_score * 1e-3
        else:
            val = central_score

        if val > best_val:
            best_val = val
            best_mask = m_bool
    return best_mask


def _normalize_mask_output(
    raw_mask: Any,
    height: int,
    width: int,
    bbox: Optional[np.ndarray] = None,
    prior_core: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Convert heterogeneous segmentor outputs into an (H, W) boolean mask."""
    if raw_mask is None:
        return None
    scores: Optional[np.ndarray] = None
    if isinstance(raw_mask, dict):
        if "scores" in raw_mask and raw_mask["scores"] is not None:
            s_raw = raw_mask["scores"]
            scores = np.asarray(s_raw.detach().cpu().numpy() if hasattr(s_raw, "detach") else s_raw, dtype=np.float32).reshape(-1)
        for key in ("mask", "masks", "person_mask"):
            if key in raw_mask and raw_mask[key] is not None:
                raw_mask = raw_mask[key]
                break
    if isinstance(raw_mask, (tuple, list)) and len(raw_mask) > 0:
        if len(raw_mask) >= 2 and raw_mask[1] is not None:
            try:
                s_cand = raw_mask[1]
                s_arr = np.asarray(s_cand.detach().cpu().numpy() if hasattr(s_cand, "detach") else s_cand, dtype=np.float32).reshape(-1)
                if s_arr.size > 0:
                    scores = s_arr
            except Exception:
                pass
        raw_mask = raw_mask[0]
    if hasattr(raw_mask, "detach"):
        raw_mask = raw_mask.detach().cpu().numpy()
    arr = np.asarray(raw_mask)
    if arr.size == 0 or arr.ndim < 2:
        return None
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 3:
        if arr.shape[0] > 1 and (arr.shape[1] == height and arr.shape[2] == width or arr.shape[0] != height):
            return _select_primary_instance_mask(arr, height, width, scores=scores, bbox=bbox, prior_core=prior_core)
        if arr.shape[0] <= 4 and arr.shape[1] == height and arr.shape[2] == width:
            arr = arr[0]
        elif arr.shape[0] == height and arr.shape[1] == width:
            arr = arr[..., 0]
        elif arr.shape[0] <= 4:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0:
        return None
    if arr.shape != (height, width):
        arr = cv2.resize(arr.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    if arr.dtype == np.bool_:
        return arr
    return arr.astype(np.float32) > 0.5


def _segment_with_human_segmentor(
    rgb: np.ndarray,
    bbox: Optional[np.ndarray],
    human_segmentor: Any,
    prior_core: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Run SAM 2 / SAM or compatible callable segmentor when provided."""
    h, w = rgb.shape[:2]
    box_arr = (
        np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        if bbox is not None
        else np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)
    )
    raw = None
    try:
        if hasattr(human_segmentor, "run_sam"):
            raw = human_segmentor.run_sam(rgb, box_arr)
        elif hasattr(human_segmentor, "segment"):
            try:
                raw = human_segmentor.segment(rgb, bbox, prior_core=prior_core)
            except TypeError:
                try:
                    raw = human_segmentor.segment(rgb, bbox)
                except TypeError:
                    raw = human_segmentor.segment(rgb)
        elif hasattr(human_segmentor, "set_image") and hasattr(human_segmentor, "predict"):
            human_segmentor.set_image(rgb)
            raw = human_segmentor.predict(box=box_arr[0], multimask_output=False)
        elif callable(human_segmentor):
            try:
                raw = human_segmentor(rgb, bbox, prior_core=prior_core)
            except TypeError:
                try:
                    raw = human_segmentor(rgb, bbox)
                except TypeError:
                    raw = human_segmentor(rgb)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError, AttributeError, OSError) as exc:
        warnings.warn(
            f"SAM/SAM2 human_segmentor failed ({exc}); falling back to MediaPipe/GrabCut.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return _normalize_mask_output(raw, h, w, bbox=bbox, prior_core=prior_core)


def _import_mediapipe(kind: str) -> Any:
    """Import mediapipe safely, warning once when installed package fails (e.g. protobuf conflict)."""
    try:
        import mediapipe as mp  # type: ignore[import-not-found]
        return mp
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "mediapipe" and "mediapipe" not in sys.modules:
            return None
        _warn_mediapipe_once(
            kind,
            f"MediaPipe import failed ({exc}); falling back to OpenCV GrabCut/skin heuristic.",
        )
        return None
    except (ImportError, OSError, AttributeError, TypeError) as exc:
        _warn_mediapipe_once(
            kind,
            f"MediaPipe initialization failed ({exc}); falling back to OpenCV GrabCut/skin heuristic.",
        )
        return None


def _get_mp_tasks_modules(mp: Any) -> Tuple[Any, Any]:
    """Resolve mediapipe.tasks.python and mediapipe.tasks.python.vision cleanly."""
    tasks_mod = getattr(mp, "tasks", None)
    if tasks_mod is None:
        try:
            import mediapipe.tasks as tasks_mod  # type: ignore[import-not-found]
        except Exception:
            tasks_mod = None
    python_mod = getattr(tasks_mod, "python", None) if tasks_mod is not None else None
    if python_mod is None and tasks_mod is not None:
        try:
            from mediapipe.tasks import python as python_mod  # type: ignore[import-not-found]
        except Exception:
            python_mod = None
    vision_mod = getattr(python_mod, "vision", None) if python_mod is not None else None
    if vision_mod is None and python_mod is not None:
        try:
            from mediapipe.tasks.python import vision as vision_mod  # type: ignore[import-not-found]
        except Exception:
            vision_mod = None
    return python_mod, vision_mod


def _resolve_mp_task_model_path(
    explicit_path: Optional[str],
    env_var: str,
    default_filenames: Tuple[str, ...],
) -> Optional[str]:
    """Locate a MediaPipe .task/.tflite model file from explicit param, env var, or ./checkpoints/."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        return None

    env_val = os.environ.get(env_var, "").strip()
    if env_val and os.path.isfile(env_val):
        return env_val

    repo_root = Path(__file__).resolve().parent.parent
    search_dirs = (
        Path("./checkpoints"),
        Path("./checkpoints/mediapipe"),
        repo_root / "checkpoints",
        repo_root / "checkpoints" / "mediapipe",
        Path("/content/sam-3d-body/checkpoints"),
        Path("/content/sam-3d-body/checkpoints/mediapipe"),
    )
    for base_dir in search_dirs:
        for fname in default_filenames:
            cand = base_dir / fname
            if cand.is_file():
                return str(cand)
    return None


def _build_mp_srgb_image(mp: Any, rgb: np.ndarray) -> Any:
    """Wrap an RGB uint8 NumPy array in a MediaPipe Image object."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    img_cls = getattr(mp, "Image", None)
    if img_cls is None:
        raise AttributeError("mediapipe.Image unavailable")
    img_fmt = getattr(getattr(mp, "ImageFormat", None), "SRGB", None)
    if img_fmt is not None:
        try:
            return img_cls(image_format=img_fmt, data=arr)
        except TypeError:
            pass
    try:
        return img_cls(data=arr)
    except TypeError:
        return img_cls(arr)


def _mp_image_to_numpy(img_obj: Any) -> np.ndarray:
    """Extract a NumPy array from a MediaPipe Image or array-like mock."""
    if hasattr(img_obj, "numpy_view"):
        return np.asarray(img_obj.numpy_view())
    return np.asarray(img_obj)


def _segment_with_mediapipe(
    rgb: np.ndarray,
    task_model_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Optional Apache 2.0 MediaPipe segmentation via mp.solutions or mp.tasks fallback."""
    mp = _import_mediapipe("segmenter")
    if mp is None:
        return None

    h, w = rgb.shape[:2]
    failures: List[str] = []

    solutions = getattr(mp, "solutions", None)
    selfie_mod = getattr(solutions, "selfie_segmentation", None) if solutions is not None else None
    if selfie_mod is not None:
        try:
            with selfie_mod.SelfieSegmentation(model_selection=0) as segmenter:
                res = segmenter.process(rgb)
            seg = getattr(res, "segmentation_mask", None)
            if seg is not None:
                return np.asarray(seg, dtype=np.float32) >= 0.5
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            failures.append(f"mp.solutions.selfie_segmentation: {exc}")
    else:
        failures.append("mp.solutions.selfie_segmentation unavailable")

    mp_python, mp_vision = _get_mp_tasks_modules(mp)
    model_path = _resolve_mp_task_model_path(
        task_model_path,
        "ALPHASCAN_MP_SEGMENTER_TASK",
        _MP_SEGMENTER_TASK_FILENAMES,
    )
    if mp_python is not None and mp_vision is not None and model_path is not None:
        segmenter = None
        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            opt_kwargs: Dict[str, Any] = {
                "base_options": base_options,
                "output_category_mask": True,
                "output_confidence_masks": True,
            }
            running_mode = getattr(getattr(mp_vision, "RunningMode", None), "IMAGE", None)
            if running_mode is not None:
                opt_kwargs["running_mode"] = running_mode
            try:
                options = mp_vision.ImageSegmenterOptions(**opt_kwargs)
            except TypeError:
                options = mp_vision.ImageSegmenterOptions(base_options=base_options)

            segmenter = mp_vision.ImageSegmenter.create_from_options(options)
            mp_img = _build_mp_srgb_image(mp, rgb)
            res = segmenter.segment(mp_img)

            conf_masks = getattr(res, "confidence_masks", None)
            if conf_masks is not None and len(conf_masks) > 0:
                if len(conf_masks) == 1:
                    fg = _mp_image_to_numpy(conf_masks[0]).astype(np.float32) >= 0.5
                elif len(conf_masks) == 2:
                    fg = _mp_image_to_numpy(conf_masks[1]).astype(np.float32) >= 0.5
                else:
                    fg = (1.0 - _mp_image_to_numpy(conf_masks[0]).astype(np.float32)) >= 0.5
                return _normalize_mask_output(fg, h, w)

            cat_mask = getattr(res, "category_mask", None)
            if cat_mask is not None:
                cat_arr = _mp_image_to_numpy(cat_mask)
                return _normalize_mask_output(cat_arr > 0, h, w)
            failures.append("mp.tasks.ImageSegmenter returned empty masks")
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            failures.append(f"mp.tasks.ImageSegmenter: {exc}")
        finally:
            if segmenter is not None and hasattr(segmenter, "close"):
                try:
                    segmenter.close()
                except Exception:
                    pass
    else:
        if mp_python is None or mp_vision is None:
            failures.append("mp.tasks.vision unavailable")
        elif model_path is None:
            failures.append(
                "no segmenter .task/.tflite found (set ALPHASCAN_MP_SEGMENTER_TASK or place model in ./checkpoints/)"
            )

    _warn_mediapipe_once(
        "segmenter",
        f"MediaPipe segmentation fell back to OpenCV GrabCut ({'; '.join(failures)}).",
    )
    return None


def _resolve_best_projection(
    pts: np.ndarray,
    cam_t: np.ndarray,
    focal: float,
    img_h: int,
    img_w: int,
    k3d: Optional[np.ndarray] = None,
    k2d: Optional[np.ndarray] = None,
    hip_pair: Tuple[int, ...] = (),
    sh_indices: Tuple[int, ...] = (),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Project 3D points into (u, v) pixels, auto-selecting CV vs Y-up sign convention."""
    cx = 0.5 * float(img_w)
    cy = 0.5 * float(img_h)
    variants: List[Tuple[float, float, float, bool]] = [
        (1.0, 1.0, 1.0, False),
        (1.0, -1.0, 1.0, True),
        (1.0, -1.0, -1.0, False),
        (1.0, 1.0, -1.0, True),
    ]

    def _project_raw(p: np.ndarray, sy: float, sz: float, flip_cam: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = cam_t
        if flip_cam:
            x = p[:, 0] + t[0]
            y = sy * p[:, 1] + t[1]
            z = sz * p[:, 2] + t[2]
        else:
            x = p[:, 0] + t[0]
            y = sy * (p[:, 1] + t[1])
            z = sz * (p[:, 2] + t[2])
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0.15)
        u = np.where(valid, focal * (x / np.maximum(z, 1e-4)) + cx, np.nan)
        v = np.where(valid, focal * (y / np.maximum(z, 1e-4)) + cy, np.nan)
        return u, v, valid

    best_idx = 0
    best_score = float("inf")
    found_calibrated = False

    if k3d is not None and k2d is not None and len(k3d) > 0 and len(k2d) > 0:
        n_k = min(len(k3d), len(k2d))
        k2_valid = (
            np.isfinite(k2d[:n_k, :2]).all(axis=1)
            & ((np.abs(k2d[:n_k, 0]) + np.abs(k2d[:n_k, 1])) > 1.0)
        )
        if np.sum(k2_valid) >= 4:
            for idx_v, (_, sy, sz, flip_cam) in enumerate(variants):
                uk, vk, val_k = _project_raw(k3d[:n_k], sy, sz, flip_cam)
                ok = k2_valid & val_k
                if np.sum(ok) < 4:
                    continue
                err = float(np.median(np.hypot(uk[ok] - k2d[:n_k, 0][ok], vk[ok] - k2d[:n_k, 1][ok])))
                if err < best_score:
                    best_score = err
                    best_idx = idx_v
                    found_calibrated = True

    if not found_calibrated:
        diag = max(float(img_h), float(img_w))
        best_heur = -float("inf")
        for idx_v, (_, sy, sz, flip_cam) in enumerate(variants):
            up, vp, val_p = _project_raw(pts, sy, sz, flip_cam)
            frac_pos = float(np.mean(val_p))
            if frac_pos < 0.25:
                continue
            in_b = val_p & (up >= -0.25 * img_w) & (up <= 1.25 * img_w) & (vp >= -0.25 * img_h) & (vp <= 1.25 * img_h)
            score = float(np.mean(in_b)) * 100.0
            if k3d is not None and hip_pair and sh_indices and len(k3d) > max(*hip_pair, *sh_indices):
                uk, vk, val_k = _project_raw(k3d, sy, sz, flip_cam)
                if val_k[sh_indices[0]] and val_k[sh_indices[1]] and val_k[hip_pair[0]] and val_k[hip_pair[1]]:
                    sh_v = 0.5 * (vk[sh_indices[0]] + vk[sh_indices[1]])
                    hip_v = 0.5 * (vk[hip_pair[0]] + vk[hip_pair[1]])
                    if sh_v < hip_v - 0.03 * diag:
                        score += 20.0
                    elif sh_v > hip_v:
                        score -= 50.0
            if score > best_heur:
                best_heur = score
                best_idx = idx_v

    _, sy_b, sz_b, flip_b = variants[best_idx]
    u_out, v_out, valid_out = _project_raw(pts, sy_b, sz_b, flip_b)
    if not np.any(valid_out):
        return None
    return u_out[valid_out], v_out[valid_out]


def _project_mhr_person_prior(
    out: Dict[str, Any],
    img_h: int,
    img_w: int,
) -> Optional[np.ndarray]:
    """Project MHR body mesh/keypoints into image pixels to seed foreground GrabCut cleanly."""
    if img_h <= 0 or img_w <= 0 or not isinstance(out, dict):
        return None

    k2d_raw = out.get("pred_keypoints_2d")
    k2d: Optional[np.ndarray] = None
    if k2d_raw is not None:
        k2_arr = np.asarray(k2d_raw, dtype=np.float32)
        if k2_arr.ndim == 2 and k2_arr.shape[1] >= 2:
            k2d = k2_arr

    k3d_raw = out.get("pred_keypoints_3d")
    k3d: Optional[np.ndarray] = None
    if k3d_raw is not None:
        k3_arr = np.asarray(k3d_raw, dtype=np.float32)
        if k3_arr.ndim == 2 and k3_arr.shape[1] >= 3:
            k3d = k3_arr

    sh_indices: Tuple[int, ...] = ()
    hip_pair: Tuple[int, ...] = ()
    if k3d is not None and len(k3d) > 0:
        sh_indices, hip_pair, _ = _get_keypoint_semantics(len(k3d))

    verts_raw = out.get("pred_vertices")
    cam_t_raw = out.get("pred_cam_t")
    focal_raw = out.get("focal_length")
    prior = np.zeros((img_h, img_w), dtype=bool)

    if verts_raw is not None and cam_t_raw is not None and focal_raw is not None:
        verts = np.asarray(verts_raw, dtype=np.float32)
        cam_t = np.asarray(cam_t_raw, dtype=np.float32).reshape(-1)
        focal = float(np.mean(focal_raw)) if np.size(focal_raw) > 0 else 0.0
        if verts.ndim == 2 and verts.shape[1] >= 3 and len(verts) >= 20 and cam_t.size == 3 and focal > 1.0:
            proj = _resolve_best_projection(
                pts=verts[:, :3],
                cam_t=cam_t,
                focal=focal,
                img_h=img_h,
                img_w=img_w,
                k3d=k3d,
                k2d=k2d,
                hip_pair=hip_pair,
                sh_indices=sh_indices,
            )
            if proj is not None:
                u_f, v_f = proj
                ui = np.round(u_f).astype(np.int32)
                vi = np.round(v_f).astype(np.int32)
                in_img = (ui >= 0) & (ui < img_w) & (vi >= 0) & (vi < img_h)
                if np.sum(in_img) >= max(24, int(0.05 * len(u_f))):
                    prior[vi[in_img], ui[in_img]] = True
                    span_h = max(12, int(vi[in_img].max() - vi[in_img].min() + 1))
                    span_w = max(8, int(ui[in_img].max() - ui[in_img].min() + 1))
                    approx_step = float(np.sqrt((0.45 * float(span_h) * float(span_w)) / float(max(1, int(np.sum(in_img))))))
                    k_sz = max(7, min(27, int(round(max(0.015 * span_h, 2.6 * approx_step))) | 1))
                    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
                    closed = cv2.morphologyEx(prior.astype(np.uint8), cv2.MORPH_CLOSE, k_close)
                    prior = closed > 0

    if k2d is not None and len(k2d) >= 4:
        val_k = (
            np.isfinite(k2d[:, :2]).all(axis=1)
            & (k2d[:, 0] > 1.0)
            & (k2d[:, 0] < img_w - 1.0)
            & (k2d[:, 1] > 1.0)
            & (k2d[:, 1] < img_h - 1.0)
        )
        if np.sum(val_k) >= 4:
            canvas = prior.astype(np.uint8).copy()
            sh_ok, hp_ok, _ = _get_keypoint_semantics(len(k2d))
            if sh_ok and hp_ok and len(k2d) > max(*sh_ok, *hp_ok):
                pts_quad = [k2d[sh_ok[0], :2], k2d[sh_ok[1], :2], k2d[hp_ok[1], :2], k2d[hp_ok[0], :2]]
                if all(np.isfinite(p).all() and (abs(p[0]) + abs(p[1]) > 1.0) for p in pts_quad):
                    poly = np.round(np.asarray(pts_quad)).astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillConvexPoly(canvas, poly, 1)
            if not np.any(prior):
                k_pts = k2d[val_k, :2]
                span = max(24.0, float(k_pts[:, 1].max() - k_pts[:, 1].min()))
                rad = max(4, min(18, int(round(0.03 * span))))
                for x_f, y_f in k_pts:
                    cv2.circle(canvas, (int(round(float(x_f))), int(round(float(y_f)))), rad, 1, thickness=-1)
            prior = canvas > 0

    area_frac = float(np.mean(prior)) if np.any(prior) else 0.0
    if area_frac <= 1e-4 or area_frac >= 0.92:
        return None
    return prior


def _prune_person_mask_components(
    mask: np.ndarray,
    prior_core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Drop detached background islands far from the primary subject / MHR prior."""
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_lbl <= 2:
        return mask_u8 > 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max())
    keep = np.zeros_like(mask_u8, dtype=bool)

    if prior_core is not None and np.any(prior_core):
        h, w = mask_u8.shape
        fg_scale = max(12.0, float(np.sqrt(float(np.sum(prior_core)))))
        max_dist = max(8.0, 0.14 * fg_scale)
        dist_map = distance_transform_edt(~np.asarray(prior_core, dtype=bool))
        dil_k_sz = max(5, min(35, int(round(0.08 * fg_scale)) | 1))
        dil_prior = cv2.dilate(
            prior_core.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k_sz, dil_k_sz)),
            iterations=1,
        ) > 0
        for lbl in range(1, num_lbl):
            comp = labels == lbl
            if np.any(comp & dil_prior):
                keep |= comp
                continue
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= max(16, int(0.03 * max_area)) and float(np.min(dist_map[comp])) <= max_dist:
                keep |= comp
        if np.any(keep):
            return keep

    min_area = max(12, int(0.025 * max_area))
    for lbl in range(1, num_lbl):
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == lbl
    return keep if np.any(keep) else (mask_u8 > 0)


def _refine_precomputed_or_seg_mask(
    mask: np.ndarray,
    prior_core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Prune detached background islands and clamp huge background floods via dilated MHR hull."""
    pruned = _prune_person_mask_components(mask, prior_core=prior_core)
    if prior_core is None:
        return pruned
    prior_b = np.asarray(prior_core, dtype=bool)
    if prior_b.shape != pruned.shape or not np.any(prior_b):
        return pruned

    prior_area = int(np.sum(prior_b))
    fg_scale = max(12.0, float(np.sqrt(float(prior_area))))
    if int(np.sum(pruned)) > int(round(1.30 * float(prior_area))):
        env_r = max(24, min(140, int(round(0.32 * fg_scale))))
        env_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * env_r + 1, 2 * env_r + 1))
        env = cv2.dilate(prior_b.astype(np.uint8), env_k, iterations=1) > 0
        clipped = pruned & env
        if int(np.sum(clipped & prior_b)) >= max(16, int(0.65 * float(np.sum(pruned & prior_b)))):
            pruned = _prune_person_mask_components(clipped, prior_core=prior_b)
    return pruned


def _segment_with_grabcut(
    rgb: np.ndarray,
    bbox: Optional[np.ndarray],
    prior_core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """OpenCV GrabCut foreground separation, mask-seeded from MHR projection when available."""
    h, w = rgb.shape[:2]
    if bbox is not None and len(bbox) >= 4:
        x0 = max(0, min(w - 1, int(np.floor(float(bbox[0])))))
        y0 = max(0, min(h - 1, int(np.floor(float(bbox[1])))))
        x1 = max(x0 + 1, min(w, int(np.ceil(float(bbox[2])))))
        y1 = max(y0 + 1, min(h, int(np.ceil(float(bbox[3])))))
    else:
        mx = max(1, int(0.05 * w))
        my = max(1, int(0.05 * h))
        x0, y0, x1, y1 = mx, my, max(mx + 2, w - mx), max(my + 2, h - my)

    core = np.asarray(prior_core, dtype=bool) if (prior_core is not None and prior_core.shape == (h, w)) else None
    if core is not None and np.any(core):
        fg_px = int(np.sum(core))
        bh = max(16, y1 - y0)
        bw = max(16, x1 - x0)
        prior_fill = float(fg_px) / float(max(1, bh * bw))
        scale_px = max(12.0, float(np.sqrt(float(fg_px))), 0.35 * float(bh))
        core_u8 = core.astype(np.uint8)

        k_bg_sz = max(15, min(101, int(round(0.26 * scale_px)) | 1))
        k_pr_sz = max(5, min(21, int(round(0.05 * scale_px)) | 1))
        k_fg_sz = max(3, min(11, int(round(0.03 * scale_px)) | 1))

        if prior_fill >= 0.22:
            pr_bg_zone = cv2.dilate(
                core_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bg_sz, k_bg_sz)),
                iterations=1,
            ) > 0
            ys_c, _ = np.where(core)
            head_pad_y = max(10, int(round(0.20 * scale_px)))
            top_ext_y0 = max(0, int(ys_c.min()) - head_pad_y)
            top_ext_y1 = min(h, int(ys_c.min()) + max(8, int(round(0.25 * scale_px))))
            top_band = core[
                int(ys_c.min()) : min(h, int(ys_c.min()) + max(6, int(round(0.12 * (ys_c.max() - ys_c.min() + 1))))), :
            ]
            if np.any(top_band):
                _, hx = np.where(top_band)
                hx0 = max(0, int(hx.min()) - max(8, int(round(0.12 * scale_px))))
                hx1 = min(w, int(hx.max()) + 1 + max(8, int(round(0.12 * scale_px))))
                pr_bg_zone[top_ext_y0:top_ext_y1, hx0:hx1] = True
        else:
            pr_bg_zone = np.zeros((h, w), dtype=bool)
            pr_bg_zone[y0:y1, x0:x1] = True
            pr_bg_zone |= core

        pr_fg_zone = cv2.dilate(
            core_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_pr_sz, k_pr_sz)),
            iterations=1,
        ) > 0
        sure_fg_zone = cv2.erode(
            core_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fg_sz, k_fg_sz)),
            iterations=1,
        ) > 0
        if int(np.sum(sure_fg_zone)) < max(16, int(0.50 * fg_px)):
            sure_fg_zone = cv2.erode(
                core_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
            ) > 0
        if int(np.sum(sure_fg_zone)) < 16:
            sure_fg_zone = core.copy()

        gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        gc_mask[pr_bg_zone] = cv2.GC_PR_BGD
        gc_mask[pr_fg_zone] = cv2.GC_PR_FGD
        gc_mask[sure_fg_zone] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        fg = np.zeros((h, w), dtype=bool)
        try:
            cv2.grabCut(rgb, gc_mask, (0, 0, 1, 1), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except cv2.error:
            fg = np.zeros((h, w), dtype=bool)

        if int(np.sum(fg)) < max(16, int(0.45 * fg_px)):
            fg = fg | sure_fg_zone
        fg = _prune_person_mask_components(fg, prior_core=core)
        if np.any(fg):
            return fg

    pad = 8 if (x0 < 2 or y0 < 2 or x1 > w - 2 or y1 > h - 2) else 0
    work_rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad > 0 else rgb
    rect = (x0 + pad, y0 + pad, max(2, x1 - x0), max(2, y1 - y0))
    gc_mask = np.zeros(work_rgb.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(work_rgb, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        if pad > 0:
            gc_mask = gc_mask[pad : pad + h, pad : pad + w]
        fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    except cv2.error:
        fg = np.zeros((h, w), dtype=bool)

    if not np.any(fg):
        # Color-contrast fallback against outer border median if GrabCut GMM degenerates.
        border_px = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0)
        bg_med = np.median(border_px.astype(np.float32), axis=0)
        diff = np.linalg.norm(rgb.astype(np.float32) - bg_med, axis=-1)
        roi = np.zeros((h, w), dtype=bool)
        roi[y0:y1, x0:x1] = True
        fg = roi & (diff > 18.0)
        if not np.any(fg):
            fg = roi
    return _prune_person_mask_components(fg, prior_core=core)


def _infer_segmentor_backend_name(human_segmentor: Any) -> str:
    """Return a concise backend label ('torchvision', 'sam2', or 'human_segmentor')."""
    cls_name = type(human_segmentor).__name__.lower()
    if "torchvision" in cls_name or "maskrcnn" in cls_name:
        return "torchvision"
    if "sam2" in cls_name or hasattr(human_segmentor, "predictor"):
        return "sam2"
    return "human_segmentor"


def _segment_person_mask_with_source(
    rgb: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    human_segmentor: Any = None,
    erode_px: int = 0,
    mediapipe_task_path: Optional[str] = None,
    prior_core: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    """Segment foreground person silhouette (H, W) bool and return `(mask, mask_source)`."""
    verify_license_compliance()
    if rgb is None or rgb.size == 0:
        return np.zeros((0, 0), dtype=bool), "empty"
    rgb_arr = np.asarray(rgb)
    if rgb_arr.ndim == 2:
        rgb_arr = np.repeat(rgb_arr[..., None], 3, axis=-1)

    mask: Optional[np.ndarray] = None
    mask_source = "grabcut"
    used_human_seg = False
    if human_segmentor is not None:
        mask = _segment_with_human_segmentor(rgb_arr, bbox, human_segmentor, prior_core=prior_core)
        if mask is not None and np.any(mask):
            used_human_seg = True
            mask_source = _infer_segmentor_backend_name(human_segmentor)
        else:
            if mask is not None and not np.any(mask):
                warnings.warn(
                    "SAM/SAM2 human_segmentor returned empty mask; falling back to MediaPipe/GrabCut.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            mask = None
    if mask is None:
        mask = _segment_with_mediapipe(rgb_arr, task_model_path=mediapipe_task_path)
        if mask is not None and np.any(mask):
            mask_source = "mediapipe"
    if mask is None or not np.any(mask):
        mask = _segment_with_grabcut(rgb_arr, bbox, prior_core=prior_core)
        mask_source = "grabcut"

    mask = _refine_precomputed_or_seg_mask(mask, prior_core=prior_core)
    mask_u8 = mask.astype(np.uint8)
    if not used_human_seg:
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_k)
    if erode_px > 0:
        k_size = 2 * int(erode_px) + 1
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask_u8 = cv2.erode(mask_u8, erode_k, iterations=1)
    return mask_u8 > 0, mask_source


def segment_person_mask(
    rgb: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    human_segmentor: Any = None,
    erode_px: int = 0,
    mediapipe_task_path: Optional[str] = None,
    prior_core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Segment foreground person silhouette (H, W) bool with optional MHR prior & boundary erosion."""
    mask, _ = _segment_person_mask_with_source(
        rgb=rgb,
        bbox=bbox,
        human_segmentor=human_segmentor,
        erode_px=erode_px,
        mediapipe_task_path=mediapipe_task_path,
        prior_core=prior_core,
    )
    return mask


def classify_view_azimuth(azimuth_deg: float, cardinal_tol_deg: float = 35.0) -> str:
    """Map continuous azimuth [0, 360) to 'front' | 'right' | 'back' | 'left' | 'oblique'."""
    az = float(azimuth_deg) % 360.0
    best_label = "oblique"
    best_dist = float("inf")
    for label, center in CARDINAL_CENTERS:
        dist = abs(((az - center + 180.0) % 360.0) - 180.0)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label if best_dist <= cardinal_tol_deg else "oblique"


def _get_keypoint_semantics(num_kpts: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, ...]]:
    """Dispatch shoulder, hip, and head indices by keypoint schema (MHR-127 vs MHR-70/COCO)."""
    if num_kpts >= 127:
        return (75, 39), (2, 18), (NECK_HEAD_JOINT, 0)
    if num_kpts == 70 or num_kpts >= 13:
        return (5, 6), (11, 12), (0,)
    if num_kpts >= 7:
        return (5, 6), (), (0,)
    return (), (), (0,)


def _resolve_lr_indices(
    kpts: np.ndarray,
    joints_lr_indices: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Pick valid (left_shoulder, right_shoulder) indices for MHR-127 vs MHR-70/COCO."""
    n = len(kpts)
    default_sh, _, _ = _get_keypoint_semantics(n)
    candidates: List[Tuple[int, int]] = []
    if n >= 127 or joints_lr_indices != (75, 39):
        candidates.append(joints_lr_indices)
    if default_sh and default_sh not in candidates:
        candidates.append(default_sh)
    if n >= 127 and (5, 6) not in candidates:
        candidates.append((5, 6))

    for pair in candidates:
        il, ir = pair
        if n > max(il, ir) and np.isfinite(kpts[il]).all() and np.isfinite(kpts[ir]).all():
            if np.linalg.norm(kpts[il] - kpts[ir]) > 1e-4:
                return pair
    return None


def _estimate_spine_up(kpts: np.ndarray, shoulder_mid: np.ndarray, verts: Optional[np.ndarray]) -> np.ndarray:
    """Estimate vertical spine-up unit vector from schema-matched pelvis/head keypoints or vertices."""
    n = len(kpts)
    _, hip_pair, head_indices = _get_keypoint_semantics(n)
    if hip_pair and n > max(hip_pair):
        h0, h1 = kpts[hip_pair[0]], kpts[hip_pair[1]]
        if np.isfinite(h0).all() and np.isfinite(h1).all() and (np.any(h0 != 0.0) or np.any(h1 != 0.0)):
            hip_mid = 0.5 * (h0 + h1)
            vec = shoulder_mid - hip_mid
            norm = float(np.linalg.norm(vec))
            if norm > 1e-4:
                return (vec / norm).astype(np.float32)

    for head_idx in head_indices:
        if n > head_idx and np.isfinite(kpts[head_idx]).all() and np.any(kpts[head_idx] != 0.0):
            vec = kpts[head_idx] - shoulder_mid
            norm = float(np.linalg.norm(vec))
            if 0.02 < norm < 0.7:
                return (vec / norm).astype(np.float32)

    if verts is not None and len(verts) >= 4:
        span_y = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))
        if span_y > 1e-3:
            return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)


def infer_view_azimuth(
    pred_keypoints_3d: Optional[np.ndarray],
    pred_vertices: Optional[np.ndarray] = None,
    joints_lr_indices: Tuple[int, int] = (75, 39),
    fallback_index: int = 0,
    total_views: int = 4,
) -> float:
    """Infer continuous view azimuth in [0, 360) deg from 3D shoulder normal and nose cues."""
    if pred_keypoints_3d is not None:
        kpts = np.asarray(pred_keypoints_3d, dtype=np.float32)
        if kpts.ndim == 2 and kpts.shape[1] >= 3 and len(kpts) > 0:
            kpts = kpts[:, :3]
            lr_pair = _resolve_lr_indices(kpts, joints_lr_indices)
            if lr_pair is not None:
                p_left, p_right = kpts[lr_pair[0]], kpts[lr_pair[1]]
                lr_vec = p_left - p_right  # Right -> Left (+X at front view)
                shoulder_mid = 0.5 * (p_left + p_right)
                spine_up = _estimate_spine_up(kpts, shoulder_mid, pred_vertices)

                nose_horiz: Optional[np.ndarray] = None
                if np.isfinite(kpts[0]).all() and np.any(kpts[0] != 0.0):
                    cand = kpts[0] - shoulder_mid
                    if abs(float(cand[1])) < 0.65 and float(np.hypot(cand[0], cand[2])) > 0.02:
                        nose_horiz = np.array([cand[0], 0.0, cand[2]], dtype=np.float32)

                if spine_up[1] > 0.2:
                    # +Y up coordinates: check handedness against anterior nose when present.
                    facing_yup = np.cross(lr_vec, spine_up)
                    use_mhr_z = nose_horiz is not None and float(
                        np.dot(np.array([facing_yup[0], 0.0, facing_yup[2]], dtype=np.float32), nose_horiz)
                    ) > 0.0
                    if use_mhr_z:
                        fx, fz = float(facing_yup[0]), float(facing_yup[2])
                    else:
                        facing_cv = np.cross(lr_vec, np.array([0.0, -1.0, 0.0], dtype=np.float32))
                        fx, fz = float(facing_cv[0]), float(-facing_cv[2])
                else:
                    up_use = spine_up if spine_up[1] < -0.2 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                    facing_cv = np.cross(lr_vec, up_use)
                    fx, fz = float(facing_cv[0]), float(-facing_cv[2])

                if np.hypot(fx, fz) > 1e-5:
                    return float(np.degrees(np.arctan2(fx, fz)) % 360.0)
                if nose_horiz is not None:
                    return float(np.degrees(np.arctan2(nose_horiz[0], -nose_horiz[2])) % 360.0)

    denom = max(int(total_views), 1)
    return float((int(fallback_index) % denom) * (360.0 / denom))


def _landmarks_to_478(face_lms: Any) -> Optional[np.ndarray]:
    """Convert MediaPipe landmark sequence into (478, 3) float32 array in RAM."""
    if face_lms is None:
        return None
    if isinstance(face_lms, np.ndarray):
        pts = np.asarray(face_lms, dtype=np.float32)
    else:
        pts = np.array([[float(lm.x), float(lm.y), float(lm.z)] for lm in face_lms], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    pts = pts[:, :3]
    if len(pts) >= 478:
        return pts[:478]
    if len(pts) == 468:
        pad = np.zeros((10, 3), dtype=np.float32)
        return np.vstack([pts, pad])
    return None


def extract_face_landmarks_ram(
    head_crop_rgb: np.ndarray,
    task_model_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Extract ephemeral (478, 3) float32 normalized facial landmarks strictly in RAM."""
    verify_license_compliance()
    if head_crop_rgb is None or head_crop_rgb.size == 0:
        return None
    crop = np.asarray(head_crop_rgb)
    if crop.ndim < 2 or min(crop.shape[:2]) < 16:
        return None

    mp = _import_mediapipe("face_landmarker")
    if mp is None:
        return None

    failures: List[str] = []

    solutions = getattr(mp, "solutions", None)
    face_mesh_mod = getattr(solutions, "face_mesh", None) if solutions is not None else None
    if face_mesh_mod is not None:
        try:
            with face_mesh_mod.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            ) as mesh:
                res = mesh.process(crop)
            multi = getattr(res, "multi_face_landmarks", None)
            if not multi:
                return None
            return _landmarks_to_478(getattr(multi[0], "landmark", multi[0]))
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            failures.append(f"mp.solutions.face_mesh: {exc}")
    else:
        failures.append("mp.solutions.face_mesh unavailable")

    mp_python, mp_vision = _get_mp_tasks_modules(mp)
    model_path = _resolve_mp_task_model_path(
        task_model_path,
        "ALPHASCAN_MP_FACE_LANDMARKER_TASK",
        _MP_FACE_LANDMARKER_TASK_FILENAMES,
    )
    if mp_python is not None and mp_vision is not None and model_path is not None:
        landmarker = None
        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            opt_kwargs: Dict[str, Any] = {
                "base_options": base_options,
                "num_faces": 1,
                "min_face_detection_confidence": 0.5,
            }
            running_mode = getattr(getattr(mp_vision, "RunningMode", None), "IMAGE", None)
            if running_mode is not None:
                opt_kwargs["running_mode"] = running_mode
            try:
                options = mp_vision.FaceLandmarkerOptions(**opt_kwargs)
            except TypeError:
                options = mp_vision.FaceLandmarkerOptions(base_options=base_options)

            landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            mp_img = _build_mp_srgb_image(mp, crop)
            res = landmarker.detect(mp_img)
            faces = getattr(res, "face_landmarks", None)
            if not faces:
                return None
            return _landmarks_to_478(faces[0])
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            failures.append(f"mp.tasks.FaceLandmarker: {exc}")
        finally:
            if landmarker is not None and hasattr(landmarker, "close"):
                try:
                    landmarker.close()
                except Exception:
                    pass
    else:
        if mp_python is None or mp_vision is None:
            failures.append("mp.tasks.vision unavailable")
        elif model_path is None:
            failures.append(
                "no face_landmarker.task found (set ALPHASCAN_MP_FACE_LANDMARKER_TASK or place model in ./checkpoints/)"
            )

    _warn_mediapipe_once(
        "face_landmarker",
        f"MediaPipe FaceLandmarker fell back to color skin heuristic ({'; '.join(failures)}).",
    )
    return None


def _detect_skin_mask(
    rgb: np.ndarray,
    face_landmarks_ram: Optional[np.ndarray],
    head_bbox: Optional[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, bool]:
    """Build pixel skin mask via YCrCb/HSV color thresholds and optional landmark hull."""
    h, w = rgb.shape[:2]
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    skin_color = (
        (ycrcb[..., 0] > 40)
        & (ycrcb[..., 1] >= 130)
        & (ycrcb[..., 1] <= 180)
        & (ycrcb[..., 2] >= 75)
        & (ycrcb[..., 2] <= 132)
        & (hsv[..., 1] >= 15)
        & (hsv[..., 2] >= 50)
    )

    lm_mask = np.zeros((h, w), dtype=bool)
    if face_landmarks_ram is not None and len(face_landmarks_ram) > 0:
        lms = np.asarray(face_landmarks_ram, dtype=np.float32)
        bx0, by0, bx1, by1 = head_bbox if head_bbox is not None else (0, 0, w, h)
        bw = max(1.0, float(bx1 - bx0))
        bh = max(1.0, float(by1 - by0))
        pts_px = np.column_stack([bx0 + lms[:, 0] * bw, by0 + lms[:, 1] * bh])
        valid = np.isfinite(pts_px).all(axis=1)
        if int(valid.sum()) >= 3:
            hull = cv2.convexHull(np.round(pts_px[valid]).astype(np.int32))
            cv2.fillPoly(lm_mask.view(np.uint8), [hull], 1)

    return skin_color | lm_mask, bool(np.any(lm_mask))


def _propagate_draped_hair_below_neck(
    rgb: np.ndarray,
    pmask: np.ndarray,
    hair_above: np.ndarray,
    skin_mask: np.ndarray,
    neck_y_int: int,
    head_h: float,
    hx0: int,
    hx1: int,
    bg_med: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extend hair_mask below neck_y_px only along color-matched components connected to hair_above."""
    h, w = pmask.shape
    if not np.any(hair_above) or neck_y_int >= h - 1:
        return np.zeros((h, w), dtype=bool)

    collar_band_top = max(0, neck_y_int - 4)
    if not np.any(hair_above[collar_band_top : neck_y_int + 1, :]):
        return np.zeros((h, w), dtype=bool)

    rgb_f = rgb.astype(np.float32)
    erode_k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_core = cv2.erode(hair_above.astype(np.uint8), erode_k5, iterations=1) > 0
    dist_to_bg_all: Optional[np.ndarray] = None
    bg_modes: Optional[np.ndarray] = None
    if bg_med is not None:
        bg_arr = np.asarray(bg_med, dtype=np.float32)
        bg_modes = bg_arr.reshape(1, 3) if bg_arr.ndim == 1 else bg_arr.reshape(-1, 3)
        dist_to_bg_all = np.min(
            np.linalg.norm(rgb_f[:, :, None, :] - bg_modes[None, None, :, :], axis=-1),
            axis=-1,
        )
        core_no_bg = hair_core & (dist_to_bg_all >= 28.0)
        if int(np.sum(core_no_bg)) >= 12:
            hair_core = core_no_bg
        else:
            above_no_bg = hair_above & (dist_to_bg_all >= 28.0)
            if int(np.sum(above_no_bg)) >= 12:
                hair_core = above_no_bg
    if int(np.sum(hair_core)) < 8:
        hair_core = hair_above

    hair_rgb = rgb_f[hair_core]
    med_col = np.median(hair_rgb, axis=0)
    spread = float(np.linalg.norm(np.std(hair_rgb, axis=0)))
    tol = min(42.0, max(24.0, 1.6 * spread))

    max_y = min(h, neck_y_int + max(24, int(round(1.1 * head_h))))
    pad_x = max(8, int(round(0.35 * max(1, hx1 - hx0))))
    x_lo = max(0, hx0 - pad_x)
    x_hi = min(w, hx1 + pad_x)

    roi_below = np.zeros((h, w), dtype=bool)
    roi_below[neck_y_int + 1 : max_y, x_lo:x_hi] = True
    roi_below &= pmask & (~skin_mask)

    dist = np.linalg.norm(rgb_f - med_col[None, None, :], axis=-1)
    cand_below = roi_below & (dist <= tol)
    if bg_modes is not None and dist_to_bg_all is not None:
        min_contrast = float(np.min(np.linalg.norm(bg_modes - med_col[None, :], axis=-1)))
        if min_contrast >= 24.0:
            cand_below &= (dist_to_bg_all >= 28.0) & (dist < dist_to_bg_all)
        else:
            cand_below &= (dist_to_bg_all >= 20.0) & (dist < 0.85 * dist_to_bg_all)
    if not np.any(cand_below):
        return np.zeros((h, w), dtype=bool)

    combined = (hair_above | cand_below).astype(np.uint8)
    _, labels = cv2.connectedComponents(combined, connectivity=8)
    touching_labels = set(np.unique(labels[hair_above]).tolist()) - {0}
    if not touching_labels:
        return np.zeros((h, w), dtype=bool)

    grown = np.isin(labels, list(touching_labels)) & cand_below
    max_below_px = int(0.65 * float(np.sum(hair_above)))
    if int(np.sum(grown)) > max_below_px:
        if max_below_px <= 0:
            return np.zeros((h, w), dtype=bool)
        ys_g, xs_g = np.where(grown)
        rank_score = (ys_g - neck_y_int).astype(np.float32) + 0.25 * dist[ys_g, xs_g]
        keep_order = np.argsort(rank_score)[:max_below_px]
        capped = np.zeros((h, w), dtype=bool)
        capped[ys_g[keep_order], xs_g[keep_order]] = True
        comb2 = (hair_above | capped).astype(np.uint8)
        _, lbl2 = cv2.connectedComponents(comb2, connectivity=8)
        touch2 = set(np.unique(lbl2[hair_above]).tolist()) - {0}
        grown = (np.isin(lbl2, list(touch2)) & capped) if touch2 else capped
    return grown


def _extract_nose_profile_curve(
    body_mask: np.ndarray,
    person_mask: np.ndarray,
    neck_y_px: float,
    view_label: str,
    face_landmarks_ram: Optional[np.ndarray],
    head_bbox: Optional[Tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
    """Extract anterior (x_px, y_px) facial profile curve on Left/Right views across eye-to-mouth band."""
    label = view_label.lower()
    if label not in ("right", "left"):
        return None

    h, w = person_mask.shape
    neck_y_int = max(1, min(h - 1, int(round(neck_y_px))))
    head_body = body_mask[: neck_y_int + 1, :]
    ys_b, _ = np.where(head_body)
    if len(ys_b) < 6:
        ys_b, _ = np.where(person_mask[: neck_y_int + 1, :])
        source_mask = person_mask
    else:
        source_mask = body_mask

    if len(ys_b) < 6:
        return None

    head_top = float(ys_b.min())
    head_h = max(8.0, float(neck_y_int) - head_top)
    y_start = int(np.floor(head_top + 0.14 * head_h))
    y_end = int(np.ceil(head_top + 0.88 * head_h))

    if face_landmarks_ram is not None and len(face_landmarks_ram) > 0:
        bx0, by0, bx1, by1 = head_bbox if head_bbox is not None else (0, 0, w, h)
        bh = max(1.0, float(by1 - by0))
        lm_y = by0 + face_landmarks_ram[:, 1] * bh
        valid_y = lm_y[np.isfinite(lm_y)]
        if len(valid_y) > 0:
            y_start = max(0, int(np.floor(float(valid_y.min()) - 0.05 * head_h)))
            y_end = min(neck_y_int, int(np.ceil(float(valid_y.max()))))

    y_start = max(0, min(neck_y_int - 2, y_start))
    y_end = max(y_start + 2, min(neck_y_int, y_end))

    pts: List[Tuple[float, float]] = []
    for y in range(y_start, y_end + 1):
        xs = np.where(source_mask[y])[0]
        if len(xs) == 0:
            continue
        x_ant = float(xs.max()) if label == "right" else float(xs.min())
        pts.append((x_ant, float(y)))

    if len(pts) < 3:
        return None
    return np.asarray(pts, dtype=np.float32)


def _compute_local_bg_modes(
    rgb: np.ndarray,
    head_fg: np.ndarray,
    pmask: np.ndarray,
    upper_lim: int,
    head_h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract deterministic local multi-modal background RGB centers around head_fg."""
    h, w = pmask.shape[:2]
    ring_in = max(2, int(round(0.03 * head_h)))
    ring_out = max(12, min(64, int(round(0.38 * head_h))))
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_in + 1, 2 * ring_in + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_out + 1, 2 * ring_out + 1))
    head_u8 = head_fg[:upper_lim, :].astype(np.uint8)
    dil_in = cv2.dilate(head_u8, k_in, iterations=1) > 0
    dil_out = cv2.dilate(head_u8, k_out, iterations=1) > 0
    ring_mask = dil_out & (~dil_in) & (~pmask[:upper_lim, :])

    if int(np.sum(ring_mask)) >= 32:
        bg_pixels = rgb[:upper_lim, :][ring_mask].astype(np.float32)
    elif int(np.sum(~pmask[:upper_lim, :])) >= 16:
        bg_pixels = rgb[:upper_lim, :][~pmask[:upper_lim, :]].astype(np.float32)
    elif np.any(~pmask):
        bg_pixels = rgb[~pmask].astype(np.float32)
    else:
        bg_pixels = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0).astype(np.float32)

    if len(bg_pixels) == 0:
        default_bg = np.array([240.0, 240.0, 240.0], dtype=np.float32)
        return default_bg, default_bg.reshape(1, 3)

    bg_med = np.median(bg_pixels, axis=0).astype(np.float32)
    stride = max(1, len(bg_pixels) // 1200)
    pts = bg_pixels[::stride]
    centers: List[np.ndarray] = [bg_med.copy()]
    for _ in range(4):
        c_arr = np.stack(centers, axis=0)
        dists = np.min(np.linalg.norm(pts[:, None, :] - c_arr[None, :, :], axis=-1), axis=1)
        max_idx = int(np.argmax(dists))
        if float(dists[max_idx]) < 24.0:
            break
        centers.append(pts[max_idx].copy())

    c_arr = np.stack(centers, axis=0)
    if len(c_arr) > 1:
        # Refine cluster centers over 2 deterministic assignment steps and drop tiny stray clusters.
        for _ in range(2):
            d_mat = np.linalg.norm(pts[:, None, :] - c_arr[None, :, :], axis=-1)
            assign = np.argmin(d_mat, axis=1)
            new_centers = []
            min_support = max(4, int(round(0.03 * len(pts))))
            for ci in range(len(c_arr)):
                sel = pts[assign == ci]
                if len(sel) >= min_support:
                    new_centers.append(np.median(sel, axis=0).astype(np.float32))
            if new_centers:
                c_arr = np.stack(new_centers, axis=0)
            else:
                c_arr = bg_med.reshape(1, 3)
                break
    return bg_med, c_arr


def extract_head_semantics(
    rgb: np.ndarray,
    person_mask: np.ndarray,
    neck_y_px: Optional[float],
    view_label: str,
    face_landmarks_ram: Optional[np.ndarray] = None,
    head_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """Separate hair_mask vs face_skin_mask, exclude hair from body_mask, and detect nose/collar cues."""
    verify_license_compliance()
    pmask = np.asarray(person_mask, dtype=bool)
    h, w = pmask.shape[:2]
    empty_bool = np.zeros((h, w), dtype=bool)
    if not np.any(pmask):
        return {
            "body_mask": empty_bool.copy(),
            "hair_mask": empty_bool.copy(),
            "face_skin_mask": empty_bool.copy(),
            "nose_profile_curve": None,
            "collar_hair_overlap": False,
            "warnings": [],
        }

    ys_fg, xs_fg = np.where(pmask)
    y_top, y_bot = int(ys_fg.min()), int(ys_fg.max())
    fg_h = max(1, y_bot - y_top + 1)

    if neck_y_px is None:
        if head_bbox is not None:
            neck_y_val = float(head_bbox[3])
        else:
            neck_y_val = float(y_top) + 0.18 * float(fg_h)
    else:
        neck_y_val = float(neck_y_px)
    neck_y_int = max(y_top + 1, min(h - 1, int(round(neck_y_val))))
    head_h = max(10.0, float(neck_y_int - y_top))

    head_fg = pmask.copy()
    head_fg[neck_y_int + 1 :, :] = False
    num_hc, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(head_fg.astype(np.uint8), connectivity=8)
    if num_hc > 2:
        largest_lbl = 1 + int(np.argmax(h_stats[1:, cv2.CC_STAT_AREA]))
        neck_lbls = set(np.unique(h_labels[max(0, neck_y_int - 3) : neck_y_int + 1, :]).tolist()) - {0}
        anchor_lbls = neck_lbls | {largest_lbl}
        head_anchor = np.isin(h_labels, list(anchor_lbls))
        bridge_px = max(3, min(15, int(round(0.10 * head_h))))
        bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge_px + 1, 2 * bridge_px + 1))
        anchor_dil = cv2.dilate(head_anchor.astype(np.uint8), bridge_k, iterations=1) > 0
        near_lbls = (set(np.unique(h_labels[anchor_dil]).tolist()) - {0}) | anchor_lbls
        head_fg = np.isin(h_labels, list(near_lbls))
    ys_h, xs_h = np.where(head_fg)
    hx0 = int(xs_h.min()) if len(xs_h) > 0 else int(xs_fg.min())
    hx1 = int(xs_h.max()) + 1 if len(xs_h) > 0 else int(xs_fg.max()) + 1

    skin_raw, has_landmarks = _detect_skin_mask(rgb, face_landmarks_ram, head_bbox)
    hsv_head = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    is_achromatic = (
        not has_landmarks
        and not np.any(skin_raw & head_fg)
        and (int(np.max(hsv_head[..., 1][head_fg])) < 10 if np.any(head_fg) else True)
    )

    upper_lim = max(neck_y_int + 1, min(h, max(neck_y_int + int(round(0.25 * head_h)), h // 2)))
    bg_med, bg_modes = _compute_local_bg_modes(
        rgb=rgb, head_fg=head_fg, pmask=pmask, upper_lim=upper_lim, head_h=head_h
    )

    if is_achromatic:
        crown_cut = y_top + int(round(0.24 * head_h))
        hair_above_raw = head_fg.copy()
        hair_above_raw[crown_cut:, :] = False
        face_skin_mask = head_fg & (~hair_above_raw)
    else:
        skin_in_head = skin_raw & head_fg
        skin_filled_raw = np.asarray(binary_fill_holes(skin_in_head), dtype=bool) & head_fg
        holes_only = skin_filled_raw & (~skin_in_head)
        if np.any(holes_only) and np.any(skin_in_head):
            skin_med_v = float(np.median(hsv_head[..., 2][skin_in_head]))
            n_holes, hole_lbls, hole_stats, _ = cv2.connectedComponentsWithStats(
                holes_only.astype(np.uint8), connectivity=8
            )
            max_hole_area = max(32, int(0.15 * float(np.sum(head_fg))))
            keep_holes = np.zeros((h, w), dtype=bool)
            for hl in range(1, n_holes):
                area_hl = int(hole_stats[hl, cv2.CC_STAT_AREA])
                comp_h = hole_lbls == hl
                comp_v = float(np.median(hsv_head[..., 2][comp_h]))
                if area_hl <= max_hole_area and (comp_v >= 0.45 * skin_med_v or area_hl <= 48):
                    keep_holes |= comp_h
            face_skin_mask = skin_in_head | keep_holes
        else:
            face_skin_mask = skin_in_head
        hair_above_raw = head_fg & (~face_skin_mask)

    erode_px_h = max(2, int(round(0.015 * head_h)))
    dist_from_bg = distance_transform_edt(pmask)
    rgb_f = rgb.astype(np.float32)
    dist_to_bg = np.min(
        np.linalg.norm(rgb_f[:, :, None, :] - bg_modes[None, None, :, :], axis=-1),
        axis=-1,
    )

    core_depth = max(float(erode_px_h) + 2.0, 0.12 * head_h)
    core_candidates = hair_above_raw & (dist_from_bg >= core_depth) & (dist_to_bg >= 28.0)
    if int(np.sum(core_candidates)) < 12:
        core_candidates = hair_above_raw & (dist_from_bg >= float(erode_px_h)) & (dist_to_bg >= 26.0)
    if int(np.sum(core_candidates)) < 8:
        core_candidates = hair_above_raw & (dist_to_bg >= 24.0)

    outer_shell_px = max(float(erode_px_h) + 3.0, 0.18 * head_h)
    in_outer_shell = dist_from_bg <= outer_shell_px
    in_border_ring = dist_from_bg <= float(erode_px_h)

    if np.any(core_candidates):
        core_hair_med = np.median(rgb_f[core_candidates], axis=0)
        dist_to_hair = np.linalg.norm(rgb_f - core_hair_med[None, None, :], axis=-1)
        bg_hair_contrast = float(np.max(np.linalg.norm(bg_modes - core_hair_med[None, :], axis=-1)))
        min_bg_hair_contrast = float(np.min(np.linalg.norm(bg_modes - core_hair_med[None, :], axis=-1)))
        # Never disable background veto completely: always strip boundary pixels matching local bg modes.
        if min_bg_hair_contrast >= 24.0:
            bg_veto = (
                (in_outer_shell & (dist_to_bg < 28.0) & (dist_to_bg < dist_to_hair))
                | (in_border_ring & (dist_to_bg < 42.0) & (dist_to_bg < 0.90 * dist_to_hair))
                | (in_border_ring & (dist_to_bg < 0.65 * dist_to_hair) & (dist_to_bg < 0.45 * bg_hair_contrast))
                | ((dist_to_bg < 22.0) & (dist_to_bg < 0.70 * dist_to_hair))
            )
        else:
            bg_veto = (
                (in_outer_shell & (dist_to_bg < 24.0) & (dist_to_bg < 0.85 * dist_to_hair))
                | (in_border_ring & (dist_to_bg < 34.0) & (dist_to_bg < 0.80 * dist_to_hair))
                | ((dist_to_bg < 18.0) & (dist_to_bg < 0.65 * dist_to_hair))
            )
        hair_above = hair_above_raw & (~bg_veto)
    else:
        hair_above = hair_above_raw & (~(in_outer_shell & (dist_to_bg < 22.0)))

    # Keep hair components anchored to face skin or upper-crown core, dropping detached poster/wall blobs.
    if not is_achromatic and np.any(hair_above):
        n_ha, ha_lbls, ha_stats, _ = cv2.connectedComponentsWithStats(hair_above.astype(np.uint8), connectivity=8)
        if n_ha > 2:
            anchor_mask = np.zeros((h, w), dtype=bool)
            if np.any(face_skin_mask):
                skin_pad = max(4, int(round(0.18 * head_h)))
                k_sp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * skin_pad + 1, 2 * skin_pad + 1))
                anchor_mask |= cv2.dilate(face_skin_mask.astype(np.uint8), k_sp, iterations=1) > 0
            cx_head = 0.5 * float(hx0 + hx1)
            crown_box = np.zeros((h, w), dtype=bool)
            cy_cut = min(neck_y_int, y_top + max(6, int(round(0.55 * head_h))))
            cx_lo = max(0, int(round(cx_head - 0.38 * head_h)))
            cx_hi = min(w, int(round(cx_head + 0.38 * head_h)))
            crown_box[y_top : cy_cut + 1, cx_lo:cx_hi] = True
            anchor_mask |= crown_box
            largest_ha = 1 + int(np.argmax(ha_stats[1:, cv2.CC_STAT_AREA]))
            keep_ha = np.zeros((h, w), dtype=bool)
            for lbl in range(1, n_ha):
                comp_ha = ha_lbls == lbl
                if lbl == largest_ha or np.any(comp_ha & anchor_mask):
                    keep_ha |= comp_ha
            if np.any(keep_ha):
                hair_above = keep_ha

    hair_below = _propagate_draped_hair_below_neck(
        rgb=rgb,
        pmask=pmask,
        hair_above=hair_above,
        skin_mask=skin_raw,
        neck_y_int=neck_y_int,
        head_h=head_h,
        hx0=hx0,
        hx1=hx1,
        bg_med=bg_modes,
    )
    hair_mask = (hair_above | hair_below) & pmask & (~face_skin_mask)
    body_mask = pmask & (~hair_mask)

    below_mask = hair_mask.copy()
    below_mask[: neck_y_int + 1, :] = False
    below_count = int(np.sum(below_mask))
    total_hair = max(1, int(np.sum(hair_mask)))
    below_frac = below_count / float(total_hair)
    max_drop_px = float(np.max(np.where(below_mask)[0]) - neck_y_int) if below_count > 0 else 0.0

    collar_hair_overlap = below_count >= 12 and (
        below_frac > 0.04 or max_drop_px >= max(5.0, 0.07 * head_h)
    )
    warnings: List[str] = []
    if collar_hair_overlap:
        warnings.append(
            "Collar hair overlap detected (hair drapes below neck collar line); "
            "follow the hair-up protocol so hair stays clear of neck and shoulders."
        )

    nose_profile_curve = _extract_nose_profile_curve(
        body_mask=body_mask,
        person_mask=pmask,
        neck_y_px=float(neck_y_int),
        view_label=view_label,
        face_landmarks_ram=face_landmarks_ram,
        head_bbox=head_bbox,
    )

    return {
        "body_mask": body_mask,
        "hair_mask": hair_mask,
        "face_skin_mask": face_skin_mask,
        "nose_profile_curve": nose_profile_curve,
        "collar_hair_overlap": collar_hair_overlap,
        "warnings": warnings,
    }


def _estimate_neck_y_from_output(
    out: Dict[str, Any],
    person_mask: np.ndarray,
    head_crop_bbox: Optional[Tuple[int, int, int, int]],
) -> Optional[float]:
    """Infer neck y-coordinate in image pixels from 2D/3D keypoints or fallback bounds."""
    k2d_raw = out.get("pred_keypoints_2d")
    if k2d_raw is not None:
        k2d = np.asarray(k2d_raw, dtype=np.float32)
        if k2d.ndim == 2 and k2d.shape[1] >= 2:
            sh_indices, _, _ = _get_keypoint_semantics(len(k2d))
            if len(k2d) >= 127 and len(k2d) > NECK_BASE_JOINT and np.isfinite(k2d[NECK_BASE_JOINT, :2]).all():
                if k2d[NECK_BASE_JOINT, 1] > 0.0:
                    return float(k2d[NECK_BASE_JOINT, 1])
            elif len(k2d) == 70 and np.isfinite(k2d[MHR70_NECK_JOINT, :2]).all():
                if k2d[MHR70_NECK_JOINT, 1] > 0.0:
                    return float(k2d[MHR70_NECK_JOINT, 1])
            if sh_indices:
                vals = [
                    float(k2d[idx, 1])
                    for idx in sh_indices
                    if len(k2d) > idx and np.isfinite(k2d[idx, :2]).all() and k2d[idx, 1] > 0.0
                ]
                if vals:
                    return float(np.mean(vals))

    k3d_raw = out.get("pred_keypoints_3d")
    cam_t_raw = out.get("pred_cam_t")
    focal = out.get("focal_length")
    if k3d_raw is not None and cam_t_raw is not None and focal is not None:
        k3d = np.asarray(k3d_raw, dtype=np.float32)
        cam_t = np.asarray(cam_t_raw, dtype=np.float32).reshape(3)
        if k3d.ndim == 2 and k3d.shape[1] >= 3 and len(k3d) > 0:
            sh_indices, hip_pair, _ = _get_keypoint_semantics(len(k3d))
            neck_3d: Optional[np.ndarray] = None
            if len(k3d) >= 127 and len(k3d) > NECK_BASE_JOINT and np.isfinite(k3d[NECK_BASE_JOINT]).all():
                neck_3d = k3d[NECK_BASE_JOINT]
            elif (
                len(k3d) == 70
                and np.isfinite(k3d[MHR70_NECK_JOINT]).all()
                and float(np.linalg.norm(k3d[MHR70_NECK_JOINT])) > 1e-4
            ):
                neck_3d = k3d[MHR70_NECK_JOINT]
            elif sh_indices and len(k3d) > max(sh_indices):
                s0, s1 = k3d[sh_indices[0]], k3d[sh_indices[1]]
                if np.isfinite(s0).all() and np.isfinite(s1).all():
                    neck_3d = 0.5 * (s0 + s1)

            if neck_3d is not None:
                pt = neck_3d + cam_t
                y_val = float(pt[1])
                # Flip Y if keypoints are in +Y-up coordinates (neck/head above hip).
                if hip_pair and len(k3d) > max(hip_pair):
                    h0, h1 = k3d[hip_pair[0]], k3d[hip_pair[1]]
                    if np.isfinite(h0).all() and np.isfinite(h1).all():
                        hip_y = 0.5 * (float(h0[1]) + float(h1[1]))
                        if float(neck_3d[1]) > hip_y:
                            y_val = -y_val
                if pt[2] > 0.1:
                    h_img = person_mask.shape[0]
                    return float(focal) * (y_val / float(pt[2])) + (h_img / 2.0)

    if head_crop_bbox is not None and len(head_crop_bbox) >= 4:
        return float(head_crop_bbox[3])
    return None


def process_views_semantics(
    raw_results: List[Dict[str, Any]],
    estimator: Any = None,
    mediapipe_segmenter_task: Optional[str] = None,
    mediapipe_face_task: Optional[str] = None,
) -> List[SegmentedView]:
    """Run segmentation, azimuth inference, and RAM-only head semantics for all views."""
    verify_license_compliance()
    human_seg = getattr(estimator, "human_segmentor", None) if estimator is not None else None
    total_views = len(raw_results)
    views: List[SegmentedView] = []

    for idx, rec in enumerate(raw_results):
        out = rec.get("output") or {}
        rgb = rec.get("image_rgb")
        if rgb is None:
            img_path = rec.get("image_path")
            if img_path and os.path.exists(img_path):
                rgb = prepare_image_rgb(img_path)
            else:
                raise ValueError(f"View {idx} missing 'image_rgb' and valid 'image_path'.")

        prior_core = _project_mhr_person_prior(out, rgb.shape[0], rgb.shape[1])
        pre_mask_raw = out.get("mask") if out.get("mask") is not None else out.get("person_mask")
        person_mask: Optional[np.ndarray] = None
        mask_source = "unknown"
        if pre_mask_raw is not None:
            norm_pre = _normalize_mask_output(pre_mask_raw, rgb.shape[0], rgb.shape[1])
            if norm_pre is not None and np.any(norm_pre):
                refined_pre = _refine_precomputed_or_seg_mask(norm_pre, prior_core)
                if np.any(refined_pre):
                    person_mask = refined_pre
                    mask_source = "precomputed"
        if person_mask is None:
            seg_kwargs: Dict[str, Any] = {
                "rgb": rgb,
                "bbox": out.get("bbox"),
                "human_segmentor": human_seg,
                "erode_px": 0,
                "prior_core": prior_core,
            }
            if mediapipe_segmenter_task is not None:
                seg_kwargs["mediapipe_task_path"] = mediapipe_segmenter_task
            person_mask, mask_source = _segment_person_mask_with_source(**seg_kwargs)

        azimuth_deg = infer_view_azimuth(
            pred_keypoints_3d=out.get("pred_keypoints_3d"),
            pred_vertices=out.get("pred_vertices"),
            fallback_index=idx,
            total_views=total_views,
        )
        view_label = classify_view_azimuth(azimuth_deg)
        slot_role = rec.get("slot_role")
        cardinal_angles = dict(CARDINAL_CENTERS)
        if slot_role in cardinal_angles and slot_role != view_label:
            view_label = slot_role
            azimuth_deg = cardinal_angles[slot_role]

        head_crop_bbox = rec.get("head_crop_bbox")
        neck_y_px = _estimate_neck_y_from_output(out, person_mask, head_crop_bbox)

        head_crop_rgb = rec.get("head_crop_rgb")
        if head_crop_rgb is None:
            ys, _ = np.where(person_mask)
            if len(ys) > 0:
                y_cut = int(round(neck_y_px)) if neck_y_px is not None else int(ys.min() + 0.20 * (ys.max() - ys.min()))
                y_cut = max(int(ys.min()) + 2, min(rgb.shape[0], y_cut))
                _, xs_head = np.where(person_mask[:y_cut, :])
                if len(xs_head) > 0:
                    head_crop_bbox = (int(xs_head.min()), int(ys.min()), int(xs_head.max()) + 1, y_cut)
                    head_crop_rgb = rgb[head_crop_bbox[1] : head_crop_bbox[3], head_crop_bbox[0] : head_crop_bbox[2]]

        if head_crop_rgb is not None and view_label in ("front", "right", "left", "oblique"):
            if mediapipe_face_task is not None:
                face_landmarks_ram = extract_face_landmarks_ram(
                    head_crop_rgb, task_model_path=mediapipe_face_task
                )
            else:
                face_landmarks_ram = extract_face_landmarks_ram(head_crop_rgb)
        else:
            face_landmarks_ram = None

        sem = extract_head_semantics(
            rgb=rgb,
            person_mask=person_mask,
            neck_y_px=neck_y_px,
            view_label=view_label,
            face_landmarks_ram=face_landmarks_ram,
            head_bbox=head_crop_bbox,
        )
        views.append(
            SegmentedView(
                view_index=idx,
                azimuth_deg=azimuth_deg,
                view_label=view_label,
                person_mask=person_mask,
                body_mask=sem["body_mask"],
                hair_mask=sem["hair_mask"],
                face_skin_mask=sem["face_skin_mask"],
                nose_profile_curve=sem["nose_profile_curve"],
                face_landmarks_ram=face_landmarks_ram,
                collar_hair_overlap=sem["collar_hair_overlap"],
                warnings=sem["warnings"],
                mask_source=mask_source,
                landmarks_active=bool(face_landmarks_ram is not None),
            )
        )
    return views


def strip_ram_only_semantics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip all RAM-only images, crops, depth maps, landmarks, and SegmentedViews before disk caching."""
    return strip_raw_images_for_cache(records)
