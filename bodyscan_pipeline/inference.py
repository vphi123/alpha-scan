"""Step 1 inference wrapper using SAM 3D Body + Detectron2 (ViTDet) + MoGe2."""

from collections import OrderedDict
import inspect
import os
import sys
import warnings
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

_EXTRA_FORBIDDEN_MODULES: Tuple[str, ...] = ("flame", "briaai", "rmbg")
_SAM2_CHECKPOINT_NAMES: Tuple[str, ...] = (
    "sam2.1_hiera_large.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_tiny.pt",
    "sam2_hiera_large.pt",
    "sam2_hiera_base_plus.pt",
    "sam2_hiera_small.pt",
    "sam2_hiera_tiny.pt",
    "sam2.pt",
    "sam_vit_h_4b8939.pth",
    "sam_vit_b_01ec64.pth",
)
_MOGE2_DEPTH_CACHE: "OrderedDict[Tuple[int, int], Tuple[Any, Tuple[Any, ...], np.ndarray]]" = OrderedDict()
_MOGE2_CACHE_MAX_ENTRIES = 16


def clear_moge2_depth_cache() -> None:
    """Clear cached MoGe2 depth maps."""
    _MOGE2_DEPTH_CACHE.clear()


def verify_license_compliance() -> None:
    """Ensure no license-restricted modules (AGPL/non-commercial) are loaded."""
    from .config import FORBIDDEN_MODULES

    forbidden = tuple(FORBIDDEN_MODULES) + _EXTRA_FORBIDDEN_MODULES
    loaded = [
        mod
        for mod in forbidden
        if mod in sys.modules or any(k.startswith(f"{mod}.") for k in sys.modules)
    ]
    if loaded:
        raise RuntimeError(f"License violation: forbidden modules loaded: {loaded}")


def _resolve_segmentor_weights(
    repo_path: str,
    checkpoint_path: str,
    segmentor_path: str,
) -> str:
    """Resolve optional SAM/SAM2 checkpoint path from argument, env vars, or local checkpoints."""
    if segmentor_path:
        return segmentor_path

    for env_name in ("ALPHASCAN_SAM2_CHECKPOINT", "SAM2_CHECKPOINT", "ALPHASCAN_SEGMENTOR_PATH"):
        env_val = os.environ.get(env_name, "").strip()
        if env_val and os.path.isfile(env_val):
            return env_val

    search_dirs: List[Path] = []
    if repo_path:
        rp = Path(repo_path)
        search_dirs.extend([rp / "checkpoints", rp / "checkpoints" / "sam2"])
    if checkpoint_path:
        cp = Path(checkpoint_path).resolve()
        search_dirs.extend([cp.parent, cp.parent / "sam2", cp.parent.parent])
    search_dirs.extend([Path("./checkpoints"), Path("./checkpoints/sam2")])

    seen: set[str] = set()
    for base_dir in search_dirs:
        key = str(base_dir)
        if key in seen:
            continue
        seen.add(key)
        for fname in _SAM2_CHECKPOINT_NAMES:
            cand = base_dir / fname
            if cand.is_file():
                return str(cand)
    return ""


def _infer_sam2_config_name(ckpt_path: str) -> str:
    """Select the Hydra config YAML corresponding to a SAM2 checkpoint filename."""
    env_cfg = os.environ.get("ALPHASCAN_SAM2_CONFIG", "").strip()
    if env_cfg:
        return env_cfg
    name = os.path.basename(ckpt_path).lower()
    is_v21 = "2.1" in name or "2_1" in name
    prefix = "configs/sam2.1/sam2.1_hiera" if is_v21 else "configs/sam2/sam2_hiera"
    if "tiny" in name or name.endswith("_t.pt"):
        return f"{prefix}_t.yaml"
    if "small" in name or name.endswith("_s.pt"):
        return f"{prefix}_s.yaml"
    if "base" in name or "b+" in name or "bplus" in name:
        return f"{prefix}_b+.yaml"
    return f"{prefix}_l.yaml"


def _call_with_supported_kwargs(fn: Any, candidates: Dict[str, Any]) -> Any:
    """Invoke factory/constructor passing only kwargs accepted by its signature."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return fn(**candidates)
        filtered = {k: v for k, v in candidates.items() if k in params}
        return fn(**filtered)
    except (ValueError, TypeError):
        return fn(**candidates)


class _SAM2SegmentorAdapter:
    """Adapt SAM2ImagePredictor (`set_image`/`predict`) to expose `.run_sam(img, boxes)`."""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def set_image(self, image: np.ndarray) -> Any:
        """Delegate `set_image` to the underlying SAM2 predictor."""
        return self.predictor.set_image(image)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate `predict` to the underlying SAM2 predictor."""
        return self.predictor.predict(*args, **kwargs)

    def run_sam(self, img: np.ndarray, boxes: Any = None, **kwargs: Any) -> np.ndarray:
        """Predict `(N, H, W)` uint8 binary masks for `img` and `boxes`."""
        h, w = img.shape[:2]
        if boxes is None:
            box_list: List[Optional[np.ndarray]] = [None]
        else:
            box_arr = np.asarray(boxes, dtype=np.float32)
            if box_arr.size == 0:
                return np.zeros((0, h, w), dtype=np.uint8)
            box_arr = box_arr.reshape(-1, 4)
            box_list = [b for b in box_arr]

        masks_out: List[np.ndarray] = []
        with torch.no_grad():
            self.predictor.set_image(img)
            for single_box in box_list:
                pred = self.predictor.predict(box=single_box, multimask_output=False)
                raw = pred[0] if isinstance(pred, tuple) and len(pred) > 0 else pred
                if hasattr(raw, "detach"):
                    raw = raw.detach().cpu().numpy()
                mask = np.asarray(raw)
                while mask.ndim > 2:
                    if mask.ndim == 3 and mask.shape[0] == h and mask.shape[1] == w:
                        mask = mask[..., 0]
                    else:
                        mask = mask[0]
                if mask.size == 0 or mask.ndim != 2:
                    masks_out.append(np.zeros((h, w), dtype=np.uint8))
                    continue
                if mask.shape != (h, w):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                masks_out.append((mask.astype(np.float32) > 0.5).astype(np.uint8))
        return np.stack(masks_out, axis=0).astype(np.uint8)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.predictor, name)


def _wrap_human_segmentor(seg: Any) -> Any:
    """Ensure initialized SAM/SAM2 predictors expose `.run_sam(img, boxes)`."""
    if seg is None or hasattr(seg, "run_sam"):
        return seg
    if hasattr(seg, "set_image") and hasattr(seg, "predict"):
        return _SAM2SegmentorAdapter(seg)
    return seg


def _build_optional_human_segmentor(
    device: torch.device,
    repo_path: str = "",
    checkpoint_path: str = "",
    segmentor_name: str = "sam2",
    segmentor_path: str = "",
) -> Any:
    """Try initializing SAM/SAM2 human segmentor cleanly; warn and fall back to None if unavailable."""
    verify_license_compliance()
    resolved_path = _resolve_segmentor_weights(repo_path, checkpoint_path, segmentor_path)
    errors: List[str] = []

    for mod_name in ("tools.build_segmentor", "tools.build_sam"):
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except ImportError as exc:
            errors.append(f"{mod_name}: {exc}")
            continue

        factory = (
            getattr(mod, "build_segmentor", None)
            or getattr(mod, "build_human_segmentor", None)
            or getattr(mod, "HumanSegmentor", None)
        )
        if callable(factory):
            try:
                seg = _call_with_supported_kwargs(
                    factory,
                    {"name": segmentor_name, "device": device, "path": resolved_path},
                )
                verify_license_compliance()
                if seg is not None:
                    return _wrap_human_segmentor(seg)
            except Exception as exc:
                if "License violation" in str(exc):
                    raise
                errors.append(f"{mod_name}: {exc}")

    try:
        from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-not-found]
    except ImportError as exc:
        errors.append(f"sam2: {exc}")
    else:
        try:
            if resolved_path and os.path.isfile(resolved_path):
                cfg_name = _infer_sam2_config_name(resolved_path)
                try:
                    sam2_model = build_sam2(cfg_name, resolved_path, device=str(device))
                except TypeError:
                    sam2_model = build_sam2(cfg_name, resolved_path, device=device)
                verify_license_compliance()
                return _wrap_human_segmentor(SAM2ImagePredictor(sam2_model))

            hf_model = os.environ.get("ALPHASCAN_SAM2_HF_MODEL", "").strip()
            if hf_model and hasattr(SAM2ImagePredictor, "from_pretrained"):
                seg = SAM2ImagePredictor.from_pretrained(hf_model, device=str(device))
                verify_license_compliance()
                return _wrap_human_segmentor(seg)

            errors.append(
                "sam2 installed but checkpoint weights not found "
                "(set ALPHASCAN_SAM2_CHECKPOINT or place sam2*.pt in ./checkpoints/)"
            )
        except Exception as exc:
            if "License violation" in str(exc):
                raise
            errors.append(f"sam2 init failed: {exc}")

    reason = "; ".join(errors) if errors else "no SAM/SAM2 backend available"
    warnings.warn(
        f"Optional SAM/SAM2 human_segmentor unavailable ({reason}); falling back to human_segmentor=None.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


DEFAULT_SAM3D_MODEL_CONFIG_YAML = """TRAIN:
  USE_FP16: true
  FP16_TYPE: bfloat16
MODEL:
  IMAGE_SIZE:
  - 512
  - 512
  IMAGE_MEAN:
  - 0.485
  - 0.456
  - 0.406
  IMAGE_STD:
  - 0.229
  - 0.224
  - 0.225
  BACKBONE:
    TYPE: dinov3_vith16plus
    PRETRAINED_WEIGHTS: ''
    FROZEN_STAGES: -1
    DROP_PATH_RATE: 0.1
  DECODER:
    TYPE: sam
    DIM: 1024
    DEPTH: 6
    HEADS: 8
    MLP_DIM: 1024
    DIM_HEAD: 64
    LAYER_SCALE_INIT: 0.0
    DROP_RATE: 0.0
    ATTN_DROP_RATE: 0.0
    DROP_PATH_RATE: 0.0
    FFN_TYPE: origin
    ENABLE_TWOWAY: false
    REPEAT_PE: true
    FROZEN: false
    CONDITION_TYPE: cliff
    USE_INTRIN_CENTER: true
    DO_INTERM_PREDS: true
    DO_INTERM_SUP: true
    DO_KEYPOINT_TOKENS: true
    DO_HAND_DETECT_TOKENS: true
    KEYPOINT_TOKEN_UPDATE: v2
    KEYPOINT_TOKEN_UPDATE_COORD_EMB_USE_MLP: true
    DO_KEYPOINT3D_TOKENS: true
  PROMPT_ENCODER:
    ENABLE: true
    MAX_NUM_CLICKS: 2
    PROMPT_KEYPOINTS: mhr70
    FROZEN: false
    KEYPOINT_SAMPLER:
      TYPE: v1
      WORST_RATIO: 0.8
      KEYBODY_RATIO: 0.8
      NEGATIVE_RATIO: 0.1
      DUMMY_RATIO: 0.1
      DISTANCE_THRESH: 0.0001
    MASK_EMBED_TYPE: v2
    MASK_PROMPT: v1
  PERSON_HEAD:
    POSE_TYPE: mhr
    CAMERA_ENABLE: true
    CAMERA_TYPE: perspective
    ZERO_POSE_INIT: true
    ZERO_POSE_INIT_BODY_FACTOR: 1
  MHR_HEAD:
    MLP_DEPTH: 2
    MLP_CHANNEL_DIV_FACTOR: 1
  CAMERA_HEAD:
    MLP_DEPTH: 2
    MLP_CHANNEL_DIV_FACTOR: 1
  DEFAULT_SCALE_FACTOR_HAND: 10
  ENABLE_BODY: true
  ENABLE_HAND: true
"""


def load_sam3d_estimator(
    sam_repo_dir: str,
    checkpoint_path: str,
    mhr_path: str,
    device: Optional[torch.device] = None,
    enable_human_segmentor: bool = True,
    segmentor_name: str = "sam2",
    segmentor_path: str = "",
) -> Tuple[Any, Any]:
    """Load SAM 3D Body model and Detectron2/MoGe2 estimator without prohibited deps."""
    verify_license_compliance()
    repo_path = str(Path(sam_repo_dir).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.build_detector import HumanDetector
    from tools.build_fov_estimator import FOVEstimator

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure checkpoints/model_config.yaml is mirrored to repo root or created from default if missing
    try:
        ckpt_p = Path(checkpoint_path).resolve()
        cand_yaml = ckpt_p.parent / "model_config.yaml"
        target_yaml = Path(sam_repo_dir).resolve() / "checkpoints" / "model_config.yaml"
        if cand_yaml.is_file() and not target_yaml.is_file():
            target_yaml.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cand_yaml, target_yaml)
        elif target_yaml.is_file() and not cand_yaml.is_file():
            cand_yaml.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target_yaml, cand_yaml)
        elif not target_yaml.is_file() and not cand_yaml.is_file():
            target_yaml.parent.mkdir(parents=True, exist_ok=True)
            target_yaml.write_text(DEFAULT_SAM3D_MODEL_CONFIG_YAML, encoding="utf-8")
            cand_yaml.parent.mkdir(parents=True, exist_ok=True)
            cand_yaml.write_text(DEFAULT_SAM3D_MODEL_CONFIG_YAML, encoding="utf-8")
        colab_default = Path("/content/sam-3d-body/checkpoints/model_config.yaml")
        if Path("/content/sam-3d-body").is_dir() and not colab_default.is_file():
            colab_default.parent.mkdir(parents=True, exist_ok=True)
            colab_default.write_text(DEFAULT_SAM3D_MODEL_CONFIG_YAML, encoding="utf-8")
    except Exception as exc:
        warnings.warn(f"Auto-config yaml provisioning notice: {exc}", RuntimeWarning, stacklevel=2)
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=dev, mhr_path=mhr_path)
    human_segmentor = (
        _build_optional_human_segmentor(
            device=dev,
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            segmentor_name=segmentor_name,
            segmentor_path=segmentor_path,
        )
        if enable_human_segmentor
        else None
    )
    if enable_human_segmentor and human_segmentor is None:
        try:
            human_segmentor = build_torchvision_human_segmentor(device=dev)
        except Exception as exc:
            if "License violation" in str(exc):
                raise
            warnings.warn(
                f"Torchvision Mask R-CNN fallback unavailable ({exc}); proceeding without human_segmentor.",
                RuntimeWarning,
                stacklevel=2,
            )
    # Build human detector safely with fallback to None
    detector = None
    try:
        from tools.build_detector import HumanDetector
        detector = HumanDetector(name="vitdet", device=dev, path="")
    except Exception as exc:
        warnings.warn(f"HumanDetector vitdet unavailable ({exc}); falling back to detector=None", RuntimeWarning)
        detector = None

    # Build FOV estimator safely with fallback to None
    fov_est = None
    try:
        from tools.build_fov_estimator import FOVEstimator
        fov_est = FOVEstimator(name="moge2", device=dev, path="")
    except Exception as exc:
        warnings.warn(f"FOVEstimator moge2 unavailable ({exc}); falling back to fov_estimator=None", RuntimeWarning)
        fov_est = None

    verify_license_compliance()
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_est,
    )
    return model, estimator


class _TorchvisionHumanSegmentor:
    """BSD-3 Mask R-CNN fallback human segmentor when SAM 2 is not installed."""

    def __init__(self, model: Any, device: torch.device, score_thresh: float = 0.45) -> None:
        self.model = model
        self.device = device
        self.score_thresh = float(score_thresh)

    def _predict_person_instances(
        self,
        rgb: np.ndarray,
    ) -> Optional[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Run Mask R-CNN once and return `(h, w, person_idx, scores, boxes, masks)`."""
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.size == 0:
            return None
        h, w = arr.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(device=self.device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1) / 255.0
        with torch.inference_mode():
            pred = self.model([tensor])[0]

        labels = pred["labels"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        boxes = pred["boxes"].detach().cpu().numpy()
        masks = pred["masks"].detach().cpu().numpy()

        person_idx = np.where((labels == 1) & (scores >= self.score_thresh))[0]
        if len(person_idx) == 0:
            person_idx = np.where((labels == 1) & (scores >= 0.25))[0]
        if len(person_idx) == 0:
            return None
        return h, w, person_idx, scores, boxes, masks

    def _select_mask_for_box(
        self,
        h: int,
        w: int,
        person_idx: np.ndarray,
        scores: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        prior_core: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select the best matching (H, W) bool mask among Mask R-CNN person detections."""
        prior_bool: Optional[np.ndarray] = None
        if prior_core is not None:
            p_arr = np.asarray(prior_core, dtype=bool)
            if p_arr.shape == (h, w) and np.any(p_arr):
                prior_bool = p_arr

        has_tight_bbox = False
        bx0 = by0 = bx1 = by1 = b_area = bcx = bcy = 0.0
        if bbox is not None and len(bbox) >= 4:
            bx0, by0, bx1, by1 = [float(v) for v in bbox[:4]]
            b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
            bcx = 0.5 * (bx0 + bx1)
            bcy = 0.5 * (by0 + by1)
            has_tight_bbox = b_area < 0.70 * float(max(1, h * w))

        def _get_inst_mask(idx_val: int) -> np.ndarray:
            m_prob = masks[idx_val, 0] if masks.ndim == 4 else masks[idx_val]
            if m_prob.shape != (h, w):
                m_prob = cv2.resize(m_prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            return np.asarray(m_prob > 0.5, dtype=bool)

        best_i = int(person_idx[0])
        best_score = -float("inf")
        for idx in person_idx:
            px0, py0, px1, py1 = [float(v) for v in boxes[idx]]
            p_area = max(1.0, (px1 - px0) * (py1 - py0))
            cx = 0.5 * (px0 + px1)
            cy = 0.5 * (py0 + py1)
            center_dist_norm = float(
                np.hypot((cx - 0.5 * float(w)) / max(1.0, float(w)), (cy - 0.5 * float(h)) / max(1.0, float(h)))
            )
            sc = float(scores[idx])
            central_score = sc * (p_area ** 0.35) * float(np.exp(-2.5 * center_dist_norm))

            if prior_bool is not None:
                m_i = _get_inst_mask(int(idx))
                inter_p = float(np.sum(m_i & prior_bool))
                union_p = float(max(1, np.sum(m_i | prior_bool)))
                iou_p = inter_p / union_p
                if iou_p > 0.02:
                    inst_score = iou_p * 100.0 + central_score * 1e-3
                else:
                    inst_score = central_score * 1e-4
            elif has_tight_bbox:
                ix = max(0.0, min(bx1, px1) - max(bx0, px0))
                iy = max(0.0, min(by1, py1) - max(by0, py0))
                inter = ix * iy
                iou = inter / max(1.0, b_area + p_area - inter)
                b_dist_norm = float(
                    np.hypot((cx - bcx) / max(1.0, float(w)), (cy - bcy) / max(1.0, float(h)))
                )
                if iou > 0.05:
                    inst_score = (0.75 * iou + 0.25 * sc) * float(np.exp(-1.5 * b_dist_norm)) * 100.0
                else:
                    inst_score = central_score * 1e-3
            else:
                inst_score = central_score

            if inst_score > best_score:
                best_score = inst_score
                best_i = int(idx)

        return _get_inst_mask(best_i)

    def segment(
        self,
        rgb: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        prior_core: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Predict an (H, W) bool primary-subject person mask (COCO label == 1)."""
        inst = self._predict_person_instances(rgb)
        if inst is None:
            return None
        h, w, person_idx, scores, boxes, masks = inst
        return self._select_mask_for_box(
            h, w, person_idx, scores, boxes, masks, bbox=bbox, prior_core=prior_core
        )

    def run_sam(self, img: np.ndarray, boxes: Any = None, **kwargs: Any) -> np.ndarray:
        """SAM-compatible `.run_sam(img, boxes)` returning `(N, H, W)` uint8 binary masks."""
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.size == 0:
            return np.zeros((0, 0, 0), dtype=np.uint8)
        h, w = arr.shape[:2]
        if boxes is None:
            box_list: List[Optional[np.ndarray]] = [None]
        else:
            box_arr = np.asarray(boxes, dtype=np.float32)
            if box_arr.size == 0:
                return np.zeros((0, h, w), dtype=np.uint8)
            box_list = [b for b in box_arr.reshape(-1, 4)]

        inst = self._predict_person_instances(arr)
        if inst is None:
            return np.zeros((len(box_list), h, w), dtype=np.uint8)
        _, _, person_idx, scores, det_boxes, masks = inst
        masks_out = [
            self._select_mask_for_box(h, w, person_idx, scores, det_boxes, masks, bbox=box).astype(np.uint8)
            for box in box_list
        ]
        return np.stack(masks_out, axis=0).astype(np.uint8)

    def __call__(
        self,
        rgb: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        prior_core: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        return self.segment(rgb, bbox=bbox, prior_core=prior_core)


def build_torchvision_human_segmentor(device: Optional[torch.device] = None) -> Any:
    """Load BSD-3 torchvision Mask R-CNN human segmentor as an optional fallback for SAM2."""
    verify_license_compliance()
    try:
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        model.eval().to(dev)
        return _TorchvisionHumanSegmentor(model=model, device=dev)
    except Exception as exc_v2:
        try:
            from torchvision.models.detection import (
                MaskRCNN_ResNet50_FPN_Weights,
                maskrcnn_resnet50_fpn,
            )

            dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
            model.eval().to(dev)
            return _TorchvisionHumanSegmentor(model=model, device=dev)
        except Exception as exc_v1:
            warnings.warn(
                f"Torchvision Mask R-CNN fallback unavailable ({exc_v2}; {exc_v1}); using MHR-prior GrabCut.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None


def save_mhr_head_assets(model: Any, output_path: str) -> None:
    """Save lightweight MHR head buffers so Step 2 can run without loading ViTDet."""
    head = model.head_pose
    torch.save(
        {
            "scale_mean": head.scale_mean.detach().cpu(),
            "scale_comps": head.scale_comps.detach().cpu(),
            "faces": head.faces.detach().cpu(),
        },
        output_path,
    )


RAM_ONLY_KEYS: Tuple[str, ...] = (
    "image_rgb",
    "head_crop_rgb",
    "moge2_depth",
    "face_landmarks_ram",
    "segmented_view",
)


def prepare_image_rgb(image_path: str, max_side: int = 1600) -> np.ndarray:
    """Load an image safely (handling EXIF orientation and oversized photos)."""
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return np.asarray(im)
    except Exception as exc:
        bgr = cv2.imread(image_path)
        if bgr is not None:
            h, w = bgr.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise ValueError(
            f"Cannot open image '{image_path}'. If sent via Zalo/Messenger, it may be HEIC disguised as .jpg."
        ) from exc


def _select_best_person(
    outputs: List[Dict[str, Any]],
    img_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Select the primary detected person scored by area, score, and image-center proximity."""
    if len(outputs) <= 1 or img_shape is None:
        return max(
            outputs,
            key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]),
        )
    h, w = float(max(1, img_shape[0])), float(max(1, img_shape[1]))

    def _score_out(o: Dict[str, Any]) -> float:
        bx0, by0, bx1, by1 = [float(v) for v in o["bbox"][:4]]
        area = max(1.0, (bx1 - bx0) * (by1 - by0))
        cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        cdist = float(np.hypot((cx - 0.5 * w) / w, (cy - 0.5 * h) / h))
        sc = float(o.get("score", o.get("bbox_score", 1.0)))
        return sc * (area ** 0.35) * float(np.exp(-2.5 * cdist))

    return max(outputs, key=_score_out)


def _extract_head_crop(
    image_path: str,
    inf_rgb: np.ndarray,
    best_output: Dict[str, Any],
    inf_max_side: int,
    head_max_side: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop the high-resolution head/hair region using bbox and optional 2D keypoints."""
    h_inf, w_inf = inf_rgb.shape[:2]
    bbox = best_output.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        bx0, by0, bx1, by1 = [float(v) for v in bbox[:4]]
    else:
        bx0, by0, bx1, by1 = 0.0, 0.0, float(w_inf), float(h_inf)

    bw = max(1.0, bx1 - bx0)
    bh = max(1.0, by1 - by0)
    hx0 = max(0.0, bx0 + 0.10 * bw)
    hx1 = min(float(w_inf), bx1 - 0.10 * bw)
    hy0 = max(0.0, by0 - 0.05 * bh)
    hy1 = min(float(h_inf), by0 + 0.28 * bh)

    k2d_raw = best_output.get("pred_keypoints_2d")
    if k2d_raw is not None:
        k2d = np.asarray(k2d_raw, dtype=np.float32)
        if k2d.ndim == 2 and k2d.shape[1] >= 2:
            sh_indices = [75, 39] if len(k2d) >= 127 else ([5, 6] if len(k2d) >= 7 else [])
            sh_pts = [
                k2d[idx, :2]
                for idx in sh_indices
                if np.isfinite(k2d[idx, :2]).all() and k2d[idx, 1] > 0.0
            ]
            if sh_pts:
                sh_y = float(np.mean([p[1] for p in sh_pts]))
                hy1 = min(float(h_inf), max(by0 + 0.18 * bh, sh_y + 0.04 * bh))

    if hx1 <= hx0 + 4.0:
        hx0, hx1 = max(0.0, bx0), min(float(w_inf), bx1)
    if hy1 <= hy0 + 4.0:
        hy0, hy1 = max(0.0, by0), min(float(h_inf), by0 + max(8.0, 0.25 * bh))

    crop_bbox = (
        int(round(hx0)),
        int(round(hy0)),
        max(int(round(hx0)) + 1, int(round(hx1))),
        max(int(round(hy0)) + 1, int(round(hy1))),
    )

    if head_max_side > max(h_inf, w_inf) and max(h_inf, w_inf) >= inf_max_side - 2:
        hi_rgb = prepare_image_rgb(image_path, max_side=head_max_side)
    else:
        hi_rgb = inf_rgb

    h_hi, w_hi = hi_rgb.shape[:2]
    sx = w_hi / float(max(w_inf, 1))
    sy = h_hi / float(max(h_inf, 1))
    ix0 = max(0, min(w_hi - 1, int(np.floor(crop_bbox[0] * sx))))
    iy0 = max(0, min(h_hi - 1, int(np.floor(crop_bbox[1] * sy))))
    ix1 = max(ix0 + 1, min(w_hi, int(np.ceil(crop_bbox[2] * sx))))
    iy1 = max(iy0 + 1, min(h_hi, int(np.ceil(crop_bbox[3] * sy))))
    return hi_rgb[iy0:iy1, ix0:ix1].copy(), crop_bbox


def _to_float32_array(val: Any) -> np.ndarray:
    """Convert a Tensor or array-like depth buffer into a squeezed float32 NumPy array."""
    arr = val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)
    return np.asarray(arr, dtype=np.float32).squeeze()


def _rgb_cache_fingerprint(rgb: np.ndarray) -> Tuple[Any, ...]:
    """Compute a fast lightweight fingerprint to guard against Python id() reuse after GC."""
    arr = np.asarray(rgb)
    flat = np.ascontiguousarray(arr).reshape(-1)
    step = max(1, flat.size // 256)
    return (arr.shape, str(arr.dtype), flat[::step][:256].tobytes())


def _cache_moge2_depth(fov_est: Any, rgb: np.ndarray, depth_arr: np.ndarray) -> np.ndarray:
    """Store depth map in bounded LRU keyed by (id(fov_est), id(rgb))."""
    if fov_est is not None:
        key = (id(fov_est), id(rgb))
        fov_ref: Any = None
        try:
            def _evict_dead_fov(
                dead_ref: Any,
                cache_key: Tuple[int, int] = key,
                cache: Any = _MOGE2_DEPTH_CACHE,
            ) -> None:
                cur = cache.get(cache_key)
                if cur is not None and cur[0] is dead_ref:
                    cache.pop(cache_key, None)

            fov_ref = weakref.ref(fov_est, _evict_dead_fov)
        except TypeError:
            fov_ref = None
        _MOGE2_DEPTH_CACHE[key] = (fov_ref, _rgb_cache_fingerprint(rgb), depth_arr)
        _MOGE2_DEPTH_CACHE.move_to_end(key)
        while len(_MOGE2_DEPTH_CACHE) > _MOGE2_CACHE_MAX_ENTRIES:
            _MOGE2_DEPTH_CACHE.popitem(last=False)
    return depth_arr


def _get_cached_moge2_depth(fov_est: Any, rgb: np.ndarray) -> Optional[np.ndarray]:
    """Retrieve cached depth map for (id(fov_est), id(rgb)) when identity and fingerprint match."""
    if fov_est is None:
        return None
    key = (id(fov_est), id(rgb))
    entry = _MOGE2_DEPTH_CACHE.get(key)
    if entry is None:
        return None
    fov_ref, fp, depth_arr = entry
    if fov_ref is not None and fov_ref() is not fov_est:
        _MOGE2_DEPTH_CACHE.pop(key, None)
        return None
    if fp != _rgb_cache_fingerprint(rgb):
        _MOGE2_DEPTH_CACHE.pop(key, None)
        return None
    _MOGE2_DEPTH_CACHE.move_to_end(key)
    return depth_arr


def _extract_moge2_depth(
    estimator: Any,
    rgb: np.ndarray,
    best_output: Dict[str, Any],
    precomputed_depth: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """Extract optional MoGe2 depth map as float32 (H, W) with single-pass reuse and caching."""
    for key in ("moge2_depth", "depth_map", "depth"):
        raw = best_output.get(key)
        if raw is not None:
            arr = _to_float32_array(raw)
            if arr.ndim >= 2:
                return arr

    fov_est = getattr(estimator, "fov_estimator", None)
    if precomputed_depth is not None:
        arr = _to_float32_array(precomputed_depth)
        if arr.ndim >= 2:
            return _cache_moge2_depth(fov_est, rgb, arr)

    if fov_est is None:
        return None

    cached = _get_cached_moge2_depth(fov_est, rgb)
    if cached is not None:
        return cached

    for method_name in ("infer_depth", "predict_depth", "extract_depth"):
        fn = getattr(fov_est, method_name, None)
        if callable(fn):
            try:
                depth = fn(rgb)
                if depth is not None:
                    arr = _to_float32_array(depth)
                    if arr.ndim >= 2:
                        return _cache_moge2_depth(fov_est, rgb, arr)
            except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as exc:
                warnings.warn(
                    f"MoGe2 depth extraction via {method_name} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    inner = getattr(fov_est, "fov_estimator", None)
    infer_fn = getattr(inner, "infer", None)
    if callable(infer_fn):
        try:
            param = next(iter(inner.parameters()), None) if hasattr(inner, "parameters") else None
            dev = param.device if param is not None else torch.device("cpu")
            img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                pred = infer_fn(img_t)
            if isinstance(pred, dict) and pred.get("depth") is not None:
                arr = _to_float32_array(pred["depth"])
                if arr.ndim >= 2:
                    return _cache_moge2_depth(fov_est, rgb, arr)
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            warnings.warn(
                f"MoGe2 depth extraction via fov_estimator.infer failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return None


def _call_process_one_image(estimator: Any, rgb: np.ndarray, bbox_thr: float) -> Any:
    """Invoke estimator.process_one_image with use_mask when human_segmentor is present."""
    fn = estimator.process_one_image
    want_mask = bool(getattr(estimator, "human_segmentor", None) is not None)
    accepts_use_mask = True
    accepts_bbox_thr = True
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        accepts_use_mask = has_varkw or ("use_mask" in params)
        accepts_bbox_thr = has_varkw or ("bbox_thr" in params)
    except (ValueError, TypeError):
        pass

    def _invoke(use_mask_flag: Optional[bool]) -> Any:
        kwargs: Dict[str, Any] = {}
        if accepts_bbox_thr:
            kwargs["bbox_thr"] = bbox_thr
        if use_mask_flag is not None and accepts_use_mask:
            kwargs["use_mask"] = use_mask_flag
        return fn(rgb, **kwargs)

    use_mask_arg: Optional[bool] = want_mask if accepts_use_mask else None
    try:
        return _invoke(use_mask_arg)
    except TypeError as exc:
        msg = str(exc)
        if use_mask_arg is not None and ("use_mask" in msg or "unexpected keyword" in msg):
            accepts_use_mask = False
            return _invoke(None)
        if want_mask and use_mask_arg is True:
            warnings.warn(
                f"process_one_image(use_mask=True) failed ({exc}); retrying with use_mask=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _invoke(False)
        raise
    except (RuntimeError, ValueError, IndexError, KeyError, AttributeError) as exc:
        if want_mask and accepts_use_mask:
            warnings.warn(
                f"process_one_image(use_mask=True) failed ({exc}); retrying with use_mask=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _invoke(False)
        raise


def _run_process_one_image_capturing_depth(
    estimator: Any,
    rgb: np.ndarray,
    bbox_thr: float,
    capture_depth: bool,
) -> Tuple[Any, Optional[Any]]:
    """Run process_one_image while capturing MoGe2 depth from any single FOV pass."""
    if not capture_depth:
        return _call_process_one_image(estimator, rgb, bbox_thr=bbox_thr), None

    fov_est = getattr(estimator, "fov_estimator", None)
    inner = getattr(fov_est, "fov_estimator", None) if fov_est is not None else None
    orig_infer = getattr(inner, "infer", None) if inner is not None else None
    if callable(orig_infer) and hasattr(inner, "__dict__"):
        captured: Dict[str, Any] = {"depth": None}
        had_instance_attr = "infer" in inner.__dict__

        def _wrapped_infer(*args: Any, **kwargs: Any) -> Any:
            pred = orig_infer(*args, **kwargs)
            if isinstance(pred, dict) and pred.get("depth") is not None:
                captured["depth"] = pred["depth"]
            return pred

        inner.infer = _wrapped_infer
        try:
            outputs = _call_process_one_image(estimator, rgb, bbox_thr=bbox_thr)
        finally:
            if had_instance_attr:
                inner.__dict__["infer"] = orig_infer
            else:
                inner.__dict__.pop("infer", None)
        return outputs, captured["depth"]

    outputs = _call_process_one_image(estimator, rgb, bbox_thr=bbox_thr)
    last_depth = (
        getattr(fov_est, "_last_depth", None) or getattr(fov_est, "last_depth", None)
        if fov_est is not None
        else None
    )
    return outputs, last_depth


def run_inference_on_images(
    estimator: Any,
    image_paths: List[str],
    bbox_thr: float = 0.5,
    max_side: int = 1600,
    keep_rgb: bool = False,
    keep_head_crop: bool = False,
    head_max_side: int = 3200,
    extract_depth: bool = False,
) -> List[Dict[str, Any]]:
    """Run SAM 3D Body inference on each view and return the primary detected person."""
    results: List[Dict[str, Any]] = []
    for path in sorted(image_paths):
        rgb = prepare_image_rgb(path, max_side=max_side)
        outputs, captured_depth = _run_process_one_image_capturing_depth(
            estimator, rgb, bbox_thr=bbox_thr, capture_depth=extract_depth
        )
        if not outputs:
            continue
        best = _select_best_person(outputs, img_shape=rgb.shape[:2])
        entry: Dict[str, Any] = {
            "image_path": path,
            "image_name": os.path.basename(path),
            "output": best,
        }
        if keep_rgb:
            entry["image_rgb"] = rgb
        if keep_head_crop:
            head_crop, head_bbox = _extract_head_crop(
                image_path=path,
                inf_rgb=rgb,
                best_output=best,
                inf_max_side=max_side,
                head_max_side=head_max_side,
            )
            entry["head_crop_rgb"] = head_crop
            entry["head_crop_bbox"] = head_bbox
        if extract_depth:
            depth_map = _extract_moge2_depth(estimator, rgb, best, precomputed_depth=captured_depth)
            if depth_map is not None:
                entry["moge2_depth"] = depth_map
        results.append(entry)
    return results


def run_inference_on_face_image(
    estimator: Any,
    face_image_path: str,
    bbox_thr: float = 0.4,
    max_side: int = 1600,
    keep_rgb: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run SAM 3D Body on an optional ~1m closeup front portrait for crisp head texturing."""
    if not face_image_path:
        return None
    rgb = prepare_image_rgb(face_image_path, max_side=max_side)
    outputs = _call_process_one_image(estimator, rgb, bbox_thr=bbox_thr)
    if not outputs:
        return None
    best = _select_best_person(outputs, img_shape=rgb.shape[:2])
    entry: Dict[str, Any] = {
        "image_path": face_image_path,
        "image_name": os.path.basename(face_image_path),
        "output": best,
        "is_closeup_face": True,
    }
    if keep_rgb:
        entry["image_rgb"] = rgb
    return entry


def strip_raw_images_for_cache(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return shallow copies without RAM-only photos, head crops, depth, landmarks, or SegmentedViews."""
    ram_set = set(RAM_ONLY_KEYS)
    cleaned: List[Dict[str, Any]] = []
    for rec in raw_results:
        item = {k: v for k, v in rec.items() if k not in ram_set}
        out_val = item.get("output")
        if isinstance(out_val, dict) and any(k in out_val for k in ram_set):
            item["output"] = {k: v for k, v in out_val.items() if k not in ram_set}
        cleaned.append(item)
    return cleaned
