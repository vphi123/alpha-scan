"""Configuration constants and anatomical landmarks for SAM 3D Body + MHR measurements."""

from dataclasses import dataclass
from typing import Dict, Tuple

SCHEMA_VERSION = "alpha_scan.body_scan.v2"
PIPELINE_VERSION = "vfl-bodyscan-0.1"
DEFAULT_SAM_CHECKPOINT_VERSION = "facebook/sam-3d-body-dinov3 (2026-11-13)"
DEFAULT_MHR_VERSION = "mhr_model.pt (Apache 2.0)"

# MHR 127-joint skeleton indices for arm removal (excludes clavicles J38 & J74).
ARM_JOINT_IDS: frozenset[int] = frozenset(range(39, 74)) | frozenset(range(75, 110))
ARM_WEIGHT_THRESHOLD = 0.35

# Multi-view fusion outlier rejection multiplier against median L2 distance.
OUTLIER_MEDIAN_MULTIPLIER = 2.5
MIN_OUTLIER_DIST_THRESHOLD = 1.0

# Neck axis joints in MHR 127-joint skeleton.
NECK_BASE_JOINT = 110
NECK_HEAD_JOINT = 113
HEAD_JOINT_IDS: frozenset[int] = frozenset(range(113, 127))
HEAD_SKIN_WEIGHT_THRESHOLD = 0.35

# Canonical MHR-127 and MHR-70 anatomical joint indices (verified against MHR70_TO_MHR127_JOINT_MAP & LIMB_SPECS).
# In MHR-127: 41/77 are forearm distal / wrist attachment joints (end of LIMB_SPECS forearm_r/l),
# while 42/78 are r_wrist/l_wrist hand roots; 4/20 are l_foot/r_foot ankle joints.
MHR127_L_HIP = 2
MHR127_L_KNEE = 3
MHR127_L_ANKLE = 4
MHR127_R_HIP = 18
MHR127_R_KNEE = 19
MHR127_R_ANKLE = 20
MHR127_R_SHOULDER = 39
MHR127_R_ELBOW = 40
MHR127_R_FOREARM_END = 41
MHR127_R_WRIST = 42
MHR127_L_SHOULDER = 75
MHR127_L_ELBOW = 76
MHR127_L_FOREARM_END = 77
MHR127_L_WRIST = 78
MHR127_NECK_BASE = NECK_BASE_JOINT
MHR127_NECK_HEAD = NECK_HEAD_JOINT
MHR127_R_EYE = 122
MHR127_L_EYE = 124

MHR127_WRIST_ELBOW_PAIRS: Tuple[Tuple[int, int], ...] = (
    (MHR127_R_WRIST, MHR127_R_ELBOW),
    (MHR127_L_WRIST, MHR127_L_ELBOW),
)
MHR127_FOREARM_ELBOW_PAIRS: Tuple[Tuple[int, int], ...] = (
    (MHR127_R_FOREARM_END, MHR127_R_ELBOW),
    (MHR127_L_FOREARM_END, MHR127_L_ELBOW),
)
MHR127_ANKLE_JOINT_IDS: Tuple[int, int] = (MHR127_L_ANKLE, MHR127_R_ANKLE)

MHR70_L_SHOULDER = 5
MHR70_R_SHOULDER = 6
MHR70_L_ELBOW = 7
MHR70_R_ELBOW = 8
MHR70_L_HIP = 9
MHR70_R_HIP = 10
MHR70_L_KNEE = 11
MHR70_R_KNEE = 12
MHR70_L_ANKLE = 13
MHR70_R_ANKLE = 14
MHR70_R_WRIST = 41
MHR70_L_WRIST = 62
MHR70_NECK_JOINT = 69

MHR70_TO_MHR127_JOINT_MAP: Tuple[Tuple[int, int], ...] = (
    (1, MHR127_L_EYE),
    (2, MHR127_R_EYE),
    (MHR70_L_SHOULDER, MHR127_L_SHOULDER),
    (MHR70_R_SHOULDER, MHR127_R_SHOULDER),
    (MHR70_L_ELBOW, MHR127_L_ELBOW),
    (MHR70_R_ELBOW, MHR127_R_ELBOW),
    (MHR70_L_HIP, MHR127_L_HIP),
    (MHR70_R_HIP, MHR127_R_HIP),
    (MHR70_L_KNEE, MHR127_L_KNEE),
    (MHR70_R_KNEE, MHR127_R_KNEE),
    (MHR70_L_ANKLE, MHR127_L_ANKLE),
    (MHR70_R_ANKLE, MHR127_R_ANKLE),
    (MHR70_R_WRIST, MHR127_R_WRIST),
    (MHR70_L_WRIST, MHR127_L_WRIST),
    (MHR70_NECK_JOINT, MHR127_NECK_BASE),
)

# Step 3 texture baking & privacy defaults.
DEFAULT_TEXTURE_SIZE = 2048
TEXTURE_MIN_VIEW_COS = 0.38
TEXTURE_ANGLE_POWER = 3.0
TEXTURE_DEPTH_TOLERANCE_M = 0.015
UV_GUTTER_DILATE_PX = 8
FACE_VIEW_WEIGHT_BOOST = 4.0
NECK_T_START = 0.25
NECK_T_END = 0.75
NECK_T_STEPS = 11

# Torso height scan fractions relative to total scaled height H (meters).
CHEST_FRAC_RANGE = (0.70, 0.75, 9)
WAIST_FEMALE_FRAC_RANGE = (0.56, 0.67, 15)
WAIST_MALE_NAVEL_FRAC = 0.585
HIP_FRAC_RANGE = (0.48, 0.55, 11)

# Shape profile loft range: 30% to 86% of total height (inclusive, step 1%).
LOFT_PCT_MIN = 30
LOFT_PCT_MAX = 86

# Elbow straightening candidate indices in body_pose_params (length 130).
ELBOW_CANDIDATES: Tuple[Tuple[str, Tuple[int, ...], int, int, int], ...] = (
    ("right", (27, 28, 29, 30, 31), 39, 40, 41),
    ("left", (37, 38, 39, 40, 41), 75, 76, 77),
)


@dataclass(frozen=True)
class LimbSpec:
    """Anatomical bone segment definition for orthogonal cross-section measurement."""

    name: str
    joint_start: int
    joint_end: int
    t_values: Tuple[float, ...]
    mode: str  # "max", "min", or "single"


LIMB_SPECS: Tuple[LimbSpec, ...] = (
    LimbSpec("bicep_l", 75, 76, (0.40, 0.45, 0.50, 0.55, 0.60), "max"),
    LimbSpec("bicep_r", 39, 40, (0.40, 0.45, 0.50, 0.55, 0.60), "max"),
    LimbSpec("forearm_l", 76, 77, (0.25, 0.30, 0.35, 0.40, 0.45), "max"),
    LimbSpec("forearm_r", 40, 41, (0.25, 0.30, 0.35, 0.40, 0.45), "max"),
    LimbSpec("wrist_l", 76, 77, (0.95,), "single"),
    LimbSpec("wrist_r", 40, 41, (0.95,), "single"),
    LimbSpec("thigh_l", 2, 3, (0.32, 0.35, 0.385, 0.42, 0.45), "max"),
    LimbSpec("thigh_r", 18, 19, (0.32, 0.35, 0.385, 0.42, 0.45), "max"),
    LimbSpec("calf_l", 3, 4, (0.25, 0.30, 0.35, 0.40, 0.45), "max"),
    LimbSpec("calf_r", 19, 20, (0.25, 0.30, 0.35, 0.40, 0.45), "max"),
    LimbSpec("ankle_l", 3, 4, (0.92,), "single"),
    LimbSpec("ankle_r", 19, 20, (0.92,), "single"),
)

# Forbidden packages that must never be imported or used due to license constraints.
FORBIDDEN_MODULES = ("ultralytics", "smplx", "nvdiffrast")

# Multi-layer reconstruction modes & anthropometric / silhouette audit thresholds.
RECONSTRUCTION_MODES: Tuple[str, ...] = ("calibrated_mhr", "hi_res_head", "dual_layer")
DEFAULT_RECONSTRUCTION_MODE: str = "calibrated_mhr"

# Studio Monochrome Clay Material Constants (Body Geometry Focus)
CLAY_STUDIO_BODY_COLOR: Tuple[float, float, float, float] = (0.52, 0.55, 0.59, 1.0)
CLAY_STUDIO_SHORTS_COLOR: Tuple[float, float, float, float] = (0.22, 0.24, 0.28, 1.0)
CLAY_STUDIO_CORE_COLOR: Tuple[float, float, float, float] = (0.50, 0.53, 0.57, 1.0)
CLAY_STUDIO_ROUGHNESS: float = 0.58
CLAY_STUDIO_METALLIC: float = 0.05
SEX_BODY_DENSITY_KG_PER_L: Dict[str, float] = {"male": 1.045, "female": 1.028}
WEIGHT_MATCH_TOL: float = 0.08
WEIGHT_MISMATCH_TOL: float = 0.12
MIN_MHR_SILHOUETTE_IOU: float = 0.78
FOREGROUND_MASK_ERODE_PX: int = 4
HEAD_TEXTURE_SIZE: int = 2048
INNER_BARRIER_M: float = 0.003
MHR_HEAD_SUBDIV_STEPS: int = 1
CARDINAL_VIEWS: Tuple[Tuple[str, float], ...] = (
    ("front", 0.0),
    ("right", 90.0),
    ("back", 180.0),
    ("left", 270.0),
)
