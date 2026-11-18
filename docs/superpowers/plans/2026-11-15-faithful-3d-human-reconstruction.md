# Alpha Scan Plus/PRO — Faithful 4-View 3D Human Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking progress. Note: `c:\Users\nvpph\Desktop\alpha scan` is not a git repository (`NOT_A_GIT_REPO`), so verification uses automated `pytest` and `node --test` checks at every task instead of git commits.

**Goal:** Build the cumulative 3-tier `Alpha Scan Plus/PRO` 3D reconstruction pipeline (`calibrated_mhr` -> `hi_res_head` -> `dual_layer`) delivering sex-specific proportions, calibrated `height_cm`/`weight_kg`, sculpted face/nose/hair geometry, clean foreground-masked textures, dual inner/outer `.glb` support, and full API/Lab integration.

**Architecture:**
- **Stage 1 Core (`segmentation.py`, `fitting.py`, `fusion.py` — Hướng 1):** Segment human silhouettes `M_v` and separate `hair_mask` vs `face_skin_mask` + `nose_profile_curve`, jointly optimize MHR parameters against hair-excluded `M_body_v = M_v \ hair_mask_v` + `height_cm` + optional `weight_kg` (`0.0` = estimate & report; `>0` = constrain & audit), slice 16 measurements strictly from this calibrated `inner_mhr_mesh`, and emit `proportion_audit`.
- **Stage 2 Head/Nose/Hair (`head_hair.py` — Hướng 3):** Subdivide MHR head vertices above `NECK_BASE_JOINT=110`, sculpt facial anatomy via RAM-only `MediaPipe` 478 landmarks + Left/Right profile silhouette curves + `MoGe2` face depth across Privacy Levels `1`, `2`, `3`, fit a collar-clear 4-view `hair_shell` above `NECK_BASE_JOINT=110`, weld seamlessly onto neck, and partition `face_blur_uv_mask` vs `hair_uv_mask`.
- **Stage 3 Dual-Layer Shell (`outer_shell.py`, `texturing.py`, `exporter.py`, `scan_service.py`, `api_server.py`, `frontend_lab/` — Hướng 2):** Reconstruct clothed/outer body surface bounded by 4-view `M_v` hull and `-3 mm` inner MHR barrier, bake eroded-mask multi-view textures without facial `TELEA` hallucination (blurring ONLY `face_blur_uv_mask` and keeping hair crisp on Level 2), export dual-material GLB + `scan_core.glb`, and surface UI mode selector, 4-cardinal gate, `proportion_audit`, and outer/core toggle in `frontend_lab/`.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch3D (GPU/Colab), `MediaPipe` (`Apache 2.0`), `SAM 2`/`SAM`, `MoGe2`, `OpenCV`, `SciPy`, `NumPy`, `Pillow`, `trimesh`, `xatlas`, `FastAPI`, `Pytest`, `Node.js v24+` (`node:test`), `Three.js r128` (`frontend_lab/`).

**Spec:** [docs/superpowers/specs/2026-11-15-faithful-3d-human-reconstruction-design.md](file:///c:/Users/nvpph/Desktop/alpha%20scan/docs/superpowers/specs/2026-11-15-faithful-3d-human-reconstruction-design.md)

## Global Constraints

- License compliance: ONLY permit `SAM 3D Body`, `MHR`, `SAM/SAM2`, `Detectron2`, `MoGe2`, `MediaPipe` (`Apache 2.0`), `PyTorch3D`, `trimesh`, `xatlas`, `OpenCV`, `SciPy`, `NumPy`, `Pillow` — NEVER import or require `ultralytics`, `smplx`, `FLAME`, `nvdiffrast`, or non-commercial weights (e.g. `briaai/RMBG`); check `verify_license_compliance()` (`FORBIDDEN_MODULES = ("ultralytics", "smplx", "nvdiffrast")`).
- Colab T4 memory & speed budget: Keep peak VRAM `< 12 GB` and total runtime `<= 90s` by releasing detector/segmentor state (`torch.cuda.empty_cache()`, `gc.collect()`) between inference, fitting, and head/shell/baking stages inside `GpuScanEngine._lock`.
- Windows CPU testability: Guard optional packages (`pytorch3d`, `mediapipe`, `trimesh`, `xatlas`, `sam_3d_body`) cleanly so every new math, geometry, masking, blur, API, and parser unit test passes 100% locally via `pytest` and `node --test` without CUDA or weights.
- `alpha_scan.body_scan.v2` compatibility & S2 landmark privacy: All new JSON keys (`reconstruction_mode`, `effective_reconstruction_mode`, `proportion_audit`, `mesh.core_file`) are additive; 478 facial landmarks remain strictly in RAM and are NEVER written to disk, JSON, `.glb`, or cache `.pt` files.
- Capture & `P0-15` privacy rules: Assume fitted athletic wear (`1 sát form`, optional `weight_kg`) and hair clear of collar (`2 gọn`); run detailed head/nose/hair geometry on Privacy Levels `1`, `2`, `3` (`3a`); blur ONLY facial skin below hairline (`face_blur_uv_mask`) while keeping hair (`hair_uv_mask`) crisp on Level 2 (`3b`); include `scan_core.glb` in Decision 3a binary deletion (`410 Gone`).
- Zero hallucination: Reject background pixels via eroded foreground masks before texture projection; NEVER run `cv2.INPAINT_TELEA` over visible facial/eye/nose/mouth UV texels; keep `Three.js r128` (`frontend_lab/`) 100% offline with zero external CDN `<script>` tags added.

## Review Focus

1. **Out-of-order / mislabeled views (`Front`, `Right`, `Back`, `Left`):** Users or raw API calls may upload the 4 angles in arbitrary order; `infer_view_azimuth` must infer canonical azimuth (`0°`, `90°`, `180°`, `270°`) from 3D keypoints/pose rather than filename sorting so side profiles never sculpt opposite ears/noses — pinned in **Task 2 (`test_infer_view_azimuth_handles_shuffled_cardinal_views`)**.
2. **Optional vs contradictory `weight_kg` (`0.0` vs `>12%` conflict):** `weight_kg == 0.0` must disable `L_vol` cleanly (`status == "estimated_only"`), while contradictory declared weights (`>12%` off photo silhouette volume) must NEVER warp the avatar distortedly; `L_sil` dominates and `proportion_audit` emits `mismatch_warning` — pinned in **Task 3 (`test_fit_mhr_weight_zero_and_contradictory_mass_audit`)**.
3. **Loose hair collar overlap & Level 2 hairline blur bleed (`2 gọn` + `3b giữ rõ`):** Hair touching neck/shoulders below `NECK_BASE_JOINT=110` must trigger a pre-flight warning without distorting body girths, and Level 2 Gaussian/mosaic blur + gutter dilation must NEVER bleed blur onto adjacent `hair_uv_mask` texels — pinned in **Task 2 (`test_extract_head_semantics_collar_overlap_warning`)** and **Task 7 (`test_level2_blur_isolates_face_skin_and_preserves_hair`)**.
4. **Incomplete cardinal views via API & missing optional GPU/landmark backends:** Submitting `1–3` photos to `POST /api/v1/bodyscan` with `dual_layer` must gracefully downgrade `effective_reconstruction_mode` to `"calibrated_mhr"` with a clear `warnings` message, and missing `MoGe2`/`MediaPipe` on an angle must degrade cleanly with zero NaNs and zero landmarks written to disk (`S2`) — pinned in **Task 5 (`test_head_hair_graceful_fallback_and_zero_landmark_persistence`)** and **Task 8 (`test_bodyscan_api_downgrades_incomplete_views_to_calibrated_mhr`)**.
5. **UV upright alignment, multi-material glTF parsing & Decision 3a purge on `scan_core.glb`:** `MockScanEngine` UV vertical convention must match `tex.flipY = false` so face/hair/shirt render right-side-up, `parseGlbBinary` must load and dispose multi-material (`body` + `head`) GLBs cleanly, and `POST .../delete` (`purge_all=false`) must delete `scan_core.glb` returning `410 Gone` while preserving `body_scan.json` — pinned in **Task 1 (`test_mock_mannequin_uv_upright_orientation`)**, **Task 8 (`test_reconstruction_mode_validation_and_core_glb_decision_3a`)**, and **Task 9 (`test_parse_glb_multi_material_body_head_and_dispose`)**.

---

### Task 1: Shared Configuration Constants, Additive Schema (`exporter.py`), & `MockScanEngine` UV Fix (`scan_service.py`)

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\config.py:1-80`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\exporter.py:25-185`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\scan_service.py:19-315,317-751`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\requirements.txt:1-43`
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_scan_service.py`

**Interfaces:**
- Produces in `bodyscan_pipeline/config.py`:
  - `RECONSTRUCTION_MODES: Tuple[str, ...] = ("calibrated_mhr", "hi_res_head", "dual_layer")`
  - `DEFAULT_RECONSTRUCTION_MODE: str = "dual_layer"`
  - `SEX_BODY_DENSITY_KG_PER_L: Dict[str, float] = {"male": 1.045, "female": 1.028}`
  - `WEIGHT_MATCH_TOL: float = 0.08`, `WEIGHT_MISMATCH_TOL: float = 0.12`, `MIN_MHR_SILHOUETTE_IOU: float = 0.78`
  - `FOREGROUND_MASK_ERODE_PX: int = 2`, `HEAD_TEXTURE_SIZE: int = 2048`, `INNER_BARRIER_M: float = 0.003`, `MHR_HEAD_SUBDIV_STEPS: int = 1`
  - `CARDINAL_VIEWS: Tuple[Tuple[str, float], ...] = (("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0))`
- Produces in `bodyscan_pipeline/exporter.py`:
  - Extended `build_body_scan_json(..., reconstruction_mode: Optional[str] = None, effective_reconstruction_mode: Optional[str] = None, proportion_audit: Optional[Dict[str, Any]] = None, core_mesh_filename: Optional[str] = None, warnings: Optional[List[str]] = None) -> Dict[str, Any]` (strictly additive; sets `mesh["core_file"]` when `core_mesh_filename` is provided)
- Produces in `bodyscan_pipeline/scan_service.py`:
  - Extended `ScanRequest(..., reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE)`
  - Extended `ScanResult(..., reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE, effective_reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE, proportion_audit: Optional[Dict[str, Any]] = None)`
  - Multi-primitive/material GLB helper: `write_multi_material_glb(primitives: List[Dict[str, Any]], out_path: str) -> str` (supports `body` + `head` primitives with distinct `Material` + `baseColorTexture` in one binary `.glb` without external libs)

- [ ] **Step 1 (RED): Write failing tests in `tests/test_scan_service.py`**

Add tests pinning upright `MockScanEngine` UV orientation (Review Focus #5), `reconstruction_mode` validation/metadata, `scan_core.glb` generation for `dual_layer`, and `write_multi_material_glb`:

```python
def test_mock_mannequin_uv_upright_orientation(self) -> None:
    from bodyscan_pipeline.scan_service import (
        MockScanEngine,
        ScanRequest,
        _build_mock_mannequin,
        _render_mock_texture,
    )

    verts, faces, _, uvs = _build_mock_mannequin(170.0, "female", 54.0)
    crown_idx = int(np.argmax(verts[:, 1]))
    ankle_idx = int(np.argmin(verts[:, 1]))
    # OpenGL/glTF raw buffer convention with Three.js tex.flipY=False:
    # row 0 (top of JPEG, crown/head) maps to v close to 0.0; lower body maps to larger v.
    self.assertLess(float(uvs[crown_idx, 1]), 0.12)
    self.assertGreater(float(uvs[ankle_idx, 1]), 0.35)

    # Privacy Level 2 blur_face=True blurs face skin band while keeping top hair crown crisp (3b giữ rõ)
    sharp_bytes = _render_mock_texture((210, 170, 145), (45, 70, 130), blur_face=False)
    blur_bytes = _render_mock_texture((210, 170, 145), (45, 70, 130), blur_face=True)
    im_sharp = np.asarray(Image.open(io.BytesIO(sharp_bytes)).convert("RGB"))
    im_blur = np.asarray(Image.open(io.BytesIO(blur_bytes)).convert("RGB"))
    # Top hair crown rows [0:28] retain crisp strand detail (identical between L2 and L3)
    self.assertTrue(np.allclose(im_sharp[:26, :, :], im_blur[:26, :, :], atol=3))
    # Facial band below hairline [36:72] is strongly smoothed in Level 2
    self.assertGreater(float(np.abs(im_sharp[36:72].astype(np.float32) - im_blur[36:72].astype(np.float32)).mean()), 4.0)

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
        _, _, _, _, _, gltf, _, _ = _parse_glb_chunks(Path(out_glb).read_bytes())
        self.assertEqual(len(gltf["meshes"][0]["primitives"]), 2)
        self.assertEqual(len(gltf["materials"]), 2)
        self.assertEqual(len(gltf["images"]), 2)

        # MockScanEngine dual_layer produces scan_core.glb + proportion_audit
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
        self.assertEqual(res.body_json["proportion_audit"]["status"], "matched")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scan_service.py -v`
Expected: FAIL (`ImportError: cannot import name 'write_multi_material_glb'` / `TypeError: ScanRequest.__init__() got an unexpected keyword argument 'reconstruction_mode'`).

- [ ] **Step 3: Implement constants, additive schema, multi-material GLB writer, and `MockScanEngine` UV + dual-layer fixes**

1. Add `RECONSTRUCTION_MODES`, `DEFAULT_RECONSTRUCTION_MODE`, `SEX_BODY_DENSITY_KG_PER_L`, `WEIGHT_MATCH_TOL`, `WEIGHT_MISMATCH_TOL`, `MIN_MHR_SILHOUETTE_IOU`, `FOREGROUND_MASK_ERODE_PX`, `HEAD_TEXTURE_SIZE`, `INNER_BARRIER_M`, `MHR_HEAD_SUBDIV_STEPS`, `CARDINAL_VIEWS` to `bodyscan_pipeline/config.py`.
2. Add `mediapipe>=0.10.14` to `requirements.txt` (under Apache 2.0 landmark/segmentation section).
3. Update `build_body_scan_json` in `bodyscan_pipeline/exporter.py` to accept optional additive keys (`reconstruction_mode`, `effective_reconstruction_mode`, `proportion_audit`, `core_mesh_filename`, `warnings`) while leaving legacy callers unchanged.
4. Implement `write_multi_material_glb` (refactoring `write_minimal_glb` to wrap it cleanly so there is zero duplicated glTF serialization logic!), fix `_build_tube_part` / `_render_mock_texture` to use unified top-down glTF UV orientation matching Three.js `tex.flipY = False` (hair crown band on rows `0..28` stays crisp on `blur_face=True` while face patch rows `36..72` blurs for Level 2), add `hair`/`sex`-distinct proportions & face/hair mock primitives, add `reconstruction_mode` validation & `<4`-view fallback notice + `scan_core.glb` + `proportion_audit` in `MockScanEngine.process`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_scan_service.py tests/test_measurements.py tests/test_texturing.py -v`
Expected: All `test_scan_service.py` and `test_texturing.py` tests PASS (with zero skips in `test_scan_service.py`).

---

### Task 2: Foreground Segmentation, Azimuth Sorting & Ephemeral Head/Nose Semantics (`segmentation.py`, `inference.py`)

**Files:**
- Create: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\segmentation.py`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\inference.py:64-146`
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_segmentation.py`

**Interfaces:**
- Produces in `bodyscan_pipeline/segmentation.py`:
  - `SegmentedView` dataclass: `view_index: int`, `azimuth_deg: float`, `view_label: str`, `person_mask: np.ndarray`, `body_mask: np.ndarray`, `hair_mask: np.ndarray`, `face_skin_mask: np.ndarray`, `nose_profile_curve: Optional[np.ndarray]`, `face_landmarks_ram: Optional[np.ndarray]`, `collar_hair_overlap: bool`, `warnings: List[str]`
  - `segment_person_mask(rgb: np.ndarray, bbox: Optional[np.ndarray] = None, human_segmentor: Any = None, erode_px: int = 0) -> np.ndarray`
  - `infer_view_azimuth(pred_keypoints_3d: Optional[np.ndarray], pred_vertices: Optional[np.ndarray] = None, joints_lr_indices: Tuple[int, int] = (75, 39), fallback_index: int = 0, total_views: int = 4) -> float`
  - `extract_face_landmarks_ram(head_crop_rgb: np.ndarray) -> Optional[np.ndarray]`
  - `extract_head_semantics(rgb: np.ndarray, person_mask: np.ndarray, neck_y_px: Optional[float], view_label: str, face_landmarks_ram: Optional[np.ndarray] = None, head_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]`
  - `process_views_semantics(raw_results: List[Dict[str, Any]], estimator: Any = None) -> List[SegmentedView]`
  - `strip_ram_only_semantics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
- Produces in `bodyscan_pipeline/inference.py`:
  - Extended `run_inference_on_images(..., keep_head_crop: bool = False, head_max_side: int = 3200, extract_depth: bool = False)` storing RAM-only `head_crop_rgb`, `head_crop_bbox`, and optional `moge2_depth` when `extract_depth=True`
  - Extended `strip_raw_images_for_cache` purging `image_rgb`, `head_crop_rgb`, `moge2_depth`, `face_landmarks_ram`, `segmented_view`

- [ ] **Step 1 (RED): Write failing tests in `tests/test_segmentation.py`**

Cover foreground separation & erosion, shuffled view azimuth sorting (Review Focus #1), hair vs face mask extraction (`M_body_v = M_v & ~hair_mask`), Left/Right profile `nose_profile_curve` extraction, collar overlap pre-flight warning (`2 gọn` Review Focus #3), RAM-only landmark stripping (`S2`), and license check:

```python
def test_segment_person_mask_erode_and_license_guard() -> None: ...
def test_infer_view_azimuth_handles_shuffled_cardinal_views() -> None:
    # Construct canonical MHR-like left/right shoulder & nose-offset keypoints rotated by [180, 270, 0, 90] deg
    # Assert inferred azimuths classify exact labels ['back', 'left', 'front', 'right'] regardless of input order
def test_extract_head_semantics_hair_exclusion_and_nose_profile() -> None:
    # Synthetic Right-profile binary silhouette with distinct prominent nose bump + upper bun hair region above neck_y_px
    # Assert body_mask excludes top hair bun, nose_profile_curve captures peak x projection > forehead baseline, collar_hair_overlap is False
def test_extract_head_semantics_collar_overlap_warning() -> None:
    # Synthetic hair mask extending well below neck_y_px into shoulders
    # Assert collar_hair_overlap is True and pre-flight warning mentions collar/hair-up protocol
def test_strip_raw_images_removes_head_crops_and_landmarks() -> None:
    # Assert stripped dict has no 'image_rgb', 'head_crop_rgb', 'moge2_depth', 'face_landmarks_ram'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'bodyscan_pipeline.segmentation'`).

- [ ] **Step 3: Implement `bodyscan_pipeline/segmentation.py` & extend `bodyscan_pipeline/inference.py`**

1. Build `segment_person_mask`: verify license (`verify_license_compliance()`), call `human_segmentor` when provided (SAM 2 / SAM `Apache 2.0`), else `MediaPipe SelfieSegmentation`, else clean OpenCV GrabCut inside `bbox` (plus optional morphological closing + `erode_px` boundary erosion).
2. Implement `infer_view_azimuth`: derive torso facing vector from shoulder/hip normal (`left_shoulder` idx `75` to `right_shoulder` idx `39` cross spine up vector) in OpenCV camera frame and snap/map continuously to `[0, 360)` with label `"front" | "right" | "back" | "left" | "oblique"`.
3. Implement `extract_face_landmarks_ram`: guarded `mediapipe` FaceLandmarker/FaceMesh extraction returning `(478, 3)` float32 in normalized crop coords (returning `None` cleanly if `mediapipe` unavailable); ensure `strip_ram_only_semantics` & `strip_raw_images_for_cache` strip all photo/depth/landmark keys.
4. Implement `extract_head_semantics`: isolate head region above `neck_y_px`, separate skin pixels (landmark hull + YCrCb/HSV skin cluster) vs hair crown/bun/side-locks (`hair_mask`), compute `body_mask = person_mask & (~hair_mask)`, extract `Left`/`Right` anterior silhouette boundary across the eye-to-mouth band as `nose_profile_curve`, and detect `collar_hair_overlap` when hair fraction below `neck_y_px` exceeds threshold (`warning`).
5. Extend `inference.py` (`load_sam3d_estimator` keeps optional `human_segmentor=None` unless configured cleanly, `run_inference_on_images` optionally preserves high-res `head_crop_rgb` up to `head_max_side=3200` and optional `moge2_depth`).

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_segmentation.py -v`
Expected: PASS (5 passed, 0 skipped).

---

### Task 3: Metric Multi-View MHR Fitting (`fitting.py`), Weight/Height Optimization & `proportion_audit` (`fusion.py`)

**Files:**
- Create: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\fitting.py`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\fusion.py:18-146`
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_fitting.py`

**Interfaces:**
- Consumes: `SegmentedView` (Task 2), `FusedSubjectParams` (`fusion.py`), `SEX_BODY_DENSITY_KG_PER_L`, `WEIGHT_MATCH_TOL`, `WEIGHT_MISMATCH_TOL`, `MIN_MHR_SILHOUETTE_IOU` (`config.py`).
- Produces in `bodyscan_pipeline/fusion.py`:
  - `FusedSubjectParams(..., expr_params: Optional[torch.Tensor] = None)`
  - `evaluate_mhr_pose(..., expr_t: Optional[torch.Tensor] = None)` and `build_scaled_apose_mesh` passing `fused.expr_params` when non-None
- Produces in `bodyscan_pipeline/fitting.py`:
  - `ProportionAudit` dataclass: `declared_weight_kg: float`, `estimated_weight_kg: float`, `weight_diff_pct: Optional[float]`, `mhr_silhouette_iou: float`, `outer_silhouette_iou: Optional[float]`, `status: str`, `warnings: List[str]`, `.to_dict() -> Dict[str, Any]`
  - `compute_closed_mesh_volume_liters(verts_m: Any, faces: Any) -> Any` (works with both NumPy arrays and gradient-tracking PyTorch tensors)
  - `estimate_weight_kg(verts_m: Any, faces: Any, sex: str) -> Any`
  - `evaluate_proportion_audit(verts_a: np.ndarray, faces: np.ndarray, sex: str, declared_weight_kg: float, mhr_silhouette_iou: float, outer_silhouette_iou: Optional[float] = None) -> ProportionAudit`
  - `compute_soft_silhouette_and_edge_loss(rendered_alpha: torch.Tensor, target_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
  - `fit_mhr_multiview(mhr_model: Any, head_assets: Dict[str, torch.Tensor], fused: FusedSubjectParams, segmented_views: List[SegmentedView], raw_results: List[Dict[str, Any]], height_cm: float, weight_kg: float, sex: str, steps: int = 50, prefer_pytorch3d: bool = True) -> Tuple[FusedSubjectParams, ProportionAudit, Dict[str, Any]]`

- [ ] **Step 1 (RED): Write failing tests in `tests/test_fitting.py`**

Cover closed mesh volume accuracy (cube/ellipsoid analytical match within `0.1%`), sex-specific body density weight estimation, soft silhouette + Sobel edge differentiable optimization converging on synthetic 3D aspect ratios using pure PyTorch tensors, and `weight_kg == 0.0` vs `>12%` mismatch audit (Review Focus #2):

```python
def test_closed_mesh_volume_liters_and_sex_weight_estimation() -> None:
    # Closed unit-scaled box 0.40m x 0.25m x 1.00m = 0.100 m^3 = 100.0 liters
    # Assert volume == 100.0 +- 0.05 L and male (104.5 kg) > female (102.8 kg) for identical volume

def test_soft_silhouette_and_edge_loss_backpropagates_shape_scale() -> None:
    # Synthetic differentiable soft renderer over 2 orthogonal target masks + L_vol
    # Optimize scale (sx, sz) with Adam for 30 steps -> IoU rises > 0.90 and L_total drops monotonically

def test_fit_mhr_weight_zero_and_contradictory_mass_audit() -> None:
    # Case A: declared_weight_kg == 0.0, mhr_silhouette_iou=0.88 -> status == "estimated_only", weight_diff_pct is None
    # Case B: declared_weight_kg within 5% -> status == "matched", warnings == []
    # Case C: declared_weight_kg differs by 35% -> status == "mismatch_warning", non-empty warning
    # Case D: poor silhouette overlap mhr_silhouette_iou=0.68 -> status == "mismatch_warning" even if weight matched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fitting.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'bodyscan_pipeline.fitting'`).

- [ ] **Step 3: Implement `bodyscan_pipeline/fitting.py` & extend `bodyscan_pipeline/fusion.py`**

1. Extend `FusedSubjectParams` (`expr_params: Optional[torch.Tensor] = None`) and `evaluate_mhr_pose` in `bodyscan_pipeline/fusion.py` preserving all existing signatures/defaults.
2. Implement exact tetrahedral volume `compute_closed_mesh_volume_liters` supporting both `np.ndarray` and differentiable `torch.Tensor`, plus `estimate_weight_kg` using `SEX_BODY_DENSITY_KG_PER_L`.
3. Implement `evaluate_proportion_audit` with exact statuses (`"estimated_only"`, `"matched"` when diff `<= 0.12` and `mhr_silhouette_iou >= 0.78`, `"mismatch_warning"` otherwise with clear warning when diff `> 0.12` or `iou < 0.78` — also attaching informational note if `0.08 < diff <= 0.12`).
4. Implement `compute_soft_silhouette_and_edge_loss` (BCE + Dice IoU + 2D Sobel gradient L1 loss) in pure PyTorch.
5. Implement `fit_mhr_multiview`: register views at inferred `azimuth_deg`, optimize `shape_params`, `scale_params`, restrained `expr_params`, and per-view `(cam_t, cam_rot)` via PyTorch3D `SoftSilhouetteShader` against `body_mask` (`M_body_v`) when PyTorch3D is installed (with `L_sil > L_vol` weighting and clamped deviation norm, skipping `L_vol` if `weight_kg <= 0.0`), and fall back cleanly on CPU/no-PyTorch3D by keeping initial fused parameters while projecting mesh silhouettes with OpenCV/NumPy to compute exact `mhr_silhouette_iou` + `ProportionAudit`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_fitting.py tests/test_measurements.py -v`
Expected: `test_fitting.py` passes completely; `test_measurements.py` preserves all existing passes.

---

### Task 4: High-Resolution Head Subdivision & Face/Nose/Profile Sculpting (`head_hair.py` Part 1)

**Files:**
- Create: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\head_hair.py`
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_head_hair.py`

**Interfaces:**
- Consumes: `SegmentedView` (Task 2), `NECK_BASE_JOINT`, `NECK_HEAD_JOINT`, `MHR_HEAD_SUBDIV_STEPS` (`config.py`), `build_head_face_vertex_mask` (`texturing.py`).
- Produces in `bodyscan_pipeline/head_hair.py`:
  - `HeadPartition` dataclass: `head_vert_idx: np.ndarray`, `collar_vert_idx: np.ndarray`, `body_face_mask: np.ndarray`, `head_face_mask: np.ndarray`, `subdiv_verts: np.ndarray`, `subdiv_faces: np.ndarray`, `collar_subdiv_idx: np.ndarray`, `subdiv_collar_ring_dist: np.ndarray`, `cranium_verts: np.ndarray`
  - `partition_and_subdivide_head(verts_a: np.ndarray, faces: np.ndarray, joints_a: np.ndarray, mhr_model: Optional[Any] = None, subdiv_steps: int = MHR_HEAD_SUBDIV_STEPS) -> HeadPartition`
  - `align_face_landmarks_3d(head_verts: np.ndarray, joints_a: np.ndarray, segmented_views: List[SegmentedView], raw_results: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]` (returns `(face_delta_verts, face_skin_subdiv_mask)`, keeping all `478` landmarks ephemeral in local scope only!)
  - `fit_nose_profile_and_depth(head_verts: np.ndarray, head_faces: np.ndarray, collar_ring_dist: np.ndarray, segmented_views: List[SegmentedView], moge2_depths: Optional[List[Optional[np.ndarray]]] = None) -> np.ndarray`

- [ ] **Step 1 (RED): Write failing tests in `tests/test_head_hair.py`**

Build a synthetic manifold humanoid prism/sphere-head mesh (`verts`, `faces`, `joints`) with a clean neck ring at `NECK_BASE_JOINT=110`:
- Test `partition_and_subdivide_head`: subdivided head vertex count increases meaningfully (`> 2x`), `collar_subdiv_idx` coordinates stay bit-identical to original `verts_a[collar_vert_idx]` (zero neck crack!), and subdivided faces form a valid edge-manifold around internal edges (`subdiv_collar_ring_dist` increases cleanly upward from 0 at collar).
- Test `fit_nose_profile_and_depth`: given Right/Left profile silhouettes with a pronounced `2.5 cm` nose tip projection vs a flat nose control, sculpted nose tip `+Z` projection increases toward observed profile curve while collar ring (`collar_ring_dist == 0`) retains exact `0.0` displacement!

```python
def test_partition_and_subdivide_head_preserves_exact_collar_ring() -> None: ...
def test_fit_nose_profile_and_depth_matches_side_silhouette_with_zero_collar_drift() -> None: ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_head_hair.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'bodyscan_pipeline.head_hair'`).

- [ ] **Step 3: Implement `partition_and_subdivide_head`, `align_face_landmarks_3d`, and `fit_nose_profile_and_depth` in `bodyscan_pipeline/head_hair.py`**

1. Implement clean localized neck partition above `NECK_BASE_JOINT=110` finding boundary collar vertices shared between body triangles and head triangles.
2. Implement 1-to-4 midpoint triangle subdivision strictly inside `head_face_mask`: keep original boundary collar edges unsplit OR add transition fan triangles connecting split internal edges to fixed collar vertices so the neck collar stitch stays 100% watertight without T-junction cracks! Compute geodesic BFS edge-ring distance `subdiv_collar_ring_dist` upward from `collar_subdiv_idx`.
3. Implement `align_face_landmarks_3d` (Procrustes + damped TPS RBF displacement mapped from front/oblique 478 MediaPipe landmarks when present, strictly bounded within anterior face ROI) and `fit_nose_profile_and_depth` (enforce sagittal profile silhouette along nose bridge/tip/chin from Right `90°` & Left `270°` `nose_profile_curve`, apply metric-scaled `MoGe2` normal relief with Laplacian smoothing, and damp displacement weight to `0` across `ring_dist <= 3` collar blend rings).

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_head_hair.py -v`
Expected: PASS (all Part 1 tests green).

---

### Task 5: 4-View Hair Volume Shell, Watertight Stitch & `3a`/`3b` Semantic Partition (`head_hair.py` Part 2)

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\head_hair.py`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\tests\test_head_hair.py`

**Interfaces:**
- Consumes: `HeadPartition`, `align_face_landmarks_3d`, `fit_nose_profile_and_depth` (Task 4), `SegmentedView` (Task 2).
- Produces in `bodyscan_pipeline/head_hair.py`:
  - `HighResHeadOutput` dataclass: `full_verts: np.ndarray`, `full_faces: np.ndarray`, `head_vert_mask: np.ndarray`, `hair_vert_mask: np.ndarray`, `face_skin_vert_mask: np.ndarray`, `front_face_vert_mask: np.ndarray`, `warnings: List[str]`
  - `build_hair_volume_shell(cranium_verts: np.ndarray, cranium_faces: np.ndarray, joints_a: np.ndarray, segmented_views: List[SegmentedView], raw_results: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]`
  - `reconstruct_highres_head_and_hair(verts_a: np.ndarray, faces: np.ndarray, joints_a: np.ndarray, segmented_views: List[SegmentedView], raw_results: List[Dict[str, Any]], mhr_model: Optional[Any] = None, subdiv_steps: int = MHR_HEAD_SUBDIV_STEPS) -> HighResHeadOutput`

- [ ] **Step 1 (RED): Add hair volume & fallback tests to `tests/test_head_hair.py`**

Cover `build_hair_volume_shell` radial cranium non-penetration clamp, tall ponytail/bun `hair_mask` contour expansion above crown, seamless watertight merge with lower body (collar gap == 0.0, all triangles finite), `hair_vert_mask` vs `face_skin_vert_mask` disjointness (`3b`), and graceful fallback when `MediaPipe`/`MoGe2` missing + zero landmark attributes in `HighResHeadOutput` (Review Focus #4):

```python
def test_hair_shell_captures_bun_contour_without_skull_penetration() -> None:
    # Assert radial distance from skull centroid for every hair shell vertex >= cranium reference radius - 1e-5
    # Assert top/back vertices expand outward toward tall back/side hair_mask envelope

def test_reconstruct_highres_head_and_hair_stitch_and_disjoint_skin_vs_hair_masks() -> None:
    # Assert len(full_verts) > len(verts_a), finite coords, body vertices below collar remain bit-exact to inner verts_a
    # Assert np.any(hair_vert_mask) and np.any(face_skin_vert_mask) and not np.any(hair_vert_mask & face_skin_vert_mask)

def test_head_hair_graceful_fallback_and_zero_landmark_persistence() -> None:
    # Pass segmented_views with face_landmarks_ram=None and moge2_depth=None -> finishes with zero NaNs and warnings list
    # Assert no dataclass field on HighResHeadOutput stores landmark coordinates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_head_hair.py -v`
Expected: FAIL (`ImportError: cannot import name 'reconstruct_highres_head_and_hair'`).

- [ ] **Step 3: Implement `build_hair_volume_shell` & `reconstruct_highres_head_and_hair` in `bodyscan_pipeline/head_hair.py`**

1. Extract scalp/crown patch above ear/forehead hairline plane (strictly above `NECK_BASE_JOINT=110`, `2 gọn`), build closed thickness cap by keeping cranium scalp ring sealed along hairline rim.
2. Project candidate shell rays radially from skull center across available cardinal `hair_mask` contours (expanding outward to match observed bangs, crown height, side bulk, bun silhouette with smooth Laplacian regularization across `45°` intermediate angles), clamping every vertex radius `>= cranium_radius`.
3. Assemble unified `full_verts`, `full_faces` (`body` + welded `sculpted_head` + `hair_shell`), tag disjoint `hair_vert_mask` vs `face_skin_vert_mask` (strictly facial skin below hairline — excludes scalp/hair so Level 2 blur `3b` leaves hair sharp!), and guarantee zero finite/NaN defects when landmarks/depths are absent.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_head_hair.py -v`
Expected: PASS (all tests green).

---

### Task 6: Dual-Layer Clothed/Outer Body Shell Reconstruction (`outer_shell.py`)

**Files:**
- Create: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\outer_shell.py`
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_outer_shell.py`

**Interfaces:**
- Consumes: `HighResHeadOutput` (Task 5), `SegmentedView` (Task 2), `INNER_BARRIER_M` (`config.py`).
- Produces in `bodyscan_pipeline/outer_shell.py`:
  - `OuterShellOutput` dataclass: `outer_verts: np.ndarray`, `outer_faces: np.ndarray`, `inner_mhr_verts: np.ndarray`, `inner_mhr_faces: np.ndarray`, `head_vert_mask: np.ndarray`, `hair_vert_mask: np.ndarray`, `face_skin_vert_mask: np.ndarray`, `front_face_vert_mask: np.ndarray`, `outer_silhouette_iou: float`, `warnings: List[str]`
  - `align_metric_depth_map(raw_depth: np.ndarray, mhr_rendered_depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray`
  - `build_dual_layer_outer_shell(inner_verts_a: np.ndarray, inner_faces: np.ndarray, joints_a: np.ndarray, highres_head: HighResHeadOutput, segmented_views: List[SegmentedView], raw_results: List[Dict[str, Any]], prefer_pytorch3d: bool = True) -> OuterShellOutput`

- [ ] **Step 1 (RED): Write failing tests in `tests/test_outer_shell.py`**

Verify metric depth affine alignment `(scale * raw + shift)` recovers exact ground-truth meters within `< 1 mm`, `-3 mm` inner MHR non-penetration barrier holds strictly on torso vertices, 4-view `person_mask` hull cap prevents ballooning beyond observed silhouette, `inner_mhr_verts` remains unmodified for 16 measurements, and `outer_silhouette_iou` is computed cleanly:

```python
def test_align_metric_depth_map_recovers_exact_scale_and_shift() -> None:
    # Synthesize raw_depth = 0.4 * true_depth - 0.15 with noise on valid_mask
    # Assert MAE(aligned, true_depth) < 0.002 m

def test_build_dual_layer_outer_shell_respects_inner_barrier_and_hull_cap() -> None:
    # Synthetic inner torso cylinder/mannequin + wider clothed person_mask silhouette & MoGe2 depth ridge (+1.5cm fold)
    # Assert outer torso vertices stay >= inner surface - INNER_BARRIER_M - 1e-4
    # Assert projected outer torso silhouettes stay inside dilated person_mask hull and outer_silhouette_iou > 0.80
    # Assert head/hair vertices mirror highres_head cleanly and inner_mhr_verts equals input inner_verts_a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_outer_shell.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'bodyscan_pipeline.outer_shell'`).

- [ ] **Step 3: Implement `bodyscan_pipeline/outer_shell.py`**

1. Implement `align_metric_depth_map`: robust median/Huber weighted linear fit `(a > 0, b)` between `MoGe2` depth and rendered MHR depth on interior foreground pixels.
2. Implement `build_dual_layer_outer_shell`: retain `highres_head` above collar cleanly, isolate torso/limb surface below `NECK_BASE_JOINT=110`, optimize normal displacement field `(d_i)` toward aligned depth + 4-view `person_mask` boundaries with `45°` biharmonic/Laplacian regularization (`0` displacement at collar boundary ring), enforce `d_i >= -INNER_BARRIER_M` and 4-view silhouette hull containment, and compute `outer_silhouette_iou` against full `M_v`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_outer_shell.py -v`
Expected: PASS (all tests green).

---

### Task 7: Masked Anti-Hallucination Texturing, Dual-Material UV & Level 2 Skin-Only Blur (`texturing.py`, `exporter.py`)

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\texturing.py:36-91,447-774`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\exporter.py:91-180`
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_texturing.py`

**Interfaces:**
- Consumes: `FOREGROUND_MASK_ERODE_PX`, `HEAD_TEXTURE_SIZE` (`config.py`), `write_multi_material_glb` (`scan_service.py` — imported lazily in `exporter.py` to avoid circular import).
- Produces in `bodyscan_pipeline/texturing.py`:
  - Extended `ViewObservation`: optional `foreground_mask: Optional[np.ndarray] = None`, `hair_mask: Optional[np.ndarray] = None`, `face_skin_mask: Optional[np.ndarray] = None`, `mask_erode_px: int = FOREGROUND_MASK_ERODE_PX`
  - Extended `TexturedScanOutputs`: optional `face_blur_uv_mask: Optional[np.ndarray] = None`, `hair_uv_mask: Optional[np.ndarray] = None`, `head_uv_layout: Optional[UVLayout] = None`, `head_tex_face_rgb: Optional[np.ndarray] = None`, `head_tex_blur_rgb: Optional[np.ndarray] = None`, `coverage_pct: float = 0.0`, `head_coverage_pct: float = 0.0`, `head_coverage_mask: Optional[np.ndarray] = None`
  - Updated `bake_multiview_texture`: rejects any projected sample `(u_img, v_img)` outside `erode(obs.foreground_mask, mask_erode_px) >= 0.5`
  - Updated `inpaint_and_dilate_texture(..., strict_no_invent: bool = True, no_inpaint_mask: Optional[np.ndarray] = None)`: NEVER calls `cv2.INPAINT_TELEA` inside `no_inpaint_mask` (`face_blur_uv_mask`/`front_face_uv_mask`); dilates small cracks via `_dilate_uv_gutters` and fills deep non-face crevices with observed local median RGB
  - Updated `generate_blurred_face_texture(..., blur_target_mask: Optional[np.ndarray] = None, preserve_sharp_mask: Optional[np.ndarray] = None)`: when `blur_target_mask` (`face_blur_uv_mask`) is supplied, blurs ONLY `blur_target_mask` while restoring/preserving `preserve_sharp_mask` (`hair_uv_mask`) crisp across UV borders (`3b`)! Keeps backwards compatibility when omitted (`head_uv_mask`).
  - Dual-material UV support in `run_step3_texturing(..., head_vert_mask: Optional[np.ndarray] = None, hair_vert_mask: Optional[np.ndarray] = None, face_skin_vert_mask: Optional[np.ndarray] = None, dual_material: bool = True)`: produces dedicated `2048×2048` head UV + textures alongside body UV + textures when partitioned head/hair masks exist.
- Produces in `bodyscan_pipeline/exporter.py`:
  - `build_texture_metadata`: includes `coverage_pct`, `head_coverage_pct`, `materials: ["body", "head"]` when `head_uv_layout` exists.
  - `export_textured_scan_artifacts`: supports dual-material GLB output cleanly (both when `trimesh` exports `Scene`/multi-primitive GLB OR delegating cleanly to `write_multi_material_glb`) + `core_mesh_filename` (`scan_core.glb`) export option!

- [ ] **Step 1 (RED): Add masked-projection, anti-hallucination, dual-material, and Level 2 skin-only blur tests to `tests/test_texturing.py`**

Keep all 10 existing `tests/test_texturing.py` tests intact; add:

```python
def test_foreground_mask_rejects_background_wall_bleed() -> None:
    # Image with bright magenta wall [255, 0, 255] outside foreground_mask covering 30% of torso edge
    # Assert baked tex_face_rgb contains zero magenta bleed along body boundary after 2px erosion

def test_strict_no_invent_skips_telea_on_unobserved_facial_texels() -> None:
    # Mark interior facial strip unobserved (accum_w == 0) + surrounding red/blue markers
    # With strict_no_invent=True and no_inpaint_mask on face, unseen gap does not invoke TELEA synthesis across eyes/nose

def test_level2_blur_isolates_face_skin_and_preserves_hair() -> None:
    # High-frequency alternating dark/light hair stripes in hair_uv_mask bordering facial skin in face_blur_uv_mask
    # Call generate_blurred_face_texture(tex, head_uv_mask, tri_id_map, blur_target_mask=face_uv, preserve_sharp_mask=hair_uv)
    # Assert face_uv variance drops strongly while hair_uv remains bit-identical to unblurred tex_face_rgb!

def test_dual_material_head_body_texturing_outputs_coverage_stats() -> None:
    # Pass partitioned head_vert_mask, hair_vert_mask, face_skin_vert_mask -> populates head_uv_layout, head_tex_face_rgb, coverage_pct, head_coverage_pct
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_texturing.py -v`
Expected: FAIL on the 4 new tests while existing 10 pass.

- [ ] **Step 3: Implement masked baking, strict anti-hallucination, skin-only Level 2 blur, dual-material UV & export in `texturing.py` & `exporter.py`**

1. Erode `obs.foreground_mask` (`cv2.erode` kernel `mask_erode_px`) once per view in `bake_multiview_texture` and mask `visible &= fg_sampled >= 0.5`.
2. Track `coverage_pct` and `head_coverage_pct` in `run_step3_texturing`.
3. Enforce `strict_no_invent` in `inpaint_and_dilate_texture` (skip `TELEA` on `no_inpaint_mask`; only gutter dilate + local observed median fill for non-face deep cavities).
4. Isolate `face_blur_uv_mask` vs `hair_uv_mask` in `_build_uv_head_masks` and `generate_blurred_face_texture`, preventing blur feather or UV gutter bleed into `hair_uv_mask`.
5. Partition `head_uv_layout` + `body_uv_layout` when `dual_material=True` and `head_vert_mask` separates clean triangles; update `exporter.export_textured_scan_artifacts` to emit valid multi-material GLBs via `write_multi_material_glb` (so even without `trimesh`, dual-material GLB export works deterministically!) plus optional `scan_core.glb` export!

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_texturing.py -v`
Expected: All tests PASS (14 passed, 0 skipped).

---

### Task 8: End-to-End Orchestration (`scan_service.py`, `run_scan.py`, `demo_ui.py`) & FastAPI Server Extensions (`api_server.py`)

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\scan_service.py:619-952`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\api_server.py:37-76,98-430,532-648`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\run_scan.py:44-205`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\demo_ui.py:42-166`
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_scan_service.py`, `c:\Users\nvpph\Desktop\alpha scan\tests\test_api_server.py`

**Interfaces:**
- Consumes: Task 1–7 modules (`segmentation`, `fitting`, `head_hair`, `outer_shell`, `texturing`, `exporter`).
- Produces in `bodyscan_pipeline/scan_service.py`:
  - `GpuScanEngine.process` executing sequential memory-safe pipeline: `inference + segmentation -> release detector VRAM -> fitting (inner_mhr_mesh & proportion_audit) -> measurements -> (if effective_mode in ("hi_res_head", "dual_layer"): head_hair) -> (if effective_mode == "dual_layer": outer_shell & scan_core.glb) -> (if privacy_level >= 2: texturing) -> export & finally shred raw images + clear landmarks`.
  - `_resolve_effective_mode(requested_mode: str, unique_cardinals: int) -> Tuple[str, List[str]]`: shared helper downgrading `hi_res_head`/`dual_layer` -> `"calibrated_mhr"` with a clear `warnings` message whenever fewer than `4` unique cardinal angles exist.
- Produces in `bodyscan_pipeline/api_server.py`:
  - `BINARY_ASSET_FILES` includes `"scan_core.glb"` (`_allowed_files_for_privacy` includes `"scan_core.glb"` for ALL Levels `1`, `2`, `3` when present)
  - `POST /api/v1/bodyscan` accepts `reconstruction_mode: str = Form(DEFAULT_RECONSTRUCTION_MODE)` (`422` on invalid mode)
  - `GET /api/v1/bodyscan/{job_id}` returns `reconstruction_mode`, `effective_reconstruction_mode`, `proportion_audit`, `warnings`, and `artifacts.core_mesh_url` (when `core_glb` produced)
  - `POST /api/v1/bodyscan/{job_id}/delete` deletes `scan_core.glb` along with other `.glb`/`.jpg`/`.png` assets (`410 Gone` afterward)

- [ ] **Step 1 (RED): Write failing tests in `tests/test_api_server.py` and `tests/test_scan_service.py`**

Add tests pinning Review Focus #4 & #5:
- `POST /api/v1/bodyscan` rejects invalid `reconstruction_mode="ultra_magic"` with `422`.
- Submitting `n_photos=2` with `reconstruction_mode="dual_layer"` returns `202` and completes cleanly with `effective_reconstruction_mode == "calibrated_mhr"` and explanatory `warnings`.
- Submitting `4` photos with `reconstruction_mode="dual_layer"` across Privacy Levels `1`, `2`, `3` exposes `artifacts["core_mesh_url"]` (`scan_core.glb`), `proportion_audit`, and zero `face_landmarks` keys anywhere in JSON; calling `POST .../delete` removes `scan_core.glb` too and subsequent `GET artifacts["core_mesh_url"]` returns `410`.
- Unit test verifying `GpuScanEngine.process` stage orchestration using a lightweight fake `SAM3DBodyEstimator` + fake `MHR` callable on CPU (`test_gpu_engine_orchestrates_3tier_pipeline_with_fake_mhr_cpu`).

```python
def test_bodyscan_api_downgrades_incomplete_views_to_calibrated_mhr(self) -> None: ...
def test_reconstruction_mode_validation_and_core_glb_decision_3a(self) -> None: ...
def test_gpu_engine_orchestrates_3tier_pipeline_with_fake_mhr_cpu(self) -> None: ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_server.py tests/test_scan_service.py -v`
Expected: FAIL (`422` expected vs `202`, missing `core_mesh_url` / `proportion_audit`).

- [ ] **Step 3: Wire full 3-tier pipeline in `scan_service.py`, `api_server.py`, `run_scan.py`, and `demo_ui.py`**

1. Implement `_resolve_effective_mode` shared across `MockScanEngine` and `GpuScanEngine`.
2. Wire `GpuScanEngine.process` through `segmentation` -> `fitting` -> `measurements` -> `head_hair` -> `outer_shell` -> `texturing` -> `exporter` with strict `torch.cuda.empty_cache()` + `gc.collect()` boundaries and guaranteed `finally:` cleanup + RAM landmark deletion.
3. Update `api_server.py` (`JobRecord`, `ScanJobManager.submit`, `build_status_payload`, `create_bodyscan`, `BINARY_ASSET_FILES` including `scan_core.glb`, `_allowed_files_for_privacy` allowing `scan_core.glb` across Levels `1/2/3`).
4. Add `--reconstruction_mode` flag to `run_scan.py` and `gr.Radio` mode selector to `demo_ui.py`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_api_server.py tests/test_scan_service.py tests/test_texturing.py tests/test_segmentation.py tests/test_fitting.py tests/test_head_hair.py tests/test_outer_shell.py -v`
Expected: All tests PASS with 0 failures!

---

### Task 9: Alpha Scan Lab UI Integration (`frontend_lab/alpha-plus-pro.js`, `frontend_lab/index.html`)

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\alpha-plus-pro.js:50-105,441-515,650-730,793-1351`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\index.html:395-465` (`I18N.en` keys)
- Test: `c:\Users\nvpph\Desktop\alpha scan\tests\test_frontend_plus_pro.mjs`

**Interfaces:**
- Produces in `frontend_lab/alpha-plus-pro.js`:
  - Extended `validatePrivacyAndConnection({ ..., reconstructionMode, cardinalSlotsCount })`: requires `cardinalSlotsCount === 4` when `reconstructionMode in ['hi_res_head', 'dual_layer']` (`reasonKey: 'pp_err_4cardinal'`), allows `1–8` when `'calibrated_mhr'`.
  - `#plusModal`:
    - Capture-protocol notice banner (`pp_protocol_hint`: `"Quy chuẩn: mặc đồ tập sát form · buộc tóc gọn lộ cổ–vai"`).
    - **"Độ chi tiết dựng 3D"** selector (`dual_layer` default, `hi_res_head`, `calibrated_mhr`) appended to `FormData` as `reconstruction_mode`.
  - Viewer state + `#detail` card:
    - Shows server mode pill (`GPU thật` vs `MOCK TEST`), active `reconstruction_mode` / `effective_reconstruction_mode`.
    - Shows `proportion_audit` (`Cân nặng ước lượng`, `Độ khớp ảnh %`, `matched`/`estimated_only`/`mismatch_warning`) + `warnings` list.
    - When `artifacts.core_mesh_url` exists, shows button **"Xem lõi đo MHR / Xem lớp ngoài thật"** switching live `realPivot` cleanly between outer surface `.glb` and inner core `.glb` with full `disposeObject3D`.
- Produces in `frontend_lab/index.html`:
  - All matching EN translations in `I18N.en` for every new `pp_*` key referenced in `alpha-plus-pro.js` (so `test_frontend_plus_pro.mjs`'s automatic `I18N.en` completeness check passes).

- [ ] **Step 1 (RED): Expand `tests/test_frontend_plus_pro.mjs`**

Add assertions pinning:
- `validatePrivacyAndConnection` blocks `dual_layer`/`hi_res_head` when fewer than 4 cardinal slots are filled (`reasonKey === 'pp_err_4cardinal'`), while allowing `calibrated_mhr` with `>=1` photos.
- `parseGlbBinary` parses a multi-material `.glb` (`body` + `head`, generated via Python `write_multi_material_glb`) producing `stats.meshes === 2`, `stats.textures === 2`, `flipYs === [false, false]`, and `disposeObject3D` disposes both geometries, materials, and textures (`disposed === 2`) (Review Focus #5).
- `renderDetailCard` (tested with lightweight DOM stub or HTML output check) / `extractAuditSummary` renders `GPU thật`/`MOCK TEST` pill, `proportion_audit`, `warnings`, and inner/outer core toggle when `core_mesh_url` exists.

```javascript
test('validatePrivacyAndConnection enforces 4 cardinal slots for hi_res_head and dual_layer', () => { ... });
test('parseGlbBinary parses multi-material body+head GLB and disposes all GPU resources', async () => { ... });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/test_frontend_plus_pro.mjs`
Expected: FAIL on new assertions (`pp_err_4cardinal`, missing `I18N.en` keys / modal selector).

- [ ] **Step 3: Implement UI updates in `frontend_lab/alpha-plus-pro.js` and `I18N.en` entries in `frontend_lab/index.html`**

1. Update `validatePrivacyAndConnection`, `_refreshGuardBanner`, `_buildModalHtml`, `_bindModalEvents`, `_submitScan`, `_pollJobUntilDone`, `_applyCompletedPayload`, `renderDetailCard`, `toggleCoreMeshView()`, and `resetSession()` in `frontend_lab/alpha-plus-pro.js`.
2. Add EN dictionary entries for every new `_t('pp_*', ...)` string in `frontend_lab/index.html` (`I18N.en`). Zero external CDN URLs added!

- [ ] **Step 4: Run test to verify pass**

Run: `node --test tests/test_frontend_plus_pro.mjs`
Expected: All Node tests PASS (`fail 0`).

---

### Task 10: Documentation, Colab Bundle Packaging & Full-System Verification

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\README.md`
- Rebuild: `c:\Users\nvpph\Desktop\alpha scan\colab_backend_only.zip`
- Rebuild: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline_bundle.zip`

**Interfaces:**
- Consumes: All files modified/created in Tasks 1–9.
- Produces: Up-to-date `README.md` documenting `reconstruction_mode` (`calibrated_mhr`, `hi_res_head`, `dual_layer`), `proportion_audit`, capture protocol (`đồ tập sát form, tóc gọn lộ cổ–vai`), and refreshed Colab zip bundles including `segmentation.py`, `fitting.py`, `head_hair.py`, `outer_shell.py`, and updated `frontend_lab/`.

- [ ] **Step 1: Update `README.md` CLI & API quick-reference**

Document `--reconstruction_mode` in `run_scan.py` examples, `POST /api/v1/bodyscan` `reconstruction_mode` field, `proportion_audit` + `core_mesh_url` response fields, and `mediapipe` dependency in Colab setup instructions.

- [ ] **Step 2: Rebuild `colab_backend_only.zip` and `bodyscan_pipeline_bundle.zip`**

Run Python `zipfile` packager excluding `__pycache__/`, `.pyc`, and output artifacts, packaging `requirements.txt`, `README.md`, `bodyscan_pipeline/*.py` (all 15 modules), and `frontend_lab/` cleanly, then inspect `zipfile.ZipFile(...).namelist()` to confirm all new modules are present.

- [ ] **Step 3: Run Full Automated Verification Suite + License & Live Mock Check**

Run:
1. `python -m pytest -v` — verify all suites pass (`test_scan_service.py`, `test_segmentation.py`, `test_fitting.py`, `test_head_hair.py`, `test_outer_shell.py`, `test_texturing.py`, `test_api_server.py`, `test_measurements.py`).
2. `node --test tests/test_frontend_plus_pro.mjs` — verify all JS & HTML `I18N.en` parity checks pass.
3. Verify `/health` + mock end-to-end upload -> `.glb` + `scan_core.glb` -> `proportion_audit` -> Decision 3a `410 Gone` works cleanly against live `python -m bodyscan_pipeline.api_server --mock --serve-frontend frontend_lab`.
4. Note for Colab GPU QA (after implementation completes): run visual acceptance gate on `sample_ai_images/female_01` verifying `< 90s`, `< 12 GB` VRAM, zero wall haloing, clear Level 2 blur strictly on facial skin while keeping hair crisp, `mhr_silhouette_iou >= 0.85`, and `outer_silhouette_iou >= 0.88`.
