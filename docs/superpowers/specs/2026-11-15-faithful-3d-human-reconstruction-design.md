# Alpha Scan Plus/PRO — Faithful 4-View 3D Human Reconstruction Design (Unified 3-Tier Plan)

**Date:** 2026-11-15
**Status:** Ready for User Approval (Builder Check Completed & Resolved)
**Workspace Root:** `c:\Users\nvpph\Desktop\alpha scan`
**Builds On:**
- `docs/plans/2026-11-13-step3-texturing.md`
- `docs/superpowers/specs/2026-11-15-plan-2a-gpu-api-lab-design.md`
- `Hướng dẫn kỹ thuật - Hình 3D người thật (SAM 3D Body + MHR) cho Alpha Scan Plus.docx`

---

## 1. Intent & Why

Currently, `GpuScanEngine` (`bodyscan_pipeline/scan_service.py`) runs `SAM 3D Body + MHR` independently per image without human foreground masks (`human_segmentor=None`, `use_mask=False`), averages raw `shape_params`/`scale_params` across views (`fusion.py`), zeroes facial expressions (`torch.zeros(1, 72)`), uniformly scales vertices solely to `height_cm` while ignoring `weight_kg` and `sex` shape priors, and paints textures using unmasked image projections plus `cv2.INPAINT_TELEA` hole-filling (`texturing.py`). Meanwhile, `MockScanEngine` produces a fixed 5-part geometric mannequin with a UV vertical flip bug.

As a result, reconstructed 3D avatars fail user fidelity expectations: different subjects look overly similar, hair volume and crisp nose/facial geometry are absent (MHR is a bald, coarse-face parametric template), background pixels bleed into body textures, `weight_kg` is unverified against true body volume, and declared gender traits are under-represented.

The user requested **one unified plan combining all three proposed directions**:
1. **Metric 4-View MHR Calibration (`calibrated_mhr` — Hướng 1):** Optimize MHR body parameters against 4 segmented silhouettes, declared `height_cm`, `weight_kg` volume constraints, and `sex`-calibrated priors, plus clean foreground-masked UV texturing.
2. **High-Resolution Face, Nose & Hair 3D Reconstruction (`hi_res_head` — Hướng 3):** Reconstruct crisp anatomical facial features (nose bridge, jaw, brow, lips) and real 4-view hair volume from un-resized head crops and stitch cleanly onto the calibrated MHR neck.
3. **Dual-Layer Metric Core + 4-View Outer Surface Shell (`dual_layer` — Hướng 2):** Under the **fitted-form & hair-up capture protocol** (`đồ tập sát form, tóc gọn lộ cổ–vai`), preserve the calibrated inner MHR mesh strictly for 16 anatomical circumferences and volumetric weight verification while reconstructing an outer 3D display surface fused from 4 calibrated depth maps, hair-up cranium hull, and garment surface micro-relief (collars, hems, seams, footwear).

Critically, every stage must follow the strict **non-hallucination rule**: reconstruct only what the **4 uploaded photos (`Front`, `Right`, `Back`, `Left`)** and initial declared inputs (`height_cm`, `weight_kg`, `sex`) physically support — **no generative 3D fabrication, no beauty retouching, and no prohibited packages**.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
1. **Strict 4-Photo Guided Intake + Full-Res Facial Preservation:** `#plusModal` requires all **4 guided cardinal photos (`Front`, `Right`, `Back`, `Left`)** for `hi_res_head` / `dual_layer` (retaining full-resolution head/face crops in memory during processing rather than relying solely on `1600px` body-downsampled arrays), while backend API keeps `1–8` photo compatibility (auto-downgrading `<4` cardinal views to `calibrated_mhr` with a clear notice).
2. **Clean Foreground Segmentation & Fitted/Collar-Clear Protocol:**
   - Capture protocol requires **fitted athletic wear (`sát form`)** and **hair tied/pinned clear of the neck and shoulders (`tóc gọn`)**.
   - Isolate person pixels cleanly from walls/floors (`M_v`) and separate `hair_mask_v` vs `face_skin_mask_v` on all views using Apache-2.0 compatible models (Meta `SAM 2` / `SAM` via `SAM3DBodyEstimator.human_segmentor`, with Apache-2.0 `MediaPipe SelfieSegmentation` fallback) so neither silhouette losses nor texture projections absorb background pixels.
3. **Metric Height, Weight & True Body Proportions (`Hướng 1` Core):**
   - Jointly optimize MHR `shape_params`, `scale_params`, `expr_params`, and per-view camera alignment `(cam_t, root_rot)` via differentiable `PyTorch3D` rendering against the **hair-excluded body silhouette masks `M_body_v = M_v \ hair_mask_v`** (`Silhouette IoU + Edge Chamfer`) so updos/buns do not warp the MHR skull or neck.
   - Enforce exact height scaling (`height_cm`) and apply a soft volumetric mass constraint linking closed MHR mesh volume $\times \rho(\text{sex})$ (`~1.045 kg/L` male / `~1.028 kg/L` female baseline) toward declared `weight_kg` (optional `0.0` allowed to mean "estimate from silhouette"; when `weight_kg > 0`, tightens & validates mass).
   - Produce a quantitative **`proportion_audit`** block reporting `estimated_weight_kg`, `mhr_silhouette_iou`, `outer_silhouette_iou`, and explicit mismatch warnings if `weight_kg > 0` contradicts photo silhouettes (`>12%` difference).
4. **Distinct Gender, Hair Volume, Face & Nose Geometry (`Hướng 3` Head Module):**
   - Replace coarse bald MHR head vertices (~500 verts) with a dense subdivided anatomical head template aligned and deformed from:
     - Apache-2.0 3D facial landmarks (`MediaPipe Face Landmarker` 478 points, **held in RAM only, never saved to disk**) projected consistently across Front and oblique/profile views;
     - Left/Right profile silhouette edges extracted cleanly for the **nose bridge, nose tip, lips, chin, forehead, and back-of-head hairline**;
     - MHR-depth-calibrated `MoGe2` normal/depth relief over the face & nose;
     - A 4-view silhouette-carved **hair-up volume shell** bounded cleanly above `NECK_BASE_JOINT=110` (capturing bangs, scalp/crown volume, side hair, and bun/ponytail silhouette from `hair_mask_v`, welded cleanly onto the neck collar).
   - Dedicate high pixel density to the head/face/hair UV island (dual glTF material `head` at `2048×2048` + `body` material) for crisp facial realism.
5. **Dual-Layer Body Separation (`Hướng 2` Outer Shell):**
   - Preserve the calibrated MHR mesh (`inner_mhr_mesh`) for all **16 circumferences**, **`loft` profile**, and **`weight_kg` validation** so hair volume, shoes, garment seams/folds, and waistbands do not inflate body girths.
   - Generate the visible display mesh (`outer_surface_mesh`) by combining the `Hướng 3` high-res head/hair mesh with an outer torso/limb shell reconstructed from the 4 foreground masks `M_v` + MHR-metric-aligned `MoGe2` depth maps (regularized softly onto the inner MHR surface across un-observed `45°` corners).
   - Export both outer display `.glb` (primary viewer model) and `scan_core.glb` (inner MHR mesh) so operators can toggle between **Outer 3D (Photo-Faithful)** and **Inner Body Core (Measurement Skeleton)** in `frontend_lab/`.
6. **Staged Runtime Modes & Privacy Compliance (`P0-15` Confirmed Rules):**
   - Support selectable `reconstruction_mode`: `"calibrated_mhr"` (Stage 1 / Hướng 1), `"hi_res_head"` (Stage 2 / Hướng 3), and `"dual_layer"` (Stage 3 / Hướng 2 + 3 combined — default full fidelity).
   - **Unified Detailed Geometry Across Privacy Levels:** Privacy Levels `1`, `2`, and `3` all share the detailed 3D head/nose/hair geometry produced by the chosen mode (`3a`: texture hidden in Level 1, **skin-only face blurred while hair stays sharp and clear** in Level 2 `tex_blur.jpg`, full clear likeness in Level 3 `tex_face.jpg`).
   - Fix `MockScanEngine` UV vertical orientation (`1 - v`) and surface server `mode` (`MOCK` vs `GPU`), `reconstruction_mode`, `proportion_audit`, and pipeline `warnings` visibly inside `frontend_lab/` (`#detail` card).

### Non-Goals
- Reconstructing **loose/baggy everyday outerwear** or **long loose hair draping over shoulders/back** (capture protocol requires fitted activewear `sát form` and hair tied clear of collar `tóc gọn`, matching `sample_ai_images/female_01` and the technical guide).
- Using generative 3D AI (`Hunyuan3D`, `TripoSR`, `Trellis`, `InstantMesh`, `Wonder3D`) to invent unseen back/side clothing, hair, or facial shapes.
- Altering Supabase Seoul schemas or building Plan 2B Edge Functions.

### Anti-Goals
- **Zero prohibited licenses:** Never import or require `ultralytics`, `smplx`, `FLAME`, `nvdiffrast`, or non-commercial weights (such as `briaai/RMBG`). Only `SAM 3D Body`, `MHR`, `SAM/SAM2`, `Detectron2`, `MoGe2`, `MediaPipe` (`Apache 2.0`), `PyTorch3D`, `trimesh`, `xatlas`, `OpenCV`, `SciPy`, `NumPy`, `Pillow` are permitted.
- **Zero facial hallucination / retouching:** Never run generic inpainting (`cv2.INPAINT_TELEA`) over missing facial landmarks (eyes, nose, mouth), and never force body shape blindly toward an impossible `weight_kg` if it violates the observed 4-view silhouettes (clamp & report mismatch instead).
- **Zero raw photo retention:** Source photos and temporary full-res head crops must be shredded in `finally:` with `deletion_audit.jsonl` intact.

---

## 3. Approaches Unified (Cumulative 3-Tier Architecture)

Instead of choosing only one trade-off, **Hướng 1**, **Hướng 2**, and **Hướng 3** are structured as **three cumulative, interoperable layers sharing one calibrated core**:

| Tier / Stage | Approach Mapped | Geometry Output | Texture Output | When Used / Tradeoff |
|---|---|---|---|---|
| **Stage 1 (`calibrated_mhr`)** | **Hướng 1** | Single watertight MHR mesh optimized to 4 masks + `height_cm` + `weight_kg` + `sex` | Foreground-masked `2048×2048` 4-view projection | Fastest (~25–40s on T4); always runs as the **mandatory metric core** even in Tiers 2 & 3 |
| **Stage 2 (`hi_res_head`)** | **Hướng 3** | Stage 1 body + dense Landmark/Profile/MoGe2 Face–Nose–Hair head mesh welded at neck | Dual high-res UV (body + un-resized head/face/hair crop) | Best likeness-per-second (~45–65s on T4) when clothes are fitted |
| **Stage 3 (`dual_layer`, Default)** | **Hướng 2 + 3** | Stage 1 inner core (measurements) + Stage 2 high-res head/hair + 4-view `MoGe2`/Visual-Hull torso-limb shell | Masked 4-view photo projection strictly on segmented outer visible mesh | Maximum photo fidelity including hair volume & clothing folds (~60–85s on T4) while keeping inner girths uncorrupted; slightly more regularized at `45°` torso seams |

*Builder Note / Settled Architecture Call:* Implementing Stage 1 first produces an immediately testable upgrade; Stage 2 plugs into Stage 1's neck boundary interface; Stage 3 wraps Stage 1 + Stage 2 behind `reconstruction_mode`. Zero throwaway code across phases.

---

## 4. Constraints

1. **Colab T4 15 GB VRAM / Single-Worker Memory Budget:** All stages run sequentially inside `GpuScanEngine._lock` (`max_workers=1`). Explicit stage lifecycle (`Step 1 SAM3D/Seg inference -> free detector VRAM -> Step 2 Differentiable PyTorch3D Fitting -> Step 3 Head/Outer Mesh & Baking -> gc.collect() + empty_cache()`) ensures peak VRAM stays under `12 GB` and total runtime stays `<= 90s` on T4/L4 GPUs.
2. **PyTorch3D Fallback & CPU Testability:** Core geometry math, segmentation interfaces, silhouette/volume loss calculations, head alignment math, UV masking, and proportion auditing must remain unit-testable on Windows CPU without CUDA or `SAM 3D Body` weights.
3. **`Three.js r128` Offline Compatibility:** All exported `.glb` files (single or multi-primitive/material for dual-island body+head UV, plus optional `scan_core.glb`) must load cleanly in `frontend_lab/alpha-plus-pro.js` (`parseGlbBinary`) with zero CDN calls.
4. **Backward Compatibility with `alpha_scan.body_scan.v2`:** Preserve all existing keys in `body_scan.json` (`circumferences_cm`, `shape_profile.loft`, `mesh`, `quality`, `textures`); new fields (`reconstruction_mode`, `proportion_audit`, `mesh.core_file`) are strictly additive.
5. **Privacy Levels (`P0-15`) Confirmed Behavior:**
   - All levels (`1`, `2`, `3`) use the **same detailed 3D mesh geometry** produced by `reconstruction_mode` (`hi_res_head` / `dual_layer` nose, jaw, hair shape preserved cleanly).
   - Level 1: plain neutral gray geometry (`scan.glb`, plus `scan_core.glb` when `dual_layer`, zero photo textures).
   - Level 2: outer faithful geometry + `tex_blur.jpg` (`scan_preview_blur.glb`), **blurring ONLY the facial skin region (`face_blur_uv_mask`) while keeping hair (`hair_uv_mask`), clothing, and silhouette sharp and unblurred**.
   - Level 3: full un-retouched facial/hair/body likeness (`tex_face.jpg`, `scan_preview_face.glb`).
   - Ephemeral landmark invariant (`S2` rule): 478 MediaPipe facial landmarks exist solely in RAM during fitting/baking and are **never serialized to JSON, `.glb`, or disk**, and raw photos + crops are shredded in `finally:`.
   - Explicit consent rules (`consent_granted=true`, `LOCAL_ONLY` loopback guard, Decision 3a server deletion of all `.glb`/`.jpg` including `scan_core.glb`) remain unchanged.

---

## 5. System Architecture & Module Design

```
4 Photos (Front, Right, Back, Left) + height_cm, weight_kg, sex, reconstruction_mode
   │
   ▼
[Module 1: segmentation.py + inference.py]
   - Retain full-res RGB (orig_rgb) + downsampled inference RGB (max_side=1600)
   - Segment foreground person mask M_v & coarse hair/skin/face semantic masks per view v in {0..3}
   - Run SAM 3D Body + Detectron2 + MoGe2 (focal_length, camera pred_cam_t, monocular depth D_v, initial params)
   - Extract Apache-2.0 MediaPipe 478 facial landmarks (front + oblique/profile) & nose/hair profile curves
   │
   ▼
[Module 2: fitting.py (Approach 1 — Metric MHR Core)]
   - Joint PyTorch3D differentiable optimization of MHR (shape_params, scale_params, expr_params, per-view delta_cam_t, delta_rot):
       L_total = w_sil * L_silhouette(M_body_0..3) + w_edge * L_edge_chamfer + w_kpt * L_kpt_2d
               + w_vol * L_volume_mass(weight_kg, sex) + w_prior * L_reg(sex)
     where M_body_v = M_v \ hair_mask_v (fitted-form body contour without hair bulge)
   - Solve straight A-pose, scale strictly head-to-heel == height_cm, validate volumetric weight vs declared weight_kg
   - Output calibrated inner_mhr_mesh (verts_a, joints_a, faces) + proportion_audit
   │
   ├───► (measurements.py) Slice 16 anatomical circumferences & loft purely from calibrated inner_mhr_mesh
   │
   ├───► If reconstruction_mode in ("hi_res_head", "dual_layer"):
   │     [Module 3: head_hair.py (Approach 3 — Dense Face, Nose & Hair)]
   │       - Subdivide canonical head zone above NECK_BASE_JOINT=110 cleanly into dense face/cranium topology
   │       - Deform facial vertices via ephemeral 2D/3D MediaPipe landmarks + Left/Right nose/chin silhouette contours
   │         + MHR-aligned MoGe2 facial depth relief
   │       - Build watertight 4-view hair-up volume shell above NECK_BASE_JOINT from hair_mask_v anchored to cranium
   │       - Laplacian collar weld seamlessly to inner_mhr_mesh neck boundary
   │
   └───► If reconstruction_mode == "dual_layer":
         [Module 4: outer_shell.py (Approach 2 — 4-View Outer Surface Shell)]
           - Re-project aligned MoGe2 depths D_v (masked by M_v) onto torso/limb surface initialized from inner_mhr_mesh
           - Constrain rays cleanly inside 4-view Visual Hull M_0..3 bounding volume (capturing garment seams/folds & footwear);
             regularize 45° blind strips toward inner_mhr_mesh so seams stay smooth without hallucinated creases
           - Merge torso/limb outer shell cleanly with Module 3 dense Head-Face-Hair mesh
   │
   ▼
[Module 5: texturing.py + exporter.py]
   - Foreground-masked PyTorch3D texture baking: reject any pixel outside eroded mask M_v (eliminates wall bleed)
   - Anti-hallucination rule: NO TELEA inpainting across facial/landmark regions; fill only un-seen micro-seams (<4px)
     or deeply self-occluded armpits using local observed skin/garment neighborhood median; track observed_coverage_pct
   - Dual/Weighted UV atlas (separate crisp high-res head/hair material or 50% UV territory + body territory)
   - Export display GLB (+ optional scan_core.glb), textures (respecting privacy_level 1/2/3), and extended body_scan.json
```

---

## 6. Component Specifications

### Component A: Segmentation & High-Resolution Evidence (`bodyscan_pipeline/segmentation.py`, `inference.py`)
1. **Image Retention Without Premature Cropping Loss:**
   - Extend `run_inference_on_images` (`inference.py`) to optionally retain full-resolution RGB (up to e.g. `max_side=3200` on head/crop extraction, while body inference runs at `max_side=1600` to keep VRAM bounded).
   - Detect and store canonical view orientation guesses alongside camera azimuth inferred from `pred_keypoints_3d` shoulder/pelvis vector (no longer trusting uploaded filename order blindly if user dropped 4 angles out of order!).
2. **Foreground & Semantic Part Masks (`segmentation.py`):**
   - Primary backend: `SAM 2` / `SAM` (`Apache 2.0`, already native ecosystem to `sam-3d-body` via `human_segmentor`), producing crisp binary person silhouette masks `M_v (H, W)` cleanly excluding studio/home background walls and floors.
   - Lightweight fallback: `MediaPipe SelfieSegmentation` + `OpenCV GrabCut` boundary refinement when SAM checkpoint is unavailable or in CPU tests.
   - Hair & Skin/Face ROI separation (`extract_head_semantics`): combine ephemeral `MediaPipe Face Landmarker` (`478` 3D points, `Apache 2.0`, held strictly in RAM) convex/hull skin mask + forehead/scalp priors + foreground mask `M_v` in head bounding region to isolate:
     - `face_skin_mask` (facial skin below hairline, ears, neck — excluding hair)
     - `hair_mask` (head foreground outside facial skin/collar, capturing bangs, crown height, side locks, ponytail/bun silhouettes above `NECK_BASE_JOINT=110`)
     - `M_body_v = M_v & (~hair_mask)` (hair-excluded silhouette passed to Component B so hair bulk never warps the calibrated MHR skull/body)
     - `collar_hair_overlap_warning`: pre-flight check verifying `hair_mask` stays clear of the neck/shoulder collar line (`2 gọn`); emits a visible warning in `warnings` if loose hair drapes below `NECK_BASE_JOINT=110`
     - `nose_profile_curve` (extracted directly from high-res `Left` and `Right` profile foreground silhouettes over the eye-to-mouth band!).

### Component B: Multi-View Silhouette & Metric Weight/Height Optimization (`bodyscan_pipeline/fitting.py`, `fusion.py`)
Replaces naive averaging in `fusion.py` when running GPU Stage 1+:
1. **Inputs:** Per-view `outputs` from `inference.py` (`1–8` photos supported by backend API: uses all views if `5–8` are given; if `<4` cardinal views are present, auto-downgrades `hi_res_head`/`dual_layer` to `calibrated_mhr` with a `warnings` notice while `#plusModal` requires all 4 cardinal views), hair-excluded body masks `M_body_0..N` (under the `1 sát form` fitted athletic wear & `2 gọn` collar-clear protocol), `height_cm`, optional `weight_kg` (`0.0` allowed), `sex`, and `mhr_model` + `head_assets`.
2. **Camera Registration Per View:** Solve initial rigid camera alignment `(cam_t_v, cam_rot_v)` unifying all input views against canonical `verts_posed`, refining focal/axis signs as already proven in `texturing.py:transform_vertices_to_camera` plus coarse azimuth rotation (`0°`, `90°`, `180°`, `270°` initial guess refined continuously).
3. **Differentiable PyTorch3D Fitting Loop (`40–60` Adam steps on GPU):**
   - Optimize `shape_params (1, 45)`, `scale_params (1, 28)`, subtle `expr_params (1, 72)` (or keep neutral expression while optimizing facial shape components), and per-view pose/camera residuals `(cam_t_v, cam_rot_v, pose_res_v)` (keeping final A-pose geometry cleanly in `shape_params + scale_params`).
   - Loss terms:
     - `L_sil`: Soft PyTorch3D `SoftSilhouetteShader` BCE + Dice IoU loss between rendered MHR silhouettes and `M_body_v` across all views (enforces true shoulder width, waist pinch, chest/belly side depth, hip curve, thigh/calf thickness).
     - `L_edge`: Sobel boundary chamfer distance between rendered silhouette edges and `M_body_v` contours.
     - `L_kpt`: 2D anatomical joint + facial anchor projection reprojection error.
     - `L_vol`: Closed watertight MHR mesh volume $V(\text{scaled to height\_cm})$ multiplied by sex-calibrated body density ($\rho_{\text{male}} = 1.045\text{ kg/L}$, $\rho_{\text{female}} = 1.028\text{ kg/L}$) penalized against declared `weight_kg` strictly when `weight_kg > 0` (`weight_kg == 0.0` skips `L_vol` and estimates weight purely from calibrated geometry).
     - `L_prior`: Sex-aware physiological Mahalanobis/L2 prior regularizing `shape_params`/`scale_params` toward plausible human proportions and penalizing extreme unrealistic deformation (`clip` deviation when `L_vol` fights `L_sil`).
4. **Weight/Height Reality Audit (`proportion_audit`):**
   - Compute final `estimated_weight_kg = V_final_liters * rho(sex)`.
   - Compute `mhr_silhouette_iou = mean(IoU(rendered_mhr_v, M_body_v))` (later paired with `outer_silhouette_iou` evaluated against full `M_v` after Components C/D).
   - Settled Builder Call (Handling Input Conflict): When `weight_kg == 0.0`, report `estimated_weight_kg` cleanly with `status = "estimated_only"`. When `weight_kg > 0`, because fitted athletic wear/hydration still introduces minor apparent volume variance, optimization weights `L_sil` above `L_vol` so photos are NEVER distorted into cartoonish thin/fat shapes if a client mis-typed `weight_kg`; instead:
     - If `|estimated_weight_kg - weight_kg| / weight_kg <= 0.08`: calibrated smoothly, `status = "matched"`.
     - If discrepancy exceeds `12%` (or `mhr_silhouette_iou < 0.78`): emit warning inside `warnings` and set `proportion_audit.status = "mismatch_warning"` (never silently fake proportions!).

### Component C: High-Resolution Face, Nose & Hair 3D Reconstruction (`bodyscan_pipeline/head_hair.py`)
Executes for `reconstruction_mode in ("hi_res_head", "dual_layer")` across **all Privacy Levels (`1`, `2`, `3`)** (`3a chi tiết`):
1. **Localized Neck Cut & Subdivided Head Cranium:**
   - Partition MHR vertices above `NECK_BASE_JOINT=110` cleanly into a head sub-mesh + shared collar ring vertices (`collar_idx`).
   - Subdivide facial/cranium triangles (`1x–2x` midpoint/Loop subdivision on head region only -> ~12k–20k head vertices, keeping body lightweight at ~10k vertices), giving high polygon density specifically along the nose bridge, nostrils, philtrum, eyelids, lips, ears, and scalp.
2. **Multi-Cue Facial & Nose Sculpting (Strict Observed Data Only, RAM-Only Landmarks):**
   - **Landmark Alignment:** Map `MediaPipe Face Landmarker` 478 3D canonical topological indices onto subdivided head mesh facial zone using Procrustes + TPS radial basis warp driven by actual front/left/right 2D landmark observations (never generic beauty templates). **Ephemeral S2 Rule:** Landmark arrays stay strictly in RAM during sculpting/masking and are never written to JSON, `.glb`, or disk.
   - **Nose Bridge & Chin Profile Enforcement:** Fit nose bridge height, tip projection, nasolabial angle, and chin depth explicitly against `nose_profile_curve` extracted from the **Right and Left 90° profile silhouettes** (and any oblique visibility).
   - **Metric Depth Relief:** Align each crop's `MoGe2` normal/depth field smoothly to MHR face scale/offset and apply Laplacian-regularized normal/displacement refinement for cheekbones, brow ridge, and eye sockets.
3. **4-View Collar-Clear Hair Volume Shell (`hair_shell`):**
   - Construct an auxiliary hair/scalp outer cap initialized from the cranium crown + side/back scalp band strictly above `NECK_BASE_JOINT=110` (`2 gọn`).
   - Fit vertices radially along vertex normals against the **4 foreground `hair_mask` contours** (Front hairline/bangs & side width, Right/Left hair projection & bun/ponytail depth, Back hair contour envelope), enforcing continuous manifold smoothness across `45°` intermediate angles and clamping inner radius `>= cranium_radius` so hair never punches inward into the skull!
   - Export explicit UV/vertex region partition separating **`face_blur_uv_mask`** (facial skin below hairline, eyes, nose, lips, jaw) vs **`hair_uv_mask`** (hair/scalp shell) so Privacy Level 2 (`generate_blurred_face_texture`) blurs **ONLY `face_blur_uv_mask` while preserving `hair_uv_mask` sharp and unblurred** (`3b giữ rõ`), Level 1 omits photo textures on the clean detailed geometry, and Level 3 renders full unblurred `tex_face.jpg`.
4. **Seamless Neck Collar Stitch:** Blend collar ring boundary displacement to `0` over `3` edge rings downward onto the neck so `head_hair` + `body` form one watertight surface with smooth normals.

### Component D: Dual-Layer Clothed/Outer Body Surface (`bodyscan_pipeline/outer_shell.py`)
Executes only when `reconstruction_mode == "dual_layer"` (Default):
1. **Metric Depth Alignment:** Scale and shift each view's `MoGe2` depth map `D_v` (`Front`, `Right`, `Back`, `Left`) per torso/limb region using linear regression against rendered depths of the calibrated `inner_mhr_mesh` (guaranteeing exact metric scale in meters, eliminating monocular scale drift).
2. **Outer Shell Optimization:**
   - Initialize `outer_surface_mesh` from calibrated `inner_mhr_mesh` subdivided lightly once along torso/limbs (paired with Component C's dense head/nose/hair mesh above the collar).
   - Optimize torso/limb surface displacement along surface normals using PyTorch3D differentiable silhouette + RGB-D depth loss against masked `(M_v, aligned_D_v)` from all 4 views, constrained by:
     - `inner_barrier`: `outer` vertices cannot penetrate deeper than `-3mm` inside `inner_mhr_mesh` (clothing/skin sits on or outside anatomical core);
     - `hull_cap`: `outer` projection stays strictly inside the intersection of the 4 foreground masks `M_0..3` (prevents bulging out past real photo silhouettes);
     - `laplacian_45deg_reg`: stronger biharmonic/Laplacian smoothness in regions unseen orthogonally (around `±45°` diagonal grazing angles) anchored gently toward `inner_mhr_mesh` shape delta so seams stay smooth without boxy octagonal steps!
3. **Dual Export Artifacts:**
   - Primary display mesh (`scan.glb` / `scan_preview_*.glb`) exports `outer_surface_mesh` (with dense head/nose/hair + outer clothes/body silhouette).
   - Secondary core artifact (`scan_core.glb`) optionally exports `inner_mhr_mesh` (plain MHR anatomical core) so operators/engineers can visually verify inner vs outer alignment.
   - All **16 circumferences + `loft` + `proportion_audit`** in `body_scan.json` are measured strictly from `inner_mhr_mesh` (keeping tape-measure comparability intact!).

### Component E: Foreground-Masked, Un-Altered Texture Baking (`bodyscan_pipeline/texturing.py`, `exporter.py`)
1. **Strict Foreground Pixel Masking:** Add `foreground_mask: Optional[np.ndarray]`, `hair_mask: Optional[np.ndarray]`, and `face_skin_mask: Optional[np.ndarray]` to `ViewObservation`. In `bake_multiview_texture`, reject any projected sample `(u_img, v_img)` whose `foreground_mask` pixel value `< 0.5` (plus `2px` morphological erosion on mask edges to eliminate background color haloing around hair/shoulders/arms!).
2. **Dedicated High-Res Head UV Space & Level 2 Hair-Preserving Blur:** Partition final UV layout cleanly into **two packed islands or two glTF materials** (`material[0] = body` at `1024–2048px`, `material[1] = head_face_hair` at `2048×2048px` OR unified `2048×2048` atlas allocating `>= 45%` texel area to `head/hair`), so nose tip, eyes, lips, and hair strands stay sharp when zooming in on the 3D viewer! Split the head UV mask cleanly into `face_blur_uv_mask` (facial skin below hairline) and `hair_uv_mask` so `generate_blurred_face_texture` (Privacy Level 2) blurs **only `face_blur_uv_mask` while keeping `hair_uv_mask` sharp and unblurred** (`3b giữ rõ`), avoiding Gaussian feather bleed across the hairline UV gutter.
3. **Anti-Hallucination Hole Handling (`strict_no_invent` mode):**
   - Do **NOT** run `cv2.INPAINT_TELEA` across visible facial/nose/eye regions! Any un-observed facial pixel fallback uses smooth valid multi-view projection only; if `< 2%` tiny seam cracks occur, dilate strictly across adjacent UV border texels (`_dilate_uv_gutters`) or fill deeply shadowed non-face cavities (e.g. armpits/soles) with local region observed median RGB.
   - Compute `coverage_pct` and `head_coverage_pct`, recording them in `body_scan.json["textures"]`.
4. **Fix `MockScanEngine` UV Bug (`scan_service.py`):** Flip `v` consistently in `_build_tube_part` / `_render_mock_texture` so local CPU `--mock` renders head/skin/shirt/shorts right-side-up instead of pasting the face square onto the pelvis and turning head/arms black.

### Component F: API & Frontend Integration (`bodyscan_pipeline/api_server.py`, `frontend_lab/`)
1. **API Contract Extensions (`api_server.py`):**
   - `POST /api/v1/bodyscan` accepts optional form field `reconstruction_mode` (`"calibrated_mhr"` | `"hi_res_head"` | `"dual_layer"`, default `"dual_layer"`).
   - `GET /api/v1/bodyscan/{job_id}` exposes `reconstruction_mode`, `warnings`, `artifacts.core_mesh_url` (when `dual_layer` produces `scan_core.glb` — included in `ALLOWED_ARTIFACT_FILES` & `BINARY_ASSET_FILES` so Decision 3a deletes it cleanly too!), and `body_scan.proportion_audit`.
2. **Frontend Lab Enhancements (`alpha-plus-pro.js`, `index.html`):**
   - `#plusModal`: Enforce all **4 guided cardinal photo slots (`Chính diện · Phải · Sau · Trái`)** before enabling submit for `hi_res_head` / `dual_layer` (keeping `<4`-photo warning/downgrade to `calibrated_mhr` if submitted only via raw API); add **"Độ chi tiết dựng 3D" selector** defaulting to **Toàn diện 2 lớp (Mặt–Mũi–Tóc + Vỏ ngoài thật)** alongside **Chỉ đầu chi tiết** (`hi_res_head`) and **Lõi chuẩn đo nhanh** (`calibrated_mhr`), plus a brief capture-protocol reminder (`đồ tập sát form, tóc gọn lộ cổ–vai`) and height/weight plausibility hint.
   - `parseGlbBinary`: Already iterates `for (const prim of (mDef && mDef.primitives) || [])` with per-primitive `buildMaterial(prim.material)` and `nodeGrp.add(mesh)`, verify & test multi-material `.glb` parsing (`body` material + `head` material) in `tests/test_frontend_plus_pro.mjs`!
   - `#detail` card: Show clear server engine pill (**`GPU thật`** vs **`MOCK TEST`**), active `reconstruction_mode`, **`proportion_audit` status** (e.g., estimated vs declared `weight_kg`, `mhr_silhouette_iou %`), visible list of `warnings` if fallback/mismatch occurred, and a quick toggle **"Xem lớp ngoài thật / Xem lõi đo MHR"** when `core_mesh_url` exists!

---

## 7. Phased Rollout Plan (Dependency Order)

| Phase | Scope | Runnable Value Delivered | Test Target |
|---|---|---|---|
| **Phase 1 (Foundations + Hướng 1)** | `segmentation.py`, `fitting.py`, masked `texturing.py` + `proportion_audit` + `MockScanEngine` UV fix + UI/API flags | Real Colab GPU fits 4 silhouettes + `height_cm`/`weight_kg`/`sex`, stops wall-color texture bleed, flags mismatched inputs | CPU unit tests (`pytest` for silhouette/volume loss math, mask rejection, GLB multi-material parser, mock UV alignment) + Colab GPU smoke on `sample_ai_images/female_01` |
| **Phase 2 (Hướng 3 — High-Res Face/Nose/Hair)** | `head_hair.py`, un-resized head crop pipeline, MediaPipe facial landmarks + nose profile + hair shell + neck collar weld + dual UV head island | True 3D likeness of hair, face, and nose bridge visible in 360° & zoom-in views without altering body measurements | Unit tests on collar boundary continuity, landmark Procrustes alignment, hair mask radius clamp (`>= skull`), Level 2 blur mask isolation + Colab visual QA |
| **Phase 3 (Hướng 2 — Dual-Layer Outer Shell + UI Switcher)** | `outer_shell.py`, `MoGe2` metric calibration, outer garment surface + `scan_core.glb` inner/outer viewer toggle + full E2E polish | Clothed/haired outer mesh matches photos while inner MHR preserves 16 girths; operators inspect outer vs inner core live | Unit tests on inner barrier (`>= inner - 3mm`) & hull cap constraints, Decision 3a deletion includes `scan_core.glb`, browser E2E (`node --test`, Playwright/browser) |

---

## 8. Review & Testing Plan

1. **Automated Offline CPU Tests (`pytest` + `node --test`):**
   - `tests/test_segmentation.py`: foreground mask boundary erosion, hair-vs-face mask extraction, view azimuth sorting order checks.
   - `tests/test_fitting.py`: synthetic box/ellipsoid silhouette IoU minimization converges toward ground-truth aspect ratio; volume-to-mass density helper computes exact `weight_kg` within `±0.5%`; mismatch warning triggers when declared weight differs by `>12%`.
   - `tests/test_head_hair.py`: subdivided head + neck collar boundary stitch preserves vertex indices and zero collar seam gap; nose-tip projection strictly respects profile silhouette boundary; hair shell radii never penetrate cranium interior.
   - `tests/test_outer_shell.py`: depth linear calibration accuracy; inner-MHR non-penetration barrier & outer visual-hull bounding cap invariants.
   - `tests/test_texturing.py`: pixels outside `foreground_mask` produce `0.0` weight (zero background bleed); strict anti-hallucination facial mode leaves facial texels unmodified by `TELEA`; multi-material GLB export preserves head vs body UVs.
   - `tests/test_api_server.py` & `tests/test_frontend_plus_pro.mjs`: `reconstruction_mode` validation (`422` on invalid mode), `scan_core.glb` signed URLs + Decision 3a deletion returning `410`, multi-primitive `.glb` loading in `parseGlbBinary`, `#detail` displaying `GPU/MOCK`, `proportion_audit`, and `warnings`.
2. **Colab GPU Verification & Visual Acceptance Gates:**
   - Benchmark on `sample_ai_images/female_01` + additional male/female AI & consented staff test sets wearing **fitted athletic wear & collar-clear hair-up** (`sát form, tóc gọn`; diverse nose profiles, short hair & tied-up buns/ponytails):
     - Total runtime `<= 90s` and peak VRAM `< 12 GB` on Colab T4 (`15 GB`).
     - Verified zero background color fringe around hair/arms, recognizable facial/nose profile in side & front views across Levels `1`, `2`, `3` with **zero stored facial landmarks (`S2`)**, and Privacy Level 2 (`tex_blur.jpg`) blurring **only facial skin (`face_blur_uv_mask`) while keeping hair (`hair_uv_mask`) crisp (`3b giữ rõ`)**.
     - `mhr_silhouette_iou >= 0.85` (inner core vs `M_body_v`), `outer_silhouette_iou >= 0.88` (display mesh vs full `M_v`), volume mass within `±5%` on fitted wear when `weight_kg > 0`, and 16 circumferences accurate within existing spec (`<= 2 cm` torso, `<= 1.5 cm` limbs).
     - License audit: verify `FORBIDDEN_MODULES` check passes cleanly at runtime (`no ultralytics/smplx/FLAME/nvdiffrast/non-commercial RMBG`).

---

## 9. Confirmed Requirements & Builder Decisions Settled

### User-Confirmed Requirements
- **Strict 4-Photo Guided Input:** `Front`, `Right`, `Back`, `Left` photos only (no video capture required); `#plusModal` requires all 4 cardinal angles for `hi_res_head` / `dual_layer`.
- **Combined 3-Approach Plan (`1 + 2 + 3`):** Integrate metric 4-view + `height_cm`/`weight_kg`/`sex` MHR fitting (`Hướng 1`), dense high-res face/nose/hair reconstruction (`Hướng 3`), and 2-layer inner core / outer photo surface separation (`Hướng 2`).
- **Zero Fabrication / Retouching (`không chỉnh sửa, không bịa đặt`):** Every geometric bump and texture pixel must trace back to real photo silhouettes, depths, and declared physical metrics; no generative AI hallucination.
- **4 Settled Capture & Privacy Rules (`1 sát form`, `2 gọn`, `3a chi tiết`, `3b giữ rõ`):**
  1. **`1 sát form` (Fitted Athletic Wear & Optional Weight):** Capture protocol requires fitted workout attire; Component B optimizes inner MHR against hair-excluded `M_body_v` without loose-garment volume estimation, while `weight_kg` stays optional (`0.0` = estimate & report; `>0` = constrain & audit).
  2. **`2 gọn` (Collar-Clear Hair-Up):** Long hair is tied/pinned clear of the neck and shoulders; `hair_shell` fits cleanly above `NECK_BASE_JOINT=110` with a pre-flight overlap warning if hair drapes below the collar.
  3. **`3a chi tiết` (Detailed Head Geometry Across All Privacy Levels):** `hi_res_head` / `dual_layer` sculpted head/nose/hair geometry runs on Levels `1`, `2`, and `3`, while the `478` MediaPipe facial landmarks stay strictly ephemeral in RAM (`S2` policy in `index.html:188` — never persisted to JSON, `.glb`, or disk).
  4. **`3b giữ rõ` (Skin-Only Face Blur on Level 2):** Privacy Level 2 (`tex_blur.jpg`) blurs ONLY `face_blur_uv_mask` (facial skin below hairline, eyes, nose, lips, jaw) while keeping hair (`hair_uv_mask`) and clothing crisp and unblurred.

### Engineering Calls Settled (Confirmed Against Codebase)
- **Unified Stacked Modes (`reconstruction_mode`):** Expose `"calibrated_mhr"`, `"hi_res_head"`, and `"dual_layer"` (default combining all 3 layers), rather than maintaining 3 isolated pipelines.
- **API `1–8` Compat vs 4-View UI Gate:** `#plusModal` enforces 4 cardinal slots (`Front`, `Right`, `Back`, `Left`) for `hi_res_head` / `dual_layer`, while `POST /api/v1/bodyscan` stays backwards-compatible with `1–8` photos (incorporating all provided angles when `5–8` photos are uploaded, and auto-downgrading `<4` cardinal views to `calibrated_mhr` with a `warnings` notice).
- **Weight vs Photo Conflict Rule:** When `weight_kg > 0` deviates by `>12%` from photo-fitted body volume (or `mhr_silhouette_iou < 0.78`), preserve true photo geometry (clamped against distortion) and surface a visible `proportion_audit` mismatch warning rather than silently warping the mesh or hard-aborting the scan.
- **Foreground Segmentation Stack:** Use Meta `SAM 2` / `SAM` (`Apache 2.0`) via `SAM3DBodyEstimator.human_segmentor` with `MediaPipe SelfieSegmentation` (`Apache 2.0`) fallback; strictly avoid non-commercial `RMBG` or `YOLO`.
- **Anti-Hallucination Texture Policy:** Replace broad facial `TELEA` inpainting with mask-eroded 4-view blending + observed `coverage_pct` reporting; local median fill allowed only for occluded underarm/sole crevices.
- **Multi-Material GLB (`body` + `head`) & `scan_core.glb`:** Already supported cleanly by `parseGlbBinary` primitive loop in `frontend_lab/alpha-plus-pro.js`, keeping facial texture razor-sharp without bloating body texture memory.
