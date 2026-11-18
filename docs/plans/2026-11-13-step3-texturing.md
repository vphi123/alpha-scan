# Alpha Scan Step 3 — UV Unwrapping, Texture Baking & Privacy Tiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Step 3 of the Alpha Scan Plus/PRO pipeline (`bodyscan_pipeline/texturing.py`) to generate MHR/xatlas UV maps, bake `2048×2048` multi-view textures using PyTorch3D (with pure-PyTorch CPU/GPU fallback), support an optional 1m closeup front face photo, export `scan.glb` + `tex_face.jpg` + `tex_blur.jpg` + preview textured GLBs, and integrate into CLI (`run_scan.py`) and Gradio UI (`demo_ui.py`).

**Architecture:** View-independent A-pose geometry (`verts_a`, `faces`) shares exact triangle indices `0..F-1` with every per-view SAM 3D Body posed prediction (`pred_vertices`, `pred_cam_t`, `focal_length`). Step 3 builds a canonical UV atlas (`xatlas` or MHR UVs), maps each texel to `(face_id, bary_coords)`, self-calibrates camera projection sign conventions via `pred_keypoints_3d`/`2d`, tests occlusion and grazing-angle visibility per view via PyTorch3D `MeshRasterizer` (or vectorized z-buffer fallback), blends RGB textures, inpaints holes + dilates UV seams with OpenCV, blurs MHR head/face regions for `tex_blur.jpg`, and exports all privacy levels cleanly.

**Tech Stack:** Python, PyTorch, PyTorch3D (primary rasterizer, GPU/Colab), xatlas, OpenCV, NumPy, Pillow, Trimesh, Gradio.

**Spec:** `Hướng dẫn kỹ thuật - Hình 3D người thật (SAM 3D Body + MHR) cho Alpha Scan Plus.docx` (Step 3, Step 4 privacy/deletion rules, Step 5 deliverables).

## Global Constraints

- License-compliant modules ONLY: `SAM 3D Body`, `MHR`, `detectron2`, `PyTorch3D`, `trimesh`, `xatlas`, `OpenCV` — NEVER import or use `ultralytics`/`yolo`, `smplx`, `FLAME`, or `nvdiffrast`.
- Output texture resolution: `2048 × 2048` (`TEX_SIZE = 2048`).
- Privacy outputs: Mức 1 (`scan.glb` with UV, no face/body photo baked), Mức 2 (`tex_blur.jpg` blurring head/face region via MHR geometry), Mức 3 (`tex_face.jpg` clear face); plus optional preview GLBs for visual inspection and explicit opt-in input deletion audit log.
- GLB UV convention: `trimesh` exports `TextureVisuals.uv` assuming `v=0` bottom / `v=1` top (auto-flipping to `1 - v` for glTF). Texture atlas pixel coordinates MUST map `(u, v)` to `col = u * (W-1)`, `row = (1 - v) * (H-1)` so `tex_face.jpg` / `tex_blur.jpg` align upright on `scan.glb`.
- Cross-platform testability: Core math, projection, blending, inpainting, face blur, and UV packing must run locally on CPU without `pytorch3d`/`xatlas`/`trimesh` installed so local `pytest` catches regressions before Colab GPU runs.

## Review Focus

- **Camera sign mismatch across SAM 3D Body builds:** `pred_vertices` + `pred_cam_t` vs `pred_keypoints_2d` sign flip (`+Y` vs `-Y`) would paste upside-down/mirrored textures unless auto-calibrated against `pred_keypoints_2d`.
- **UV seam tearing on `xatlas`:** `xatlas.parametrize` duplicates vertices along seams (`len(uvs) > len(orig_verts)`). Exporting GLB with un-split vertices corrupts UVs; must export `verts_a[vmapping]` with `uv_faces`.
- **Grazing silhouette smearing:** Pixels near silhouette edges (`cos(theta) < 0.25`) stretch background wall color across body flanks; must strictly reject grazing angles and occluded backfaces.
- **UV gutter bleeding & seam dark lines:** Bilinear filtering in glTF viewers samples 1–2px outside UV islands; without post-inpaint UV island dilation, dark edges appear along seams.
- **Accidental deletion of local test datasets:** Auto-deletion for temporary Gradio uploads must NOT erase user's local `--image_folder` originals unless `--delete_inputs` is explicitly set.

---

### Task 1: Configuration Constants, Data Types & Head-Face Vertex Mask

**Files:**
- Modify: `bodyscan_pipeline/config.py:1-70`
- Create: `bodyscan_pipeline/texturing.py` (skeleton + dataclasses + UV/head mask helpers)
- Test: `tests/test_texturing.py`

**Interfaces:**
- Produces in `bodyscan_pipeline/config.py`: `DEFAULT_TEXTURE_SIZE = 2048`, `HEAD_JOINT_IDS`, `TEXTURE_MIN_VIEW_COS = 0.25`, `TEXTURE_ANGLE_POWER = 3.0`, `TEXTURE_DEPTH_TOLERANCE_M = 0.015`, `UV_GUTTER_DILATE_PX = 8`.
- Produces in `bodyscan_pipeline/texturing.py`: `UVLayout`, `ViewObservation`, `TexturedScanOutputs`, `build_head_face_vertex_mask(...)`.

- [ ] **Step 1: Write failing test for `build_head_face_vertex_mask` and texture config constants**
- [ ] **Step 2: Run `pytest tests/test_texturing.py -v` to confirm expected failure**
- [ ] **Step 3: Add Step 3 constants to `config.py` and dataclasses + `build_head_face_vertex_mask` to `texturing.py`**
- [ ] **Step 4: Run `pytest tests/test_texturing.py -v` to verify pass**

### Task 2: UV Layout Resolution (`xatlas` / MHR UVs) & Fast Atlas Triangle Rasterizer

**Files:**
- Modify: `bodyscan_pipeline/texturing.py`
- Test: `tests/test_texturing.py`

**Interfaces:**
- Consumes: `UVLayout`.
- Produces: `extract_mhr_uvs(...)`, `build_uv_layout(verts, faces, ...) -> UVLayout`, `rasterize_uv_atlas_map(uv_layout, tex_size=2048) -> Tuple[np.ndarray, np.ndarray]` returning `(tri_id_map, bary_map)` matching `(u, v) -> (col, (1-v) row)` OpenGL/GLB convention.

- [ ] **Step 1: Write failing test checking barycentric reconstruction on a planar UV quad/cube & GLB `(1-v)` row orientation**
- [ ] **Step 2: Run `pytest tests/test_texturing.py -v` to confirm failure**
- [ ] **Step 3: Implement `extract_mhr_uvs`, `build_uv_layout`, and `rasterize_uv_atlas_map` using vectorized OpenCV/NumPy**
- [ ] **Step 4: Run `pytest tests/test_texturing.py -v` to verify pass**

### Task 3: Self-Calibrating Camera Projection & Visibility Z-Buffer (`PyTorch3D` + CPU Fallback)

**Files:**
- Modify: `bodyscan_pipeline/texturing.py`
- Test: `tests/test_texturing.py`

**Interfaces:**
- Consumes: `ViewObservation`.
- Produces: `resolve_camera_sign(...)`, `project_points_pinhole(...)`, `render_view_depth_buffer(...)` (using `pytorch3d.renderer.MeshRasterizer` when available, pure-PyTorch/NumPy vectorized depth buffer fallback otherwise).

- [ ] **Step 1: Write failing tests on synthetic posed mesh + known rendered view testing both `+Y` and `-Y` camera conventions plus self-occlusion depth test**
- [ ] **Step 2: Run `pytest tests/test_texturing.py -v` to confirm failure**
- [ ] **Step 3: Implement `resolve_camera_sign`, `project_points_pinhole`, and `render_view_depth_buffer`**
- [ ] **Step 4: Run `pytest tests/test_texturing.py -v` to verify pass**

### Task 4: Multi-View Texture Baking, Optional Closeup Face Injection, Inpainting & Privacy Blur

**Files:**
- Modify: `bodyscan_pipeline/texturing.py`
- Test: `tests/test_texturing.py`

**Interfaces:**
- Consumes: `UVLayout`, `rasterize_uv_atlas_map`, `render_view_depth_buffer`, `build_head_face_vertex_mask`.
- Produces: `bake_multiview_texture(...)`, `inpaint_and_dilate_texture(...)`, `generate_blurred_face_texture(...)`, `run_step3_texturing(...)`.

- [ ] **Step 1: Write failing end-to-end synthetic multi-view baking test verifying front/back color recovery, occlusion handling, hole inpainting, seam padding, face blur isolation (`tex_blur` != `tex_face` only inside head region), and optional closeup face priority**
- [ ] **Step 2: Run `pytest tests/test_texturing.py -v` to confirm failure**
- [ ] **Step 3: Implement multi-view weighted baking, closeup face fusion, OpenCV `TELEA` inpaint + gutter dilation, and head-mask Gaussian/pixelate blur**
- [ ] **Step 4: Run `pytest tests/test_texturing.py -v` to verify pass**

### Task 5: GLB / Texture Export, Input Deletion Audit Log, CLI & Gradio UI Integration

**Files:**
- Modify: `bodyscan_pipeline/inference.py` (attach image dimensions / helper to collect closeup face observation cleanly)
- Modify: `bodyscan_pipeline/exporter.py` (export UV `scan.glb`, `tex_face.jpg`, `tex_blur.jpg`, optional `scan_preview_face.glb` / `scan_preview_blur.glb`, JSON `textures` & deletion log)
- Modify: `bodyscan_pipeline/run_scan.py` (add `--enable_texture`, `--face_image`, `--texture_size`, `--export_preview_glb`, `--delete_inputs`)
- Modify: `bodyscan_pipeline/demo_ui.py` (add optional closeup face upload, privacy level selector Mức 1/2/3 for 3D viewer, display `tex_face.jpg` & `tex_blur.jpg` + download artifacts + auto-clean temp uploads)
- Modify: `requirements.txt` & `README.md`
- Test: `tests/test_texturing.py` & `tests/test_measurements.py`

- [ ] **Step 1: Write failing tests for `build_body_scan_json` texture metadata, license compliance (`nvdiffrast` blocked), and deletion audit log**
- [ ] **Step 2: Run `pytest -v` to verify failure**
- [ ] **Step 3: Implement `exporter.py`, `inference.py`, `run_scan.py`, `demo_ui.py` integrations, `requirements.txt`, `README.md`**
- [ ] **Step 4: Run full test suite `PYTHONPATH=. pytest -v` and `python -m py_compile ...` and rebuild `bodyscan_pipeline_bundle.zip`**
