# Plan 2A Design Specification — Alpha Scan Plus/PRO GPU API & Alpha Scan Lab Integration

**Date:** 2026-11-15
**Status:** Ready for User Approval (Builder Check Completed & Resolved)
**Workspace Root:** `c:\Users\nvpph\Desktop\alpha scan`
**Reference Frontend Source:** `c:\Users\nvpph\Downloads\AlphaScanLab_SuaPreviewMo_14-11` -> copied to `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\`
**Technical Reference:** `Hướng dẫn kỹ thuật - Hình 3D người thật (SAM 3D Body + MHR) cho Alpha Scan Plus.docx`

---

## 1. Intent & Why

An end-to-end **Alpha Scan Plus/PRO** workflow needs to work live inside **Alpha Scan Lab** for next week's CEO demo and **internal testing only** (AI sample sets and consented internal staff). The target workflow is explicit: **provide 4 body photos -> process on GPU backend -> load back onto the web viewer with default 360° auto-rotation while keeping interactive mouse drag/zoom/angle controls, and display the 16 primary circumferences on the left panel (`#left`)**. Right now, `bodyscan_pipeline` only exposes a CLI (`run_scan.py`) and an internal Gradio tool (`demo_ui.py`), while `AlphaScanLab` (`index.html`) shows a static animated placeholder (`./assets/overview.webp`) on its Overview tab and displays *"Alpha Scan Plus/PRO (hình 3D toàn thân trên máy chủ): sắp có."* inside the measurement modal (`akBuildBodyEdit`).

At the same time, inspecting [appkhoe_netlify_1411_SONGNGU_p18 (1)/index.html](file:///c:/Users/nvpph/Downloads/appkhoe_netlify_1411_SONGNGU_p18%20(1)/index.html) and [AlphaScanLab/index.html](file:///c:/Users/nvpph/Downloads/AlphaScanLab_SuaPreviewMo_14-11/index.html) shows that the live Supabase project (`bdctqdpkogsthhfbbsse`) is hosted in **Seoul**, and under spec requirement `P0-15`, raw photos and biometric facial/body textures (`tex_blur.jpg`, `tex_face.jpg`) must never traverse or reside in Seoul—nor should internal demo prototyping alter production Supabase tables before Edge Functions and Vietnam storage are ready.

**Plan 2A** bridges this gap cleanly: build a standalone **FastAPI GPU API** around `bodyscan_pipeline` and wire **Alpha Scan Plus/PRO** directly into a workspace copy of Alpha Scan Lab (`frontend_lab/`), keeping exact schema and privacy compatibility with future Phase 2B (`P0-15` Supabase Edge Functions).

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
1. **Shared Scan Service:** Extract core pipeline orchestration out of `demo_ui.py` into a reusable `bodyscan_pipeline/scan_service.py` supporting both **GPU/SAM-3D mode** (`GpuScanEngine`) and a zero-dependency **CPU `--mock` mode** (`MockScanEngine`) for local Windows testing without GPU/weights.
2. **Async Job-Based FastAPI Backend:** Expose `POST /api/v1/bodyscan`, `GET /api/v1/bodyscan/{job_id}`, `GET /api/v1/bodyscan/{job_id}/files/{filename}`, `POST /api/v1/bodyscan/{job_id}/delete`, and `GET /health` in `bodyscan_pipeline/api_server.py`.
3. **Privacy & Compliance by Design (`P0-15` + Confirmed Internal-Test Rules):**
   - **Internal Testing Mode (Decision 1):** Default `--region-mode internal_test` tags `/health` and `body_scan` with `region_mode: "internal_test"` / `test_only: true`, so Colab tunnel or Cloud GPU can be used for AI samples and consented staff demos while displaying a clear `"THỬ NGHIỆM NỘI BỘ (P0-15)"` badge. Also supports `--region-mode vn_prod` for future Vietnam VPS lock.
   - **`LOCAL_ONLY` Loopback Guard (Decision 2a):** When intake consent is `SAFETY.consent === 'local'`, Alpha Scan Plus/PRO ONLY runs if the configured GPU Server URL targets loopback (`localhost`, `127.0.0.1`, `[::1]`). Any non-loopback URL is blocked before upload with a clear prompt to switch intake consent to server mode or use a local server.
   - Require explicit `consent_granted=true` and explicit `privacy_level` (`1`, `2`, or `3`).
   - Immediately shred uploaded source photos in a `finally:` block via `secure_delete_files` and record `deletion_audit.jsonl`.
   - Serve exported artifacts (`scan.glb`, `scan_preview_blur.glb`, `scan_preview_face.glb`, textures, JSON) only via short-lived HMAC-SHA256 signed URLs (`300s` = 5-minute TTL, refreshed when polling status).
   - **Image/3D Purge Retaining Measurements (Decision 3a):** `POST /api/v1/bodyscan/{job_id}/delete` deletes all 3D mesh and image files (`*.glb`, `*.jpg`, `*.png`) for that job immediately, keeping non-biometric `body_scan.json` + audit log (`server_deletion_audit.jsonl`), with optional `?purge_all=true` to remove the entire directory; background TTL reaper cleans expired jobs (default 24h).
4. **Workspace Copy of Alpha Scan Lab (`frontend_lab/`):** Copy `c:\Users\nvpph\Downloads\AlphaScanLab_SuaPreviewMo_14-11` cleanly into `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\`, leaving the original folder in Downloads untouched.
5. **Alpha Scan Plus/PRO Flow in Lab (4 Photos -> 360° Auto-Rotating Interactive 3D + Left-Panel 16 Circumferences):**
   - Enable **Alpha Scan Plus/PRO** alongside *Thước dây* and *Camera AI* (`vongdo.html`) inside `akBuildBodyEdit`.
   - Prioritize guided **4-angle photo upload/capture** (Front / Right / Back / Left — or 1–8 photos batch) + optional ~1 m front closeup face photo (for Privacy Mức 3), height/sex/weight pre-filled from `SAFETY.client`, configurable GPU Server URL & optional API Token (`localStorage`), and asynchronous progress polling.
   - Render the real scanned `.glb` (Level 1 plain, Level 2 blurred face, or Level 3 clear face) in Three.js **r128** directly inside `#scene` on the **Tổng quan** (`cover`) tab using an offline zero-CDN binary glTF 2.0 loader (`alpha-plus-pro.js`), **automatically starting smooth 360° rotation (`spin = true`, `#viewSeg` Spin active) while preserving mouse drag-to-orbit (`spin = false` on pointerdown), scroll zoom, and Front/Back/Rotate controls**.
   - Display all **16 Vietnamese circumferences prominently in the left panel (`#left` -> `#plus16Box`)**, auto-fill `waist`/`neck`/`hip` into `SAFETY.client` (updating US Navy estimation if InBody is absent, without overwriting `CLIENT_INBODY`), show compact 3D status/actions (`3D thật` vs `Hologram` toggle, deletion audit badge, `.glb`/`.json` download, `Xoá ảnh/3D trên server`) in `#detail` (right panel), preserve the 16 circumferences & `waist`/`neck`/`hip` in `#left` + `buildReportHTML()` PDF report when server 3D/image assets are deleted (Decision 3a).

### Non-Goals (Deferred to Plan 2B)
- Scanning external paying or trial gym members (`P0-15` requires Vietnam production storage & legal sign-off first; Plan 2A is internal testing & CEO demo only).
- Calling or modifying Supabase Seoul (`bdctqdpkogsthhfbbsse`) schemas, RPCs (`lab_submit_assessment`, `bodyscan-tao`), or storing 16 circumferences/GLBs in Supabase.
- Integrating with App KHOẺ (`appkhoe_netlify_1411_SONGNGU_p18`) client mobile views.
- Automatic multi-node S3/MinIO object storage sync in Vietnam (Plan 2A stores artifacts on the GPU server's local job directory with TTL cleanup).

### Anti-Goals
- Never import or depend on license-restricted modules (`ultralytics`, `smplx`, `nvdiffrast`).
- Never introduce external CDN scripts for 3D GLB loading (must work offline with bundled `./three.min.js` r128).
- Never persist uploaded raw photos past the inference/mock job execution window, even on error.
- Never silently pre-check consent or default privacy consent to Mức 3 without explicit selection.

---

## 3. Approaches Considered

| Option | Summary | Tradeoff | Decision |
|---|---|---|---|
| **Option 2A (Selected)** | FastAPI GPU server wrapping `bodyscan_pipeline` + copied `frontend_lab/` talking directly over HTTP + signed artifact URLs | Ready for demo/pilot next week on Windows (`--mock`), Colab tunnel, or Cloud GPU; zero changes to live Supabase Seoul; identical payload shapes drop behind `bodyscan-tao` later | **Chosen** |
| **Option 2B (Deferred)** | Full Supabase Edge Functions (`bodyscan-tao`, `bodyscan/{job_id}`, `bodyscan-xoa-anh`) + VN S3/MinIO + SQL migrations | Blocked on Supabase admin access, VN storage infra, and legal sign-off (`P0-15`); slower iteration before CEO demo | Deferred to Phase 2B |

---

## 4. Constraints

1. **License Compliance:** Only use approved packages (`FastAPI`, `uvicorn`, `python-multipart`, `SAM 3D Body`, `MHR`, `Detectron2`, `PyTorch3D`, `trimesh`, `xatlas`, `OpenCV`, `NumPy`, `SciPy`, `Pillow`, `Three.js r128`). Enforce `verify_license_compliance()` on GPU engine startup.
2. **Offline Three.js r128 Compatibility:** `frontend_lab/three.min.js` is standalone Three.js r128 (`THREE.BufferGeometry`, `THREE.MeshStandardMaterial`, `THREE.Texture`, no `GLTFLoader` bundled in `Downloads`). The glTF 2.0 `.glb` loader must work directly on global `THREE` (`window.THREE`) with zero network calls.
3. **Cross-Platform Windows Dev (`--mock` without `trimesh`):** Local Windows environment currently has `fastapi`, `uvicorn`, `python-multipart`, `httpx`, `pytest`, `scipy`, `PIL`, `cv2`, `torch`, but **not** `trimesh`. Therefore `MockScanEngine` MUST be able to synthesize valid binary `.glb` files using Python stdlib (`struct`, `json`) + `numpy`/`Pillow` when `trimesh` is not installed.
4. **InBody Integrity Rule:** Receiving `waist`, `neck`, `hip` from Alpha Scan Plus/PRO may populate `SAFETY.client` girths and recalculate US Navy estimate, but MUST NEVER overwrite `CLIENT_INBODY` real device measurements.
5. **Single-GPU Memory Safety:** Colab T4 (15 GB VRAM / 12.7 GB RAM) and single-GPU VPS cannot run concurrent SAM 3D Body jobs safely. Inference requests must execute sequentially through a single-worker queue (`max_workers=1`).
6. **File Boundaries:** Leave `c:\Users\nvpph\Downloads\AlphaScanLab_SuaPreviewMo_14-11` unmodified; all edited frontend files live in `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\`.
7. **`LOCAL_ONLY` Network Boundary (Decision 2a):** When `SAFETY.consent === 'local'`, Plus/PRO client calls may only hit loopback URLs (`localhost`, `127.0.0.1`, `[::1]`) and must reject external hosts before uploading photos.

---

## 5. System Architecture & Module Design

```
+-------------------------------------------------------------------+
| frontend_lab/ (Static Copy of AlphaScanLab + alpha-plus-pro.js)   |
|                                                                   |
|  akBuildBodyEdit ---> [Alpha Scan Plus/PRO Modal (#plusModal)]    |
|     - Pre-fills height/sex/weight/person_id from SAFETY.client    |
|     - Enforces LOCAL_ONLY loopback URL guard + explicit consent   |
|     - Guided 4-angle photos (Front/Right/Back/Left) + Level 1/2/3 |
|     - Stores GPU URL + Bearer token in localStorage('ak_gpu_cfg') |
|     - Polls job status -> fetches signed .glb & body_scan.json    |
|     - Parses glTF 2.0 GLB -> mounts realPivot on #scene (spin=360)|
|     - Displays 16 VI/EN circumferences in LEFT panel (#plus16Box) |
+------------------------------+------------------------------------+
                               | HTTPS / HTTP (CORS + Bearer/X-API-Key)
                               v
+-------------------------------------------------------------------+
| bodyscan_pipeline/api_server.py (FastAPI + Uvicorn)               |
|                                                                   |
|  - Auth middleware (optional --api-key / ALPHASCAN_API_KEY)       |
|  - Upload validation (<=8 photos, <=15MB/file, image magic bytes) |
|  - JobStore + single-worker ThreadPoolExecutor (FIFO GPU lock)    |
|  - HMAC-SHA256 signed 5-min file tokens + strict filename whitelist|
|  - Background reaper (purges expired jobs after TTL, default 24h) |
|  - Optional static mount (--serve-frontend frontend_lab at /lab)  |
+------------------------------+------------------------------------+
                               |
                               v
+-------------------------------------------------------------------+
| bodyscan_pipeline/scan_service.py (Shared Pipeline Engine)        |
|                                                                   |
|  - BODY_LABELS_VI & ANATOMICAL_ORDER (16 circumferences)          |
|  - ScanRequest / ScanResult dataclasses                           |
|  - GpuScanEngine (shared by demo_ui.py & api_server.py)           |
|  - MockScanEngine (CPU/Windows mock mesh + GLB writer + JSON)     |
|  - Always runs secure_delete_files on temp uploaded photos        |
+-------------------------------------------------------------------+
```

---

## 6. Component Specifications

### Component A: `bodyscan_pipeline/scan_service.py`

#### Data Contracts
- `ANATOMICAL_ORDER: Tuple[str, ...]`: Exact ordered tuple of the 16 circumference keys matching `measurements.py` (`"neck"`, `"chest"`, `"waist"`, `"hip"`, `"bicep_l"`, `"bicep_r"`, `"forearm_l"`, `"forearm_r"`, `"wrist_l"`, `"wrist_r"`, `"thigh_l"`, `"thigh_r"`, `"calf_l"`, `"calf_r"`, `"ankle_l"`, `"ankle_r"`).
- `BODY_LABELS_VI: Dict[str, str]` and `BODY_LABELS_EN: Dict[str, str]` defined in one place (imported by both `demo_ui.py` and `api_server.py`).
- `@dataclass class ScanRequest`:
  - `image_paths: List[str]` (1 to 8 paths on disk in temp upload directory)
  - `face_image_path: Optional[str] = None`
  - `height_cm: float = 165.0` (validated `80.0 <= height_cm <= 230.0`)
  - `sex: str = "female"` (`"male"` | `"female"`)
  - `weight_kg: float = 0.0` (`>= 0.0`, default `0.0`)
  - `person_id: str = "guest_01"`
  - `privacy_level: int = 2` (`1` | `2` | `3`)
  - `consent_granted: bool = False` (must be `True` for `api_server.py`)
  - `output_dir: str = ""`
  - `actor: str = "api_server"`
  - `save_debug_plots: bool = False`
- `@dataclass class ScanResult`:
  - `body_json: Dict[str, Any]` (exact `alpha_scan.body_scan.v2` schema from `exporter.build_body_scan_json`)
  - `artifacts: Dict[str, str]` (file paths produced in `output_dir`, e.g., `"glb"`, `"json"`, `"tex_blur"`, `"tex_face"`, `"glb_preview_blur"`, `"glb_preview_face"`, `"deletion_audit"`)
  - `viewer_artifact_key: str` (`"glb_preview_face"` if `privacy_level == 3` and generated, `"glb_preview_blur"` if `privacy_level == 2` and generated, else `"glb"`)
  - `warnings: List[str]`
  - `deletion_audit: Dict[str, Any]`
  - `raw_height_cm: float`
  - `elapsed_sec: float`
  - `mode: str` (`"gpu"` | `"mock"`)

#### `GpuScanEngine`
- Consolidates `ScanDemoEngine` logic currently in `bodyscan_pipeline/demo_ui.py`:
  - Loads SAM 3D Body + MHR once on init, caches `head_assets` (`scale_mean`, `scale_comps`, `faces`) and `mhr_model = head.mhr`.
  - `process(req: ScanRequest, progress_cb: Optional[Callable[[int, str], None]] = None) -> ScanResult`:
    1. `(10%, "inference")`: Runs `run_inference_on_images(keep_rgb=enable_texture)` and optional `run_inference_on_face_image` under `self._lock`. Calls `torch.cuda.empty_cache()` if CUDA available.
    2. `(45%, "fusion")`: Runs `fuse_multiview_params` and `build_scaled_apose_mesh`.
    3. `(65%, "measurements")`: Computes 16 circumferences, `loft`, and `validate_sanity`.
    4. `(75%, "texturing")`: If `privacy_level >= 2`, builds view observations & runs `run_step3_texturing` + `build_texture_metadata`. Falls back cleanly to Level 1 with warning if camera extrinsics unavailable.
    5. `(90%, "export")`: Calls `export_textured_scan_artifacts` (if textured) or `export_scan_artifacts`, plus optional `save_debug_plots`.
    6. `finally`: Always calls `secure_delete_files(temp_inputs, os.path.join(output_dir, "deletion_audit.jsonl"), actor=req.actor)` even on exception, clears image buffers, runs `gc.collect()`.
- Refactor `bodyscan_pipeline/demo_ui.py` to wrap `GpuScanEngine` so no pipeline orchestration logic is duplicated.

#### `MockScanEngine`
- Runs cleanly on CPU without `SAM 3D Body`, `xatlas`, `PyTorch3D`, or `trimesh`.
- Reads uploaded images (`PIL.Image.open`) just to validate they are readable images (raises `ValueError` on corrupted non-image bytes) and optionally samples median garment/skin colors for realistic preview texture tinting; still always executes `secure_delete_files` in `finally:`!
- Generates deterministic/realistic anthropometric circumferences scaled smoothly from `height_cm`, `sex`, and `weight_kg` (matching all 16 keys in `ANATOMICAL_ORDER`), plus a 57-entry `loft` (`30..86%`).
- Synthesizes an A-pose procedural human mannequin mesh (lathed torso + head + A-pose arms at ~25° + separated legs grounded at `y = 0`, scaled strictly to `height_cm / 100.0` meters) with outward normals and UVs `[0..1]`.
- Writes pure binary **glTF 2.0 `.glb`** via helper `write_minimal_glb(verts, faces, normals, uvs, texture_rgb, out_path)` (standard little-endian glTF 2.0 header `0x46546C67` version `2`, JSON chunk `0x4E4F534A` padded with spaces `0x20`, BIN chunk `0x004E4942` padded with zeros `0x00`, using `baseColorFactor` for Level 1 `scan.glb` and embedded JPEG `bufferView` for Level 2/3 textured `.glb`). Also exports `tex_blur.jpg` / `tex_face.jpg` via Pillow (explicitly visibly labeled/blurred face band for Level 2) + `body_scan.json`.
- Simulates short progress steps (configurable `step_delay_sec`, default `0.15s` per stage, `0.0s` in unit tests) so async job state transitions (`queued -> running -> completed`) are genuinely exercised.

---

### Component B: `bodyscan_pipeline/api_server.py`

#### Auth, Security & CORS
- CLI / env config:
  - `--mock` (boolean, runs `MockScanEngine` instead of `GpuScanEngine`)
  - `--sam_repo_dir`, `--checkpoint_path`, `--mhr_path` (passed to `GpuScanEngine` when not `--mock`)
  - `--host` (default `0.0.0.0`), `--port` (default `8000`)
  - `--api-key` (or env `ALPHASCAN_API_KEY`; optional for pure local dev, required if non-empty on all `/api/v1/bodyscan*` endpoints except HMAC-signed `/files/{filename}`)
  - `--region-mode` (default `internal_test`; values `internal_test` | `vn_prod` via `--region-mode` or `ALPHASCAN_REGION_MODE`)
  - `--signing-secret` (or random 32-byte process secret generated cleanly per start if omitted — configurable via env `ALPHASCAN_SIGNING_SECRET` so restarts preserve links if needed)
  - `--file-token-ttl` (default `300` seconds = 5 minutes per `P0-15`)
  - `--job-ttl-minutes` (default `1440` = 24 hours)
  - `--data-dir` (default temp root `tempfile.gettempdir()/alphascan_jobs`)
  - `--cors-origins` (default `*` when `--mock` or explicitly configured; comma-separated list supported)
  - `--serve-frontend` (optional path, e.g. `./frontend_lab`, mounted at `/lab` with SPA/static index)

#### Endpoints

1. **`GET /health`** (no auth required so Lab "Test Connection" works gracefully, or optionally indicates `auth_required: bool`):
   ```json
   {
     "ok": true,
     "mode": "mock",
     "region_mode": "internal_test",
     "test_only": true,
     "pipeline_version": "vfl-bodyscan-0.1",
     "cuda_available": false,
     "auth_required": false,
     "queue_depth": 0
   }
   ```

2. **`POST /api/v1/bodyscan`**
   - Auth check (`Bearer` or `X-API-Key` if configured) -> `401 Unauthorized` on mismatch.
   - `multipart/form-data` fields:
     - `photos`: `List[UploadFile]` (1 to 8 files required; reject `0` or `>8` with `422`; reject file `> 15 MB` with `413`; check Pillow/magic bytes or extension `.jpg/.jpeg/.png/.webp/.bmp` and reject invalid files immediately deleting any written temp files).
     - `closeup_face`: `Optional[UploadFile]` (optional, max 15 MB, only used if `privacy_level == 3`; still deleted immediately if provided!).
     - `height_cm`: `float` (`80 <= height_cm <= 230`, else `422`)
     - `sex`: `Literal["male", "female"]`
     - `weight_kg`: `Optional[float] = 0.0`
     - `person_id`: `str = "guest_01"` (sanitized for logging/JSON)
     - `privacy_level`: `int = 2` (must be `1`, `2`, or `3`, else `422`)
     - `consent_granted`: `bool` (MUST be `true`, else **`403 Forbidden`** with `{"detail": "Explicit client consent (consent_granted=true) is required."}`)
   - Creates `job_id` (UUID4 hex), writes uploaded files to `<data_dir>/<job_id>/uploads/`, creates output folder `<data_dir>/<job_id>/outputs/`, submits background task to 1-worker `ThreadPoolExecutor`, returns **`202 Accepted`**:
     ```json
     {
       "job_id": "...",
       "status": "queued",
       "progress": 0,
       "step": "queued"
     }
     ```
   - Note on photo privacy: Even if server shuts down or inference throws, `uploads/` directory is shredded via `secure_delete_files` inside the job worker's `finally:` block and logged in `outputs/deletion_audit.jsonl`.

3. **`GET /api/v1/bodyscan/{job_id}`**
   - Auth check if `--api-key` set.
   - Returns `404` if unknown or purged; returns `{ job_id, status: "deleted", deleted_at }` if deleted.
   - When `status == "completed"`, generates **fresh 5-minute (`file_token_ttl`) HMAC-SHA256 signed URLs** relative or absolute to request base URL:
     ```json
     {
       "job_id": "...",
       "status": "completed",
       "progress": 100,
       "step": "completed",
       "mode": "mock",
       "raw_height_cm": 161.4,
       "elapsed_sec": 0.48,
       "warnings": [],
       "body_scan": { "schema": "alpha_scan.body_scan.v2", "...": "..." },
       "artifacts": {
         "mesh_url": "/api/v1/bodyscan/.../files/scan_preview_blur.glb?exp=...&sig=...",
         "glb_url": "/api/v1/bodyscan/.../files/scan.glb?exp=...&sig=...",
         "json_url": "/api/v1/bodyscan/.../files/body_scan.json?exp=...&sig=...",
         "texture_url": "/api/v1/bodyscan/.../files/tex_blur.jpg?exp=...&sig=..."
       },
       "deletion_audit": {
         "timestamp_utc": "...",
         "actor": "api_server",
         "deleted_count": 4,
         "deleted_files": ["01.jpg", "02.jpg", "03.jpg", "04.jpg"]
       },
       "created_at": "...",
       "expires_at": "..."
     }
     ```
   - Privacy scoping: If `privacy_level == 1`, `artifacts` ONLY exposes `mesh_url` -> `scan.glb`, `glb_url` -> `scan.glb`, `json_url` -> `body_scan.json` (no `texture_url`). If `privacy_level == 2`, `mesh_url` points to `scan_preview_blur.glb` (fallback `scan.glb` if texturing fell back) and NEVER exposes `tex_face.jpg` or `scan_preview_face.glb` (404 on `/files/tex_face.jpg` because privacy level 2 never generated/allows it!).

4. **`GET /api/v1/bodyscan/{job_id}/files/{filename}?exp=<ts>&sig=<hex>`**
   - Validates `exp >= now` (else `403 "Signed URL expired"`) and constant-time HMAC-SHA256 over `f"{job_id}:{filename}:{exp}"` (else `403 "Invalid file signature"`).
   - Validates `filename` against strict whitelist (`ALLOWED_ARTIFACT_FILES`) and privacy level allowed for that job (blocking path traversal and blocking clear-face files when `privacy_level < 3`).
   - Returns file with accurate MIME type (`model/gltf-binary` for `.glb`, `image/jpeg` for `.jpg`, `application/json` for `.json`).

5. **`POST /api/v1/bodyscan/{job_id}/delete`** (also alias `DELETE /api/v1/bodyscan/{job_id}`)
   - Auth check if `--api-key` set.
   - Optional query param `purge_all: bool = False`.
   - **Default (`purge_all=false`, Decision 3a matching `P0-15` `bodyscan-xoa-anh`):** deletes all 3D mesh and image files under `<data_dir>/<job_id>/` (`*.glb`, `*.jpg`, `*.png`), preserving non-biometric `body_scan.json` and `deletion_audit.jsonl`; marks `assets_deleted = true`, `status = "deleted"`; appends `"manual_delete_assets"` record to `server_deletion_audit.jsonl`. Subsequent `GET /files/{filename}` requests for `.glb`/`.jpg`/`.png` return **`410 Gone`**.
   - **Full purge (`purge_all=true`):** removes the entire `<data_dir>/<job_id>/` directory and logs `"manual_purge_all"`.
   - Response:
     ```json
     {
       "ok": true,
       "job_id": "...",
       "status": "deleted",
       "assets_deleted": true,
       "measurements_retained": true,
       "deleted_files": ["scan.glb", "scan_preview_blur.glb", "tex_blur.jpg"],
       "deleted_at": "..."
     }
     ```

6. **Background Reaper Thread**
   - Daemon thread waking every 5 minutes (configurable for tests) removing job directories whose `expires_at < now`, recording auto-expiry in `server_deletion_audit.jsonl`.

---

### Component C: `frontend_lab/` (Alpha Scan Lab Copy + `alpha-plus-pro.js`)

#### Files Modified / Added in `frontend_lab/`
- Copy all assets & scripts from `c:\Users\nvpph\Downloads\AlphaScanLab_SuaPreviewMo_14-11\` into `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\`.
- **Create:** `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\alpha-plus-pro.js`
- **Modify:** `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\index.html` (minimal targeted edits):
  1. Add `<script src="./alpha-plus-pro.js"></script>` after `<script src="./three.min.js">` (or before closing `</body>`).
  2. Activate the 3rd card/button in `akBuildBodyEdit()` (`#be_m_plus`) cleanly right inside the "Cách lấy số đo / How to get the measurements" section (replacing the gray `"sắp có / coming soon"` note with a real interactive button: **"Alpha Scan Plus/PRO"** — *"3D người thật + 16 vòng đo (Server GPU)"*).
  3. Add `<div id="plus16Box"></div>` inside the left panel (`#left .pad`, directly beneath `#metricsBox`) so the **16 Vietnamese/English circumferences** render prominently on the **left side of the screen** regardless of whether `#metricsBox` is showing `renderLeftEmpty()`, `renderLeftReal()`, or `renderLeft()` (keeping `akUpdateTiles()` isolated from circumference rows).
  4. Extend `init()` / `animate()` / `applyAppearance()` / `frameModel()` / `setView()` / `renderLeft()` / `girthSrcLine()` / `buildReportHTML()` cleanly via well-defined hooks on `window.AlphaPlusPro` (avoiding brittle rewrites of existing Anatomy/InBody/Posture code):
     - Keep a dedicated `THREE.Group` `realPivot` attached directly to `scene` at `(0, 0, 0)`.
     - In `animate()`: rotate `realPivot.rotation.y = ry; realPivot.rotation.x = rx;` in sync with `root`. Because `pointerdown` sets `spin = false` and `pointermove`/`wheel` update `ry`/`rx`/`dist`, mouse orbit/pitch/zoom automatically override spin seamlessly!
     - On scan completion: switch `appearance = 'cover'`, activate **360° auto-rotation by default** (`spin = true`, mark `#viewSeg button[data-view=spin]` active `.on`, keep Front `Trước` and Back `Sau` buttons functional on `realPivot` via `#viewSeg` click handler), hide `#coverImg` (`overview.webp`) & `sysGroups`, show `realPivot`, and call `frameRealModel()` so the scanned body fills `#scene`.
     - Update `.hint` text dynamically when real 3D is active: *"Đang tự xoay 360° · kéo chuột để chỉnh góc nhìn · lăn chuột để phóng to/thu nhỏ"*.
     - Inside `#detail` (right panel) when `appearance === 'cover'` and `AlphaPlusPro.hasMesh()`: keep the right panel clean with a compact **3D scan status & action card** (privacy level, `✓ Đã xoá X ảnh gốc trên server`, `TEST_ONLY` internal-test badge if server reports `test_only`, toggle `Hình 3D thật (Plus/PRO)` vs `Hologram mẫu`, download `.glb` / `.json`, and button **`Xoá hình 3D/ảnh trên server`**).
     - **Decision 3a Delete Behavior in Lab:** Clicking **`Xoá hình 3D/ảnh trên server`** calls `POST /api/v1/bodyscan/{job_id}/delete` (`purge_all=false`), disposes `realPivot` WebGL resources, switches `#scene` (`cover` tab) back to `./assets/overview.webp` hologram, updates `#detail` badge to *"✓ Đã xoá hình 3D & ảnh trên server (vẫn giữ 16 số đo)"*, and **keeps `#plus16Box` (16 circumferences in `#left`), `SAFETY.client.waist`/`neck`/`hip`, US Navy calculation, and the 16-measurement PDF table**.
     - **Tabs Isolation:** Switching to **Giải phẫu thật**, **Theo dữ liệu**, or **Tư thế** hides `realPivot`, turns `spin = false`, and displays `BodyParts3D` untouched.
     - In `startNewPerson()` and `akEnterWithPerson()`: call `window.AlphaPlusPro.resetSession()` so the previous client's 3D mesh & 16 circumferences never leak to the next client!
     - In `akBuildPanel()` ("Kết nối" settings drawer): add a collapsible/compact section **3.5 · Server 3D Alpha Scan Plus/PRO** (Server URL + optional API Key + Test `/health`) synced with `localStorage('ak_gpu_cfg')`, while ALSO allowing quick server config directly inside `#plusModal`.
     - In `buildReportHTML()`: if `window.AlphaPlusPro.getResult()` exists (even after Decision 3a server mesh deletion), include section **"Số đo 16 vòng cơ thể (Alpha Scan Plus/PRO 3D)"** in the printable PDF report; capture 3D real-body snapshot when `realPivot` is currently active, otherwise capture anatomy canvas as before.
     - Add EN translations in `I18N.en` for all new keys so `toggleLang()` switches `#plusModal`, `#plus16Box`, and `#detail` cleanly between Vietnamese and English!

#### Pure-JS Three.js r128 glTF 2.0 Binary `.glb` Loader (`alpha-plus-pro.js`)
Let's make sure the `.glb` loader handles real `trimesh` exports AND mock `.glb` exports cleanly without external libraries:
- Validate 12-byte glTF header: `magic === 0x46546C67` (`"glTF"`), `version === 2`, parse chunks cleanly (`0x4E4F534A` JSON + `0x004E4942` BIN).
- Helper `readAccessor(gltf, binView, accessorIdx)` supporting componentTypes `5120 (BYTE)`, `5121 (UNSIGNED_BYTE)`, `5122 (SHORT)`, `5123 (UNSIGNED_SHORT)`, `5125 (UNSIGNED_INT)`, `5126 (FLOAT)` and `type` (`SCALAR`, `VEC2`, `VEC3`, `VEC4`), `normalized` flag, and `bufferView.byteStride`.
- For each scene/node/mesh/primitive:
  - Read `POSITION` (`VEC3`), optional `NORMAL` (else `geometry.computeVertexNormals()`), optional `TEXCOORD_0` (`VEC2`), optional `indices`.
  - Read material `pbrMetallicRoughness`:
    - `baseColorFactor` (default `[1, 1, 1, 1]`), `metallicFactor` (`0.0` default for skin/garment matte look instead of default shiny `1.0`), `roughnessFactor` (default `0.78` for natural lighting on beige canvas).
    - If `baseColorTexture` present: resolve `textures[idx].source -> images[imgIdx]` (supporting BOTH embedded `bufferView` + `mimeType` AND `data:` URI -> create `Blob` -> `Image.decode()` / `onload` -> `THREE.Texture(img)` with `tex.flipY = false`, `tex.encoding = THREE.sRGBEncoding` so colors match photo brightness on Three.js r128!).
    - Construct `THREE.MeshStandardMaterial({ color, map, roughness, metalness })`, create `THREE.Mesh`, apply node TRS/matrix if present.
- Auto-center the loaded `THREE.Group` horizontally on `X/Z` and center vertically on `Y` around `0` (`Box3().setFromObject(grp)`), compute exact `hY` and `hX` for `frameModel()`, and dispose previous `BufferGeometry` / `Material` / `Texture` whenever replacing or resetting a scan so repeated scans don't leak GPU WebGL memory!

#### Modal `#plusModal` UX & Guards
- Opens from `#be_m_plus` inside `akBuildBodyEdit` (or closes `#bodyEditModal` cleanly and opens `#plusModal` on top; closing `#plusModal` can reopen or refresh cleanly).
- Checks `SAFETY.client` (`sex` + `height` required; prompts inline if missing, allows editing `height_cm` & `weight_kg` right there).
- **Photo Inputs (Guided 4-Angle Workflow):**
  - Displays **4 guided angle slots** matching the target capture workflow: **1. Chính diện (Front)**, **2. Nghiêng phải (Right)**, **3. Phía sau (Back)**, **4. Nghiêng trái (Left)** — plus a convenient **"Chọn nhanh 4–8 ảnh cùng lúc"** multi-file input (auto-filling empty slots in order) and drag/drop, with thumbnail previews, remove buttons, and ability to attach up to 8 body photos total.
  - Optional ~1 m front closeup face `<input type="file" accept="image/*">` unlocked visibly when Mức 3 is selected.
- **`LOCAL_ONLY` Loopback Guard (Decision 2a):**
  - Helper `isLoopbackUrl(urlStr)` checks hostname against `localhost`, `127.0.0.1`, `[::1]`.
  - If `SAFETY.consent === 'local'` and the configured GPU Server URL is non-loopback: display a clear inline warning inside `#plusModal` (*"Chế độ LOCAL_ONLY chỉ cho phép gửi ảnh về máy chủ trên chính thiết bị này (`localhost` / `127.0.0.1`). Hãy đổi URL về localhost hoặc chuyển đồng thuận khách sang Lưu hồ sơ ở cổng đầu."*), and disable the Submit button until resolved.
- **Consent & Privacy Selector (`P0-15` strict):**
  - 3 distinct radio cards (**never pre-selected on fresh modal if client hasn't chosen yet, or requires deliberate choice + checkbox** — let's require selecting 1 of 3 levels AND ticking an unchecked consent checkbox *"Khách hàng đã được giải thích và đồng ý dựng hình 3D theo mức đã chọn; ảnh gốc sẽ xoá ngay sau khi xử lý"* before the Submit button enables!).
    - **Mức 1:** Lưới 3D trơn + 16 số đo *(nhanh nhất, không dán ảnh)*
    - **Mức 2:** Hình 3D dán ảnh thật + Làm mờ khuôn mặt (`tex_blur.jpg`) *(khuyên dùng)*
    - **Mức 3:** Hình 3D dán ảnh thật + Giữ rõ khuôn mặt (`tex_face.jpg`)
- **Server Config Drawer inside `#plusModal` (collapsed by default once saved, auto-expanded if unconfigured):**
  - Default GPU Server URL smartly initialized: if `location.protocol.startsWith('http')` and hosted on same server (e.g., opened via `http://localhost:8000/lab`), pre-fill `location.origin`, else `http://127.0.0.1:8000`; saved in `localStorage('ak_gpu_cfg')`.
  - Optional Bearer/API Token field.
  - "Kiểm tra kết nối" button (`GET /health`) showing green pill `● Sẵn sàng (GPU)` or `● Sẵn sàng (Mock test)` or clear troubleshooting hint if CORS/Mixed-Content/Offline!
  - Helpful hint if Lab is open on `https://` Netlify while URL is `http://localhost`: note Cloudflare Tunnel `https://...trycloudflare.com` requirement for remote HTTPS -> local HTTP!
- **Result Application:**
  - Automatically injects `waist`, `neck`, `hip` (if present in `circumferences_cm`) into `SAFETY.client` (preserving `weight`/`height` edits made in modal), sets `SAFETY.client.girth_src` entries to `'plus_pro'`, stores `SAFETY.client.alpha_plus_scan` (without keeping photo binaries), renders `#plus16Box` in the **left panel (`#left`)**, updates US Navy preview via `renderLeft()`, updates `updateClientHeader()`, switches to `appearance = 'cover'`, enables **default 360° auto-rotation (`spin = true`, `#viewSeg` `Xoay` active, mouse drag/zoom still interactive)**, and displays `#detail` controls in the right panel!

---

## 7. Review & Testing Plan

1. **Backend Unit & API Integration Tests (`tests/test_api_server.py`):**
   - `MockScanEngine` produces valid 16 circumferences, `loft` length `57` (`30..86%`), valid glTF 2.0 `.glb` headers & JSON/BIN chunks for all 3 privacy levels (`1`, `2`, `3`), and `tex_blur.jpg` / `tex_face.jpg` files.
   - `POST /api/v1/bodyscan` rejects `consent_granted=false` (`403`), invalid height (`422`), `>8` photos (`422`), invalid non-image payload (`400`/`422`), and wrong API key (`401`).
   - Happy path `POST -> GET poll -> completed`: verifies uploaded temp photo files are **deleted on disk** and logged in `deletion_audit.jsonl`, and verifies `test_only: true` / `region_mode: "internal_test"`.
   - Privacy isolation test: Level 1 & Level 2 jobs reject requests for `tex_face.jpg` or `scan_preview_face.glb` (`404`/`403`); Level 2 exposes only blurred assets.
   - Signed URL verification test: expired `exp` or tampered `sig` returns `403`, path traversal (`../`) rejected (`400`/`404`).
   - Manual delete test (`Decision 3a`): `POST /api/v1/bodyscan/{job_id}/delete` deletes `*.glb`/`*.jpg`/`*.png` from disk while retaining `body_scan.json`, and subsequent `GET /files/*.glb` or `*.jpg` returns `410 Gone` (`?purge_all=true` erases the entire folder -> `404`/`410`).
   - Regression test: existing `tests/test_texturing.py` remains 11/11 passing (`demo_ui.py` refactoring preserves `ScanDemoEngine` contract).
2. **Frontend glTF Loader & UI Hook Verification:**
   - Binary `.glb` generated by `MockScanEngine` parses cleanly (automated Node/Python verification of glTF 2.0 JSON chunk + accessor bounds + embedded image MIME).
   - Live browser verification with `fastapi --mock --serve-frontend frontend_lab` at `http://127.0.0.1:8000/lab/`:
     - Check `LOCAL_ONLY` loopback guard (`Decision 2a`: remote URL blocked with clear prompt; loopback URL allowed).
     - Complete 4-photo upload -> run Level 2 scan -> verify real 3D mesh renders on `#scene` and **rotates 360° automatically (`spin = true`)**, verify mouse drag stops spin and orbits manually + scroll zooms + `Trước/Sau/Xoay` buttons work.
     - Verify all **16 circumferences appear cleanly in the left panel (`#left` / `#plus16Box`)** and `waist`/`neck`/`hip` update US Navy without overwriting InBody.
     - Click `Xoá ảnh/3D trên server` (`Decision 3a`): verify `#scene` returns to `./assets/overview.webp` hologram, while `#plus16Box` in `#left`, `SAFETY.client` girths, and `buildReportHTML()` 16-measurement PDF table remain intact!
     - Verify `VI/EN` toggle + zero console errors.

---

## 8. Confirmed Decisions & Reasonable Defaults Settled

### User-Confirmed Decisions
- **Scope (`Decision 1`):** Internal testing & CEO demo only (`AI samples` + consented staff); server runs in `--region-mode internal_test` (`test_only: true`, banner displayed in UI) so Colab/Cloud GPU tunnels work next week without external gym member traffic.
- **`LOCAL_ONLY` Behavior (`Decision 2a`):** Loopback GPU server URLs (`localhost`, `127.0.0.1`, `[::1]`) are allowed in `LOCAL_ONLY` mode; remote URLs are blocked before upload.
- **Delete Semantics (`Decision 3a`):** Deleting server assets (`POST .../delete`) erases 3D meshes and image/texture files (`*.glb`, `*.jpg`, `*.png`), resets `#scene` back to the static hologram, and **keeps the 16 circumferences + `waist`/`neck`/`hip` + PDF table**.
- **Workflow & Screen Layout:** Guided 4-photo upload -> backend processing -> real 3D model returned to web viewer **auto-rotating 360° with interactive mouse drag/zoom/angle control**, and **16 primary circumferences displayed on the left panel (`#left`)**.

### Engineering Calls Settled for Builder (Please Review)
- **API Route Naming:** `/api/v1/bodyscan`, `/api/v1/bodyscan/{job_id}`, `/api/v1/bodyscan/{job_id}/files/{filename}`, `/api/v1/bodyscan/{job_id}/delete`, `/health` — mirrors future Phase 2B Edge Functions (`bodyscan-tao`, `bodyscan/{job_id}`, `bodyscan-xoa-anh`).
- **Sequential Single-Worker Queue (`max_workers=1`):** Prevents GPU OOM on 15 GB Colab T4; status shows queue/stage progress cleanly.
- **Upload & Token Limits:** Max `8` body photos + `1` closeup face photo, max `15 MB` per file, short-lived `300s` (5-min) HMAC file tokens refreshed automatically by status polling, job artifacts auto-purged after `24h` by default (`--job-ttl-minutes 1440`).
- **Pure-JS glTF 2.0 `.glb` Loader (`alpha-plus-pro.js`):** Written directly against local `Three.js r128` (`window.THREE`) so `frontend_lab/` works 100% offline with zero external CDN scripts; uses `tex.flipY = false` and `THREE.sRGBEncoding` matching `trimesh` glTF exports.
- **Auto-fill Behavior on `SAFETY.client`:** Fills `waist`, `neck`, `hip` (tagged `'plus_pro'`) and updates `height`/`weight` if edited in `#plusModal`; never overwrites `CLIENT_INBODY`.
- **Single-Origin Convenience Flag:** `--serve-frontend frontend_lab` mounts `frontend_lab/` at `/lab` on FastAPI so Windows `--mock` and Colab tunnel testing work cleanly with zero CORS/mixed-content hurdles. Also add `fastapi`, `uvicorn[standard]`, `python-multipart` to `requirements.txt`.
