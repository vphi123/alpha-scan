# Plan 2A Implementation Plan — Alpha Scan Plus/PRO GPU API & Alpha Scan Lab Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking progress. Note: `c:\Users\nvpph\Desktop\alpha scan` is not a git repository (`NOT_A_GIT_REPO`), so verification relies on automated test commands rather than git commits.

**Goal:** Deliver end-to-end **Plan 2A** integrating the Alpha Scan Plus/PRO Python pipeline (`bodyscan_pipeline`) with an isolated workspace copy of Alpha Scan Lab (`frontend_lab/`), supporting guided 4-angle photo upload, FastAPI GPU/mock job processing, a **360° auto-rotating + mouse-interactive real `.glb` 3D body viewer**, **16 Vietnamese/English body circumferences displayed on the left panel (`#left`)**, and strict internal-test privacy & deletion controls.

**Architecture:**
- **Shared Core (`bodyscan_pipeline/scan_service.py`):** Single source of truth for the 16-key anatomical schema (`ANATOMICAL_ORDER`, `BODY_LABELS_VI`, `BODY_LABELS_EN`, `PRIVACY_CHOICES`), `GpuScanEngine` (refactoring `ScanDemoEngine` out of `demo_ui.py` without duplication), `MockScanEngine` (pure-Python/Pillow procedural A-pose mannequin glTF 2.0 `.glb` writer working natively on CPU/Windows even without `trimesh`), and guaranteed `finally:` shredding of uploaded raw photos via `exporter.secure_delete_files`.
- **Async HTTP API (`bodyscan_pipeline/api_server.py`):** FastAPI server with `--mock` / `--region-mode internal_test|vn_prod` / `--serve-frontend`, `ThreadPoolExecutor(max_workers=1)` FIFO queue preventing Colab T4 OOM, consent/validation guards (`403`/`422`/`413`/`401`), short-lived HMAC-SHA256 signed artifact tokens (`300s` default), strict privacy scoping (Levels 1/2/3), `POST .../delete` (`purge_all=false` Decision 3a keeping `body_scan.json` while returning `410 Gone` for erased binaries), and background TTL reaper.
- **Lab Frontend (`frontend_lab/` + `frontend_lab/alpha-plus-pro.js` + `frontend_lab/index.html` hooks):** Workspace copy of `AlphaScanLab_SuaPreviewMo_14-11` paired with a self-contained Three.js r128 pure-JS binary glTF 2.0 `.glb` parser (`tex.flipY = false`, `THREE.sRGBEncoding`, GPU buffer disposal), guided 4-angle upload modal (`#plusModal`) enforcing `LOCAL_ONLY` loopback-only URLs (`Decision 2a`), `realPivot` mounted on `scene` with default 360° spin + mouse drag/zoom + Front/Back/Rotate toggle, left-panel `#plus16Box` table, US Navy autofill without overwriting `CLIENT_INBODY`, printable 16-measurement PDF section, `startNewPerson` reset, and EN/VI i18n parity.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, `python-multipart`, `httpx`, Pillow, NumPy, PyTorch, Trimesh (optional in mock mode), Pytest, Node.js `v24+` (`node:test`), Vanilla ES2020 JS + local Three.js r128 (`frontend_lab/three.min.js`).

**Spec:** [docs/superpowers/specs/2026-11-15-plan-2a-gpu-api-lab-design.md](file:///c:/Users/nvpph/Desktop/alpha%20scan/docs/superpowers/specs/2026-11-15-plan-2a-gpu-api-lab-design.md)

**Global Constraints:**
- Never modify anything inside `c:\Users\nvpph\Downloads\AlphaScanLab_SuaPreviewMo_14-11` or query/mutate live Supabase production project `bdctqdpkogsthhfbbsse`.
- Do not add external CDN `<script>` tags to `frontend_lab/` (`three.min.js` r128 + `alpha-plus-pro.js` must run 100% offline / single-origin).
- Keep all Plan 2A defaults in internal-test posture (`--region-mode internal_test`, `test_only: true`).
- Existing `tests/test_measurements.py` and `tests/test_texturing.py` must stay 100% green after all edits.

**Review Focus:**
1. **Consent, `LOCAL_ONLY`, and Zero Photo Retention:** Verify `POST /api/v1/bodyscan` returns `403` when `consent_granted=false`, `#plusModal` blocks non-loopback GPU URLs under `SAFETY.consent === 'local'` (`Decision 2a`), and uploaded raw photos in `uploads/` are shredded via `secure_delete_files` in `finally:` even when inference raises (`tests/test_scan_service.py`, `tests/test_api_server.py`, `tests/test_frontend_plus_pro.mjs`).
2. **Privacy Level Isolation:** Verify Level 1 exposes only `scan.glb` + `body_scan.json`, Level 2 exposes only blurred `scan_preview_blur.glb` + `tex_blur.jpg`, and valid-signature requests for `tex_face.jpg` or `scan_preview_face.glb` under Level 1/2 are rejected (`404`) (`tests/test_api_server.py`).
3. **Signed File Security & Decision 3a `410 Gone`:** Verify expired `exp`, tampered HMAC `sig`, and traversal paths fail (`403`/`404`), and `POST .../delete` (`purge_all=false`) erases `*.glb`/`*.jpg`/`*.png` while preserving `body_scan.json` and returning `410 Gone` on subsequent asset downloads (`tests/test_api_server.py`).
4. **Pure-JS `.glb` Parsing, Orientation & Spin/Mouse Interactivity:** Verify `MockScanEngine` `.glb` parses cleanly in Three.js r128 (`flipY === false`, `THREE.sRGBEncoding`, valid vertex/normal/UV bounds, resource disposal), and live Overview shows `realPivot` auto-spinning 360° (`spin=true`) while stopping spin on pointer drag + supporting Front/Back/Rotate (`tests/test_frontend_plus_pro.mjs` + Task 5 live check).
5. **Client State Isolation & Decision 3a Retention in Lab UI:** Verify server deletion returns `#scene` to `./assets/overview.webp` while keeping `#plus16Box` (16 girths), `SAFETY.client` (`'plus_pro'`), US Navy, and PDF report rows intact, never overwriting `CLIENT_INBODY`, and clearing cleanly on `startNewPerson` (`tests/test_frontend_plus_pro.mjs` + Task 5 live check).

---

### Task 1: Shared Pipeline Core (`scan_service.py`), Pure Binary `.glb` Writer & `demo_ui.py` Refactor

**Files:**
- Create: `c:\Users\nvpph\Desktop\alpha scan\tests\test_scan_service.py`
- Create: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\scan_service.py`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\demo_ui.py`

**Interfaces:**
- Constants in `bodyscan_pipeline/scan_service.py`:
  - `ANATOMICAL_ORDER: Tuple[str, ...]` (16 ordered keys matching `measurements.py`: `"neck"`, `"chest"`, `"waist"`, `"hip"`, `"bicep_l"`, `"bicep_r"`, `"forearm_l"`, `"forearm_r"`, `"wrist_l"`, `"wrist_r"`, `"thigh_l"`, `"thigh_r"`, `"calf_l"`, `"calf_r"`, `"ankle_l"`, `"ankle_r"`)
  - `BODY_LABELS_VI: Dict[str, str]`, `BODY_LABELS_EN: Dict[str, str]`
  - `PRIVACY_CHOICES: Tuple[str, ...]`, `PRIVACY_LEVEL_MAP: Dict[str, int]`
- Dataclasses in `bodyscan_pipeline/scan_service.py`:
  - `ScanRequest(image_paths, face_image_path=None, height_cm=165.0, sex="female", weight_kg=0.0, person_id="guest_01", privacy_level=2, consent_granted=False, output_dir="", actor="api_server", save_debug_plots=False)`
  - `ScanResult(body_json, artifacts, viewer_artifact_key, warnings, deletion_audit, raw_height_cm, elapsed_sec, mode)`
- Functions & Classes:
  - `write_minimal_glb(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, uvs: Optional[np.ndarray], texture_jpeg_bytes: Optional[bytes], out_path: str, base_color_rgba: Tuple[float, float, float, float] = (0.76, 0.72, 0.66, 1.0)) -> str`
  - `BaseScanEngine`: `process(req: ScanRequest, progress_cb: Optional[Callable[[int, str], None]] = None) -> ScanResult`
  - `GpuScanEngine(BaseScanEngine)`: wraps SAM 3D Body + MHR + `fusion` + `measurements` + `texturing` + `exporter` with thread `Lock` and `finally:` `secure_delete_files` cleanup
  - `MockScanEngine(BaseScanEngine)`: validates inputs & image readability via Pillow, builds A-pose mannequin mesh (`y >= 0`, height `== height_cm / 100.0`), computes 16 realistic girths & 57 `loft` rows, emits glTF 2.0 `.glb` + `tex_blur.jpg`/`tex_face.jpg` + `body_scan.json`, always runs `secure_delete_files` in `finally:`
- `bodyscan_pipeline/demo_ui.py`: imports `ANATOMICAL_ORDER`, `BODY_LABELS_VI`, `PRIVACY_CHOICES`, `PRIVACY_LEVEL_MAP`, `GpuScanEngine`, `ScanRequest` from `scan_service`; `ScanDemoEngine` keeps identical public `__init__` and `process(...)` tuple return contract.

**Steps:**
- [ ] **Step 1 (RED):** Create `tests/test_scan_service.py` covering:
  - `ANATOMICAL_ORDER` length == 16 and exact key parity with `BODY_LABELS_VI` and `BODY_LABELS_EN`.
  - `write_minimal_glb` creates valid glTF 2.0 binary (`magic == 0x46546C67`, `version == 2`, padded JSON/BIN chunk alignment `chunk_len % 4 == 0`, valid JSON `accessors`/`POSITION` min-max, embedded `image/jpeg` bufferView when `texture_jpeg_bytes` supplied).
  - `MockScanEngine.process` across Privacy Levels `1`, `2`, `3`: verifies `body_json["schema"] == SCHEMA_VERSION`, all 16 circumferences positive, `len(loft) == 57`, `viewer_artifact_key` equals `"glb"` (L1), `"glb_preview_blur"` (L2), `"glb_preview_face"` (L3), Level 2 does not write `tex_face.jpg` or `scan_preview_face.glb`, and progress callback receives strictly increasing percentages ending at Export/Complete.
  - Cleanup on error: calling `MockScanEngine.process` with corrupted non-image file raises `ValueError` AND still runs `secure_delete_files` in `finally:`, removing the temp upload file and appending `deletion_audit.jsonl`.
  - `demo_ui.ScanDemoEngine` still imports cleanly without circular errors and preserves its `PRIVACY_CHOICES` / `PRIVACY_LEVEL_MAP` attributes.
- [ ] **Step 2:** Run `python -m pytest tests/test_scan_service.py -v` and confirm failure (`ModuleNotFoundError: No module named 'bodyscan_pipeline.scan_service'`).
- [ ] **Step 3 (GREEN):** Implement `bodyscan_pipeline/scan_service.py`:
  - Keep pure-Python `struct` + `json` `write_minimal_glb` (no `trimesh` dependency so `--mock` works anywhere).
  - Implement procedural A-pose humanoid synthesizer (`_build_mock_mannequin`) with lathed torso/head rings, abducted arms (~25°), separated legs, clean triangle winding, outward vertex normals, and cylindrical UV coordinates `[0..1]`, plus Pillow texture generator (`_render_mock_texture`) that draws visible skin/garment shading and clearly blurs the face UV band (`y` near head) on `tex_blur.jpg` (`privacy_level == 2`).
  - Factor shared cleanup & validation helpers (`_validate_request`, `_cleanup_temp_inputs`) cleanly for both `GpuScanEngine` and `MockScanEngine`.
- [ ] **Step 4:** Refactor `bodyscan_pipeline/demo_ui.py` to delegate core inference/fusion/measurement/texturing/export to `GpuScanEngine` and import shared constants from `scan_service.py`, eliminating duplicated pipeline code.
- [ ] **Step 5:** Run `python -m pytest tests/test_scan_service.py tests/test_measurements.py tests/test_texturing.py -v` and verify all tests pass with 0 failures/skips.

---

### Task 2: Async FastAPI Server (`api_server.py`), Signed Artifacts, Privacy Isolation & TTL Reaper

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\requirements.txt`
- Create: `c:\Users\nvpph\Desktop\alpha scan\tests\test_api_server.py`
- Create: `c:\Users\nvpph\Desktop\alpha scan\bodyscan_pipeline\api_server.py`

**Interfaces:**
- Factory & App in `bodyscan_pipeline/api_server.py`:
  - `ServerConfig`: `mock: bool = False`, `region_mode: str = "internal_test"`, `api_key: str = ""`, `signing_secret: str = ""`, `file_token_ttl: int = 300`, `job_ttl_minutes: int = 1440`, `reaper_interval_sec: int = 300`, `data_dir: str = ...`, `cors_origins: List[str] = ["*"]`, `serve_frontend: Optional[str] = None`, `sam_repo_dir: str = ""`, `checkpoint_path: str = ""`, `mhr_path: str = ""`
  - `create_app(config: ServerConfig, engine: Optional[BaseScanEngine] = None) -> FastAPI`
  - Endpoints:
    - `GET /health` -> `{ok, mode, region_mode, test_only, pipeline_version, cuda_available, auth_required, queue_depth}`
    - `POST /api/v1/bodyscan` (multipart: `photos`, optional `closeup_face`, `height_cm`, `sex`, `weight_kg`, `person_id`, `privacy_level`, `consent_granted`) -> `202 {job_id, status: "queued", progress: 0, step: "queued"}`
    - `GET /api/v1/bodyscan/{job_id}` -> fresh HMAC-signed URLs (`exp`, `sig`) scoped strictly by job `privacy_level`
    - `GET /api/v1/bodyscan/{job_id}/files/{filename}?exp=...&sig=...` -> MIME-typed artifact download (`403` expired/bad sig, `404` unknown/blocked by privacy, `410` after Decision 3a delete)
    - `POST /api/v1/bodyscan/{job_id}/delete` & `DELETE /api/v1/bodyscan/{job_id}` (query `purge_all: bool = False`) -> deletes binary assets (`Decision 3a`, preserving `body_scan.json`) or full folder (`purge_all=true`), logs `server_deletion_audit.jsonl`
    - Optional `/lab` static mount when `--serve-frontend` points to an existing directory
  - CLI entrypoint `main(argv=None)` (`python -m bodyscan_pipeline.api_server --mock ...`)

**Steps:**
- [ ] **Step 1:** Update `requirements.txt` appending FastAPI runtime deps (`fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`, `python-multipart>=0.0.9`, `httpx>=0.27.0`).
- [ ] **Step 2 (RED):** Create `tests/test_api_server.py` using FastAPI `TestClient` + `MockScanEngine(step_delay_sec=0.0)` testing:
  - `GET /health`: confirms `region_mode == "internal_test"`, `test_only is True`, `auth_required` reflects `--api-key`, `mode == "mock"`.
  - Auth guard: with `api_key="secret123"`, `GET /health` stays open (`200`), while `POST /api/v1/bodyscan`, `GET /api/v1/bodyscan/{id}`, `POST .../delete` reject missing/wrong key with `401` and accept `Authorization: Bearer secret123` or `X-API-Key: secret123`.
  - Input & Consent guards: `consent_granted=false` -> `403`; invalid `height_cm` (`50` or `250`) -> `422`; `0` or `9` photos -> `422`; photo `> 15 MB` -> `413`; non-image garbage bytes -> `422` AND temp upload folder on disk contains zero leftover files.
  - Async poll & Privacy scoping (`privacy_level=1`, `2`, `3`): polls until `completed`, confirms uploaded photos were deleted on disk (`deletion_audit.deleted_count == len(photos)`), downloads signed `mesh_url` + `json_url`, and confirms Level 1 & Level 2 return `404` when requesting `tex_face.jpg` or `scan_preview_face.glb` (even with a server-minted signature!).
  - Signed URL security: expired `exp` (`403`), flipped `sig` char (`403`), path traversal / unlisted file name (`400`/`404`).
  - Decision 3a Delete & Full Purge: `POST .../delete` (`purge_all=false`) returns `{ok: true, assets_deleted: true, measurements_retained: true}`, erases `*.glb`/`*.jpg` from disk while keeping `body_scan.json`, appends to `server_deletion_audit.jsonl`, and makes `GET /files/scan.glb` / `tex_blur.jpg` return `410 Gone`; `POST .../delete?purge_all=true` erases the job directory completely (`404`).
  - TTL Reaper: calling `app.state.scan_manager.reap_expired_jobs(now_override=...)` removes expired job directory and logs reaper event.
  - `--serve-frontend`: verifies `/lab/` returns `200` when configured with a temporary frontend dir.
- [ ] **Step 3:** Run `python -m pytest tests/test_api_server.py -v` and confirm failure (`ModuleNotFoundError`).
- [ ] **Step 4 (GREEN):** Implement `bodyscan_pipeline/api_server.py`:
  - `ALLOWED_ARTIFACT_FILES` whitelist + `_allowed_files_for_privacy(privacy_level)` mapping.
  - `hmac.new` + `hmac.compare_digest` signing (`f"{job_id}:{filename}:{exp}"`).
  - Upload streaming size cap (`15 * 1024 * 1024` bytes per file) + Pillow `Image.open(io).verify()` before enqueueing; clean unlink on early rejection + `secure_delete_files` inside worker `finally:`.
  - Single-worker `ThreadPoolExecutor(max_workers=1)` tracking `queue_depth` and per-job `(status, progress, step, result, assets_deleted, created_at, expires_at, deleted_at)` guarded by a threading lock.
  - Background daemon reaper thread stopped cleanly on FastAPI lifespan shutdown.
- [ ] **Step 5:** Run `python -m pytest tests/test_api_server.py tests/test_scan_service.py tests/test_measurements.py tests/test_texturing.py -v` and confirm 100% pass.

---

### Task 3: Copy Lab Bundle (`frontend_lab/`) & Build Pure-JS `.glb` Parser + Privacy/Client Logic (`alpha-plus-pro.js`)

**Files:**
- Copy to: `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\` (from `c:\Users\nvpph\Downloads\AlphaScanLab_SuaPreviewMo_14-11\`, leaving source untouched)
- Create: `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\alpha-plus-pro.js`
- Create: `c:\Users\nvpph\Desktop\alpha scan\tests\test_frontend_plus_pro.mjs`

**Interfaces (`window.AlphaPlusPro` in `frontend_lab/alpha-plus-pro.js`):**
- Pure helpers (also exported for CommonJS/Node tests when `module.exports` exists):
  - `isLoopbackUrl(urlStr: string): boolean`
  - `validatePrivacyAndConnection({consentMode, serverUrl, privacyLevel, consentChecked, photoCount, heightCm}): {ok: boolean, reasonKey: string}`
  - `parseGlbBinary(arrayBuffer: ArrayBuffer, THREE, imageFactory?): Promise<{group: THREE.Group, metrics: {hY: number, hX: number, center: number[]}, stats: {meshes: number, textures: number, flipYs: boolean[]}}>`
  - `disposeObject3D(obj): void`
  - `extract16GirthRows(bodyScan, lang): Array<{key: string, label: string, cm: number}>`
  - `applyScanResultToClient(client, bodyScan, isInBodyClient): {waist?: number, neck?: number, hip?: number, keptInBody: boolean}`
  - `renderLeft16Html(scanResult, lang, helpers): string`
  - `renderReport16Html(scanResult, lang, helpers): string`
- Runtime controller methods attached to `window.AlphaPlusPro`:
  - `init(hostBridge)`, `openModal()`, `closeModal()`, `resetSession()`, `hasMesh()`, `isReal3DActive()`, `getResult()`, `getDeletedFlag()`, `setRealActive(flag)`, `renderDetailCard(detailEl)`, `renderLeft16Box()`, `onHealthCheck()`, `deleteServerAssets()`

**Steps:**
- [ ] **Step 1:** Copy `c:\Users\nvpph\Downloads\AlphaScanLab_SuaPreviewMo_14-11` cleanly into `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\` using `Copy-Item -Recurse` and verify `assets/`, `models/`, `index.html`, `three.min.js`, `vongdo.html`, `khoe-qr.js`, `vfl_*.bin` are all present with identical counts and untouched timestamps/files in `Downloads`.
- [ ] **Step 2 (RED):** Create `tests/test_frontend_plus_pro.mjs` (using `node:test` + `node:assert`) loading `frontend_lab/three.min.js` + `frontend_lab/alpha-plus-pro.js`:
  - Test `isLoopbackUrl`: `localhost`, `127.0.0.1`, `[::1]` -> `true`; LAN IP, `.trycloudflare.com`, empty/invalid URL -> `false`.
  - Test `Decision 2a` guard: `consentMode === 'local'` with `https://abc.trycloudflare.com` returns `{ok: false, reasonKey: 'pp_err_local_only'}`; same with `http://127.0.0.1:8000` succeeds (`ok: true`).
  - Test glTF 2.0 binary `.glb` parser against real `.glb` bytes emitted via `python` `MockScanEngine` (`subprocess`/temp `.glb` for Level 1 & Level 2):
    - Parses header (`0x46546C67`, version `2`), JSON & BIN chunks, `POSITION`, `NORMAL`, `TEXCOORD_0`, `indices`.
    - Creates `THREE.Texture` for Level 2 with `flipY === false` and `encoding === THREE.sRGBEncoding`.
    - Centers mesh bounding box at `(0,0,0)` and returns accurate `metrics.hY` (~height in meters) and `metrics.hX`.
    - Calls `disposeObject3D` and verifies `.dispose()` invoked on geometries, materials, and textures.
  - Test `extract16GirthRows` returns exact 16 items in `ANATOMICAL_ORDER` in both `vi` and `en`.
  - Test `applyScanResultToClient`: populates `waist`, `neck`, `hip` with `girth_src = 'plus_pro'` on standard client, and preserves `CLIENT_INBODY` flag untouched when `isInBodyClient === true`.
- [ ] **Step 3:** Run `node --test tests/test_frontend_plus_pro.mjs` and confirm failure (`alpha-plus-pro.js` missing).
- [ ] **Step 4 (GREEN):** Implement `frontend_lab/alpha-plus-pro.js`:
  - Full pure-JS glTF 2.0 binary parser (`parseGlbBinary` + `readAccessor` handling `5120/5121/5122/5123/5125/5126`, `SCALAR/VEC2/VEC3/VEC4`, normalized integer attributes, `byteStride`, interleaved buffers, optional `matrix`/`translation`/`rotation`/`scale` TRS node transform, `DoubleSide` / `sRGBEncoding` / `flipY = false`).
  - Clean modal UI generator (`#plusModal`), guided 4-angle upload slots (`Front`, `Right`, `Back`, `Left`) + quick 4–8 multi-file selector + conditional Level-3 closeup face input, explicit Mức 1/2/3 radio cards, unchecked consent checkbox, `LOCAL_ONLY` warning, `ak_gpu_cfg` config drawer + `Test Connection` (`/health` showing `TEST_ONLY` internal badge + HTTPS->HTTP Cloudflare hint when applicable), async job submit/poll + fresh signed `.glb` load, right-panel `#detail` action card + `Decision 3a` delete handler, `#plus16Box` left-panel 16-girth card renderer, and `buildReportHTML()` 16-measurement table generator.
- [ ] **Step 5:** Run `node --test tests/test_frontend_plus_pro.mjs` and confirm all JS unit & real-`.glb` integration checks pass.

---

### Task 4: Wire Plus/PRO Workflow, 360° Auto-Rotate Scene & 16 Left-Panel Girths into `frontend_lab/index.html`

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\index.html`
- Modify: `c:\Users\nvpph\Desktop\alpha scan\frontend_lab\alpha-plus-pro.js` (only if small bridge refinements are needed)
- Modify: `c:\Users\nvpph\Desktop\alpha scan\tests\test_frontend_plus_pro.mjs` (add static contract check verifying all `index.html` <-> `alpha-plus-pro.js` hook IDs/methods match)

**Steps:**
- [ ] **Step 1 (RED):** Extend `tests/test_frontend_plus_pro.mjs` with a DOM/HTML contract test inspecting `frontend_lab/index.html` to verify:
  - Script tag `<script src="./alpha-plus-pro.js"></script>` exists.
  - `#plus16Box` exists at the top of `#left` (between `#cmpBox` and `#akTiles`, above `#metricsBox`) so the 16 primary girths sit in the top-left corner of the screen without scrolling.
  - `#be_m_plus` button exists inside `akBuildBodyEdit()`, AND `#ppQuickBtn` shortcut button exists on the main toolbar/left bar, both calling `AlphaPlusPro.openModal()`.
  - Every `AlphaPlusPro.<method>` referenced in `frontend_lab/index.html` is defined on `window.AlphaPlusPro`.
  - Every `t('pp_...')` key referenced in `alpha-plus-pro.js` and `index.html` has matching EN translation in `I18N.en`.
- [ ] **Step 2:** Run `node --test tests/test_frontend_plus_pro.mjs` to confirm the contract test fails before editing `index.html`.
- [ ] **Step 3 (GREEN):** Apply targeted edits to `frontend_lab/index.html`:
  1. Extend `I18N.en` (closing right before `lang_name:'EN'}`) with all `pp_*` keys and updated `be_plus` / `'plus_pro'` strings so VI <-> EN switching translates modal, `#plus16Box`, `#detail`, `beTag`, `girthSrcLine`, and PDF table seamlessly.
  2. Insert `<div id="plus16Box"></div>` immediately after `<div id="cmpBox"></div>` (and above `#akTiles` / `#metricsBox`) inside `#left` so the 16 primary circumferences appear prominently at the top-left of the screen whenever Plus/PRO scan data exists. Also expose a clean direct `#ppQuickBtn` ("Quét 3D · 4 ảnh") action button on the main toolbar/left card so PTs can launch the 4-photo upload flow in one click as well as from `#be_m_plus`.
  3. Extend 3D viewer state & loop:
     - Declare `realPivot = new THREE.Group(); scene.add(realPivot);` alongside `root` in `init()`.
     - In `animate()`: apply `ry`/`rx` to `realPivot` as well as `root`.
     - Add small helper `frameRealModel(m)` that positions camera using `realPivot` metrics (`hY`, `hX`) via the same vertical/horizontal FOV math as `frameModel()`.
     - In `applyAppearance()`: when `appearance === 'cover'` and `AlphaPlusPro.isReal3DActive()`, hide `#coverImg` and `sysGroups`, show `realPivot`, set warm neutral studio lighting + `#EDE7DB` background, call `frameRealModel`, leave **360° auto-rotation enabled (`spin = true`, update `#viewSeg` `Xoay` button `.on`, allow `#viewSeg` `Trước`/`Sau`/`Xoay` + mouse pointer drag/wheel to take over freely)**, update `.hint` text, and call `AlphaPlusPro.renderDetailCard($('#detail'))`. When `!isReal3DActive()` (e.g., before scan or after Decision 3a server deletion), ensure `realPivot.visible = false` and show `./assets/overview.webp` hologram cleanly.
     - In `anat`, `data`, `posture` branches of `applyAppearance()`: always hide `realPivot` cleanly (`realPivot.visible = false`) and preserve existing `BodyParts3D` behavior.
  4. In `renderLeft()`: keep `#plus16Box` synced cleanly via `AlphaPlusPro.renderLeft16Box()` across `CLIENT_INBODY`, `MEASURED`, and `!MEASURED` branches without colliding with `akUpdateTiles()`.
  5. Update `beTag(src)` & `girthSrcLine(c)` so `src === 'plus_pro'` renders `3D Plus/PRO` badge (and `BE.meta` / `bePaint` preserve `'plus_pro'` until user manually types an override).
  6. Update `akBuildBodyEdit()` (`#be_m_plus` 3rd source card) and `akBuildPanel()` (section `3.5 · Server 3D Alpha Scan Plus/PRO` with URL + API key + `Kiểm tra / Test` -> `/health`).
  7. Update `buildReportHTML()`: capture canvas when `realPivot.visible` (without forcing temporary switch to `anat`) and append `AlphaPlusPro.renderReport16Html()` inside report output.
  8. Call `AlphaPlusPro.resetSession()` inside `startNewPerson()` and `akEnterWithPerson()`.
  9. Add `<script src="./alpha-plus-pro.js"></script>` and initialize `AlphaPlusPro.init(...)` providing controlled getters/setters (`getSceneState`, `setCoverSpin`, `applyResult`, `refreshLeft`, etc.).
- [ ] **Step 4:** Re-run `node --test tests/test_frontend_plus_pro.mjs` and verify 100% pass (including zero `SyntaxError` parsing inline JS from `index.html`!).

---

### Task 5: End-to-End Integration, Live Browser Verification & Documentation (`README.md`)

**Files:**
- Modify: `c:\Users\nvpph\Desktop\alpha scan\README.md`

**Steps:**
- [ ] **Step 1:** Run full backend + frontend automated suites:
  - `python -m pytest tests/ -v` (`test_measurements.py`, `test_texturing.py`, `test_scan_service.py`, `test_api_server.py`)
  - `node --test tests/test_frontend_plus_pro.mjs`
- [ ] **Step 2:** Start `python -m bodyscan_pipeline.api_server --mock --region-mode internal_test --serve-frontend frontend_lab --host 127.0.0.1 --port 8000` locally and execute a live interactive browser verification on `http://127.0.0.1:8000/lab/`:
  - Verify gate -> new test client -> `LOCAL_ONLY` consent blocks non-loopback GPU URL and accepts `http://127.0.0.1:8000` (`Decision 2a`).
  - Open `Cân / Đo` -> click **Alpha Scan Plus/PRO** -> upload 4 test images (e.g., synthetic sample images from artifacts/temp) across the 4 angle slots (`Front`, `Right`, `Back`, `Left`) -> choose Level 2 + tick consent -> submit -> verify `GET /health` `TEST_ONLY` badge, job completion, `.glb` rendered on `#scene` **auto-rotating 360°**, mouse drag pauses spin and orbits freely, `Trước`/`Sau`/`Xoay` buttons work.
  - Verify left panel (`#left` -> `#plus16Box`) renders all **16 Vietnamese circumferences**, `waist`/`neck`/`hip` update US Navy preview, and switching to `Giải phẫu` / `Theo dữ liệu` / `Tư thế` hides real mesh cleanly.
  - Click **`Xoá hình 3D/ảnh trên server`** (`Decision 3a`): verify `#scene` returns to `./assets/overview.webp`, server returns `410 Gone` for `.glb`/`.jpg`, while `#plus16Box` in `#left` + 16-measurement PDF preview (`buildReportHTML`) remain intact.
  - Switch language VI <-> EN (`toggleLang`) and verify zero uncaught browser console errors.
- [ ] **Step 3:** Stop background test server cleanly; clean up any temporary verification test artifacts.
- [ ] **Step 4:** Update `README.md` with Plan 2A Quickstart (`--mock --serve-frontend frontend_lab` on Windows and GPU + Cloudflare Tunnel instructions for Colab internal testing).
- [ ] **Step 5:** Run final verification commands (`python -m pytest tests/ -v`, `node --test tests/test_frontend_plus_pro.mjs`) after final edit and report exact observed outputs.

---

### Self-Review Checklist
- [x] **Spec coverage:** Covers `scan_service.py` (Component A), `api_server.py` (Component B), `frontend_lab/` + `alpha-plus-pro.js` + `index.html` (Component C), 4-angle upload -> backend -> 360° auto-spin + mouse drag + 16 girths on `#left`, Decision 1 (`internal_test`), Decision 2a (`LOCAL_ONLY` loopback guard), Decision 3a (asset delete retaining 16 girths + PDF), and all Section 7 tests.
- [x] **No placeholders / no full code dumps:** Concrete file paths, exact module interfaces, test assertions, and surgical hook locations without copying hundreds of lines of implementation source into the plan.
- [x] **Type & name consistency:** `ANATOMICAL_ORDER`, `BODY_LABELS_VI`, `BODY_LABELS_EN`, `ScanRequest`, `ScanResult`, `MockScanEngine`, `GpuScanEngine`, `write_minimal_glb`, `AlphaPlusPro`, `#plus16Box`, `#plusModal`, `#be_m_plus`, `realPivot`, `'plus_pro'`, `/api/v1/bodyscan*` match across Tasks 1–5 and spec.
- [x] **DRY & modular boundaries:** `demo_ui.py` and `api_server.py` share `scan_service.py`; `alpha-plus-pro.js` cleanly separates glTF `.glb` parsing & Plus/PRO workflow from `index.html`.
- [x] **Git safety:** Explicitly avoids `git commit` steps because workspace is `NOT_A_GIT_REPO`.
