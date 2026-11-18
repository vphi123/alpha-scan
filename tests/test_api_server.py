"""Integration tests for Async FastAPI bodyscan server, auth, privacy scoping, signed URLs, Decision 3a delete, and TTL reaper."""

import io
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
from PIL import Image


def _make_jpeg_bytes(color=(190, 140, 115), size=(64, 96)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _submit_sample_scan(
    client: TestClient,
    privacy_level: int = 2,
    headers=None,
    n_photos: int = 4,
    with_face: bool = False,
    reconstruction_mode: str = None,
    weight_kg: str = "56.0",
):
    img_bytes = _make_jpeg_bytes()
    files = [("photos", (f"angle_{i}.jpg", img_bytes, "image/jpeg")) for i in range(n_photos)]
    if with_face:
        files.append(("closeup_face", ("face.jpg", _make_jpeg_bytes((225, 185, 165)), "image/jpeg")))
    data = {
        "height_cm": "168.0",
        "sex": "female",
        "weight_kg": weight_kg,
        "person_id": "test_pt_01",
        "privacy_level": str(privacy_level),
        "consent_granted": "true",
    }
    if reconstruction_mode is not None:
        data["reconstruction_mode"] = reconstruction_mode
    return client.post("/api/v1/bodyscan", data=data, files=files, headers=headers or {})


def _wait_completed(client: TestClient, job_id: str, headers=None, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/api/v1/bodyscan/{job_id}", headers=headers or {})
        if resp.status_code != 200:
            return resp.json()
        payload = resp.json()
        if payload["status"] in ("completed", "failed"):
            return payload
        time.sleep(0.02)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_s}s")


class TestApiServer(unittest.TestCase):
    """Verify FastAPI Plan 2A server endpoints and security invariants."""

    def test_health_defaults_internal_test(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                r = client.get("/health")
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertTrue(body["ok"])
                self.assertEqual(body["mode"], "mock")
                self.assertEqual(body["region_mode"], "internal_test")
                self.assertTrue(body["test_only"])
                self.assertFalse(body["auth_required"])
                self.assertIn("pipeline_version", body)
                self.assertIn("queue_depth", body)

    def test_api_key_auth_except_health_and_signed_files(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ServerConfig(mock=True, api_key="secret_pt_key", data_dir=tmp)
            app = create_app(cfg, engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                # /health stays open for Lab connectivity diagnostics
                h = client.get("/health")
                self.assertEqual(h.status_code, 200)
                self.assertTrue(h.json()["auth_required"])

                # Unauthenticated / wrong key rejected with 401
                r_no_auth = _submit_sample_scan(client, privacy_level=1)
                self.assertEqual(r_no_auth.status_code, 401)
                r_bad = _submit_sample_scan(client, privacy_level=1, headers={"X-API-Key": "wrong"})
                self.assertEqual(r_bad.status_code, 401)

                # Accepted via Bearer
                auth_hdr = {"Authorization": "Bearer secret_pt_key"}
                r_ok = _submit_sample_scan(client, privacy_level=1, headers=auth_hdr)
                self.assertEqual(r_ok.status_code, 202)
                job_id = r_ok.json()["job_id"]

                self.assertEqual(client.get(f"/api/v1/bodyscan/{job_id}").status_code, 401)
                done = _wait_completed(client, job_id, headers=auth_hdr)
                self.assertEqual(done["status"], "completed")

                # Signed file URL works WITHOUT Bearer header
                r_glb = client.get(done["artifacts"]["mesh_url"])
                self.assertEqual(r_glb.status_code, 200)
                self.assertTrue(r_glb.content.startswith(b"glTF"))

                # Delete requires key
                self.assertEqual(client.post(f"/api/v1/bodyscan/{job_id}/delete").status_code, 401)
                self.assertEqual(client.post(f"/api/v1/bodyscan/{job_id}/delete", headers={"X-API-Key": "secret_pt_key"}).status_code, 200)

    def test_consent_and_input_validation_guards(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                img_bytes = _make_jpeg_bytes()

                # consent_granted=false -> 403 Forbidden
                r_consent = client.post(
                    "/api/v1/bodyscan",
                    data={"height_cm": "165", "sex": "female", "privacy_level": "2", "consent_granted": "false"},
                    files=[("photos", ("front.jpg", img_bytes, "image/jpeg"))],
                )
                self.assertEqual(r_consent.status_code, 403)

                # Invalid height -> 422
                r_height = client.post(
                    "/api/v1/bodyscan",
                    data={"height_cm": "50", "sex": "female", "privacy_level": "2", "consent_granted": "true"},
                    files=[("photos", ("front.jpg", img_bytes, "image/jpeg"))],
                )
                self.assertEqual(r_height.status_code, 422)

                # > 8 photos -> 422
                r_many = _submit_sample_scan(client, n_photos=9)
                self.assertEqual(r_many.status_code, 422)

                # File > 15 MB -> 413
                big_payload = b"\xff\xd8\xff\xe0" + (b"0" * (15 * 1024 * 1024 + 32))
                r_big = client.post(
                    "/api/v1/bodyscan",
                    data={"height_cm": "165", "sex": "female", "privacy_level": "1", "consent_granted": "true"},
                    files=[("photos", ("huge.jpg", big_payload, "image/jpeg"))],
                )
                self.assertEqual(r_big.status_code, 413)

                # Corrupted non-image bytes -> 422 and zero leftover files
                r_bad_img = client.post(
                    "/api/v1/bodyscan",
                    data={"height_cm": "165", "sex": "female", "privacy_level": "2", "consent_granted": "true"},
                    files=[("photos", ("evil.jpg", b"not_an_image_at_all", "image/jpeg"))],
                )
                self.assertEqual(r_bad_img.status_code, 422)
                for root, _, files in os.walk(tmp):
                    self.assertEqual(files, [], f"Leftover rejected files on disk: {root} -> {files}")

    def test_privacy_scoping_blocks_clear_face_assets(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import ANATOMICAL_ORDER, MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                # Level 1: geometry only
                r1 = _submit_sample_scan(client, privacy_level=1, n_photos=4)
                self.assertEqual(r1.status_code, 202)
                j1 = _wait_completed(client, r1.json()["job_id"])
                self.assertEqual(j1["status"], "completed")
                self.assertEqual(list(j1["body_scan"]["circumferences_cm"].keys()), list(ANATOMICAL_ORDER))
                self.assertEqual(len(j1["body_scan"]["shape_profile"]["loft"]), 57)
                self.assertNotIn("texture_url", j1["artifacts"])
                self.assertIn("scan.glb", j1["artifacts"]["mesh_url"])
                self.assertEqual(j1["deletion_audit"]["deleted_count"], 4)

                # Even with a valid server signature, Level 1 blocks tex_face & tex_blur & preview GLBs
                sig_face = app.state.scan_manager.sign_file_url(r1.json()["job_id"], "tex_face.jpg")
                self.assertEqual(client.get(sig_face).status_code, 404)
                sig_blur = app.state.scan_manager.sign_file_url(r1.json()["job_id"], "tex_blur.jpg")
                self.assertEqual(client.get(sig_blur).status_code, 404)

                # Level 2: blurred face only
                r2 = _submit_sample_scan(client, privacy_level=2, n_photos=4)
                j2 = _wait_completed(client, r2.json()["job_id"])
                self.assertIn("scan_preview_blur.glb", j2["artifacts"]["mesh_url"])
                self.assertIn("tex_blur.jpg", j2["artifacts"]["texture_url"])
                self.assertEqual(client.get(j2["artifacts"]["texture_url"]).status_code, 200)
                sig_face_2 = app.state.scan_manager.sign_file_url(r2.json()["job_id"], "tex_face.jpg")
                self.assertEqual(client.get(sig_face_2).status_code, 404)
                sig_glb_face_2 = app.state.scan_manager.sign_file_url(r2.json()["job_id"], "scan_preview_face.glb")
                self.assertEqual(client.get(sig_glb_face_2).status_code, 404)

                # Level 3: clear face available
                r3 = _submit_sample_scan(client, privacy_level=3, n_photos=4, with_face=True)
                j3 = _wait_completed(client, r3.json()["job_id"])
                self.assertIn("scan_preview_face.glb", j3["artifacts"]["mesh_url"])
                self.assertIn("tex_face.jpg", j3["artifacts"]["texture_url"])
                self.assertEqual(client.get(j3["artifacts"]["mesh_url"]).status_code, 200)
                self.assertEqual(client.get(j3["artifacts"]["texture_url"]).status_code, 200)

    def test_signed_url_expiry_tamper_traversal_and_410_after_delete(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                r = _submit_sample_scan(client, privacy_level=2, n_photos=4)
                job_id = r.json()["job_id"]
                done = _wait_completed(client, job_id)
                mesh_url = done["artifacts"]["mesh_url"]
                tex_url = done["artifacts"]["texture_url"]
                json_url = done["artifacts"]["json_url"]

                # Tamper signature -> 403
                parsed = urlparse(mesh_url)
                qs = parse_qs(parsed.query)
                bad_sig = "0" if qs["sig"][0][0] != "0" else "1"
                tampered = f"{parsed.path}?exp={qs['exp'][0]}&sig={bad_sig}{qs['sig'][0][1:]}"
                self.assertEqual(client.get(tampered).status_code, 403)

                # Expired timestamp -> 403
                expired_url = app.state.scan_manager.sign_file_url(job_id, "scan.glb", exp_ts=int(time.time()) - 10)
                self.assertEqual(client.get(expired_url).status_code, 403)

                # Traversal / disallowed filename -> 404 or 400
                bad_file = app.state.scan_manager.sign_file_url(job_id, "../secret.txt")
                self.assertIn(client.get(bad_file).status_code, (400, 404))

                # Decision 3a delete (purge_all=false) -> deletes binaries, keeps JSON, returns 410 on binaries
                d_resp = client.post(f"/api/v1/bodyscan/{job_id}/delete")
                self.assertEqual(d_resp.status_code, 200)
                d_json = d_resp.json()
                self.assertTrue(d_json["ok"])
                self.assertTrue(d_json["assets_deleted"])
                self.assertTrue(d_json["measurements_retained"])

                out_dir = os.path.join(tmp, job_id, "outputs")
                self.assertFalse(os.path.exists(os.path.join(out_dir, "scan.glb")))
                self.assertFalse(os.path.exists(os.path.join(out_dir, "scan_preview_blur.glb")))
                self.assertFalse(os.path.exists(os.path.join(out_dir, "tex_blur.jpg")))
                self.assertTrue(os.path.exists(os.path.join(out_dir, "body_scan.json")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "server_deletion_audit.jsonl")))

                # Binary assets return 410 Gone
                self.assertEqual(client.get(mesh_url).status_code, 410)
                self.assertEqual(client.get(tex_url).status_code, 410)
                # Retained non-biometric JSON still accessible
                self.assertEqual(client.get(json_url).status_code, 200)
                status_after = client.get(f"/api/v1/bodyscan/{job_id}").json()
                self.assertEqual(status_after["status"], "deleted")
                self.assertTrue(status_after["assets_deleted"])
                self.assertIn("circumferences_cm", status_after["body_scan"])

                # purge_all=true deletes entire job folder
                p_resp = client.post(f"/api/v1/bodyscan/{job_id}/delete?purge_all=true")
                self.assertEqual(p_resp.status_code, 200)
                self.assertFalse(os.path.exists(os.path.join(tmp, job_id)))
                self.assertEqual(client.get(f"/api/v1/bodyscan/{job_id}").status_code, 404)

    def test_reaper_expires_old_jobs_and_serve_frontend_mount(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as fe_dir:
            with open(os.path.join(fe_dir, "index.html"), "w", encoding="utf-8") as fp:
                fp.write("<html><body>Alpha Lab OK</body></html>")

            app = create_app(
                ServerConfig(mock=True, data_dir=tmp, job_ttl_minutes=60, serve_frontend=fe_dir),
                engine=MockScanEngine(0.0),
            )
            with TestClient(app) as client:
                fe_r = client.get("/lab/")
                self.assertEqual(fe_r.status_code, 200)
                self.assertIn("Alpha Lab OK", fe_r.text)

                r = _submit_sample_scan(client, privacy_level=1, n_photos=2)
                job_id = r.json()["job_id"]
                _wait_completed(client, job_id)
                self.assertTrue(os.path.exists(os.path.join(tmp, job_id)))

                future = datetime.now(timezone.utc) + timedelta(minutes=65)
                reaped = app.state.scan_manager.reap_expired_jobs(now_utc=future)
                self.assertEqual(reaped, 1)
                self.assertFalse(os.path.exists(os.path.join(tmp, job_id)))
                self.assertEqual(client.get(f"/api/v1/bodyscan/{job_id}").status_code, 404)

                with open(os.path.join(tmp, "server_deletion_audit.jsonl"), encoding="utf-8") as fp:
                    rows = [json.loads(line) for line in fp if line.strip()]
                self.assertTrue(any(row.get("reason") == "ttl_expired" for row in rows))

    def test_delete_in_flight_job_returns_409_conflict(self) -> None:
        from bodyscan_pipeline.api_server import JobRecord, ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                mgr = app.state.scan_manager
                out_dir = os.path.join(tmp, "job_inflight", "outputs")
                os.makedirs(out_dir, exist_ok=True)
                rec = JobRecord(
                    job_id="job_inflight",
                    job_dir=os.path.join(tmp, "job_inflight"),
                    outputs_dir=out_dir,
                    person_id="p_test",
                    privacy_level=2,
                    mode="mock",
                    status="running",
                    progress=40,
                    step="inference",
                )
                with mgr._lock:
                    mgr._jobs[rec.job_id] = rec

                r_del = client.post(f"/api/v1/bodyscan/{rec.job_id}/delete")
                self.assertEqual(r_del.status_code, 409)
                self.assertEqual(client.get(f"/api/v1/bodyscan/{rec.job_id}").json()["status"], "running")

    def test_bodyscan_api_downgrades_incomplete_views_to_calibrated_mhr(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                r = _submit_sample_scan(
                    client,
                    privacy_level=2,
                    n_photos=2,
                    reconstruction_mode="dual_layer",
                )
                self.assertEqual(r.status_code, 202, r.text)
                job_id = r.json()["job_id"]
                payload = _wait_completed(client, job_id)
                self.assertEqual(payload["status"], "completed")
                self.assertEqual(payload["reconstruction_mode"], "dual_layer")
                self.assertEqual(payload["effective_reconstruction_mode"], "calibrated_mhr")
                self.assertEqual(payload["body_scan"]["reconstruction_mode"], "dual_layer")
                self.assertEqual(payload["body_scan"]["effective_reconstruction_mode"], "calibrated_mhr")
                self.assertTrue(any("calibrated_mhr" in w for w in payload["warnings"]))
                self.assertTrue(any("calibrated_mhr" in w for w in payload["body_scan"]["warnings"]))
                self.assertNotIn("core_mesh_url", payload["artifacts"])
                self.assertIn("mesh_url", payload["artifacts"])
                self.assertEqual(
                    payload["proportion_audit"]["status"],
                    payload["body_scan"]["proportion_audit"]["status"],
                )
                self.assertNotIn("face_landmarks", json.dumps(payload))

                core_probe = client.get(f"/api/v1/bodyscan/{job_id}/files/scan_core.glb")
                self.assertNotEqual(core_probe.status_code, 200)

    def test_reconstruction_mode_validation_and_core_glb_decision_3a(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                bad_r = _submit_sample_scan(
                    client,
                    privacy_level=2,
                    n_photos=4,
                    reconstruction_mode="invalid_xyz",
                )
                self.assertEqual(bad_r.status_code, 422)

                last_job_id = None
                last_core_url = None
                last_json_url = None
                for level in (1, 2, 3):
                    r = _submit_sample_scan(
                        client,
                        privacy_level=level,
                        n_photos=4,
                        with_face=(level == 3),
                        reconstruction_mode="dual_layer",
                        weight_kg="0.0" if level == 1 else "58.0",
                    )
                    self.assertEqual(r.status_code, 202, r.text)
                    job_id = r.json()["job_id"]
                    payload = _wait_completed(client, job_id)
                    self.assertEqual(payload["status"], "completed")
                    self.assertEqual(payload["reconstruction_mode"], "dual_layer")
                    self.assertEqual(payload["effective_reconstruction_mode"], "dual_layer")
                    audit = payload["proportion_audit"]
                    self.assertIn(audit["status"], ("estimated_only", "matched", "mismatch_warning"))
                    self.assertEqual(audit, payload["body_scan"]["proportion_audit"])
                    if level == 1:
                        self.assertEqual(audit["status"], "estimated_only")
                    self.assertNotIn("face_landmarks", json.dumps(payload))

                    artifacts = payload["artifacts"]
                    self.assertIn("core_mesh_url", artifacts)
                    mesh_resp = client.get(artifacts["mesh_url"])
                    core_resp = client.get(artifacts["core_mesh_url"])
                    self.assertEqual(mesh_resp.status_code, 200)
                    self.assertEqual(core_resp.status_code, 200)
                    self.assertEqual(mesh_resp.content[:4], b"glTF")
                    self.assertEqual(core_resp.content[:4], b"glTF")
                    self.assertNotEqual(mesh_resp.content, core_resp.content)

                    last_job_id = job_id
                    last_core_url = artifacts["core_mesh_url"]
                    last_json_url = artifacts["json_url"]

                # Decision 3a delete wipes scan_core.glb alongside display GLBs/JPEGs and returns 410 Gone
                d_resp = client.post(f"/api/v1/bodyscan/{last_job_id}/delete")
                self.assertEqual(d_resp.status_code, 200)
                self.assertTrue(d_resp.json()["assets_deleted"])
                out_dir = os.path.join(tmp, last_job_id, "outputs")
                self.assertFalse(os.path.exists(os.path.join(out_dir, "scan_core.glb")))
                self.assertTrue(os.path.exists(os.path.join(out_dir, "body_scan.json")))
                self.assertEqual(client.get(last_core_url).status_code, 410)
                self.assertEqual(client.get(last_json_url).status_code, 200)

    def test_async_non_mock_engine_warmup_health_and_queued_job(self) -> None:
        import threading
        from unittest.mock import patch
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        # 1. Successful delayed GPU warmup: create_app returns immediately, /health reports loading -> ready
        gate = threading.Event()
        real_mock = MockScanEngine(0.0)

        def _slow_gpu_loader(*args, **kwargs):
            gate.wait(timeout=5.0)
            return real_mock

        with tempfile.TemporaryDirectory() as tmp1, patch(
            "bodyscan_pipeline.api_server.GpuScanEngine", side_effect=_slow_gpu_loader
        ):
            t0 = time.perf_counter()
            app = create_app(ServerConfig(mock=False, data_dir=tmp1))
            self.assertLess(time.perf_counter() - t0, 1.0)
            try:
                with TestClient(app) as client:
                    h_loading = client.get("/health")
                    self.assertEqual(h_loading.status_code, 200)
                    hj = h_loading.json()
                    self.assertTrue(hj["ok"])
                    self.assertEqual(hj["status"], "loading")
                    self.assertFalse(hj["model_ready"])
                    self.assertEqual(hj["mode"], "gpu")

                    # Queue job while model is still warming up
                    r_sub = _submit_sample_scan(client, privacy_level=2, n_photos=4, reconstruction_mode="dual_layer")
                    self.assertEqual(r_sub.status_code, 202, r_sub.text)
                    jid = r_sub.json()["job_id"]
                    mid = client.get(f"/api/v1/bodyscan/{jid}").json()
                    self.assertIn(mid["status"], ("queued", "running"))

                    # Release engine loader
                    gate.set()
                    deadline = time.time() + 5.0
                    h_ready = {}
                    while time.time() < deadline:
                        h_ready = client.get("/health").json()
                        if h_ready.get("model_ready") is True:
                            break
                        time.sleep(0.02)
                    self.assertTrue(h_ready.get("ok"))
                    self.assertEqual(h_ready.get("status"), "ready")
                    self.assertTrue(h_ready.get("model_ready"))

                    done = _wait_completed(client, jid)
                    self.assertEqual(done["status"], "completed")
            finally:
                gate.set()

        # 2. Failed GPU warmup: /health reports status='error', queued job fails cleanly, new POST -> 503
        fail_gate = threading.Event()

        def _failing_loader(*args, **kwargs):
            fail_gate.wait(timeout=5.0)
            raise RuntimeError("simulated CUDA checkpoint load failure")

        with tempfile.TemporaryDirectory() as tmp2, patch(
            "bodyscan_pipeline.api_server.GpuScanEngine", side_effect=_failing_loader
        ):
            app_err = create_app(ServerConfig(mock=False, data_dir=tmp2))
            try:
                with TestClient(app_err) as client_err:
                    r_queued = _submit_sample_scan(client_err, privacy_level=1, n_photos=1)
                    self.assertEqual(r_queued.status_code, 202)
                    jid_err = r_queued.json()["job_id"]

                    fail_gate.set()
                    deadline = time.time() + 5.0
                    h_err = {}
                    while time.time() < deadline:
                        h_err = client_err.get("/health").json()
                        if h_err.get("status") == "error":
                            break
                        time.sleep(0.02)
                    self.assertFalse(h_err.get("ok"))
                    self.assertEqual(h_err.get("status"), "error")
                    self.assertFalse(h_err.get("model_ready"))
                    self.assertIn("simulated CUDA checkpoint load failure", h_err.get("detail", ""))

                    failed_job = _wait_completed(client_err, jid_err)
                    self.assertEqual(failed_job["status"], "failed")
                    self.assertIn("Model initialization failed", failed_job["error"])

                    r_503 = _submit_sample_scan(client_err, privacy_level=1, n_photos=1)
                    self.assertEqual(r_503.status_code, 503)
            finally:
                fail_gate.set()

    def test_data_dir_restart_persistence_tombstones_secret_and_cli_ttl(self) -> None:
        from pathlib import Path
        from bodyscan_pipeline.api_server import ServerConfig, _parse_args, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        # Preserve ServerConfig 300s default for Python callers while CLI defaults to 1800s
        self.assertEqual(ServerConfig().file_token_ttl, 300)
        self.assertEqual(_parse_args([]).file_token_ttl, 1800)
        self.assertEqual(_parse_args(["--file-token-ttl", "900"]).file_token_ttl, 900)

        with tempfile.TemporaryDirectory() as tmp:
            # Session 1 (auto-generated signing secret persisted lazily to data_dir/.signing_secret)
            app1 = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app1) as c1:
                r_keep = _submit_sample_scan(c1, privacy_level=3, n_photos=4, reconstruction_mode="dual_layer")
                self.assertEqual(r_keep.status_code, 202)
                keep_id = r_keep.json()["job_id"]
                keep_done = _wait_completed(c1, keep_id)
                self.assertEqual(keep_done["status"], "completed")
                keep_mesh_url = keep_done["artifacts"]["mesh_url"]
                keep_json_url = keep_done["artifacts"]["json_url"]

                r_del = _submit_sample_scan(c1, privacy_level=3, n_photos=4, reconstruction_mode="dual_layer")
                self.assertEqual(r_del.status_code, 202)
                del_id = r_del.json()["job_id"]
                del_done = _wait_completed(c1, del_id)
                del_mesh_url = del_done["artifacts"]["mesh_url"]
                del_json_url = del_done["artifacts"]["json_url"]
                d_res = c1.post(f"/api/v1/bodyscan/{del_id}/delete")
                self.assertEqual(d_res.status_code, 200)

            secret_file = Path(tmp) / ".signing_secret"
            index_file = Path(tmp) / "jobs_index.json"
            self.assertTrue(secret_file.is_file())
            self.assertTrue(index_file.is_file())
            index_raw = index_file.read_text(encoding="utf-8")
            self.assertNotIn("face_landmarks", index_raw)
            self.assertNotIn("source_filenames", index_raw)

            # Inject an interrupted running job with an orphaned upload file to verify restart cleanup
            crash_id = "crashedjob001"
            crash_uploads = Path(tmp) / crash_id / "uploads"
            crash_outputs = Path(tmp) / crash_id / "outputs"
            crash_uploads.mkdir(parents=True, exist_ok=True)
            crash_outputs.mkdir(parents=True, exist_ok=True)
            orphan_photo = crash_uploads / "orphan.jpg"
            orphan_photo.write_bytes(_make_jpeg_bytes())
            idx_doc = json.loads(index_raw)
            idx_doc["jobs"].append(
                {
                    "job_id": crash_id,
                    "job_dir": str(Path(tmp) / crash_id),
                    "outputs_dir": str(crash_outputs),
                    "person_id": "crash_pt",
                    "privacy_level": 2,
                    "mode": "mock",
                    "reconstruction_mode": "calibrated_mhr",
                    "status": "running",
                    "progress": 30,
                    "step": "fitting",
                    "assets_deleted": False,
                }
            )
            index_file.write_text(json.dumps(idx_doc, indent=2), encoding="utf-8")

            # Session 2: fresh create_app pointing at the same data_dir without passing signing_secret
            app2 = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app2) as c2:
                # Crashed in-flight job became failed & orphaned photo shredded
                self.assertFalse(orphan_photo.exists())
                crash_st = c2.get(f"/api/v1/bodyscan/{crash_id}")
                self.assertEqual(crash_st.status_code, 200)
                self.assertEqual(crash_st.json()["status"], "failed")

                # Completed job + pre-restart HMAC URLs remain valid
                keep_st = c2.get(f"/api/v1/bodyscan/{keep_id}")
                self.assertEqual(keep_st.status_code, 200)
                self.assertEqual(keep_st.json()["status"], "completed")
                self.assertEqual(c2.get(keep_mesh_url).status_code, 200)
                self.assertEqual(c2.get(keep_json_url).status_code, 200)

                # Decision 3a tombstone preserved across restart: binary -> 410 Gone, JSON -> 200 OK
                del_st = c2.get(f"/api/v1/bodyscan/{del_id}")
                self.assertEqual(del_st.status_code, 200)
                self.assertEqual(del_st.json()["status"], "deleted")
                self.assertTrue(del_st.json()["assets_deleted"])
                self.assertTrue(del_st.json()["measurements_retained"])
                self.assertEqual(c2.get(del_mesh_url).status_code, 410)
                self.assertEqual(c2.get(del_json_url).status_code, 200)

    def test_api_duplicate_uploads_downgrade_dual_layer_to_calibrated_mhr(self) -> None:
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                solid = _make_jpeg_bytes()
                r_dup = client.post(
                    "/api/v1/bodyscan",
                    data={
                        "height_cm": "168.0",
                        "sex": "female",
                        "weight_kg": "56.0",
                        "person_id": "dup_test",
                        "privacy_level": "2",
                        "reconstruction_mode": "dual_layer",
                        "consent_granted": "true",
                    },
                    files=[("photos", ("front.jpg", solid, "image/jpeg")) for _ in range(4)],
                )
                self.assertEqual(r_dup.status_code, 202, r_dup.text)
                payload = _wait_completed(client, r_dup.json()["job_id"])
                self.assertEqual(payload["status"], "completed")
                self.assertEqual(payload["reconstruction_mode"], "dual_layer")
                self.assertEqual(payload["effective_reconstruction_mode"], "calibrated_mhr")
                self.assertNotIn("core_mesh_url", payload["artifacts"])

    def test_health_and_job_status_expose_backend_telemetry(self) -> None:
        import threading
        from unittest.mock import patch
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        # 1. Ready mock server exposes structured backend capability metadata on /health and /api/v1/bodyscan/{job_id}
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(ServerConfig(mock=True, data_dir=tmp), engine=MockScanEngine(0.0))
            with TestClient(app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                hj = health.json()
                self.assertIn("backends", hj)
                b = hj["backends"]
                for k in (
                    "gpu_name",
                    "segmentation",
                    "rasterizer",
                    "pytorch3d_available",
                    "moge2_available",
                    "mediapipe_available",
                    "xatlas_available",
                    "trimesh_available",
                ):
                    self.assertIn(k, b)
                self.assertEqual(b["segmentation"], "mock")
                self.assertEqual(b["rasterizer"], "mock")

                sub = _submit_sample_scan(client, privacy_level=2, n_photos=4, reconstruction_mode="dual_layer")
                self.assertEqual(sub.status_code, 202, sub.text)
                payload = _wait_completed(client, sub.json()["job_id"])
                self.assertEqual(payload["status"], "completed")
                bu = payload.get("backends_used")
                self.assertIsInstance(bu, dict)
                self.assertEqual(bu, payload["body_scan"].get("backends_used"))
                for k in (
                    "segmentation",
                    "fitting_silhouette",
                    "texture_rasterizer",
                    "moge2_depth_used",
                    "facial_landmarks_used",
                ):
                    self.assertIn(k, bu)
                self.assertNotIn("face_landmarks", json.dumps(payload))

        # 2. Loading GPU warmup /health also includes backends cleanly without failing before engine loads
        gate = threading.Event()

        def _delayed_gpu(*args, **kwargs):
            gate.wait(timeout=5.0)
            return MockScanEngine(0.0)

        with tempfile.TemporaryDirectory() as tmp_load, patch(
            "bodyscan_pipeline.api_server.GpuScanEngine", side_effect=_delayed_gpu
        ):
            app_gpu = create_app(ServerConfig(mock=False, data_dir=tmp_load))
            try:
                with TestClient(app_gpu) as c_gpu:
                    hj_load = c_gpu.get("/health").json()
                    self.assertEqual(hj_load["status"], "loading")
                    self.assertEqual(hj_load["mode"], "gpu")
                    self.assertIn("backends", hj_load)
                    self.assertIn("gpu_name", hj_load["backends"])
                    self.assertIn("rasterizer", hj_load["backends"])
            finally:
                gate.set()

    def test_nan_inf_dimensions_return_422_and_slot_roles_validation(self):
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                ServerConfig(mock=True, data_dir=tmp),
                engine=MockScanEngine(0.05),
            )
            with TestClient(app) as client:
                for bad_height, bad_weight in [("nan", "56.0"), ("166.0", "inf"), ("166.0", "-inf")]:
                    files = [("photos", ("front.jpg", _make_jpeg_bytes(), "image/jpeg"))]
                    resp = client.post(
                        "/api/v1/bodyscan",
                        files=files,
                        data={
                            "height_cm": bad_height,
                            "weight_kg": bad_weight,
                            "sex": "female",
                            "person_id": "p1",
                            "privacy_level": "2",
                            "consent_granted": "true",
                        },
                    )
                    self.assertEqual(resp.status_code, 422)

                files2 = [
                    ("photos", ("p0.jpg", _make_jpeg_bytes(), "image/jpeg")),
                    ("photos", ("p1.jpg", _make_jpeg_bytes((150, 120, 110)), "image/jpeg")),
                ]
                resp_ok = client.post(
                    "/api/v1/bodyscan",
                    files=files2,
                    data={
                        "height_cm": "166.0",
                        "weight_kg": "56.0",
                        "sex": "female",
                        "person_id": "p1",
                        "privacy_level": "2",
                        "consent_granted": "true",
                        "slot_roles": json.dumps(["front", "right"]),
                    },
                )
                self.assertEqual(resp_ok.status_code, 202)
                done_ok = _wait_completed(client, resp_ok.json()["job_id"])
                self.assertEqual(done_ok["status"], "completed")

                resp_bad_len = client.post(
                    "/api/v1/bodyscan",
                    files=files2,
                    data={
                        "height_cm": "166.0",
                        "weight_kg": "56.0",
                        "sex": "female",
                        "person_id": "p1",
                        "privacy_level": "2",
                        "consent_granted": "true",
                        "slot_roles": json.dumps(["front"]),
                    },
                )
                self.assertEqual(resp_bad_len.status_code, 422)

                resp_bad_role = client.post(
                    "/api/v1/bodyscan",
                    files=files2,
                    data={
                        "height_cm": "166.0",
                        "weight_kg": "56.0",
                        "sex": "female",
                        "person_id": "p1",
                        "privacy_level": "2",
                        "consent_granted": "true",
                        "slot_roles": ["front", "upside_down"],
                    },
                )
                self.assertEqual(resp_bad_role.status_code, 422)

    def test_persisted_expired_jobs_reaped_on_startup_and_missing_file_returns_404(self):
        from pathlib import Path
        from bodyscan_pipeline.api_server import ServerConfig, create_app
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ServerConfig(mock=True, data_dir=tmp, job_ttl_minutes=1)
            app1 = create_app(cfg, engine=MockScanEngine(0.02))
            with TestClient(app1) as c1:
                sub = _submit_sample_scan(c1)
                jid = sub.json()["job_id"]
                done = _wait_completed(c1, jid)
                self.assertEqual(done["status"], "completed")
                mesh_url = done["artifacts"]["mesh_url"]
                preview_path = Path(app1.state.scan_manager._jobs[jid].outputs_dir) / "scan_preview_blur.glb"
                if preview_path.exists():
                    preview_path.unlink()
                dl_missing = c1.get(mesh_url)
                self.assertEqual(dl_missing.status_code, 404)

                state_path = Path(tmp) / "jobs_index.json"
                raw = json.loads(state_path.read_text(encoding="utf-8"))
                past_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
                if isinstance(raw.get("jobs"), list):
                    for rec_d in raw["jobs"]:
                        if rec_d.get("job_id") == jid:
                            rec_d["expires_at"] = past_iso
                elif jid in raw:
                    raw[jid]["expires_at"] = past_iso
                state_path.write_text(json.dumps(raw), encoding="utf-8")

            app2 = create_app(cfg, engine=MockScanEngine(0.02))
            with TestClient(app2) as c2:
                r = c2.get(f"/api/v1/bodyscan/{jid}")
                self.assertEqual(r.status_code, 404)

    def test_concurrent_delete_job_guard_prevents_double_delete_error(self):
        from pathlib import Path
        from fastapi import HTTPException
        from bodyscan_pipeline.api_server import ScanJobManager, ServerConfig
        from bodyscan_pipeline.scan_service import MockScanEngine

        with tempfile.TemporaryDirectory() as tmp:
            mgr = ScanJobManager(
                config=ServerConfig(mock=True, data_dir=tmp, job_ttl_minutes=1),
                engine=MockScanEngine(0.01),
            )
            try:
                job_dir = Path(tmp) / "j_race"
                out_dir = job_dir / "outputs"
                out_dir.mkdir(parents=True, exist_ok=True)
                p = job_dir / "front.jpg"
                p.write_bytes(_make_jpeg_bytes())
                rec = mgr.submit(
                    [str(p)],
                    None,
                    job_id="j_race",
                    job_dir=str(job_dir),
                    outputs_dir=str(out_dir),
                    height_cm=166.0,
                    sex="female",
                    weight_kg=56.0,
                    person_id="p1",
                    privacy_level=2,
                )
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    if rec.status == "completed":
                        break
                    time.sleep(0.02)
                self.assertEqual(rec.status, "completed")

                rec.deleting = True
                with self.assertRaises(HTTPException) as ctx:
                    mgr.delete_job(rec.job_id, purge_all=True)
                self.assertEqual(ctx.exception.status_code, 409)
                rec.deleting = False
                deleted = mgr.delete_job(rec.job_id, purge_all=True)
                self.assertEqual(deleted["job_id"], rec.job_id)
            finally:
                mgr.shutdown()


if __name__ == "__main__":
    unittest.main()
