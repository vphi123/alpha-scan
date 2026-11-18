"""Async FastAPI GPU/mock server for Alpha Scan Plus/PRO (Plan 2A — internal test ready)."""

from __future__ import annotations

import argparse
import hmac
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

from .config import DEFAULT_RECONSTRUCTION_MODE, PIPELINE_VERSION, RECONSTRUCTION_MODES
from .exporter import secure_delete_files
from .inbody_ocr import InBodyServiceError, OpenAIInBodyService
from .scan_service import (
    BaseScanEngine,
    GpuScanEngine,
    MockScanEngine,
    ScanRequest,
    ScanResult,
    get_backend_capabilities,
)

load_dotenv()

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB per file
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BINARY_ASSET_FILES = {
    "scan.glb",
    "scan_core.glb",
    "scan_preview_blur.glb",
    "scan_preview_face.glb",
    "tex_blur.jpg",
    "tex_face.jpg",
}
ALLOWED_ARTIFACT_FILES = BINARY_ASSET_FILES | {"body_scan.json", "deletion_audit.jsonl"}

MIME_TYPES = {
    ".glb": "model/gltf-binary",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _sanitize_person_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]", "_", (value or "").strip())
    return cleaned[:64] or "guest_01"


def _allowed_files_for_privacy(privacy_level: int) -> Set[str]:
    if privacy_level == 1:
        return {"scan.glb", "scan_core.glb", "body_scan.json", "deletion_audit.jsonl"}
    if privacy_level == 2:
        return {
            "scan.glb",
            "scan_core.glb",
            "scan_preview_blur.glb",
            "tex_blur.jpg",
            "body_scan.json",
            "deletion_audit.jsonl",
        }
    return set(ALLOWED_ARTIFACT_FILES)


@dataclass
class ServerConfig:
    """Configuration options for the Plan 2A FastAPI server."""

    mock: bool = False
    sam_repo_dir: str = "."
    checkpoint_path: str = "./checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_path: str = "./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    region_mode: str = "internal_test"
    signing_secret: str = ""
    file_token_ttl: int = 300
    job_ttl_minutes: int = 1440
    reaper_interval_sec: int = 300
    data_dir: str = ""
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    serve_frontend: Optional[str] = None
    inbody_model: str = "gpt-5.6-luna"
    inbody_timeout_seconds: float = 60.0
    inbody_max_file_size_mb: int = 20
    inbody_max_pdf_pages: int = 5


@dataclass
class JobRecord:
    """In-memory + on-disk state for a single bodyscan job."""

    job_id: str
    job_dir: str
    outputs_dir: str
    person_id: str
    privacy_level: int
    mode: str
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE
    status: str = "queued"
    progress: int = 0
    step: str = "queued"
    error: Optional[str] = None
    assets_deleted: bool = False
    deleting: bool = False
    created_at: datetime = field(default_factory=_utc_now)
    expires_at: datetime = field(default_factory=lambda: _utc_now() + timedelta(days=1))
    deleted_at: Optional[datetime] = None
    result: Optional[ScanResult] = None
    retained_body_scan: Optional[Dict[str, Any]] = None


class ScanJobManager:
    """Thread-safe FIFO job queue, HMAC signer, Decision 3a deleter, and TTL reaper."""

    def __init__(
        self,
        config: ServerConfig,
        engine: Optional[BaseScanEngine] = None,
        engine_loader: Optional[Callable[[], BaseScanEngine]] = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self._engine_loader = engine_loader
        self._persist_enabled = bool((config.data_dir or "").strip())
        self.data_dir = os.path.abspath(
            config.data_dir or os.path.join(tempfile.gettempdir(), "alphascan_jobs")
        )
        os.makedirs(self.data_dir, exist_ok=True)
        self._secret_path = os.path.join(self.data_dir, ".signing_secret")
        self._index_path = os.path.join(self.data_dir, "jobs_index.json")
        self.server_audit_path = os.path.join(self.data_dir, "server_deletion_audit.jsonl")

        self._lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alphascan_worker")
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._reaper_thread: Optional[threading.Thread] = None
        self._engine_thread: Optional[threading.Thread] = None
        self._engine_error: Optional[str] = None
        self._secret_persisted = False

        if config.signing_secret:
            self.signing_secret = config.signing_secret.encode("utf-8")
            self._secret_persisted = True
        elif self._persist_enabled and os.path.isfile(self._secret_path):
            try:
                loaded_secret = Path(self._secret_path).read_text(encoding="utf-8").strip()
            except Exception:
                loaded_secret = ""
            if loaded_secret:
                self.signing_secret = loaded_secret.encode("utf-8")
                self._secret_persisted = True
            else:
                self.signing_secret = secrets.token_hex(32).encode("utf-8")
        else:
            self.signing_secret = secrets.token_hex(32).encode("utf-8")

        if engine is not None:
            self._engine_status = "ready"
            self._engine_mode = "mock" if isinstance(engine, MockScanEngine) else "gpu"
            self._ready_event.set()
        elif engine_loader is not None:
            self._engine_status = "loading"
            self._engine_mode = "mock" if config.mock else "gpu"
        elif config.mock:
            self.engine = MockScanEngine()
            self._engine_status = "ready"
            self._engine_mode = "mock"
            self._ready_event.set()
        else:
            self._engine_status = "loading"
            self._engine_mode = "gpu"
            self._engine_loader = lambda: GpuScanEngine(
                sam_repo_dir=config.sam_repo_dir,
                checkpoint_path=config.checkpoint_path,
                mhr_path=config.mhr_path,
            )

        self._load_persisted_state()

    def start_engine_loader(self) -> None:
        with self._lock:
            if (
                self.engine is not None
                or self._engine_loader is None
                or self._engine_status in ("ready", "error")
                or (self._engine_thread is not None and self._engine_thread.is_alive())
            ):
                return
            loader = self._engine_loader

            def _load() -> None:
                try:
                    loaded = loader()
                    if self._stop_event.is_set():
                        return
                    with self._lock:
                        self.engine = loaded
                        self._engine_mode = "mock" if isinstance(loaded, MockScanEngine) else "gpu"
                        self._engine_status = "ready"
                        self._engine_error = None
                except Exception as exc:
                    with self._lock:
                        self.engine = None
                        self._engine_status = "error"
                        self._engine_error = str(exc) or exc.__class__.__name__
                finally:
                    self._ready_event.set()

            self._engine_thread = threading.Thread(
                target=_load, daemon=True, name="alphascan_engine_loader"
            )
            self._engine_thread.start()

    def engine_health_state(self) -> Tuple[str, str, Optional[str]]:
        with self._lock:
            return self._engine_status, self._engine_mode, self._engine_error

    def current_engine(self) -> Optional[BaseScanEngine]:
        with self._lock:
            return self.engine

    def _ensure_secret_persisted(self) -> None:
        if not self._persist_enabled or self._secret_persisted:
            return
        with self._persist_lock:
            if self._secret_persisted:
                return
            os.makedirs(self.data_dir, exist_ok=True)
            if os.path.isfile(self._secret_path):
                try:
                    existing = Path(self._secret_path).read_text(encoding="utf-8").strip()
                    if existing:
                        self.signing_secret = existing.encode("utf-8")
                        self._secret_persisted = True
                        return
                except Exception:
                    pass
            tmp_path = f"{self._secret_path}.{threading.get_ident()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                fp.write(self.signing_secret.decode("utf-8"))
            try:
                os.chmod(tmp_path, 0o600)
            except Exception:
                pass
            os.replace(tmp_path, self._secret_path)
            self._secret_persisted = True

    def _serialize_job(self, job: JobRecord) -> Dict[str, Any]:
        res = job.result
        artifact_filenames: Dict[str, str] = {}
        if res and res.artifacts:
            for key, path in res.artifacts.items():
                if path:
                    artifact_filenames[key] = os.path.basename(path)
        return {
            "job_id": job.job_id,
            "job_dir": job.job_dir,
            "outputs_dir": job.outputs_dir,
            "person_id": job.person_id,
            "privacy_level": int(job.privacy_level),
            "mode": job.mode,
            "reconstruction_mode": job.reconstruction_mode,
            "effective_reconstruction_mode": (
                res.effective_reconstruction_mode if res is not None else job.reconstruction_mode
            ),
            "status": job.status,
            "progress": int(job.progress),
            "step": job.step,
            "error": job.error,
            "assets_deleted": bool(job.assets_deleted),
            "created_at": _iso_utc(job.created_at),
            "expires_at": _iso_utc(job.expires_at),
            "deleted_at": _iso_utc(job.deleted_at) if job.deleted_at else None,
            "raw_height_cm": res.raw_height_cm if res is not None else None,
            "elapsed_sec": res.elapsed_sec if res is not None else None,
            "warnings": list(res.warnings) if res is not None else [],
            "proportion_audit": res.proportion_audit if res is not None else None,
            "viewer_artifact_key": res.viewer_artifact_key if res is not None else "glb",
            "artifact_filenames": artifact_filenames,
            "deletion_audit": res.deletion_audit if res is not None else {},
            "images_used": res.images_used if res is not None else 0,
            "images_total": res.images_total if res is not None else 0,
            "texture_summary": res.texture_summary if res is not None else "",
        }

    def _save_job_index(self) -> None:
        if not self._persist_enabled:
            return
        with self._persist_lock:
            with self._lock:
                rows = [self._serialize_job(job) for job in self._jobs.values()]
            os.makedirs(self.data_dir, exist_ok=True)
            payload = {"version": 1, "jobs": rows}
            tmp_path = f"{self._index_path}.{threading.get_ident()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._index_path)

    def _load_persisted_state(self) -> None:
        if not self._persist_enabled or not os.path.isfile(self._index_path):
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception:
            return
        rows = payload.get("jobs", []) if isinstance(payload, dict) else []
        dirty = False
        loaded: Dict[str, JobRecord] = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("job_id"):
                continue
            jid = str(row["job_id"])
            job_dir = str(row.get("job_dir") or os.path.join(self.data_dir, jid))
            outputs_dir = str(row.get("outputs_dir") or os.path.join(job_dir, "outputs"))
            status = str(row.get("status") or "failed")
            step = str(row.get("step") or status)
            error = row.get("error")
            assets_deleted = bool(row.get("assets_deleted", False))

            if status in ("queued", "running"):
                uploads_dir = os.path.join(job_dir, "uploads")
                if os.path.isdir(uploads_dir):
                    leftovers = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir)]
                    if leftovers:
                        os.makedirs(outputs_dir, exist_ok=True)
                        secure_delete_files(
                            leftovers,
                            os.path.join(outputs_dir, "deletion_audit.jsonl"),
                            actor="api_server_restart",
                        )
                status = "failed"
                step = "failed"
                error = "Interrupted by server restart"
                dirty = True

            body_json: Optional[Dict[str, Any]] = None
            body_json_path = os.path.join(outputs_dir, "body_scan.json")
            if os.path.isfile(body_json_path):
                try:
                    with open(body_json_path, "r", encoding="utf-8") as bfp:
                        body_json = json.load(bfp)
                except Exception:
                    body_json = None

            rebuilt_result: Optional[ScanResult] = None
            if status == "completed" and body_json is not None:
                raw_map = row.get("artifact_filenames") or {}
                artifacts: Dict[str, str] = {}
                if isinstance(raw_map, dict):
                    for key, fname in raw_map.items():
                        if not fname:
                            continue
                        candidate = os.path.join(outputs_dir, os.path.basename(str(fname)))
                        if os.path.isfile(candidate):
                            artifacts[str(key)] = candidate
                if "json" not in artifacts and os.path.isfile(body_json_path):
                    artifacts["json"] = body_json_path
                rebuilt_result = ScanResult(
                    body_json=body_json,
                    artifacts=artifacts,
                    viewer_artifact_key=str(row.get("viewer_artifact_key") or "glb"),
                    warnings=list(row.get("warnings") or []),
                    deletion_audit=dict(row.get("deletion_audit") or {}),
                    raw_height_cm=float(row.get("raw_height_cm") or body_json.get("height_cm", 0.0)),
                    elapsed_sec=float(row.get("elapsed_sec") or 0.0),
                    mode=str(row.get("mode") or "mock"),
                    images_used=int(row.get("images_used") or 0),
                    images_total=int(row.get("images_total") or 0),
                    texture_summary=str(row.get("texture_summary") or ""),
                    reconstruction_mode=str(row.get("reconstruction_mode") or DEFAULT_RECONSTRUCTION_MODE),
                    effective_reconstruction_mode=str(
                        row.get("effective_reconstruction_mode")
                        or row.get("reconstruction_mode")
                        or DEFAULT_RECONSTRUCTION_MODE
                    ),
                    proportion_audit=row.get("proportion_audit"),
                )

            created_at = _parse_iso_utc(row.get("created_at")) or _utc_now()
            expires_at = _parse_iso_utc(row.get("expires_at")) or (created_at + timedelta(days=1))
            deleted_at = _parse_iso_utc(row.get("deleted_at"))
            loaded[jid] = JobRecord(
                job_id=jid,
                job_dir=job_dir,
                outputs_dir=outputs_dir,
                person_id=str(row.get("person_id") or "guest_01"),
                privacy_level=int(row.get("privacy_level") or 2),
                mode=str(row.get("mode") or "mock"),
                reconstruction_mode=str(row.get("reconstruction_mode") or DEFAULT_RECONSTRUCTION_MODE),
                status=status,
                progress=int(row.get("progress") or 0),
                step=step,
                error=str(error) if error else None,
                assets_deleted=assets_deleted,
                created_at=created_at,
                expires_at=expires_at,
                deleted_at=deleted_at,
                result=rebuilt_result,
                retained_body_scan=body_json,
            )

        with self._lock:
            self._jobs.update(loaded)
        if dirty:
            self._save_job_index()
        self.reap_expired_jobs()

    def start_reaper(self) -> None:
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            return

        def _loop() -> None:
            interval = max(1, int(self.config.reaper_interval_sec))
            while not self._stop_event.wait(interval):
                self.reap_expired_jobs()

        self._reaper_thread = threading.Thread(target=_loop, daemon=True, name="alphascan_reaper")
        self._reaper_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._ready_event.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=2.0)
        if self._engine_thread is not None:
            self._engine_thread.join(timeout=0.5)
        self._executor.shutdown(wait=True, cancel_futures=False)

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def _append_server_audit(self, record: Dict[str, Any]) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        line = json.dumps({"timestamp_utc": _iso_utc(), **record}, ensure_ascii=False)
        with self._lock:
            with open(self.server_audit_path, "a", encoding="utf-8") as fp:
                fp.write(line + "\n")

    def _sign(self, job_id: str, filename: str, exp_ts: int) -> str:
        msg = f"{job_id}:{filename}:{exp_ts}".encode("utf-8")
        return hmac.new(self.signing_secret, msg, hashlib.sha256).hexdigest()

    def sign_file_url(self, job_id: str, filename: str, exp_ts: Optional[int] = None) -> str:
        self._ensure_secret_persisted()
        ttl = max(30, int(self.config.file_token_ttl))
        exp = int(time.time()) + ttl if exp_ts is None else int(exp_ts)
        sig = self._sign(job_id, filename, exp)
        return f"/api/v1/bodyscan/{job_id}/files/{filename}?exp={exp}&sig={sig}"

    def verify_file_signature(self, job_id: str, filename: str, exp: int, sig: str) -> None:
        if int(exp) < int(time.time()):
            raise HTTPException(status_code=403, detail="Signed URL expired")
        expected = self._sign(job_id, filename, int(exp))
        if not hmac.compare_digest(expected, sig or ""):
            raise HTTPException(status_code=403, detail="Invalid file signature")

    def submit(
        self,
        image_paths: List[str],
        face_image_path: Optional[str],
        job_id: str,
        job_dir: str,
        outputs_dir: str,
        height_cm: float,
        sex: str,
        weight_kg: float,
        person_id: str,
        privacy_level: int,
        reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
        source_filenames: Optional[List[str]] = None,
        slot_roles: Optional[List[str]] = None,
    ) -> JobRecord:
        self._ensure_secret_persisted()
        now = _utc_now()
        ttl_min = max(1, int(self.config.job_ttl_minutes))
        with self._lock:
            mode = (
                "mock"
                if isinstance(self.engine, MockScanEngine)
                else self._engine_mode
            )
            step_init = "queued" if self._ready_event.is_set() else "warming_up_model"
        rec = JobRecord(
            job_id=job_id,
            job_dir=job_dir,
            outputs_dir=outputs_dir,
            person_id=person_id,
            privacy_level=privacy_level,
            mode=mode,
            reconstruction_mode=reconstruction_mode,
            step=step_init,
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_min),
        )
        with self._lock:
            self._jobs[job_id] = rec
        self._save_job_index()

        req = ScanRequest(
            image_paths=image_paths,
            face_image_path=face_image_path,
            height_cm=height_cm,
            sex=sex,
            weight_kg=weight_kg,
            person_id=person_id,
            privacy_level=privacy_level,
            reconstruction_mode=reconstruction_mode,
            consent_granted=True,
            output_dir=outputs_dir,
            actor="api_server",
            save_debug_plots=False,
            source_filenames=source_filenames,
            slot_roles=slot_roles,
        )
        self._executor.submit(self._run_job, job_id, req)
        return rec

    def _run_job(self, job_id: str, req: ScanRequest) -> None:
        def _progress(pct: int, step: str) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status != "deleted":
                    job.progress = pct
                    job.step = step

        def _cleanup_uploads() -> None:
            uploads_dir = os.path.join(req.output_dir and os.path.dirname(req.output_dir) or "", "uploads")
            if os.path.isdir(uploads_dir):
                leftovers = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir)]
                if leftovers:
                    os.makedirs(req.output_dir, exist_ok=True)
                    secure_delete_files(
                        leftovers,
                        os.path.join(req.output_dir, "deletion_audit.jsonl"),
                        actor="api_server",
                    )

        if not self._ready_event.is_set():
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status != "deleted":
                    job.step = "warming_up_model"
            while not self._ready_event.wait(timeout=0.05):
                if self._stop_event.is_set():
                    return
        if self._stop_event.is_set():
            return

        with self._lock:
            engine = self.engine
            eng_status = self._engine_status
            eng_err = self._engine_error
            eng_mode = self._engine_mode
            job = self._jobs.get(job_id)
            if not job or job.status == "deleted":
                return

        if engine is None or eng_status != "ready":
            _cleanup_uploads()
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status != "deleted":
                    job.status = "failed"
                    job.error = f"Model initialization failed: {eng_err or 'unknown error'}"
                    job.step = "failed"
            self._save_job_index()
            return

        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status == "deleted":
                return
            job.mode = eng_mode
            job.status = "running"
            job.progress = 5
            job.step = "running"

        try:
            res = engine.process(req, progress_cb=_progress)
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status != "deleted":
                    job.result = res
                    job.retained_body_scan = res.body_json
                    job.status = "completed"
                    job.progress = 100
                    job.step = "completed"
        except Exception as exc:
            _cleanup_uploads()
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status != "deleted":
                    job.status = "failed"
                    job.error = str(exc)
                    job.step = "failed"
        self._save_job_index()

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        return job

    def build_status_payload(self, job: JobRecord) -> Dict[str, Any]:
        with self._lock:
            job_id = job.job_id
            status = job.status
            progress = job.progress
            step = job.step
            mode = job.mode
            privacy_level = job.privacy_level
            reconstruction_mode = job.reconstruction_mode
            created_at = job.created_at
            expires_at = job.expires_at
            error = job.error
            assets_deleted = job.assets_deleted
            deleted_at = job.deleted_at
            res = job.result
            retained_body_scan = job.retained_body_scan
            outputs_dir = job.outputs_dir

        base: Dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "step": step,
            "mode": mode,
            "privacy_level": privacy_level,
            "reconstruction_mode": reconstruction_mode,
            "created_at": _iso_utc(created_at),
            "expires_at": _iso_utc(expires_at),
        }
        if status == "failed":
            base["error"] = error or "Processing failed"
            return base

        if status == "deleted":
            base["assets_deleted"] = assets_deleted
            base["measurements_retained"] = bool(retained_body_scan)
            base["deleted_at"] = _iso_utc(deleted_at) if deleted_at else _iso_utc()
            if retained_body_scan:
                base["body_scan"] = retained_body_scan
                if isinstance(retained_body_scan, dict) and retained_body_scan.get("backends_used") is not None:
                    base["backends_used"] = retained_body_scan.get("backends_used")
                if os.path.exists(os.path.join(outputs_dir, "body_scan.json")):
                    base["artifacts"] = {"json_url": self.sign_file_url(job_id, "body_scan.json")}
            return base

        if status != "completed" or res is None:
            return base

        artifacts_map: Dict[str, str] = {}
        allowed = _allowed_files_for_privacy(privacy_level)

        viewer_path = res.artifacts.get(res.viewer_artifact_key) or res.artifacts.get("glb")
        viewer_name = os.path.basename(viewer_path) if viewer_path else "scan.glb"
        if viewer_name not in allowed:
            viewer_name = "scan.glb"

        artifacts_map["mesh_url"] = self.sign_file_url(job_id, viewer_name)
        if "core_glb" in res.artifacts:
            core_name = os.path.basename(res.artifacts["core_glb"])
            if core_name in allowed:
                artifacts_map["core_mesh_url"] = self.sign_file_url(job_id, core_name)
        if "glb" in res.artifacts:
            artifacts_map["glb_url"] = self.sign_file_url(job_id, os.path.basename(res.artifacts["glb"]))
        if "json" in res.artifacts:
            artifacts_map["json_url"] = self.sign_file_url(job_id, os.path.basename(res.artifacts["json"]))

        if privacy_level == 3 and "tex_face" in res.artifacts:
            tex_name = os.path.basename(res.artifacts["tex_face"])
            if tex_name in allowed:
                artifacts_map["texture_url"] = self.sign_file_url(job_id, tex_name)
        elif privacy_level >= 2 and "tex_blur" in res.artifacts:
            tex_name = os.path.basename(res.artifacts["tex_blur"])
            if tex_name in allowed:
                artifacts_map["texture_url"] = self.sign_file_url(job_id, tex_name)

        backends_used = (res.body_json or {}).get("backends_used")
        base.update(
            {
                "raw_height_cm": res.raw_height_cm,
                "elapsed_sec": res.elapsed_sec,
                "warnings": list(res.warnings),
                "reconstruction_mode": res.reconstruction_mode,
                "effective_reconstruction_mode": res.effective_reconstruction_mode,
                "proportion_audit": res.proportion_audit,
                "backends_used": backends_used,
                "body_scan": res.body_json,
                "artifacts": artifacts_map,
                "deletion_audit": res.deletion_audit,
            }
        )
        return base

    def resolve_artifact_path(self, job_id: str, filename: str) -> str:
        if (
            not filename
            or os.path.basename(filename) != filename
            or "/" in filename
            or "\\" in filename
            or ".." in filename
            or filename not in ALLOWED_ARTIFACT_FILES
        ):
            raise HTTPException(status_code=404, detail="Invalid artifact filename")

        job = self.get_job(job_id)
        allowed = _allowed_files_for_privacy(job.privacy_level)
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="Artifact unavailable for this privacy level")

        if job.assets_deleted and filename in BINARY_ASSET_FILES:
            raise HTTPException(status_code=410, detail="3D model and image assets were deleted from server")

        target = os.path.join(job.outputs_dir, filename)
        if not os.path.isfile(target):
            raise HTTPException(status_code=404, detail="Artifact file not found")
        return target

    def delete_job(self, job_id: str, purge_all: bool = False) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found or expired")
            if job.status in ("queued", "running"):
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete a scan job while it is queued or running",
                )
            if job.deleting:
                raise HTTPException(
                    status_code=409,
                    detail="Deletion is already in progress for this job",
                )
            job.deleting = True
            job_dir = job.job_dir
            outputs_dir = job.outputs_dir
        now = _utc_now()

        try:
            if purge_all:
                if os.path.isdir(job_dir):
                    shutil.rmtree(job_dir, ignore_errors=True)
                with self._lock:
                    self._jobs.pop(job_id, None)
                self._save_job_index()
                self._append_server_audit({"job_id": job_id, "reason": "manual_purge_all", "purged": True})
                return {
                    "ok": True,
                    "job_id": job_id,
                    "status": "purged",
                    "assets_deleted": True,
                    "measurements_retained": False,
                    "deleted_files": [],
                    "deleted_at": _iso_utc(now),
                }

            # Decision 3a: delete binary 3D meshes and image textures while preserving body_scan.json
            deleted_names: List[str] = []
            to_erase: List[str] = []
            if os.path.isdir(job_dir):
                for root, _, files in os.walk(job_dir):
                    for fname in files:
                        if fname.lower().endswith((".glb", ".jpg", ".jpeg", ".png")):
                            p_full = os.path.join(root, fname)
                            if os.path.isfile(p_full):
                                to_erase.append(p_full)
                                deleted_names.append(fname)

            if to_erase:
                audit_log = os.path.join(outputs_dir, "deletion_audit.jsonl")
                secure_delete_files(to_erase, audit_log, actor="manual_delete_assets")

            with self._lock:
                job.assets_deleted = True
                job.status = "deleted"
                job.deleted_at = now
                if job.retained_body_scan is None and job.result is not None:
                    job.retained_body_scan = job.result.body_json
                retained_flag = bool(job.retained_body_scan)

            self._save_job_index()
            self._append_server_audit(
                {
                    "job_id": job_id,
                    "reason": "manual_delete_assets",
                    "deleted_files": deleted_names,
                    "measurements_retained": True,
                }
            )
            return {
                "ok": True,
                "job_id": job_id,
                "status": "deleted",
                "assets_deleted": True,
                "measurements_retained": retained_flag,
                "deleted_files": sorted(deleted_names),
                "deleted_at": _iso_utc(now),
            }
        finally:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].deleting = False

    def reap_expired_jobs(self, now_utc: Optional[datetime] = None) -> int:
        now = now_utc or _utc_now()
        with self._lock:
            expired_ids = [
                jid
                for jid, j in self._jobs.items()
                if j.expires_at <= now and j.status not in ("queued", "running") and not j.deleting
            ]
        count = 0
        for jid in expired_ids:
            with self._lock:
                cur = self._jobs.get(jid)
                if cur is None or cur.deleting:
                    continue
                job = self._jobs.pop(jid, None)
            if job is None:
                continue
            if os.path.isdir(job.job_dir):
                shutil.rmtree(job.job_dir, ignore_errors=True)
            self._append_server_audit({"job_id": jid, "reason": "ttl_expired"})
            count += 1
        if count > 0:
            self._save_job_index()
        return count


def _parse_bool_form(raw: str) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _verify_api_key_factory(expected_key: str):
    def _dep(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if not expected_key:
            return
        token = ""
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        elif x_api_key:
            token = x_api_key.strip()
        if not token or not hmac.compare_digest(token.encode("utf-8"), expected_key.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return _dep


_VALID_SLOT_ROLES = {"front", "right", "back", "left", ""}


def _parse_slot_roles(raw_roles: Optional[List[str]], expected_count: int) -> Optional[List[str]]:
    """Parse slot_roles from repeated FormData fields or a JSON array string and validate."""
    if not raw_roles:
        return None
    if len(raw_roles) == 1 and str(raw_roles[0]).strip().startswith("["):
        try:
            parsed = json.loads(raw_roles[0])
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid slot_roles JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise HTTPException(status_code=422, detail="slot_roles JSON must be an array of strings.")
        items = [str(x or "").strip().lower() for x in parsed]
    else:
        items = [str(x or "").strip().lower() for x in raw_roles]

    if len(items) != expected_count:
        raise HTTPException(
            status_code=422,
            detail=f"slot_roles count ({len(items)}) must match uploaded photos count ({expected_count}).",
        )
    for role in items:
        if role not in _VALID_SLOT_ROLES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid slot_role {role!r}; expected one of 'front', 'right', 'back', 'left', or ''.",
            )
    return items


def _verify_and_write_upload_sync(raw: bytes, ext: str, uploads_dir: str, target_stem: str) -> str:
    """Validate image bytes with PIL and write to disk (safe for threadpool execution)."""
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
            fmt = (img.format or "JPEG").lower()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image file: {exc}") from exc
    safe_ext = ext if ext in ALLOWED_EXTENSIONS else (f".{fmt}" if f".{fmt}" in ALLOWED_EXTENSIONS else ".jpg")
    out_path = os.path.join(uploads_dir, f"{target_stem}{safe_ext}")
    with open(out_path, "wb") as fp:
        fp.write(raw)
    return out_path


def create_app(
    config: ServerConfig,
    engine: Optional[BaseScanEngine] = None,
    engine_loader: Optional[Callable[[], BaseScanEngine]] = None,
    inbody_service: Optional[Any] = None,
) -> FastAPI:
    """Build and configure the FastAPI application."""
    if engine is None and engine_loader is None:
        if config.mock:
            engine = MockScanEngine()
        else:
            def _default_gpu_loader() -> BaseScanEngine:
                return GpuScanEngine(
                    sam_repo_dir=config.sam_repo_dir,
                    checkpoint_path=config.checkpoint_path,
                    mhr_path=config.mhr_path,
                )

            engine_loader = _default_gpu_loader

    manager = ScanJobManager(config=config, engine=engine, engine_loader=engine_loader)
    if engine_loader is not None and engine is None:
        manager.start_engine_loader()
    require_auth = _verify_api_key_factory(config.api_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.reap_expired_jobs()
        manager.start_engine_loader()
        manager.start_reaper()
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="Alpha Scan Plus/PRO API", version=PIPELINE_VERSION, lifespan=lifespan)
    app.state.scan_manager = manager
    app.state.config = config
    app.state.inbody_ocr = inbody_service or OpenAIInBodyService(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=config.inbody_model,
        timeout_seconds=config.inbody_timeout_seconds,
        max_file_size_mb=config.inbody_max_file_size_mb,
        max_pdf_pages=config.inbody_max_pdf_pages,
    )

    origins = config.cors_origins or ["*"]
    allow_all = origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        cuda_ok = False
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        region_mode = config.region_mode or "internal_test"
        eng_status, eng_mode, eng_err = manager.engine_health_state()
        backends = get_backend_capabilities(
            manager.current_engine(),
            engine_mode=eng_mode,
            engine_status=eng_status,
        )
        payload: Dict[str, Any] = {
            "ok": eng_status != "error",
            "status": eng_status,
            "model_ready": eng_status == "ready",
            "mode": eng_mode,
            "region_mode": region_mode,
            "test_only": region_mode != "vn_prod",
            "pipeline_version": PIPELINE_VERSION,
            "cuda_available": bool(backends.get("cuda_available", cuda_ok)),
            "auth_required": bool(config.api_key),
            "queue_depth": manager.queue_depth(),
            "backends": backends,
            "inbody_ocr": app.state.inbody_ocr.health(),
        }
        if eng_err:
            payload["error"] = eng_err
            payload["detail"] = eng_err
        return payload

    @app.post("/api/v1/inbody/extract", dependencies=[Depends(require_auth)])
    async def extract_inbody(
        file: UploadFile = File(...),
        consent_granted: str = Form("false"),
    ) -> Dict[str, Any]:
        """Extract one InBody report without persisting the source document."""
        if not _parse_bool_form(consent_granted):
            raise HTTPException(
                status_code=403,
                detail="Explicit client consent (consent_granted=true) is required.",
            )
        max_bytes = config.inbody_max_file_size_mb * 1024 * 1024
        try:
            raw = await file.read(max_bytes + 1)
        finally:
            await file.close()
        if len(raw) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded document exceeds {config.inbody_max_file_size_mb} MB limit.",
            )
        try:
            return await app.state.inbody_ocr.extract(raw, file.filename or "inbody-upload")
        except InBodyServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/v1/bodyscan", status_code=202, dependencies=[Depends(require_auth)])
    async def create_bodyscan(
        photos: List[UploadFile] = File(default=[]),
        closeup_face: Optional[UploadFile] = File(default=None),
        height_cm: float = Form(...),
        sex: str = Form("female"),
        weight_kg: float = Form(0.0),
        person_id: str = Form("guest_01"),
        privacy_level: int = Form(2),
        reconstruction_mode: str = Form(DEFAULT_RECONSTRUCTION_MODE),
        consent_granted: str = Form("false"),
        slot_roles: Optional[List[str]] = Form(default=None),
    ) -> Dict[str, Any]:
        if not _parse_bool_form(consent_granted):
            raise HTTPException(
                status_code=403,
                detail="Explicit client consent (consent_granted=true) is required.",
            )
        height_val = float(height_cm)
        if not math.isfinite(height_val) or not (80.0 <= height_val <= 230.0):
            raise HTTPException(status_code=422, detail="height_cm must be a finite number between 80 and 230.")
        sex_clean = (sex or "").strip().lower()
        if sex_clean not in ("male", "female"):
            raise HTTPException(status_code=422, detail="sex must be 'male' or 'female'.")
        if int(privacy_level) not in (1, 2, 3):
            raise HTTPException(status_code=422, detail="privacy_level must be 1, 2, or 3.")
        mode_clean = (reconstruction_mode or DEFAULT_RECONSTRUCTION_MODE).strip()
        if mode_clean not in RECONSTRUCTION_MODES:
            raise HTTPException(
                status_code=422,
                detail=f"reconstruction_mode must be one of {RECONSTRUCTION_MODES}.",
            )
        weight_val = float(weight_kg)
        if not math.isfinite(weight_val) or weight_val < 0.0:
            raise HTTPException(status_code=422, detail="weight_kg must be a finite non-negative number.")
        if not (1 <= len(photos) <= 8):
            raise HTTPException(status_code=422, detail="Between 1 and 8 body photos are required.")
        parsed_slot_roles = _parse_slot_roles(slot_roles, len(photos))

        eng_status, _, eng_err = manager.engine_health_state()
        if eng_status == "error":
            raise HTTPException(
                status_code=503,
                detail=f"Model initialization failed: {eng_err or 'unknown error'}",
            )

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(manager.data_dir, job_id)
        uploads_dir = os.path.join(job_dir, "uploads")
        outputs_dir = os.path.join(job_dir, "outputs")
        os.makedirs(uploads_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

        async def _validate_and_save(up: UploadFile, target_stem: str) -> str:
            raw = await up.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded image exceeds 15 MB limit.")
            ext = os.path.splitext(up.filename or "")[1].lower()
            if ext and ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=422, detail=f"Unsupported file extension: {ext}")
            return await run_in_threadpool(_verify_and_write_upload_sync, raw, ext, uploads_dir, target_stem)

        saved_paths: List[str] = []
        source_filenames: List[str] = []
        saved_face: Optional[str] = None
        try:
            for idx, photo in enumerate(photos):
                saved_paths.append(await _validate_and_save(photo, f"photo_{idx:02d}"))
                source_filenames.append(photo.filename or f"photo_{idx:02d}")
            if closeup_face is not None and (closeup_face.filename or "").strip():
                saved_face = await _validate_and_save(closeup_face, "closeup_face")
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        rec = manager.submit(
            image_paths=saved_paths,
            face_image_path=saved_face,
            job_id=job_id,
            job_dir=job_dir,
            outputs_dir=outputs_dir,
            height_cm=height_val,
            sex=sex_clean,
            weight_kg=weight_val,
            person_id=_sanitize_person_id(person_id),
            privacy_level=int(privacy_level),
            reconstruction_mode=mode_clean,
            source_filenames=source_filenames,
            slot_roles=parsed_slot_roles,
        )
        return {
            "job_id": rec.job_id,
            "status": rec.status,
            "progress": rec.progress,
            "step": rec.step,
        }

    @app.get("/api/v1/bodyscan/{job_id}", dependencies=[Depends(require_auth)])
    def get_bodyscan_status(job_id: str) -> Dict[str, Any]:
        job = manager.get_job(job_id)
        return manager.build_status_payload(job)

    @app.get("/api/v1/bodyscan/{job_id}/files/{filename}")
    def get_bodyscan_file(
        job_id: str,
        filename: str,
        exp: int = Query(...),
        sig: str = Query(...),
    ) -> FileResponse:
        manager.verify_file_signature(job_id, filename, exp, sig)
        path = manager.resolve_artifact_path(job_id, filename)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Artifact not found or already deleted")
        ext = os.path.splitext(filename)[1].lower()
        media_type = MIME_TYPES.get(ext, "application/octet-stream")
        try:
            return FileResponse(path=path, media_type=media_type, filename=filename)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=404, detail="Artifact not found or already deleted") from exc

    @app.post("/api/v1/bodyscan/{job_id}/delete", dependencies=[Depends(require_auth)])
    def post_delete_bodyscan(job_id: str, purge_all: bool = Query(default=False)) -> Dict[str, Any]:
        return manager.delete_job(job_id, purge_all=purge_all)

    @app.delete("/api/v1/bodyscan/{job_id}", dependencies=[Depends(require_auth)])
    def delete_bodyscan_alias(job_id: str, purge_all: bool = Query(default=False)) -> Dict[str, Any]:
        return manager.delete_job(job_id, purge_all=purge_all)

    if config.serve_frontend:
        fe_path = Path(config.serve_frontend).resolve()
        if not fe_path.is_dir():
            raise ValueError(f"--serve-frontend directory does not exist: {config.serve_frontend}")
        app.mount("/lab", StaticFiles(directory=str(fe_path), html=True), name="frontend_lab")

    return app


def _parse_args(argv: Optional[List[str]] = None) -> ServerConfig:
    parser = argparse.ArgumentParser(description="Alpha Scan Plus/PRO GPU & Mock FastAPI Server")
    parser.add_argument("--mock", action="store_true", help="Run CPU mock engine (no GPU/SAM3D needed)")
    parser.add_argument("--sam_repo_dir", type=str, default=".")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr_path", type=str, default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default=os.environ.get("ALPHASCAN_API_KEY", ""))
    parser.add_argument(
        "--region-mode",
        type=str,
        choices=["internal_test", "vn_prod"],
        default=os.environ.get("ALPHASCAN_REGION_MODE", "internal_test"),
    )
    parser.add_argument("--signing-secret", type=str, default=os.environ.get("ALPHASCAN_SIGNING_SECRET", ""))
    parser.add_argument(
        "--file-token-ttl",
        type=int,
        default=int(os.environ.get("ALPHASCAN_FILE_TOKEN_TTL", "1800")),
        help="Signed artifact URL TTL in seconds (CLI default: 1800)",
    )
    parser.add_argument("--job-ttl-minutes", type=int, default=1440)
    parser.add_argument("--reaper-interval-sec", type=int, default=300)
    parser.add_argument("--data-dir", type=str, default=os.environ.get("ALPHASCAN_DATA_DIR", ""))
    parser.add_argument("--cors-origins", type=str, default="*")
    parser.add_argument("--serve-frontend", type=str, default=None, help="Optional frontend dir mounted at /lab")
    parser.add_argument(
        "--inbody-model",
        type=str,
        default=os.environ.get("OPENAI_INBODY_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument(
        "--inbody-timeout-seconds",
        type=float,
        default=float(os.environ.get("INBODY_OCR_TIMEOUT_SECONDS", "60")),
    )
    parser.add_argument(
        "--inbody-max-file-size-mb",
        type=int,
        default=int(os.environ.get("INBODY_OCR_MAX_FILE_SIZE_MB", "20")),
    )
    parser.add_argument(
        "--inbody-max-pdf-pages",
        type=int,
        default=int(os.environ.get("INBODY_OCR_MAX_PDF_PAGES", "5")),
    )
    args = parser.parse_args(argv)

    origins = [o.strip() for o in args.cors_origins.split(",") if o.strip()] or ["*"]
    return ServerConfig(
        mock=args.mock,
        sam_repo_dir=args.sam_repo_dir,
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        region_mode=args.region_mode,
        signing_secret=args.signing_secret,
        file_token_ttl=args.file_token_ttl,
        job_ttl_minutes=args.job_ttl_minutes,
        reaper_interval_sec=args.reaper_interval_sec,
        data_dir=args.data_dir,
        cors_origins=origins,
        serve_frontend=args.serve_frontend,
        inbody_model=args.inbody_model,
        inbody_timeout_seconds=args.inbody_timeout_seconds,
        inbody_max_file_size_mb=args.inbody_max_file_size_mb,
        inbody_max_pdf_pages=args.inbody_max_pdf_pages,
    )


def main(argv: Optional[List[str]] = None) -> None:
    import uvicorn

    cfg = _parse_args(argv)
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
