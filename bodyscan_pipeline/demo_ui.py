"""Gradio web demo UI for internal testing of Alpha Scan Step 1 + Step 2 + Step 3 (AI/consented images only)."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, List, Optional, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import gradio as gr
except ImportError:  # Allow importing constants and ScanDemoEngine without Gradio installed
    gr = None  # type: ignore[assignment]

from .config import DEFAULT_RECONSTRUCTION_MODE, RECONSTRUCTION_MODES
from .scan_service import (
    ANATOMICAL_ORDER,
    BODY_LABELS_VI,
    PRIVACY_CHOICES,
    PRIVACY_LEVEL_MAP,
    GpuScanEngine,
    ScanRequest,
)


def _raise_ui_error(msg: str) -> None:
    if gr is not None:
        raise gr.Error(msg)
    raise ValueError(msg)


def _prune_stale_demo_dirs(max_age_sec: float = 3600.0, keep_recent: int = 3) -> None:
    """Remove old alphascan_demo_* temp directories while preserving the most recent outputs."""
    tmp_root = tempfile.gettempdir()
    try:
        entries = [
            os.path.join(tmp_root, name)
            for name in os.listdir(tmp_root)
            if name.startswith("alphascan_demo_")
        ]
        dirs = [(p, os.path.getmtime(p)) for p in entries if os.path.isdir(p)]
    except OSError:
        return
    dirs.sort(key=lambda item: item[1], reverse=True)
    now = time.time()
    for p, mtime in dirs[keep_recent:]:
        if (now - mtime) >= max_age_sec:
            shutil.rmtree(p, ignore_errors=True)


class ScanDemoEngine:
    """Thread-safe wrapper holding preloaded SAM 3D Body & MHR models for Gradio."""

    def __init__(self, sam_repo_dir: str, checkpoint_path: str, mhr_path: str) -> None:
        self._engine = GpuScanEngine(sam_repo_dir, checkpoint_path, mhr_path)
        self._lock = self._engine._lock
        self.model = self._engine.model
        self.estimator = self._engine.estimator
        self.head_assets = self._engine.head_assets
        self.mhr_model = self._engine.mhr_model

    def process(
        self,
        uploaded_files: List[Any],
        face_file: Optional[Any],
        height_cm: float,
        sex: str,
        weight_kg: float,
        person_id: str,
        privacy_choice: str = PRIVACY_CHOICES[2],
        reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
        consent_granted: bool = True,
    ) -> Tuple[str, List[List[Any]], str, List[str], List[str]]:
        """Run full Step 1 + Step 2 (+ Step 3 texturing) pipeline and delete temp uploaded photos afterward."""
        if not consent_granted:
            _raise_ui_error("Vui lòng tích chọn xác nhận đồng ý xử lý ảnh (Consent) trước khi chạy.")
        if not uploaded_files:
            _raise_ui_error("Vui lòng tải lên ít nhất 1–4 ảnh (tốt nhất 4 góc: trước, phải, sau, trái).")
        if height_cm is None or not math.isfinite(float(height_cm)) or float(height_cm) < 80.0 or float(height_cm) > 230.0:
            _raise_ui_error("Chiều cao cần nằm trong khoảng 80 – 230 cm.")
        weight_val = float(weight_kg or 0.0)
        if not math.isfinite(weight_val) or weight_val < 0.0:
            _raise_ui_error("Cân nặng không hợp lệ (phải là số không âm hữu hạn).")

        privacy_level = PRIVACY_LEVEL_MAP.get(privacy_choice, 3)
        image_paths = [f if isinstance(f, str) else f.name for f in uploaded_files]
        face_path: Optional[str] = None
        if face_file is not None:
            face_path = face_file if isinstance(face_file, str) else getattr(face_file, "name", None)

        _prune_stale_demo_dirs()
        out_dir = tempfile.mkdtemp(prefix="alphascan_demo_")
        try:
            res = self._engine.process(
                ScanRequest(
                    image_paths=image_paths,
                    face_image_path=face_path,
                    height_cm=float(height_cm),
                    sex=sex,
                    weight_kg=weight_val,
                    person_id=(person_id or "demo_subject").strip(),
                    privacy_level=privacy_level,
                    reconstruction_mode=reconstruction_mode or DEFAULT_RECONSTRUCTION_MODE,
                    consent_granted=bool(consent_granted),
                    output_dir=out_dir,
                    actor="gradio_demo",
                    save_debug_plots=True,
                )
            )
        except Exception as exc:
            shutil.rmtree(out_dir, ignore_errors=True)
            _raise_ui_error(str(exc))

        circs = res.body_json.get("circumferences_cm", {})
        rows = [[BODY_LABELS_VI.get(k, k), circs[k]] for k in ANATOMICAL_ORDER if k in circs]
        for k, v in circs.items():
            if k not in ANATOMICAL_ORDER:
                rows.append([BODY_LABELS_VI.get(k, k), v])

        warn_text = "\n".join(f"- ⚠️ {w}" for w in res.warnings) if res.warnings else "- ✅ Các tỉ lệ số đo hợp lý."
        backends_used = getattr(res, "backends_used", None) or res.body_json.get("backends_used") or {}
        backends_line = (
            f"- **Pipeline Backends:** `seg={backends_used.get('segmentation')}`, "
            f"`tex={backends_used.get('texture_rasterizer')}`, "
            f"`outer={backends_used.get('outer_shell_renderer')}`, "
            f"`uv={backends_used.get('uv_source')}`, "
            f"`moge2={backends_used.get('moge2_depth_used')}`, "
            f"`fit={backends_used.get('fitting_silhouette')}`\n"
            if isinstance(backends_used, dict) and backends_used
            else ""
        )
        summary = (
            f"### Kết quả xử lý ({res.elapsed_sec:.1f} giây)\n"
            f"- **Số ảnh hợp lệ đã gộp:** {res.images_used}/{res.images_total} ảnh\n"
            f"- **Chế độ dựng 3D:** `{res.effective_reconstruction_mode}` (yêu cầu: `{res.reconstruction_mode}`)\n"
            f"{backends_line}"
            f"- **Chiều cao gốc AI → Quy đổi:** {res.raw_height_cm:.1f} cm → **{float(height_cm):.1f} cm**\n"
            f"- **Dán ảnh thật (Bước 3):** {res.texture_summary}\n"
            f"- **Bảo mật ảnh gốc:** Đã xoá tạm {res.deletion_audit.get('deleted_count', 0)} ảnh upload sau khi xử lý\n"
            f"{warn_text}"
        )

        artifacts = res.artifacts
        viewer_glb = artifacts.get(res.viewer_artifact_key, artifacts.get("glb", ""))
        qa_candidates = [
            os.path.join(out_dir, "debug_apose.png"),
            os.path.join(out_dir, "debug_slices.png"),
        ]
        if "tex_blur" in artifacts:
            qa_candidates.append(artifacts["tex_blur"])
        if "tex_face" in artifacts:
            qa_candidates.append(artifacts["tex_face"])
        qa_gallery = [p for p in qa_candidates if p and os.path.isfile(p)]

        dl_keys = ["glb", "core_glb", "json", "tex_blur", "tex_face", "glb_preview_blur", "glb_preview_face"]
        download_files = [artifacts[k] for k in dl_keys if k in artifacts and os.path.isfile(artifacts[k])]
        return viewer_glb, rows, summary, qa_gallery, download_files


def build_ui(engine: ScanDemoEngine) -> "gr.Blocks":
    """Construct the Gradio interface."""
    if gr is None:
        raise RuntimeError("gradio package is required to build the interactive Gradio UI.")
    with gr.Blocks(title="Alpha Scan Plus/PRO — Demo Bước 1, 2 & 3") as demo:
        gr.Markdown(
            "# Alpha Scan Plus/PRO — Demo 3D Người Thật & Đo Vòng (SAM 3D Body + MHR)\n"
            "> ⚠️ **Lưu ý bảo mật:** Chỉ dùng **ảnh AI mẫu** hoặc **ảnh nhân viên đã ký đồng ý** để test. "
            "Bản upload tạm sẽ tự xoá ngay sau khi xuất kết quả."
        )
        with gr.Row():
            with gr.Column(scale=1):
                files_in = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="1. Tải lên 4–8 ảnh toàn thân (trước, phải, sau, trái — đồ bó, chữ A 20–30°)",
                )
                face_in = gr.File(
                    file_count="single",
                    file_types=["image"],
                    label="1b. Ảnh cận mặt chính diện ~1 m (tuỳ chọn — tăng độ nét mặt ở Mức 3)",
                )
                height_in = gr.Number(value=162.0, label="2. Chiều cao thật chân trần (cm)", precision=1)
                sex_in = gr.Radio(choices=["female", "male"], value="female", label="3. Giới tính")
                weight_in = gr.Number(value=50.0, label="4. Cân nặng (kg) — tuỳ chọn", precision=1)
                pid_in = gr.Textbox(value="ai_female_01", label="5. Tên/Mã mẫu thử")
                privacy_in = gr.Radio(
                    choices=PRIVACY_CHOICES,
                    value=PRIVACY_CHOICES[2],
                    label="6. Chế độ hiển thị & xuất texture (Bước 3)",
                )
                mode_in = gr.Radio(
                    choices=list(RECONSTRUCTION_MODES),
                    value=DEFAULT_RECONSTRUCTION_MODE,
                    label="7. Chế độ dựng hình 3D",
                )
                consent_in = gr.Checkbox(
                    value=False,
                    label="8. Tôi xác nhận đây là ảnh AI mẫu hoặc có sự đồng ý xử lý 3D tạm thời",
                )
                run_btn = gr.Button("Dựng Hình 3D, Dán Ảnh & Đo Số Đo", variant="primary")
                summary_out = gr.Markdown()

            with gr.Column(scale=2):
                model_out = gr.Model3D(label="Hình 3D chữ A (kéo chuột để xoay 360°)")
                table_out = gr.Dataframe(
                    headers=["Vị trí đo", "Số đo (cm)"],
                    datatype=["str", "number"],
                    label="Bảng 16 số đo vòng",
                )
                qa_out = gr.Gallery(label="Ảnh kiểm chứng tách tay, mặt cắt & Texture UV 2048px", columns=2)
                dl_out = gr.File(file_count="multiple", label="Tải về scan.glb, scan_core.glb, body_scan.json & tex_*.jpg")

        run_btn.click(
            fn=engine.process,
            inputs=[files_in, face_in, height_in, sex_in, weight_in, pid_in, privacy_in, mode_in, consent_in],
            outputs=[model_out, table_out, summary_out, qa_out, dl_out],
        )
    return demo


def main() -> None:
    """Launch Gradio web server."""
    parser = argparse.ArgumentParser(description="Alpha Scan Gradio Demo UI")
    parser.add_argument("--sam_repo_dir", type=str, default=".")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr_path", type=str, default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    parser.add_argument("--share", action="store_true", help="Create public gradio.live link for remote testing")
    parser.add_argument("--server_name", type=str, default=None, help="Bind host for VPS/cloud GPU (e.g. 0.0.0.0)")
    parser.add_argument("--server_port", type=int, default=None, help="Port for VPS/cloud GPU (e.g. 7860)")
    parser.add_argument("--auth", type=str, default="", help="Optional 'user:password' login gate for remote demo")
    args = parser.parse_args()

    auth_tuple = None
    if args.auth:
        if ":" not in args.auth:
            raise ValueError("--auth must be formatted as 'username:password'")
        user, pwd = args.auth.split(":", 1)
        auth_tuple = (user, pwd)

    engine = ScanDemoEngine(args.sam_repo_dir, args.checkpoint_path, args.mhr_path)
    demo = build_ui(engine)
    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        auth=auth_tuple,
        debug=False,
    )


if __name__ == "__main__":
    main()
