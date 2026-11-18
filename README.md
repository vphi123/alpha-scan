# Alpha Scan Plus/PRO — Body Scan Pipeline (SAM 3D Body + MHR)

Pipeline xử lý ảnh đa góc $\rightarrow$ phân tách ngữ nghĩa tóc/da mặt (`segmentation.py`) $\rightarrow$ tối ưu hình thể 3D đa góc & kiểm định tỷ lệ (`fitting.py`) $\rightarrow$ tái tạo đầu/tóc độ phân giải cao + lớp vỏ áo quần hai tầng (`head_hair.py` + `outer_shell.py`) $\rightarrow$ đo 16 vòng cơ thể & `loft` từ lõi MHR $\rightarrow$ xuất `scan.glb`, `scan_core.glb` và JSON chuẩn `alpha_scan.body_scan.v2`.

## Cấu trúc thư mục (`bodyscan_pipeline/` — 15 module)
* `bodyscan_pipeline/__init__.py`: Khởi tạo gói `bodyscan_pipeline`.
* `bodyscan_pipeline/config.py`: Hằng số khung xương 127 khớp MHR, bảng vị trí 16 vòng đo cơ thể, ngưỡng bản quyền, và hằng số 3 chế độ dựng hình (`RECONSTRUCTION_MODES`, `DEFAULT_RECONSTRUCTION_MODE = "dual_layer"`, `HEAD_TEXTURE_SIZE = 2048`, `MHR_HEAD_SUBDIV_STEPS = 1`).
* `bodyscan_pipeline/inference.py`: Nạp `SAM 3D Body` + `Detectron2 (ViTDet)` + `MoGe2` (`extract_depth=True` cho `hi_res_head` / `dual_layer`, `keep_head_crop=True` chỉ trong RAM). Tuyệt đối không dùng YOLO/`ultralytics`, `smplx`, `FLAME`, `nvdiffrast`, hoặc `RMBG`.
* `bodyscan_pipeline/segmentation.py`: Suy luận góc phương vị liên tục (`infer_view_azimuth`), tách `body_mask`, `hair_mask`, `face_skin_mask`, đường sống mũi (`nose_profile_curve`), cảnh báo tóc xoã cổ (`collar_hair_overlap`), và điểm mốc khuôn mặt `MediaPipe` chỉ trong RAM (`extract_face_landmarks_ram`, bị xoá sạch sau khi chạy).
* `bodyscan_pipeline/fusion.py`: Gộp tham số `shape` & `scale` theo trọng số tin cậy đa góc, đưa khuỷu tay về A-pose chuẩn, chuẩn hoá chiều cao `height_cm` và tạo mặt nạ thân/tay.
* `bodyscan_pipeline/fitting.py`: Tối ưu đa góc PyTorch3D + cân bằng tỷ lệ cổ/eo/đùi đối với bóng người chụp và xuất báo cáo `ProportionAudit` (`proportion_audit`: `matched` / `mismatch_warning` / `estimated_only`, `mhr_silhouette_iou`).
* `bodyscan_pipeline/measurements.py`: Cắt lát 2D tính chu vi bao lồi cho 16 vòng đo giải phẫu và bảng `loft` (`0.30`–`0.86`) từ lưới lõi MHR đã hiệu chỉnh.
* `bodyscan_pipeline/head_hair.py`: Chia mịn lưới vùng đầu (`subdiv_steps`), áp độ sâu metric MoGe2 vùng mặt, và tách vỏ tóc ngoài bám sát đường viền tóc/búi tóc.
* `bodyscan_pipeline/outer_shell.py`: Dựng lớp vỏ trang phục ngoài (`outer_verts`) bao quanh lõi MHR trần (`scan_core.glb`) và tính `outer_silhouette_iou` trong `dual_layer`.
* `bodyscan_pipeline/texturing.py`: Bước 3 — Trải UV tách 2 vật liệu `["body", "head"]` (atlas đầu `2048×2048`, thân `2048×2048`), chiếu đa góc ưu tiên `PyTorch3D` (hoặc CPU Z-Buffer fallback), **Quyết định 3b** làm mờ chỉ vùng da mặt ở Mức 2 trong khi giữ sắc nét phần tóc, inpaint chặt không bịa chi tiết vùng mặt khuất.
* `bodyscan_pipeline/exporter.py`: Xuất `scan.glb`, `scan_core.glb` (chế độ `dual_layer`), `tex_blur.jpg`, `tex_face.jpg`, `scan_preview_blur.glb`, `scan_preview_face.glb`, `body_scan.json`, nhật ký xoá ảnh (`deletion_audit.jsonl`) và ảnh QA.
* `bodyscan_pipeline/scan_service.py`: Bộ điều phối tuần tự qua khóa `GpuScanEngine._lock`, tự động hạ cấp về `calibrated_mhr` kèm cảnh báo nếu thiếu 4 góc chuẩn, giữ đồng bộ hình học chi tiết ở Mức 1/2/3 (**Quyết định 3a**), và `MockScanEngine` cho test CPU.
* `bodyscan_pipeline/api_server.py`: Máy chủ FastAPI (`/health`, `POST /api/v1/bodyscan`, `GET /api/v1/bodyscan/{job_id}`, tải file có chữ ký, và xoá tài sản 3D theo Quyết định 3a trả `410 Gone`), hỗ trợ `--mock` và `--serve-frontend frontend_lab`.
* `bodyscan_pipeline/run_scan.py`: CLI chạy toàn bộ pipeline từ `--image_folder` hoặc từ cache `step2_raw_outputs.pt` với `--reconstruction_mode`.
* `bodyscan_pipeline/demo_ui.py`: Giao diện Gradio demo chọn `--reconstruction_mode` và tải về cả `scan.glb` lẫn `scan_core.glb`.

## Quy trình chụp ảnh, Chế độ dựng hình & Quyền riêng tư

### Quy trình chụp ảnh chuẩn (Capture Protocol)
* Chụp đủ **4 góc toàn thân chuẩn**: **Chính diện (`0°`) · Nghiêng phải (`90°`) · Sau (`180°`) · Nghiêng trái (`270°`)** (có thể thêm ảnh cận mặt ~1m `face_image` / `closeup_face` cho Mức 3). Nếu số góc chuẩn hợp lệ `< 4`, hệ thống tự động hạ cấp về `calibrated_mhr` và ghi cảnh báo trong `warnings`.
* Trang phục & tóc: mặc **đồ tập sát form, tóc gọn lộ cổ–vai** trên nền đơn sắc tương phản cao để thuật toán tách tóc, cổ và vòng eo/đùi chính xác, tránh cảnh báo `Collar hair overlap`.

### 3 Chế độ dựng hình (`reconstruction_mode`) & `proportion_audit`
* `dual_layer` *(Mặc định)*: Dựng 2 lớp — vỏ trang phục/tóc/đầu chi tiết (`scan.glb`) bên ngoài và **lõi cơ thể MHR đã hiệu chỉnh (`scan_core.glb`, trả qua `artifacts.core_mesh_url`)** dùng để đo 16 vòng cơ thể chính xác.
* `hi_res_head`: Đầu chia mịn chi tiết + độ sâu MoGe2 vùng mặt + vỏ tóc + UV 2 vật liệu `["body", "head"]` (`2048×2048`), không xuất `scan_core.glb`.
* `calibrated_mhr`: Lưới MHR tối ưu đa góc và 16 số đo chuẩn (1 vật liệu, không `scan_core.glb`), đồng thời là mức fallback an toàn khi `< 4` góc chuẩn.
* **`proportion_audit`**: Trả kèm trong `body_scan.json` và `GET /api/v1/bodyscan/{job_id}` gồm `status` (`matched`, `mismatch_warning`, `estimated_only`), `mhr_silhouette_iou`, `outer_silhouette_iou`, sai số tỷ lệ cổ/eo/đùi và kiểm tra khớp cân nặng.

### Các mức quyền riêng tư (Privacy Levels 1 / 2 / 3) & Bảo mật RAM (S2)
* **Mức 1 (Không texture, Quyết định 3a)**: `textures = null`, không tạo/lưu JPEG khuôn mặt, nhưng `scan.glb` vẫn giữ nguyên hình học 2 vật liệu `["body", "head"]` chia mịn giống hệt Mức 2 & Mức 3 (cộng `scan_core.glb` nếu là `dual_layer`).
* **Mức 2 (Mờ da mặt, giữ tóc nét — Quyết định 3b)**: Xuất `tex_blur.jpg` & `scan_preview_blur.glb`; chỉ vùng da mặt bị làm mờ, phần tóc trên đầu vẫn giữ sắc nét.
* **Mức 3 (Đầy đủ cho chủ thể uỷ quyền)**: Xuất thêm `tex_face.jpg` & `scan_preview_face.glb` (ưu tiên ảnh cận mặt ~1m nếu có).
* **Bảo mật RAM-only (S2)**: Điểm mốc MediaPipe (`face_landmarks_ram`), đường sống mũi (`nose_profile_curve`) và ảnh cắt đầu (`head_crop`) **tuyệt đối không lưu vào đĩa hay JSON**; mọi biến RAM và ảnh tải lên đều bị xoá/ghi đè trong khối `finally:` (`deletion_audit.jsonl`).

## Cách chạy trên Colab / Máy chủ GPU Linux

### 0. Chọn đúng gói Zip & cài phụ thuộc Colab

#### Phân biệt 2 gói nén (`bodyscan_pipeline_bundle.zip` vs `colab_backend_only.zip`)
| Gói nén | Thành phần bên trong | Dùng khi nào? | Lưu ý quan trọng |
| :--- | :--- | :--- | :--- |
| **`bodyscan_pipeline_bundle.zip`** | `bodyscan_pipeline/*.py`, `frontend_lab/*` (gồm `three.min.js`), `tests/*`, `sample_ai_images/female_01/*`, `AlphaScan_Plan2A_Colab.ipynb`, `requirements.txt`, `README.md` | **Khuyên dùng** cho lần chạy đầu trong `AlphaScan_Plan2A_Colab.ipynb`, chạy `--serve-frontend frontend_lab` (`/lab/`), chạy benchmark `sample_ai_images/female_01` và `pytest` | Bắt buộc giải nén gói này trước nếu bật `/lab/` |
| **`colab_backend_only.zip`** | Chỉ gồm `bodyscan_pipeline/*.py`, `requirements.txt`, `README.md` (~142 KB) | Dùng làm **bản vá nóng (Fast Hotfix Overlay)** trong Cell 3 khi `frontend_lab/three.min.js` đã giải nén sẵn trong `/content/sam-3d-body`, hoặc cho máy chủ API thuần | Không chạy riêng với `--serve-frontend frontend_lab` nếu chưa giải nén `bodyscan_pipeline_bundle.zip` trước đó |

#### Phụ thuộc trong `requirements.txt` vs Kho mã nguồn ngoài trên Colab GPU
* `requirements.txt` cài đặt các thư viện Python chuẩn (`fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`, `python-multipart>=0.0.9`, `httpx>=0.27.0`, `mediapipe>=0.10.14,<0.11`, `trimesh>=4.0`, `xatlas` và các gói nền của `sam-3d-body`). Trên Colab, luôn giữ nguyên PyTorch/CUDA có sẵn (không downgrade) và ghim `protobuf>=4.25.3,<5` để tương thích `mediapipe`.
* Các kho mã nguồn / binary GPU ngoài (tự động cài trong **Cell 2** của `AlphaScan_Plan2A_Colab.ipynb`, không đóng gói qua PyPI `requirements.txt`):
  * `facebookresearch/sam-3d-body` (`sam_3d_body`)
  * `facebookresearch/detectron2` (`detectron2` — ViTDet)
  * `microsoft/MoGe` (`moge` — MoGe2 FOV & depth)
  * Tuỳ chọn `facebookresearch/sam2` (`sam2` — Apache 2.0 `human_segmentor`, tự fallback sạch nếu thiếu checkpoint)
  * `pytorch3d` wheel biên dịch sẵn (nếu chưa có wheel khớp bản Torch/CUDA thì fallback sang Z-buffer PyTorch/CPU; tuyệt đối không dùng `nvdiffrast`)
  * `cloudflared` (tạo HTTPS tunnel công khai cho Colab)
* Tuân thủ bản quyền (`verify_license_compliance()`): **không bao giờ** cài hoặc import `ultralytics`, `smplx`, `nvdiffrast`, `flame`, hoặc `briaai/RMBG`.

```bash
pip install -r requirements.txt
```
*(Nếu cài thủ công trên GPU Colab CUDA mà không qua notebook, cài thêm wheel `pytorch3d` phù hợp bản Torch/CUDA hiện tại:)*
```python
import sys, torch
if torch.version.cuda:
    pyt_ver = torch.__version__.split("+")[0].replace(".", "")
    tag = f"py3{sys.version_info.minor}_cu{torch.version.cuda.replace('.', '')}_pyt{pyt_ver}"
    !pip install --no-index --no-cache-dir --no-deps pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{tag}/download.html
else:
    print("Đang dùng PyTorch CPU — bỏ qua PyTorch3D, pipeline tự dùng CPU Fallback.")
```

### 1. Chạy từ thư mục ảnh mới (`run_scan.py` + `--reconstruction_mode`)
```bash
PYTHONPATH=. python -m bodyscan_pipeline.run_scan \
  --image_folder /content/sam-3d-body/sample_ai_images/female_01 \
  --face_image /content/sam-3d-body/anh_mat_1m.jpg \
  --sam_repo_dir /content/sam-3d-body \
  --checkpoint_path /content/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path /content/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --height_cm 165 \
  --sex female \
  --weight_kg 55 \
  --person_id female_01_benchmark \
  --reconstruction_mode dual_layer \
  --enable_texture \
  --privacy_level 2 \
  --delete_inputs \
  --output_dir ./ket_qua_step3 \
  --save_debug_plots
```
*(Tuỳ chọn `--reconstruction_mode`: `dual_layer` | `hi_res_head` | `calibrated_mhr`. Lưu ý: CLI `run_scan.py` dùng cờ **`--sex`** (`male`/`female`, không phải `--gender`), và cờ **`--enable_texture`** mang tính **opt-in**: ở CLI phải truyền `--enable_texture` thì Mức 2/3 mới xuất `tex_blur.jpg`/`tex_face.jpg`, trong khi API `api_server.py` (`GpuScanEngine`) và `demo_ui.py` tự động bật texture khi `privacy_level >= 2`.)*

### 2. Chạy nhanh từ file cache đã suy luận (`step2_raw_outputs.pt`)
```bash
PYTHONPATH=. python -m bodyscan_pipeline.run_scan \
  --cached_outputs /content/sam-3d-body/step2_raw_outputs.pt \
  --head_assets /content/sam-3d-body/mhr_head_assets.pt \
  --mhr_path /content/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --height_cm 178 \
  --sex male \
  --reconstruction_mode dual_layer \
  --output_dir ./ket_qua_step2 \
  --save_debug_plots
```

### 3. Bật Web Demo (`demo_ui.py` Gradio)
```bash
pip install -q gradio
PYTHONPATH=. python -m bodyscan_pipeline.demo_ui \
  --sam_repo_dir /content/sam-3d-body \
  --checkpoint_path /content/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path /content/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --share
```
Giao diện cho phép chọn chế độ `reconstruction_mode` (`calibrated_mhr`, `hi_res_head`, `dual_layer`) và tải về cả `scan.glb` lẫn `scan_core.glb` (nếu có).

### 4. Chạy 1-Click trên Google Colab (`AlphaScan_Plan2A_Colab.ipynb`)

Mở [`AlphaScan_Plan2A_Colab.ipynb`](./AlphaScan_Plan2A_Colab.ipynb) trên Google Colab, chọn **Runtime -> Change runtime type -> T4 GPU (15 GB VRAM)** hoặc **L4 GPU (24 GB VRAM)**, rồi chạy tuần tự 8 cell:

* **Thư mục bền vững trên Google Drive:**
  * Workspace Plan 2A: `/content/drive/MyDrive/AlphaScan_Plan2A/{bundles,jobs,outputs}` (đặt sẵn `bodyscan_pipeline_bundle.zip` trong `bundles/`, nếu chưa có Cell 3 sẽ kiểm tra `/content/bodyscan_pipeline_bundle.zip` hoặc mở hộp thoại tải lên).
  * Checkpoint `sam-3d-body-dinov3`: `/content/drive/MyDrive/AlphaScan_SAM3D/sam-3d-body-dinov3/` (chứa `model.ckpt` và `assets/mhr_model.pt`). Cell 4 tự phát hiện từ Drive; nếu chưa có sẽ tải repo gated `facebook/sam-3d-body-dinov3` qua `huggingface_hub` dùng Colab Secret `HF_TOKEN` (hoặc đăng nhập tương tác) và sao lưu lại vào Drive.
* **Tóm tắt 8 Cell trong `AlphaScan_Plan2A_Colab.ipynb`:**
  1. **Cell 1:** Kiểm tra `nvidia-smi` / CUDA VRAM, tự động thử mount Google Drive (`USE_GOOGLE_DRIVE = True`) để tạo `/content/drive/MyDrive/AlphaScan_Plan2A/{bundles,jobs,outputs}` (tự động fallback về `/content/AlphaScan_Plan2A` nếu OAuth Drive bị chặn).
  2. **Cell 2:** Clone `facebookresearch/sam-3d-body`, cài `detectron2`, `MoGe2`, tuỳ chọn Meta `sam2`, `trimesh`, `xatlas`, `mediapipe>=0.10.14,<0.11` (trên Python `< 3.13`), wheel `pytorch3d` dựng sẵn khớp ABI Torch/CUDA (tự fallback nếu chưa khớp) và `cloudflared` (kèm chốt kiểm tra không cài gói cấm bản quyền).
  3. **Cell 3:** Giải nén `bodyscan_pipeline_bundle.zip` (~44 MB; hoặc đè bản vá nhanh `colab_backend_only.zip` ~158 KB chỉ khi `colab_backend_only.zip` mới hơn bundle hoặc bật `APPLY_HOTFIX_OVERLAY = True`) vào `/content/sam-3d-body`, dọn `__pycache__` / file `.py` cũ khi bung full bundle, ghi vĩnh viễn `/content/sam-3d-body` vào `alphascan.pth` trong `site-packages`, và cài `requirements.txt` với file ràng buộc giữ nguyên PyTorch/CUDA/NumPy của Colab.
  4. **Cell 4:** Đồng bộ `model.ckpt`, `assets/mhr_model.pt` và `sam2.1_hiera_small.pt` (`ALPHASCAN_SAM2_CHECKPOINT`) từ Drive hoặc tải tự động.
  5. **Cell 5:** Kiểm tra pre-flight (`torch` CUDA, `sam_3d_body`, `moge`, `mediapipe`, `pytorch3d`, `trimesh`, `xatlas`, `verify_license_compliance()`, `frontend_lab/three.min.js`, `sample_ai_images/female_01`) và chạy nhanh `python -m pytest tests/test_api_server.py -q`.
  6. **Cell 6:** Chạy benchmark CLI nghiệm thu trên `sample_ai_images/female_01` (`dual_layer`, `--privacy_level 2`, `--enable_texture`, `--height_cm 165`, `--weight_kg 55`, `--sex female`), lưu kết quả về `/content/drive/MyDrive/AlphaScan_Plan2A/outputs/female_01_benchmark` và in thời gian chạy (`< 90s`), VRAM đỉnh (`< 12 GB`), `backends_used`, `proportion_audit` và 16 vòng đo.
  7. **Cell 7 & Cell 7b (⚡ 1-Cell Fast Restart):** Khởi chạy nền `bodyscan_pipeline.api_server` với `--serve-frontend frontend_lab --data-dir /content/drive/MyDrive/AlphaScan_Plan2A/jobs --file-token-ttl 1800`, chờ `/health` chuyển `loading` -> `ready` kèm thông tin `backends`, bật `cloudflared tunnel --url http://127.0.0.1:8000` và hiển thị nút bấm HTML **`🚀 Open Alpha Scan Lab 3D (/lab/)`** tự động kết nối trình duyệt với GPU Colab mà không cần copy-paste URL hay điền mật khẩu.
  8. **Cell 8:** Tiện ích xem `/health` trực tiếp, đọc 40 dòng cuối `/content/alphascan_api.log`, hoặc đặt `ACTION = "stop"` để tắt sạch API server và tunnel.

### 5. Chạy Alpha Scan Lab + Backend FastAPI

#### Chạy nhanh trên máy local (`--mock`, CPU, không cần GPU)
```bash
pip install -r requirements.txt
python -m bodyscan_pipeline.api_server \
  --mock \
  --region-mode internal_test \
  --serve-frontend frontend_lab \
  --host 127.0.0.1 \
  --port 8000
```
Mở `http://127.0.0.1:8000/lab/` trên trình duyệt:
* Chọn chế độ quét (**Dual Layer**, **Hi-Res Head** hoặc **Calibrated MHR**) và tải 4 ảnh toàn thân (Chính diện · Vai phải hướng về camera · Sau · Vai trái hướng về camera; mặc đồ tập sát form, tóc gọn lộ cổ–vai). *(Lưu ý: "Nghiêng phải/trái" luôn tính theo bên phải/trái của người mẫu, không tính theo mép phải/trái khung ảnh.)*
* Khung 3D hỗ trợ xem lớp vỏ ngoài (`scan.glb`) hoặc chuyển sang lõi đo **Inner Core** (`scan_core.glb`), xoay **360°** / **Trước · Sau · Xoay**; cảnh báo `proportion_audit` và thiếu góc chụp hiển thị ngay cạnh kết quả.
* Bấm **Xoá hình 3D/ảnh trên server** để thực thi Quyết định 3a trên server và đưa khung 3D về hologram mà vẫn giữ 16 vòng đo.

#### Chạy trên GPU Colab / Linux Server + nối từ xa (bền vững trên Drive)
```bash
PYTHONPATH=. python -m bodyscan_pipeline.api_server \
  --sam_repo_dir . \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --region-mode internal_test \
  --serve-frontend frontend_lab \
  --data-dir /content/drive/MyDrive/AlphaScan_Plan2A/jobs \
  --file-token-ttl 1800 \
  --host 0.0.0.0 \
  --port 8000
```
*(Bắt buộc giải nén **`bodyscan_pipeline_bundle.zip`** nếu truyền `--serve-frontend frontend_lab`; với `colab_backend_only.zip` hãy bỏ cờ `--serve-frontend` và chạy API thuần.)*

Mở tunnel HTTPS (`cloudflared tunnel --url http://127.0.0.1:8000`) rồi sử dụng một trong hai cách:
1. **Cùng nguồn (`<tunnel-url>/lab/` — Khuyên dùng, Tự động kết nối 100%):** Mở trực tiếp `<tunnel-url>/lab/` trên trình duyệt — hệ thống tự nhận diện origin, tự bỏ qua các URL `127.0.0.1` hoặc `.trycloudflare.com` cũ lưu trong `localStorage`, tự kiểm tra `/health` khi mở modal và thu gọn phần cài đặt server nâng cao. Cũng hỗ trợ ghi đè nhanh qua query params `?gpu_url=...&api_key=...`.
2. **Frontend tách rời (`<tunnel-url>`):** Dán URL gốc `<tunnel-url>` (không kèm `/lab/`) vào mục **3.5 · Server 3D Alpha Scan Plus/PRO** trong nút **Kết nối** của Lab (hoặc mở kèm `?gpu_url=<tunnel-url>`). **Lưu ý Quyết định 2a:** Chế độ `LOCAL_ONLY` chặn gửi ảnh tới server ngoài loopback (`127.0.0.1`/`localhost`); nếu đang ở `LOCAL_ONLY`, cảnh báo trong modal Quét 3D cung cấp nút 1-click **`⚡ Bật chế độ Server GPU (cho phép gửi ảnh tới Colab)`** để chuyển ngay sang `SAFETY.consent = 'server'`.

#### Lưu trữ Drive, Token hết hạn & Nạp mô hình nền
* **`--data-dir` (Drive persistence):** Lưu bền vững **`jobs_index.json`** và khoá ký **`.signing_secret`** trên `/content/drive/MyDrive/AlphaScan_Plan2A/jobs`, giúp danh sách job và token ký URL tải file tiếp tục hợp lệ sau khi khởi động lại API server.
* **`--file-token-ttl 1800` & tự làm mới URL:** CLI `api_server.py` mặc định `--file-token-ttl 1800` (30 phút; giá trị nội bộ của `ServerConfig` là `300` giây), đồng thời `frontend_lab/alpha-plus-pro.js` tự làm mới signed artifact URLs mỗi **240 giây** (`ARTIFACT_URL_REFRESH_MS`) và trước khi tải tài sản 3D.
* **Khởi động không chặn (`/health` warmup):** Ở chế độ GPU, model SAM-3D-Body nạp nền để `/health` phản hồi ngay (`status: "loading"`, `model_ready: false`) trước khi sẵn sàng (`status: "ready"`, `model_ready: true`).

#### Tóm tắt HTTP API (`api_server.py`)
* `GET /health`: Trả `{ok, status ("loading"|"ready"|"error"), model_ready, mode, region_mode, test_only, pipeline_version, cuda_available, auth_required, queue_depth, backends}` *(kèm `error`, `detail` nếu lỗi)*.
* `POST /api/v1/bodyscan` *(multipart/form-data)*: Nhận `photos` (1–8 ảnh), `slot_roles` *(tuỳ chọn, chuỗi JSON mảng vai trò góc tương ứng thứ tự `photos`, ví dụ `'["front","right","back","left"]'`)*, `closeup_face` (tuỳ chọn), `height_cm`, `sex`, `weight_kg`, `person_id`, `privacy_level` (`1|2|3`), `consent_granted=true`, và `reconstruction_mode` (`calibrated_mhr` | `hi_res_head` | `dual_layer`, mặc định `dual_layer`; giá trị khác trả `422`). Với GPU API, texture tự động bật khi `privacy_level >= 2`. Trả `202` với `{job_id, status, progress, step}`.
* `GET /api/v1/bodyscan/{job_id}`: Trả trạng thái job, `reconstruction_mode`, `effective_reconstruction_mode`, `proportion_audit`, `backends_used` (`segmentation`, `texture_rasterizer`, `moge2_depth_used`, `fitting_silhouette`), `warnings`, `body_scan`, và URL tải có chữ ký trong `artifacts` gồm `artifacts.mesh_url`, `artifacts.glb_url`, `artifacts.json_url`, tuỳ chọn `artifacts.core_mesh_url` *(chỉ xuất hiện ở chế độ `dual_layer`, trỏ tới `scan_core.glb`)*, và `artifacts.texture_url` *(trỏ tới `tex_blur.jpg` ở Mức 2 hoặc `tex_face.jpg` ở Mức 3; không có ở Mức 1)*.
* `POST /api/v1/bodyscan/{job_id}/delete` *(hoặc `DELETE /api/v1/bodyscan/{job_id}`)*: Theo **Quyết định 3a**, xoá toàn bộ tệp `.glb` (kể cả **`scan_core.glb`**) và texture `.jpg` trên server khiến mọi yêu cầu tải tài sản 3D sau đó trả **`410 Gone`**, đồng thời giữ nguyên số đo `body_scan` (thêm `?purge_all=true` để xoá toàn bộ hồ sơ job).

## Kiểm thử tự động & Nghiệm thu QA
```bash
python -m pytest -v
node --test tests/test_frontend_plus_pro.mjs
```
*(Ghi chú: trên môi trường CPU không cài tuỳ chọn `trimesh`, các bài kiểm thử trong `tests/test_measurements.py` tự bỏ qua sạch sẽ bằng `unittest.skipUnless(HAS_TRIMESH, ...)`; 7 bộ kiểm thử Python còn lại — `test_api_server.py`, `test_scan_service.py`, `test_segmentation.py`, `test_fitting.py`, `test_head_hair.py`, `test_outer_shell.py`, `test_texturing.py` — cùng bộ kiểm thử frontend `tests/test_frontend_plus_pro.mjs` đều chạy đầy đủ trên CPU.)*

### Kiểm tra nghiệm thu GPU trên Colab (`sample_ai_images/female_01`)
Khi chạy thực tế trên GPU Colab (T4/L4/A100), kiểm tra các ngưỡng nghiệm thu sau:
* Thời gian xử lý `< 90s` và đỉnh bộ nhớ VRAM `< 12 GB`.
* Không bị lem tường nền (zero wall haloing) ở đường viền nhân vật/tóc.
* Texture Mức 2 chỉ làm mờ vùng da mặt trong khi phần tóc giữ nguyên độ sắc nét.
* `proportion_audit.mhr_silhouette_iou >= 0.85` và `proportion_audit.outer_silhouette_iou >= 0.88`.
