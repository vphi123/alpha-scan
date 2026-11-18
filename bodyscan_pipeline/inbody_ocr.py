"""Server-side InBody extraction for the Alpha Scan API.

The module deliberately keeps provider credentials and document bytes on the
server. Uploaded documents are validated in memory and are never persisted.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Identity(StrictModel):
    inbody_sheet_id: Optional[str] = Field(default=None, max_length=100)


class Measurement(StrictModel):
    test_datetime: Optional[str] = Field(
        default=None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"
    )
    device_model: Optional[str] = Field(default=None, max_length=100)
    height_cm: Optional[float] = Field(default=None, ge=50, le=250)
    age: Optional[int] = Field(default=None, ge=1, le=120)
    gender: Optional[Literal["male", "female"]] = None


class BodyComposition(StrictModel):
    weight_kg: Optional[float] = Field(default=None, ge=1, le=500)
    total_body_water_l: Optional[float] = Field(default=None, ge=0, le=300)
    protein_kg: Optional[float] = Field(default=None, ge=0, le=100)
    minerals_kg: Optional[float] = Field(default=None, ge=0, le=50)
    body_fat_mass_kg: Optional[float] = Field(default=None, ge=0, le=300)


class MuscleFatObesity(StrictModel):
    smm_kg: Optional[float] = Field(default=None, ge=0, le=300)
    pbf_percent: Optional[float] = Field(default=None, ge=0, le=100)
    bmi: Optional[float] = Field(default=None, ge=5, le=100)


class SegmentMetrics(StrictModel):
    lean_kg: Optional[float] = Field(default=None, ge=0, le=200)
    lean_percent: Optional[float] = Field(default=None, ge=0, le=1000)
    fat_kg: Optional[float] = Field(default=None, ge=0, le=200)
    fat_percent: Optional[float] = Field(default=None, ge=0, le=1000)


class Segmental(StrictModel):
    left_arm: SegmentMetrics
    right_arm: SegmentMetrics
    trunk: SegmentMetrics
    left_leg: SegmentMetrics
    right_leg: SegmentMetrics


class WeightControl(StrictModel):
    target_weight_kg: Optional[float] = Field(default=None, ge=1, le=500)
    weight_control_kg: Optional[float] = Field(default=None, ge=-500, le=500)
    fat_control_kg: Optional[float] = Field(default=None, ge=-500, le=500)
    muscle_control_kg: Optional[float] = Field(default=None, ge=-500, le=500)


class OtherMetrics(StrictModel):
    inbody_score: Optional[float] = Field(default=None, ge=0, le=100)
    visceral_fat_level: Optional[int] = Field(default=None, ge=1, le=20)
    visceral_fat_area_cm2: Optional[float] = Field(default=None, ge=0, le=1000)
    bmr_kcal: Optional[float] = Field(default=None, ge=0, le=10000)
    fat_free_mass_kg: Optional[float] = Field(default=None, ge=0, le=500)
    smi_kg_m2: Optional[float] = Field(default=None, ge=0, le=50)
    obesity_degree_percent: Optional[float] = Field(default=None, ge=0, le=1000)
    waist_hip_ratio: Optional[float] = Field(default=None, ge=0, le=5)
    recommended_calorie_intake_kcal: Optional[float] = Field(
        default=None, ge=0, le=20000
    )


class ExtractedInBody(StrictModel):
    document_status: Literal[
        "single_inbody_report", "not_inbody_report", "ambiguous_multiple_reports"
    ]
    identity: Identity
    measurement: Measurement
    body_composition: BodyComposition
    muscle_fat_obesity: MuscleFatObesity
    segmental: Segmental
    weight_control: WeightControl
    other_metrics: OtherMetrics


class Evidence(StrictModel):
    field: str = Field(min_length=1, max_length=150)
    raw_text: Optional[str] = Field(default=None, max_length=300)
    page: Optional[int] = Field(default=None, ge=1, le=25)
    region: Optional[str] = Field(default=None, max_length=100)


class InBodyExtraction(StrictModel):
    data: ExtractedInBody
    evidence: list[Evidence]


class InBodyServiceError(RuntimeError):
    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


_DATE_FORMATS = (
    "%d/%m/%Y %H:%M",
    "%d.%m.%Y %H:%M",
    "%d-%m-%Y %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%m/%d/%Y %I:%M %p",
)


def normalize_datetime(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    compact = re.sub(r"\s*([./:-])\s*", r"\1", value.strip())
    compact = re.sub(r"\s+", " ", compact)
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(compact, date_format).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            continue
    return value if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value) else None


def _normalize_evidence_datetime(extraction: InBodyExtraction) -> list[str]:
    """Let deterministic code, rather than the model, own date conversion."""
    flags: list[str] = []
    model_value = extraction.data.measurement.test_datetime
    for item in extraction.evidence:
        if item.field != "measurement.test_datetime":
            continue
        evidence_value = normalize_datetime(item.raw_text)
        if evidence_value:
            if model_value and evidence_value != model_value:
                flags.append(
                    "Ngày giờ đã được hiệu chỉnh theo chuỗi gốc trên phiếu; vui lòng đối chiếu lại."
                )
            extraction.data.measurement.test_datetime = evidence_value
            return flags
    if model_value:
        flags.append("Không có bằng chứng ngày giờ gốc; vui lòng đối chiếu ngày đo.")
    return flags


def _validation_flags(data: ExtractedInBody) -> tuple[dict[str, Any], list[str]]:
    required = {
        "measurement.test_datetime": data.measurement.test_datetime,
        "body_composition.weight_kg": data.body_composition.weight_kg,
        "body_composition.body_fat_mass_kg": data.body_composition.body_fat_mass_kg,
        "muscle_fat_obesity.smm_kg": data.muscle_fat_obesity.smm_kg,
        "muscle_fat_obesity.pbf_percent": data.muscle_fat_obesity.pbf_percent,
    }
    missing = [key for key, value in required.items() if value is None]
    failed: list[str] = []
    bc = data.body_composition
    mfo = data.muscle_fat_obesity
    if bc.weight_kg and bc.body_fat_mass_kg is not None and mfo.pbf_percent is not None:
        if abs(bc.body_fat_mass_kg / bc.weight_kg * 100 - mfo.pbf_percent) > 0.5:
            failed.append("body_fat_percentage")
    if bc.weight_kg and data.measurement.height_cm and mfo.bmi is not None:
        calculated = bc.weight_kg / ((data.measurement.height_cm / 100) ** 2)
        if abs(calculated - mfo.bmi) > 0.3:
            failed.append("bmi")
    flags = []
    if missing:
        flags.append("Thiếu trường cần kiểm tra: " + ", ".join(missing))
    if failed:
        flags.append("Số liệu chưa khớp phép đối chiếu: " + ", ".join(failed))
    return {"passed": not missing and not failed, "missing_required_fields": missing, "failed_checks": failed}, flags


def _segment_payload(segment: SegmentMetrics) -> dict[str, Optional[float]]:
    return {"lean_kg": segment.lean_kg, "fat_kg": segment.fat_kg}


def adapt_for_alpha(
    extraction: InBodyExtraction, *, provider: str, model: str
) -> dict[str, Any]:
    if extraction.data.document_status == "not_inbody_report":
        raise InBodyServiceError("Tệp tải lên không phải phiếu InBody.", 422)
    if extraction.data.document_status == "ambiguous_multiple_reports":
        raise InBodyServiceError("Tệp chứa nhiều phiếu InBody; hãy tải từng phiếu riêng.", 422)

    flags = _normalize_evidence_datetime(extraction)
    data = extraction.data
    validation, validation_flags = _validation_flags(data)
    flags.extend(validation_flags)
    dt = data.measurement.test_datetime
    return {
        "ok": True,
        "data": {
            "measured_datetime": dt,
            "measured_date": dt[:10] if dt else None,
            "device_model": data.measurement.device_model,
            "weight_kg": data.body_composition.weight_kg,
            "pbf_percent": data.muscle_fat_obesity.pbf_percent,
            "smm_kg": data.muscle_fat_obesity.smm_kg,
            "fat_mass_kg": data.body_composition.body_fat_mass_kg,
            "visceral_fat_level": data.other_metrics.visceral_fat_level,
            "inbody_score": data.other_metrics.inbody_score,
            "segmental": {
                "arm_left": _segment_payload(data.segmental.left_arm),
                "arm_right": _segment_payload(data.segmental.right_arm),
                "trunk": _segment_payload(data.segmental.trunk),
                "leg_left": _segment_payload(data.segmental.left_leg),
                "leg_right": _segment_payload(data.segmental.right_leg),
            },
        },
        "validation": validation,
        "flags": flags,
        "provider": provider,
        "model": model,
    }


def validate_document(raw: bytes, filename: str, max_bytes: int, max_pdf_pages: int) -> tuple[bytes, str]:
    if not raw:
        raise InBodyServiceError("Tệp tải lên rỗng.", 400)
    if len(raw) > max_bytes:
        raise InBodyServiceError(
            f"Tệp vượt quá giới hạn {max_bytes // (1024 * 1024)} MB.", 413
        )
    if raw.startswith(b"%PDF-"):
        try:
            import pymupdf

            with pymupdf.open(stream=raw, filetype="pdf") as document:
                if document.page_count < 1 or document.page_count > max_pdf_pages:
                    raise InBodyServiceError(
                        f"PDF phải có từ 1 đến {max_pdf_pages} trang.", 400
                    )
        except InBodyServiceError:
            raise
        except ImportError as exc:
            raise InBodyServiceError("Máy chủ chưa cài PyMuPDF để đọc PDF.", 503) from exc
        except Exception as exc:
            raise InBodyServiceError("PDF bị lỗi hoặc không đọc được.", 400) from exc
        return raw, "application/pdf"

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image_format = image.format
            image.verify()
        if image_format not in {"JPEG", "PNG"}:
            raise InBodyServiceError("Chỉ hỗ trợ PDF, JPG, JPEG và PNG.", 415)
        with Image.open(io.BytesIO(raw)) as image:
            orientation = image.getexif().get(274, 1)
            if orientation != 1:
                image = ImageOps.exif_transpose(image)
                output = io.BytesIO()
                image.save(output, format=image_format)
                raw = output.getvalue()
    except InBodyServiceError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InBodyServiceError("Ảnh bị lỗi hoặc không đọc được.", 400) from exc
    return raw, "image/jpeg" if image_format == "JPEG" else "image/png"


class OpenAIInBodyService:
    """One-pass structured extraction; validation warnings trigger human review."""

    provider = "openai"

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-5.6-luna",
        timeout_seconds: float = 60.0,
        max_file_size_mb: int = 20,
        max_pdf_pages: int = 5,
        client: Any = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_file_size_mb * 1024 * 1024
        self.max_pdf_pages = max_pdf_pages
        self._client = client
        self._prompt = (Path(__file__).parent / "prompts" / "inbody_extraction.md").read_text(
            encoding="utf-8"
        )

    def health(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.api_key or self._client is not None),
            "provider": self.provider,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }

    def _client_instance(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise InBodyServiceError("OPENAI_API_KEY chưa được cấu hình trên server.", 503)
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds, max_retries=0)
        return self._client

    def _extract_sync(self, raw: bytes, filename: str, mime_type: str) -> InBodyExtraction:
        content: dict[str, Any]
        encoded = base64.b64encode(raw).decode("ascii")
        if mime_type == "application/pdf":
            content = {
                "type": "input_file",
                "filename": filename or "inbody.pdf",
                "file_data": f"data:application/pdf;base64,{encoded}",
            }
        else:
            content = {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": "original",
            }
        try:
            response = self._client_instance().responses.parse(
                model=self.model,
                instructions=self._prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Extract this single InBody report using the required schema.",
                            },
                            content,
                        ],
                    }
                ],
                reasoning={"effort": "none"},
                text_format=InBodyExtraction,
                store=False,
                timeout=self.timeout_seconds,
            )
            if response.output_parsed is None:
                raise InBodyServiceError("OpenAI không trả về structured output.", 502)
            return InBodyExtraction.model_validate(response.output_parsed)
        except InBodyServiceError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            name = type(exc).__name__.lower()
            if status == 429 or "ratelimit" in name:
                raise InBodyServiceError("OpenAI hết quota hoặc đang bị giới hạn tốc độ.", 429) from exc
            if "timeout" in name:
                raise InBodyServiceError("OCR vượt quá thời gian xử lý cho phép.", 504) from exc
            if status in {408, 409, 500, 502, 503, 504} or "connection" in name:
                raise InBodyServiceError("OpenAI tạm thời không khả dụng.", 503) from exc
            raise InBodyServiceError(f"OpenAI xử lý thất bại: {exc}", 502) from exc

    async def extract(self, raw: bytes, filename: str) -> dict[str, Any]:
        raw, mime_type = validate_document(
            raw, filename, self.max_bytes, self.max_pdf_pages
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                extraction = await asyncio.to_thread(
                    self._extract_sync, raw, filename, mime_type
                )
        except TimeoutError as exc:
            raise InBodyServiceError(
                f"OCR vượt quá giới hạn {self.timeout_seconds:g} giây.", 504
            ) from exc
        return adapt_for_alpha(extraction, provider=self.provider, model=self.model)
