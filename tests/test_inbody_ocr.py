"""Tests for the InBody OCR adapter and Alpha Scan API integration."""

import io
import asyncio
import tempfile
import unittest

from PIL import Image

from bodyscan_pipeline.inbody_ocr import (
    InBodyExtraction,
    InBodyServiceError,
    OpenAIInBodyService,
    adapt_for_alpha,
)

try:
    from fastapi.testclient import TestClient
    from bodyscan_pipeline.api_server import ServerConfig, create_app
    from bodyscan_pipeline.scan_service import MockScanEngine

    API_TESTS_AVAILABLE = True
except ModuleNotFoundError:
    API_TESTS_AVAILABLE = False


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 48), (245, 245, 240)).save(output, format="JPEG")
    return output.getvalue()


def _extraction() -> InBodyExtraction:
    return InBodyExtraction.model_validate(
        {
            "data": {
                "document_status": "single_inbody_report",
                "identity": {"inbody_sheet_id": "00123"},
                "measurement": {
                    "test_datetime": "2026-08-27T06:18",
                    "device_model": "InBody270S",
                    "height_cm": 170,
                    "age": 30,
                    "gender": "male",
                },
                "body_composition": {
                    "weight_kg": 80,
                    "total_body_water_l": 42,
                    "protein_kg": 11,
                    "minerals_kg": 4,
                    "body_fat_mass_kg": 20,
                },
                "muscle_fat_obesity": {
                    "smm_kg": 33,
                    "pbf_percent": 25,
                    "bmi": 27.7,
                },
                "segmental": {
                    "left_arm": {"lean_kg": 3.1, "lean_percent": 100, "fat_kg": 1.2, "fat_percent": 110},
                    "right_arm": {"lean_kg": 3.2, "lean_percent": 101, "fat_kg": 1.3, "fat_percent": 111},
                    "trunk": {"lean_kg": 25, "lean_percent": 99, "fat_kg": 10, "fat_percent": 120},
                    "left_leg": {"lean_kg": 9, "lean_percent": 98, "fat_kg": 2.5, "fat_percent": 115},
                    "right_leg": {"lean_kg": 9.1, "lean_percent": 99, "fat_kg": 2.6, "fat_percent": 116},
                },
                "weight_control": {
                    "target_weight_kg": 72,
                    "weight_control_kg": -8,
                    "fat_control_kg": -8,
                    "muscle_control_kg": 0,
                },
                "other_metrics": {
                    "inbody_score": 75,
                    "visceral_fat_level": 8,
                    "visceral_fat_area_cm2": None,
                    "bmr_kcal": 1700,
                    "fat_free_mass_kg": 60,
                    "smi_kg_m2": 8,
                    "obesity_degree_percent": 115,
                    "waist_hip_ratio": 0.9,
                    "recommended_calorie_intake_kcal": 2300,
                },
            },
            "evidence": [
                {
                    "field": "measurement.test_datetime",
                    "raw_text": "07.08.2026 06:18",
                    "page": 1,
                    "region": "header",
                }
            ],
        }
    )


class FakeInBodyService:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    def health(self):
        return {"enabled": True, "provider": "fake", "model": "test", "timeout_seconds": 1}

    async def extract(self, raw, filename):
        self.calls.append((raw, filename))
        if self.failure:
            raise self.failure
        return adapt_for_alpha(_extraction(), provider="fake", model="test")


class TestInBodyAdapter(unittest.TestCase):
    def test_adapter_corrects_date_from_evidence_and_maps_segments(self):
        result = adapt_for_alpha(_extraction(), provider="openai", model="gpt-test")
        self.assertEqual(result["data"]["measured_datetime"], "2026-08-07T06:18")
        self.assertEqual(result["data"]["measured_date"], "2026-08-07")
        self.assertEqual(result["data"]["segmental"]["arm_left"]["lean_kg"], 3.1)
        self.assertEqual(result["data"]["segmental"]["leg_right"]["fat_kg"], 2.6)
        self.assertTrue(any("hiệu chỉnh" in flag for flag in result["flags"]))

    def test_openai_service_uses_structured_response_without_network(self):
        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def parse(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_parsed": _extraction()})()

        responses = FakeResponses()
        fake_client = type("Client", (), {"responses": responses})()
        service = OpenAIInBodyService(
            None, model="gpt-test", timeout_seconds=5, client=fake_client
        )
        result = asyncio.run(service.extract(_jpeg(), "inbody.jpg"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "gpt-test")
        self.assertEqual(responses.kwargs["model"], "gpt-test")
        self.assertEqual(responses.kwargs["store"], False)
        content = responses.kwargs["input"][0]["content"]
        self.assertEqual(content[1]["type"], "input_image")


@unittest.skipUnless(API_TESTS_AVAILABLE, "Alpha Scan runtime dependencies are not installed")
class TestInBodyEndpoint(unittest.TestCase):
    def test_endpoint_reuses_auth_and_requires_consent(self):
        fake = FakeInBodyService()
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                ServerConfig(mock=True, data_dir=tmp, api_key="secret"),
                engine=MockScanEngine(0),
                inbody_service=fake,
            )
            with TestClient(app) as client:
                files = {"file": ("inbody.jpg", _jpeg(), "image/jpeg")}
                self.assertEqual(
                    client.post("/api/v1/inbody/extract", files=files, data={"consent_granted": "true"}).status_code,
                    401,
                )
                self.assertEqual(
                    client.post(
                        "/api/v1/inbody/extract",
                        files=files,
                        data={"consent_granted": "false"},
                        headers={"Authorization": "Bearer secret"},
                    ).status_code,
                    403,
                )
                response = client.post(
                    "/api/v1/inbody/extract",
                    files=files,
                    data={"consent_granted": "true"},
                    headers={"Authorization": "Bearer secret"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["ok"])
                self.assertEqual(response.json()["data"]["measured_date"], "2026-08-07")
                self.assertEqual(len(fake.calls), 1)

    def test_provider_error_status_is_preserved(self):
        fake = FakeInBodyService(InBodyServiceError("quota", 429))
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                ServerConfig(mock=True, data_dir=tmp),
                engine=MockScanEngine(0),
                inbody_service=fake,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/inbody/extract",
                    files={"file": ("inbody.jpg", _jpeg(), "image/jpeg")},
                    data={"consent_granted": "true"},
                )
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response.json()["detail"], "quota")


if __name__ == "__main__":
    unittest.main()
