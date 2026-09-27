import base64
from io import BytesIO

from PIL import Image

from app.main import VisionReport, _encode_vision_image, _normalize_vision_report_payload


def test_normalize_nested_vision_analysis():
    raw = {
        "analysis": {
            "geometry": "The legs are too vertical and read as simple posts.",
            "proportions": "The camera head is too small relative to the torso.",
            "recommendations": ["Angle the legs outward", "Increase the camera head"],
        }
    }
    report = VisionReport.model_validate(_normalize_vision_report_payload(raw))
    assert report.summary
    assert report.recommended_modeling_strategy == "procedural"
    assert len(report.issues) >= 2
    assert any("legs" in issue.issue.lower() for issue in report.issues)


def test_normalize_native_vision_report():
    raw = {
        "summary": "Readable blockout with proportion issues.",
        "recommended_modeling_strategy": "procedural",
        "observations": ["Four legs are visible."],
        "issues": [
            {
                "object": "tool arm",
                "issue": "Too short",
                "severity": "high",
                "suggested_change": "Extend the forearm",
            }
        ],
        "priority_actions": ["Extend the forearm"],
    }
    report = VisionReport.model_validate(_normalize_vision_report_payload(raw))
    assert report.issues[0].severity == "high"


def test_vision_image_encoder_downscales_payload_without_touching_source(tmp_path):
    path = tmp_path / "large-reference.png"
    Image.new("RGB", (2400, 1200), (160, 170, 180)).save(path)

    encoded = _encode_vision_image(path)
    payload = base64.b64decode(encoded)

    with Image.open(BytesIO(payload)) as prepared:
        assert prepared.format == "JPEG"
        assert max(prepared.size) <= 640

    with Image.open(path) as original:
        assert original.size == (2400, 1200)

    assert len(payload) < 250_000
