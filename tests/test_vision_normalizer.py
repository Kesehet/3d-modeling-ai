from app.main import VisionReport, _normalize_vision_report_payload


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
