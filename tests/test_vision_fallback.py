from app.config import VISION_MODEL, VISION_MODELS


def test_vision_models_use_included_multimodal_primary_and_fallbacks():
    assert VISION_MODEL == "qwen3-vl:235b-cloud"
    assert VISION_MODELS[0] == VISION_MODEL
    assert "gemma4:31b-cloud" in VISION_MODELS
    assert "gemma4:cloud" in VISION_MODELS
    assert "kimi-k3:cloud" not in VISION_MODELS
    assert "mistral-large-3:675b-cloud" not in VISION_MODELS
