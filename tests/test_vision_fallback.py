from app.config import VISION_MODEL, VISION_MODELS


def test_vision_models_have_current_frontier_multimodal_primary_and_fallbacks():
    assert VISION_MODEL == "kimi-k3:cloud"
    assert VISION_MODELS[0] == VISION_MODEL
    assert "mistral-large-3:675b-cloud" in VISION_MODELS
    assert "gemma4:31b-cloud" in VISION_MODELS
    assert "qwen3-vl:235b-cloud" not in VISION_MODELS
