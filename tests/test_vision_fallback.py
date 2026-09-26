from app.config import VISION_MODEL, VISION_MODELS


def test_vision_models_have_current_primary_and_fallbacks():
    assert VISION_MODEL == "glm-5.3-flash:cloud"
    assert VISION_MODELS[0] == VISION_MODEL
    assert "gemma4:cloud" in VISION_MODELS
    assert "minimax-m3:cloud" in VISION_MODELS
