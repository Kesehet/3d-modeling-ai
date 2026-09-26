from app.config import VISION_MODEL, VISION_MODELS


def test_vision_models_have_working_primary_and_fallbacks():
    assert VISION_MODEL == "gemma4:cloud"
    assert VISION_MODELS[0] == VISION_MODEL
    assert "glm-5.3-flash:cloud" in VISION_MODELS
    assert "minimax-m3:cloud" in VISION_MODELS
