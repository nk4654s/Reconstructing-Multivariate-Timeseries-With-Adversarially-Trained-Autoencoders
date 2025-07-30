import onnxruntime
from Tools.utils import search_models


def test_model_labels():
    path = "environment/production/models"
    for f in search_models(path):
        session = onnxruntime.InferenceSession(f)
        output = [o.name for o in session.get_outputs()]
        assert "score" in output, f"{f} has no 'score' output label."
