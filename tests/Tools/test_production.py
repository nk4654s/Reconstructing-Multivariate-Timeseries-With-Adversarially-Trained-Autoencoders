import os
import onnxruntime


def test_model_labels():
    path = "environment/production/models"
    files = os.listdir(path)

    for f in files:
        if not f.endswith("onnx"):
            continue
        model = os.path.join(path, f)
        session = onnxruntime.InferenceSession(model)
        output = [o.name for o in session.get_outputs()]
        assert "score" in output, f"{f} has no 'score' output label."
