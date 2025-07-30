import numpy as np
from Tools.utils import search_models
from onnxruntime import InferenceSession


def test_all_models_have_score_label():
    for model in search_models("environment/production"):
        session = InferenceSession(model)
        names = [o.name for o in session.get_outputs()]
        assert "score" in names, model


def test_all_models_have_3_dims():
    for model in search_models("environment/production"):
        session = InferenceSession(model)
        dims = np.array([len(o.shape) for o in session.get_outputs()])
        assert (dims == 3).all(), model


def test_all_models_consistent_dims():
    for model in search_models("environment/production"):
        session = InferenceSession(model)
        last_dim = np.array([o.shape[-1] for o in session.get_outputs()])
        assert (last_dim == last_dim[0]).all(), model
