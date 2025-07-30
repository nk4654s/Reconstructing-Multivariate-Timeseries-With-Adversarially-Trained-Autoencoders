import os
import yaml

from CompriskV2.model.utils import OnnxArgs
from Tools.utils import (
    safe_cast,
    get_model_type,
    filter_models,
    search_models,
    copy_folder,
    check_model_deployed,
    check_model_consistency,
)
from Tools.tests import ProductionFolder


def test_safe_cast_success():
    res = safe_cast("1", int)
    assert res == 1

    res = safe_cast("1.0", float)
    assert res == 1.0


def test_get_model_type():
    path = "some-uuid-here_model.onnx"
    model = get_model_type(path)
    assert model == "model"

    nested_path = "path/to/model/some-uuid-here_model.onnx"
    model = get_model_type(nested_path)
    assert model == "model"

    path = "some-uuid-here_support.onnx"
    model = get_model_type(path)
    assert model == "support"

    invalid = "some-uuid-here_invalid-extension"
    model = get_model_type(invalid)
    assert model is None

    invalid = "path/model.onnx"
    model = get_model_type(invalid)
    assert model == "model"


def test_search_models():
    with ProductionFolder() as context:
        models = search_models(context.prod)
        assert len(models) == len(context.deployed)

        models = search_models(context.dev)
        assert context.undeployed[0]

        models = search_models(context.path)
        assert len(models) == 6


def test_filter_models():
    with ProductionFolder() as context:
        filter = context.deployed[0]
        models = search_models(context.prod)

        filtered = filter_models(models, filter)
        assert len(filtered) == 1

        model = OnnxArgs.load_from_model(filtered[0])
        assert filter.equals(model)

        # filter multiple models, models[1] has 'model' and 'support' type
        filter = context.deployed[1]
        filtered = filter_models(models, filter)
        assert len(filtered) == 2


def test_check_model_deployed():
    with ProductionFolder() as context:
        model = context.deployed[0]
        path, deployed = check_model_deployed(model, "model", context.prod)

        assert path
        assert model.equals(deployed)

        model = context.undeployed[0]
        path, deployed = check_model_deployed(model, "model", context.prod)


def test_check_model_consistency():
    with ProductionFolder() as context:
        models = sorted(search_models(context.prod))
        assert check_model_consistency(models[0], models[0])
        other = os.path.join(context.undeployed[1], "model.onnx")
        assert not check_model_consistency(models[0], other)


def test_copy_config():
    with ProductionFolder() as context:
        onnx, path = context.undeployed
        destination = os.path.join(context.prod, "configs", onnx.machine_uuid, "model")
        copy_folder(source=path, destination=destination, exclude_files=["model.onnx"])

        assert "model.onnx" in os.listdir(path)
        assert "model.onnx" not in os.listdir(destination)

        with open(os.path.join(destination, "config.yaml"), "r") as f:
            data = yaml.safe_load(f)
            print(data)
            assert data["env"] == "production"

            data = data["data"]
            assert data["train"] == os.path.join(destination, "train.yaml")
            assert data["validate"] == os.path.join(destination, "validate.yaml")
