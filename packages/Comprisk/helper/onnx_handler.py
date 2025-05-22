import json
from dataclasses import dataclass, field

import onnx
from onnx.onnx_ml_pb2 import ModelProto

from Dataflux.pytorch import Dataset, dataset_type_from_string


def parse_model(path_or_model: str | ModelProto) -> ModelProto:
    if isinstance(path_or_model, str):
        return onnx.load(path_or_model)
    elif isinstance(path_or_model, ModelProto):
        return path_or_model

    raise ValueError(f"Unable to parse type {type(path_or_model)}")


def extract_shape(model: ModelProto):
    def shape_generator(inputs):
        for dim in inputs.shape.dim:
            if dim.dim_param:
                yield 1
                continue

            yield dim.dim_value

    shapes = {}
    for i in model.graph.input:
        shapes[i.name] = tuple([_ for _ in shape_generator(i.type.tensor_type)])

    return shapes


@dataclass
class ONNXArgs:
    producer: str = "NEA-X"
    machine_uuid: str = "N/A"
    creation_date: str = "N/A"
    version: int = 0
    input_length: int = 0
    output_length: int = 0
    input_names: list[str] = field(init=False)
    output_names: list[str] = field(init=False)
    type: str = field(init=False)
    value_corridor: dict = field(init=False)

    def set_value_corridor(self, data: Dataset) -> None:
        self.value_corridor = dict()
        for key, value in data.metadata.state_alpha.items():
            self.value_corridor[key] = value
            self.value_corridor[key]["delay"] *= 1_000

    def equals(self, other: "ONNXArgs") -> bool:
        value = (
                self.machine_uuid == other.machine_uuid
                and self.creation_date == other.creation_date
                and self.version == other.version
                and self.producer == other.producer
                and self.input_names == other.input_names
                and self.output_names == other.output_names
                and self.type == other.type
                and self.value_corridor == other.value_corridor
                and self.input_length == other.input_length
                and self.output_length == other.output_length
        )

        return value

    @staticmethod
    def load_from_model(path_or_model: str | ModelProto) -> "ONNXArgs":
        model = parse_model(path_or_model)

        args = ONNXArgs()
        args.producer = model.producer_name or "NEA-X"
        args.machine_uuid = model.domain
        args.creation_date = model.doc_string or "N/A"
        args.version = model.model_version

        # on first load, the model might not have any metadata
        if len(model.metadata_props) == 4:
            dec = json.decoder.JSONDecoder()
            args.input_names = dec.decode(model.metadata_props[0].value)
            args.output_names = dec.decode(model.metadata_props[1].value)
            args.type = model.metadata_props[2].value
            args.value_corridor = dec.decode(model.metadata_props[3].value)

        # get the input/output sequence length of the model
        shapes = extract_shape(model)
        args.input_length = shapes["x"][1]
        args.output_length = shapes["y"][1]

        return args

    def save_to_model(self, path: str) -> None:
        model = parse_model(path)

        model.producer_name = self.producer
        model.domain = self.machine_uuid
        model.doc_string = self.creation_date
        model.model_version = self.version

        # clear metadata_props
        n = len(model.metadata_props)
        for _ in range(n):
            model.metadata_props.pop(0)

        enc = json.encoder.JSONEncoder()
        meta = model.metadata_props.add()
        meta.key = "input_names"
        meta.value = enc.encode(self.input_names)

        meta = model.metadata_props.add()
        meta.key = "output_names"
        meta.value = enc.encode(self.output_names)

        meta = model.metadata_props.add()
        meta.key = "type"
        meta.value = self.type

        meta = model.metadata_props.add()
        meta.key = "value_corridor"
        meta.value = enc.encode(self.value_corridor)

        onnx.save(model, path)

    def build_dataset(self, path: str) -> Dataset:
        dtype = dataset_type_from_string(self.type)
        d = Dataset(
            path=path,
            sequence_in=self.input_length,
            sequence_out=self.output_length,
            feature_names_in=self.input_names,
            feature_names_out=self.output_names,
            behavior=dtype.value
        )
        return d
