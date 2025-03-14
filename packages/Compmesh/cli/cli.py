from lightning.pytorch.cli import (
    LightningArgumentParser,
LightningCLI
)


class CompmeshCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--tag", type=str, default=None)
        parser.add_argument("--environment", type=str, default="dev")

        parser.add_argument("--prefix", type=str, default=None)
        parser.add_argument("--support-points", type=str, default=None)
        parser.add_argument("--name-mapping", type=str, default=None)

        # link the model's input/output features with the dataset's configuration
        parser.link_arguments("data.features_in", "model.init_args.features_in", apply_on="instantiate")
        parser.link_arguments("data.features_out", "model.init_args.features_out", apply_on="instantiate")
        parser.link_arguments("data.behavior", "model.init_args.behavior", apply_on="instantiate")
        parser.link_arguments("data.sequence_out", "model.init_args.sequence_out", apply_on="instantiate")
