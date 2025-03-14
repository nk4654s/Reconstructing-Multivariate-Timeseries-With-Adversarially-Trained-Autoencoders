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
        parser.add_argument("--names-mapping", type=str, default=None)
