from CompriskV2.tests.utils.fixtures import CliContext
from CompriskV2.model.ko3 import SupportPointModel


def test_gasunie():
    path = "environment/dev/support_points/gasunie"
    with CliContext(SupportPointModel, model={"ko3_table": path}) as context:
        assert context.cli.model._ko3_table == path

        context.cli.trainer.fit(context.cli.model, context.cli.datamodule)


def test_thyssen_gas():
    path = "environment/dev/support_points/thyssen_gas"
    with CliContext(SupportPointModel, model={"ko3_table": path}) as context:
        assert context.cli.model._ko3_table == path

        context.cli.trainer.fit(context.cli.model, context.cli.datamodule)
