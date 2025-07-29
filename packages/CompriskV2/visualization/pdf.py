import os
import shutil
from typing import Any
from uuid import UUID
from tqdm import tqdm

from markdown_pdf import MarkdownPdf, Section
from matplotlib import pyplot as plt

from CompriskV2.model.utils import IteratorResult, OnnxArgs
from CompriskV2.visualization import visualize_single_result
from Dataflux import get_logger
from Dataflux.helper import get_sensors

logger = get_logger(__name__)


class Report:
    """
    Class to generate a comprehensive report based on the analysis of sensor data and AI model outputs.

    Args:
        onnx_path (str): The path where the model is located and the report will be saved.
        report_name (str): The name of the report file.
    """

    def __init__(
        self,
        onnx_path: str,
        report_name: str,
        visualization_fn: Any | None = None,
        visualization_kwargs: dict | None = None,
        model_name: str = "model",
    ):
        self._folder = os.path.dirname(onnx_path)
        self._name = report_name
        self._tmp = os.path.join(self._folder, "tmp")

        self._visualization_fn = visualization_fn or visualize_single_result
        self._visualization_kwargs = visualization_kwargs or {"std": 7}

        self._dpi = 120
        self._size = (16, 8)

        # get the features used by the model
        self._onnx_args = OnnxArgs.load_from_model(onnx_path)
        self._features = self._onnx_args.output_names
        self._sensor_dict = get_sensors(UUID(self._onnx_args.machine_uuid))

    def _generate_report_graphs(self, result: IteratorResult) -> str:
        """
        Generates various graphs for the report based on the provided analysis results.

        Args:
            result (IteratorResult): The analysis result containing scores, inputs, and outputs.

        Returns:
            str: The markdown containing AI result graphs.
        """
        os.mkdir(self._tmp)

        # plot deeper analysis for each sensor
        graphs = []
        for sensor in tqdm(result.names, desc="Exporting Graph Images"):
            readable = self._sensor_dict.get(sensor, f"{sensor} N/A").rsplit(
                " ", maxsplit=1
            )
            name, unit = readable[0], readable[1] if len(readable) >= 2 else "N/A"

            df = result.by_name(sensor).to_dataframe()
            fig = self._visualization_fn(
                df, name, unit, result.ts, **self._visualization_kwargs
            )
            fig.savefig(os.path.join(self._tmp, sensor) + ".jpg")
            plt.close()

            graphs.append((name, os.path.join(self._tmp, sensor) + ".jpg"))

        # plot all diagrams from generate_graphs()
        markdown = "## Sensors"
        for name, path in sorted(graphs, key=lambda g: g[0]):
            markdown += f"![{name}]({path})\n"
            logger.debug("Add graph %s", path)

        return markdown

    def _generate_model_info(self) -> str:
        markdown = "# Model Information\n"
        markdown += f"Training date: {self._onnx_args.creation_date}\n\n"
        markdown += f"I/O sequence: {self._onnx_args.input_length}/{self._onnx_args.output_length}\n\n"
        markdown += "## Delays\n"

        value_corridor = self._onnx_args.value_corridor
        for id, corridor in value_corridor.items():
            delay = corridor["delay"] / 1_000
            lower, upper = corridor["lower"], corridor["upper"]
            markdown += f"{id}:\n\n"
            markdown += f"- delay: {delay}s (bounds: {lower} | {upper})\n\n"

        return markdown

    def generate_pdf(self, result: IteratorResult):
        """
        Generates a PDF report based on the provided graphs.

        Args:
            result (IteratorResult): The analysis result containing scores, inputs, and outputs.
        """
        # get the graphs created and initialize the PDF
        pdf = MarkdownPdf()
        pdf.meta["title"] = f"Report: {self._name}"

        model_section = self._generate_model_info()
        pdf.add_section(Section(model_section, toc=True))

        graphs_section = self._generate_report_graphs(result)
        pdf.add_section(Section(graphs_section, toc=True))

        pdf.save(os.path.join(self._folder, f"{self._name}.pdf"))
        shutil.rmtree(self._tmp)
