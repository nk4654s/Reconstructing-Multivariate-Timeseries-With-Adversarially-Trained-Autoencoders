import os
import shutil
from typing import Any
from uuid import UUID
from tqdm import tqdm

from markdown_pdf import MarkdownPdf, Section
from matplotlib import pyplot as plt

from Comprisk.helper import ONNXArgs
from CompriskV2.model.utils import IteratorResult
from CompriskV2.visualization import visualize_single_result
from Dataflux import get_logger
from Dataflux.helper import get_sensors

logger = get_logger(__name__)


class Report:
    """
    Class to generate a comprehensive report based on the analysis of sensor data and AI model outputs.

    Args:
        output_folder (str): The folder where the report will be saved.
        report_name (str): The name of the report file.
    """

    def __init__(self, output_folder: str,
                 report_name: str,
                 visualization_fn: Any | None = None,
                 visualization_kwargs: dict | None = None,
                 model_name: str = "model"):
        self._folder = output_folder
        self._name = report_name

        self._visualization_fn = visualization_fn or visualize_single_result
        self._visualization_kwargs = visualization_kwargs or {
            "std": 7
        }

        self._dpi = 120
        self._size = (16, 8)

        # get the features used by the model
        onnx_path = os.path.join(self._folder, f"{model_name}.onnx")
        onnx_args = ONNXArgs.load_from_model(onnx_path)
        self._features = onnx_args.output_names
        self._sensor_dict = get_sensors(UUID(onnx_args.machine_uuid))

    def _generate_report_graphs(self, result: IteratorResult) -> tuple[list, str]:
        """
        Generates various graphs for the report based on the provided analysis results.

        Args:
            result (IteratorResult): The analysis result containing scores, inputs, and outputs.

        Returns:
            list: A list of tuples containing sensor data and related statistics.
        """
        tmp = os.path.join(self._folder, "tmp")
        os.mkdir(tmp)

        # plot deeper analysis for each sensor
        graphs = []
        for sensor in tqdm(result.names, desc="Exporting Graph Images"):
            readable = self._sensor_dict.get(sensor, f"{sensor} N/A").rsplit(" ", maxsplit=1)
            name, unit = readable[0], readable[1] if len(readable) >= 2 else "N/A"

            df = result.by_name(sensor).to_dataframe()
            fig = self._visualization_fn(df, name, unit, result.ts, **self._visualization_kwargs)
            fig.savefig(os.path.join(tmp, sensor) + ".jpg")
            plt.close()

            graphs.append((name, os.path.join(tmp, sensor) + ".jpg"))

        return graphs, tmp

    def generate_pdf(self, result: IteratorResult):
        """
        Generates a PDF report based on the provided graphs.

        Args:
            result (IteratorResult): The analysis result containing scores, inputs, and outputs.
        """
        graphs, tmp = self._generate_report_graphs(result)

        # also write to a raw md file
        md = open(os.path.join(self._folder, "report.md"), "w")  # noqa: SIM115

        # get the graphs created and initialize the PDF
        pdf = MarkdownPdf()
        pdf.meta["title"] = f"Report: {self._name}"

        # plot all diagrams from generate_graphs()
        markdown = "# Sensors"
        for name, path in sorted(graphs, key=lambda g: g[0]):
            markdown += f"![{name}]({path})\n"
            md.write(markdown)
            logger.debug("Add graph %s", path)
        pdf.add_section(Section(markdown, toc=True))

        pdf.save(os.path.join(self._folder, f"{self._name}.pdf"))
        md.close()

        os.remove(os.path.join(self._folder, "report.md"))
        shutil.rmtree(tmp)
