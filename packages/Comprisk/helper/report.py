import json
import os.path
import shutil
from uuid import UUID

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import pandas as pd
import yaml
from Dataflux.helper.sensor_register import get_sensors
from markdown_pdf import MarkdownPdf, Section
from matplotlib.pyplot import figure
import matplotlib.dates as mdates

from Comprisk.helper import IteratorResult


def sensor_fig(
    s_name: str,
    s_inputs: pd.Series,
    s_outputs: pd.Series,
    scores: pd.Series,
    size: tuple[int, int] = (16, 8),
    dpi: int = 120,
) -> figure:
    """This method visualizes a single given sensor data in three sub-graphs. The upper graph shows a comparison
    between the real sensor value and its prediction. The middle graph shows the absolute error of the prediction.
    The lower graph shows the Healthscore calculated by the model.

    Args:
        s_name (str): The sensor name to be shown as the graphs title. Note that the name should contain the unit.
        s_inputs (pd.Series): The real sensor values as pd.Series.
        s_outputs (pd.Series): The predicted sensor values as pd.Series.
        scores (pd.Series): The healthscores as pd.Series.
        size (tuple): The graphs size.
        dpi (int): The graphs' resolution.

    Returns:
        figure: The matplotlib's figure to be shown or saved.
    """
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=size, dpi=dpi)
    s_name, unit = s_name.rsplit(" ", maxsplit=1)
    s_inputs.index = pd.to_datetime(s_inputs.index, unit="s")
    s_outputs.index = pd.to_datetime(s_inputs.index, unit="s")
    scores.index = pd.to_datetime(s_inputs.index, unit="s")


    axs[0].set_title("Sensor")
    axs[0].plot(s_inputs.index, s_inputs.values, label="Real Value", color="teal", alpha=0.7)
    axs[0].plot(s_outputs.index, s_outputs.values, label="AI Reconstruction", color="orange", alpha=0.7)
    axs[0].set_ylabel(unit[1:-1])
    axs[0].legend(loc="lower left")

    axs[1].set_title("Error Rate")
    axs[1].plot(s_inputs.index, abs(s_inputs - s_outputs), label="Model Error", color="teal")
    axs[1].set_ylabel("Abs. Error")
    axs[1].set_ylim(bottom=0)
    axs[1].legend(loc="lower left")

    axs[2].set_title("Healthscore")
    axs[2].plot(scores.index, scores, label="Healthscore", color="teal")
    axs[2].set_ylabel("Higher is better")
    axs[2].set_xlabel("Time")
    axs[2].tick_params(axis='x', rotation=45)
    axs[2].set_ylim(bottom=0, top=1)
    axs[2].legend(loc="lower left")

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y %H:%M"))
    fig.suptitle(f"{s_name} Analysis")
    return fig


class Report:
    """
    Class to generate a comprehensive report based on the analysis of sensor data and AI model outputs.

    Args:
        output_folder (str): The folder where the report will be saved.
        report_name (str): The name of the report file.
    """

    def __init__(self, output_folder: str, report_name: str, model_name: str = "model"):
        self._folder = output_folder
        self._name = report_name
        self._path = os.path.join(self._folder, self._name)
        self._sensors = os.path.join(self._path, "sensors")

        self._dpi = 120
        self._size = (16, 8)

        # create the output folder path
        for path in [self._path, self._sensors]:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)

        # get the features used by the model
        onnx_path = os.path.join(self._folder, f"{model_name}.onnx")
        onnx = onnxruntime.InferenceSession(onnx_path)

        meta = onnx.get_modelmeta()
        decoder = json.decoder.JSONDecoder()
        self._features = decoder.decode(meta.custom_metadata_map["output_names"])
        self._sensor_dict = get_sensors(UUID(meta.domain))

    def generate_report_graphs(self, result: IteratorResult):
        """
        Generates various graphs for the report based on the provided analysis results.

        Args:
            result (IteratorResult): The analysis result containing scores, inputs, and outputs.

        Returns:
            list: A list of tuples containing sensor data and related statistics.
        """
        scores, s_inputs, s_outputs = result.to_dataframe()
        scores = scores.rename(columns=self._sensor_dict, errors="ignore")
        s_inputs = s_inputs.rename(columns=self._sensor_dict, errors="ignore")
        s_outputs = s_outputs.rename(columns=self._sensor_dict, errors="ignore")

        # plot the total health scores
        figure(figsize=self._size, dpi=self._dpi)
        plt.plot(scores, label=scores.columns)
        plt.title("Healthscore")
        plt.legend()
        plt.savefig(os.path.join(self._path, "total_healthscore.jpg"))
        plt.close()

        # plot deeper analysis for each sensor
        graphs = []
        for i, sensor in enumerate(scores.columns.values):
            fig = sensor_fig(
                sensor,
                s_inputs[sensor],
                s_outputs[sensor],
                scores[sensor],
                self._size,
                self._dpi,
            )
            name = sensor.rsplit(maxsplit=1)[0].replace(" ", "_").replace("/", "-")
            fig.savefig(os.path.join(self._sensors, f"ai_{name}.jpg"))
            plt.close()

            # save histogram of data
            figure(figsize=self._size, dpi=self._dpi)
            values = s_inputs[sensor].to_numpy()

            bins, vals = np.histogram(values, bins=1000)
            plt.title(f"Histogram: {name}")
            plt.bar(vals[:-1], bins, color="teal", label="Real Data")
            plt.legend()
            plt.savefig(os.path.join(self._sensors, f"ds_{name}.jpg"))
            plt.close()

            bins_rndm, vals_rndm = np.histogram(
                np.random.normal(values.mean(), values.std(), len(values)), bins=1000
            )
            max_bin = np.max([bins, bins_rndm])
            euclidean = np.sqrt(np.power((bins / max_bin) - (bins_rndm / max_bin), 2).sum()) / len(bins)
            graphs.append((i, scores.columns[i], self._sensors, euclidean))

        return graphs

    def generate_pdf(self, graphs):
        """
        Generates a PDF report based on the provided graphs.

        Args:
            graphs (list): A list of tuples containing sensor data and related statistics.
        """
        # also write to a raw md file
        md = open(os.path.join(self._path, "report.md"), "w")  # noqa: SIM115

        # get the graphs created and initialize the PDF
        graphs = sorted(graphs, key=lambda x: x[-1], reverse=True)
        pdf = MarkdownPdf()
        pdf.meta["title"] = "Report"

        max_error = max([graphs[i][-1] for i in range(len(graphs))])
        acceptable = "yes" if max_error < 0.2 else "no"
        markdown = "# Report\n"
        markdown += f"- Acceptable loss deviation (<0.2 for each sensor)? **{acceptable}**\n"

        mean_gauss = round(sum([graphs[i][-1] for i in range(len(graphs))]) / len(graphs), 5)
        markdown += f"- Dataset quality (mean distance from the gaussian distribution): **{mean_gauss}**\n"

        # is there any sensor abnormal?
        markdown += "## AI Quality Indicators\n"
        markdown += "|Sensor|Acceptable Deviation|Data Quality|\n|-:|:-:|:-:|\n"
        for graph in graphs:
            score = round(graph[-1], 4)
            data = round(graph[-1], 4)
            if graph[-1] < 0.2:
                markdown += f"|{graph[1]}|{score}|{data}|\n"
            else:
                markdown += f"|**{graph[1]}**|**{score}**|{data}|\n"

        md.write(markdown)
        pdf.add_section(Section(markdown, toc=True))

        # plot all diagrams from generate_graphs()
        for _, name, path, _ in graphs:
            markdown = f"### {name}\n"
            name = name.rsplit(maxsplit=1)[0].replace(" ", "_").replace("/", "-")
            markdown += f"![{name}]({path}/ai_{name}.jpg)\n"
            markdown += f"![{name}]({path}/ds_{name}.jpg)\n"
            md.write(markdown)
            pdf.add_section(Section(markdown, toc=True))

        pdf.save(os.path.join(self._path, "report.pdf"))
        md.close()
