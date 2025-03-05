import json
import os.path
import shutil
from uuid import UUID
from markdown_pdf import MarkdownPdf, Section
import matplotlib.pyplot as plt
import onnxruntime
from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
import yaml

from Comprisk.helper import IteratorResult
from Dataflux.helper.sensor_register import get_sensors


def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# return the chi-squared distance
	return d


class Report:
    def __init__(self, output_folder: str, report_name: str):
        self._folder = output_folder
        self._name = report_name
        self._path = os.path.join(self._folder, self._name)
        self._sensors = os.path.join(self._path, "sensors")

        self._dpi = 80
        self._size = (16, 8)

        # create the output folder path
        for path in [self._path, self._sensors]:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)

        # get the features used by the model
        onnx_path = os.path.join(self._folder, "model.onnx")
        onnx = onnxruntime.InferenceSession(onnx_path)

        meta = onnx.get_modelmeta()
        decoder = json.decoder.JSONDecoder()
        self._features = decoder.decode(meta.custom_metadata_map["output_names"])
        self._sensor_dict = get_sensors(UUID(meta.domain))

    def _revert_score_calculation(self, scores: pd.DataFrame):
        path = os.path.join(self._folder, "score.yaml")
        with open(path, "r") as f:
            score = yaml.safe_load(f)
            intercept, slope = score["intercept"], score["slope"]

        lossy_loss = (scores - intercept) / slope
        noise_floor = (1 - np.array(intercept)) / slope
        sigma_loss = -0.2 / np.array(slope) + noise_floor
        return lossy_loss, noise_floor, sigma_loss

    def generate_report_graphs(self, result: IteratorResult):
        scores, s_inputs, s_outputs = result.to_dataframe()
        scores = scores.rename(columns=self._sensor_dict, errors="ignore")
        s_inputs = s_inputs.rename(columns=self._sensor_dict, errors="ignore")
        s_outputs = s_outputs.rename(columns=self._sensor_dict, errors="ignore")
        loss, noise_floor, sigma_loss = self._revert_score_calculation(scores)

        # plot the total health scores
        figure(figsize=self._size, dpi=self._dpi)
        plt.plot(scores, label=scores.columns)
        plt.title("Healthscore")
        plt.legend()
        plt.savefig(os.path.join(self._path, "total_healthscore.png"))
        plt.close()

        # plot error deviation
        figure(figsize=self._size, dpi=self._dpi)
        plt_data = [(name, sigma_loss[i], noise_floor[i]) for i, name in enumerate(scores.columns)]
        plt_data = sorted(plt_data, key=lambda x: x[1], reverse=True)
        plt.title("Loss Deviation")
        plt.bar([d[0] for d in plt_data], [d[1] for d in plt_data], color="orange", label="Acceptable Deviation")
        plt.bar([d[0] for d in plt_data], [d[2] for d in plt_data], bottom=[d[1] for d in plt_data], color="teal", label="Noise Floor")
        plt.xticks(scores.columns, rotation=90)
        plt.legend()
        plt.savefig(os.path.join(self._path, "loss_deviation.png"))
        plt.close()

        # plot deeper analysis for each sensor
        graphs = []
        for i, sensor in enumerate(scores.columns.values):
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=self._size, dpi=self._dpi)

            axs[0].set_title("Sensor")
            axs[0].plot(s_inputs[sensor], label="Real Value", color="teal", alpha=0.7)
            axs[0].plot(
                s_outputs[sensor], label="AI Reconstruction", color="orange", alpha=0.7
            )
            axs[0].legend(loc="lower right")

            axs[1].set_title("Error Rate")
            axs[1].plot(loss[sensor], label="Model Error", color="teal")
            axs[1].plot(
                scores.index,
                [noise_floor[i] for _ in scores.index],
                label="Noise Floor",
                color="black",
                alpha=0.7,
            )
            axs[1].plot(
                scores.index,
                [sigma_loss[i] for _ in scores.index],
                label="Acceptable Deviation",
                color="orange",
            )
            axs[1].set_ylim(bottom=0)
            axs[1].legend(loc="lower right")

            axs[2].set_title("Healthscore")
            axs[2].plot(scores[sensor], label="Healthscore", color="teal")
            axs[2].plot(
                scores.index,
                [0.8 for _ in scores[sensor]],
                label="Acceptable",
                linestyle="dashed",
                color="black",
                alpha=0.7,
            )
            axs[2].set_ylim(bottom=0, top=1)
            axs[2].legend(loc="lower right")
            fig.suptitle(f"{sensor} Analysis")

            # save ai analysis
            fig.suptitle(f"{sensor} Analysis")
            name = sensor.split()[0]
            fig.savefig(os.path.join(self._sensors, f"ai_{name}.png"))
            plt.close()

            # save histogram
            figure(figsize=self._size, dpi=self._dpi)
            values = s_inputs[sensor].to_numpy()
            bins, vals = np.histogram(values, bins=1000)
            plt.title(f"Histogram: {name}")
            plt.bar(vals[:-1], bins, color="teal", label="Real Data")
            plt.legend()
            plt.savefig(os.path.join(self._sensors, f"ds_{name}.png"))
            plt.close()

            bins_rndm, vals_rndm = np.histogram(np.random.normal(values.mean(), values.std(), len(values)), bins=1000)
            euclidean = np.sqrt(np.power(bins - bins_rndm, 2).sum()) / len(bins)
            graphs.append((i, scores.columns[i], self._sensors, euclidean, sigma_loss[i]))

        return graphs

    def generate_pdf(self, graphs):
        # get the graphs created and initialize the PDF
        graphs = sorted(graphs, key=lambda x: x[-1], reverse=True)
        pdf = MarkdownPdf()
        pdf.meta["title"] = "Report"

        max_error = max([graphs[i][-1] for i in range(len(graphs))])
        acceptable = "yes" if max_error < 0.2 else "no"
        markdown = "# Report\n"
        markdown += f"- Acceptable loss deviation (<0.2 for each sensor)? **{acceptable}**\n"

        mean_gauss = round(sum([graphs[i][-2] for i in range(len(graphs))]) / len(graphs), 2)
        markdown += f"- Dataset quality (mean distance from the gaussian distribution): **{mean_gauss}**\n"

        # is there any sensor abnormal?
        markdown += "## Score Quality: Worst to Best\n"
        markdown += "|Sensor|Acceptable Deviation|\n|--|--|\n"
        for i, graph in enumerate(graphs):
            if graph[-1] < 0.2:
                markdown += f"|{graph[1]}|{graph[-1]}|\n"
            else:
                markdown += f"|**{graph[1]}**|**{graph[-1]}**|\n"

        markdown += "## Data Quality: Worst to Best\n"
        markdown += "|Sensor|Distance to Gaussian|\n|--|--|\n"
        for i, graph in enumerate(sorted(graphs, key=lambda g: g[-2], reverse=True)):
            markdown += f"|{graph[1]}|{graph[-2]}|\n"

        pdf.add_section(Section(markdown, toc=True))

        for i, name, path, n, v in graphs:
            markdown = f"### {name}\n"
            name = name.split()[0]
            markdown += f"![{name}]({path}/ai_{name}.png)\n"
            markdown += f"![{name}]({path}/ds_{name}.png)\n"
            pdf.add_section(Section(markdown, toc=True))

        pdf.save(os.path.join(self._path, "report.pdf"))

