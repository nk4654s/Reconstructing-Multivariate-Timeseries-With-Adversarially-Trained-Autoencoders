import json
import os.path
import shutil
from uuid import UUID

import matplotlib.pyplot as plt
import onnxruntime
from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
import yaml

from Comprisk.helper import IteratorResult
from Dataflux.helper.sensor_register import get_sensors


class Report:
    def __init__(self, output_folder: str, report_name: str):
        self._folder = output_folder
        self._name = report_name
        self._path = os.path.join(self._folder, self._name)

        self._dpi = 210

        # create the output folder path
        if os.path.exists(self._path):
            shutil.rmtree(self._path, ignore_errors=True)
        os.makedirs(self._path)

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

        # plot the total health scores
        figure(figsize=(22, 10), dpi=60)
        plt.plot(scores, label=scores.columns)
        plt.title("Healthscore")
        plt.legend()
        plt.savefig(os.path.join(self._path, "total_score.png"))

        # plot deeper analysis for each sensor
        loss, noise_floor, sigma_loss = self._revert_score_calculation(scores)
        for i, sensor in enumerate(scores.columns.values):
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(22, 10), dpi=self._dpi)

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

            fig.suptitle(f"{sensor} Analysis")
            name = sensor.split()[0]
            fig.savefig(os.path.join(self._path, f"{name}.png"))
