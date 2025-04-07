"""
## Overview
The `anomaly` submodule contains various `ScoreTransformers` which convert the loss value produced from the model into
a range between 0 and 1. Different types of transformers are available (detailed in `variants`), which can be
configured within the `comprisk-pipeline` configuration file. Alternatively, you can also train your own transformer
on specific models by using the `comprisk-score` command. Use the `--help` option to learn more about its usage.
"""
from Comprisk.anomaly.model_based import ScoreWrapper
from Comprisk.anomaly.adjusted_r2 import R2ScoreWrapper

__all__ = ["ScoreWrapper", "R2ScoreWrapper"]
