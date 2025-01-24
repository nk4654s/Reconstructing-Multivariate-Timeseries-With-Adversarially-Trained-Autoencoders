"""
## Overview
The `anomaly` submodule contains various `ScoreTransformers` which convert the loss value produced from the model into
a range between 0 and 1. Different types of transformers are available (detailed in `variants`), which can be
configured within the `comprisk-pipeline` configuration file. Alternatively, you can also train your own transformer
on specific models by using the `comprisk-train-score` command. Use the `--help` option to learn more about its usage.
"""
from Comprisk.anomaly.score_transformer import ScoreTransformer, BaseTransformer
import Comprisk.anomaly.variants as variants

__all__ = ["BaseTransformer", "ScoreTransformer", "variants"]
