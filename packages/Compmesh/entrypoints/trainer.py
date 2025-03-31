import os

import numpy as np
import pandas as pd
import yaml
from Comprisk.entrypoints import EntrypointBase, EntrypointContext
from Comprisk.entrypoints.trainer import get_trainer_callback
from Comprisk.helper import BasicCheckpointCallback
from Dataflux import get_logger, logging_mode

logger = get_logger(__name__, path="logfile", level=logging_mode)


class SPTrainEntrypoint(EntrypointBase):
    def __init__(self):
        super().__init__()

        self._x_values = dict()
        self._y_values = dict()

        self._logging_path = None

    def load_name_mapping(self, context: EntrypointContext) -> None:
        """This method loads the mapping of KO3 names to Xplore's sensor UUIDs."""
        path = context.config.name_mapping
        with open(path, "r") as f:
            mapping = yaml.safe_load(f)
            self._x_values = mapping["inputs"]
            self._y_values = mapping["outputs"]

    def load_support_points_table(self, context: EntrypointContext) -> pd.DataFrame:
        """This method loads the partitioned KO3 export as pandas DataFrame.

        Returns:
            pd.DataFrame: The KO3 export.
        """
        path = context.config.support_points
        files = [os.path.join(path, f) for f in os.listdir(path) if ".csv" in f]

        # read the support points export
        df = pd.DataFrame()
        for p in sorted(files):
            csv = pd.read_csv(
                p,
                delimiter=";",
                decimal=",",
                dtype=np.float32,
                skiprows=[0, 1, 2, 3, 5],
                header=0,
                encoding="utf-8",
            )
            df = pd.concat([df, csv], ignore_index=True)

        return df

    def _set_checkpoint_path(self, context: EntrypointContext) -> None:
        # set up the model saving as configuring the checkpoint path in the yaml config does not
        # work. The config file does not know the model's output folder name with the increasing
        # suffix number
        checkpoint = get_trainer_callback(context.trainer, BasicCheckpointCallback.__name__)
        if checkpoint:
            self._checkpoint_path = os.path.join(context.config_folder, "checkpoints")
            os.makedirs(self._checkpoint_path)
            checkpoint.dirpath = self._checkpoint_path

        else:
            logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    def check_dataset_compatibility(self, context: EntrypointContext) -> bool:
        """This method checks if the dataset for training is compatible given the manual mapping of
        KO3 names to sensor UUIDs.

        Returns:
            bool: True if it is safe to proceed, otherwise False.
        """
        data = context.datamodule.trainset
        args = data.get_args()

        for x in self._x_values.values():
            if x not in args.names_in:
                return False

        return all(y in args.names_out for y in self._y_values.values())

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        # load the support points data
        self.load_name_mapping(context)
        ok = self.check_dataset_compatibility(context)
        if not ok:
            logger.warning("Dataset configuration does not match model configuration. Adjust columns used.")
            return context

        # setup the logging folder and paths required by a training session
        self._set_checkpoint_path(context)

        # initialize the support point mode
        df = self.load_support_points_table(context)
        context.model.init_grid(df, self._x_values, self._y_values)

        # train/calibrate the model. This will not affect the loaded grid
        context.datamodule.setup("fit")
        context.trainer.fit(context.model, context.datamodule)

        context.config.data.shuffle = False
        return context