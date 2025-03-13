import os

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from Compmesh.model import SupportPointModel
from Comprisk.entrypoints import EntrypointBase, EntrypointContext


class SPTrainEntrypoint(EntrypointBase):
    def __init__(self):
        super().__init__()

    def load_name_mapping(self) -> tuple[dict, dict]:
        # TODO: replace with loading from config folder
        x_label = {
            "Saugdruck": "40266713-aba1-42f3-a8ce-ed76e9072034",
            "Enddruck": "0fde253b-8465-40f8-84ac-4f385096d467",
            "Drehzahl": "c3f86569-e2fc-494a-9700-d03ce4bc11d2",
            "Saugtemperatur": "023a0e7a-849e-4acd-ad7f-c011be125745",
            "Rückkühltemperatur": "c5991d25-d729-4a9f-93f0-cebe60de6672",
        }
        y_label = {
            "Massenstrom": "369afc93-5604-45a1-92ff-ae9915ba9cf4",
            "Enddruck Stufe 1": "6d6f8beb-4e55-4b00-913c-5e87a1a8078d",
            "Endtemperatur Stufe 1": "9cf511a4-7a9a-488f-b642-4a946f27c808",
            "Saugdruck Stufe 2": "0269fbad-b2db-4677-9fc0-2814a0bf193b",
            "Endtemperatur Stufe 2": "5630cba4-cc34-4b94-b6a7-0b55f250ffca",
        }

        return x_label, y_label

    def load_support_points_table(self) -> pd.DataFrame:
        # TODO: get the folder from a config
        path = "/Users/jonas/Dev/StützstellenExport"
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

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        # load the support points data
        x_label, y_label = self.load_name_mapping()
        df = self.load_support_points_table()

        # initialize the support point mode
        model = SupportPointModel(df, x_label, y_label)

        # train/calibrate the model. This will not affect the loaded grid
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.L1Loss()  # Mean Absolute Error (L1 Loss)

        # Training Loop
        trainset = context.datamodule.trainset
        trainset.set_stride(trainset.get_args().sequence_in)

        epoch_loss = []
        for x, y in tqdm(context.datamodule.trainset):
            x_train = torch.tensor(x)
            y_train = torch.tensor(y)
            x_train = x_train.unsqueeze(dim=0)
            y_train = y_train.unsqueeze(dim=0)

            optimizer.zero_grad()

            y_pred, y_train = model.predict_step(x_train, y_train)
            loss = loss_fn(y_pred, y_train)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        # pass on
        context.model = model
        return context