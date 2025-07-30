from Dataflux.helper import get_sensors
from CompriskV2.thunder import DataModule


train_path = "environment/dev/configs/vd2-version-3/1_dataset_train.yml"


def print_scales():
    data = DataModule(train=train_path)

    mean, scale = data.standardizer_params()
    mapping = get_sensors(data.trainset.UUID)

    features = data.trainset.get_args().names
    [(mapping[feature], scale[i]) for i, feature in enumerate(features)]
