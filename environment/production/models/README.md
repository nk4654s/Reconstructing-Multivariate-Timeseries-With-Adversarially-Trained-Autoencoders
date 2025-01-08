# Production Models
The models trained for production are stored here and included in git. When building a docker for a machine,
the corresponding model is loaded into the docker container. A distinct folder structure is required.

```
production/
    models/
        <UUID_MACHINE_01>/
            checkpoints/
                *.ckpt
            metadata/
                normalizer.pkl
                score_transformer.pkl
            config.yaml
            hparams.yaml
            upload.yaml
        <UUID_MACHINE_02>/
            ...
```

Each folder in `models` is assigned to a machine given the `UUID`. Inside the `UUID` folder, the

 - `config.yaml` defines how the model was trained and is crucial to prepare the software when loading a trained model
 - `hparams.yaml` defines which hyperparameters were set when training the model
 - `upload.yaml` defines where to download and how to process the inputs for inference
 - `metadata/normalizer.pkl` holds the pickled normalizer instance
 - `metadata/score_transformer.pkl` holds the pickled `ScoreTransformer` instance, required to generate malfunction 
 probability scores
 - `*.ckpt` contains the trained model parameters