from Comprisk.anomaly import ScoreWrapper
from Comprisk.entrypoints import EntrypointBase, EntrypointContext


class SPExportEntrypoint(EntrypointBase):
    def __init__(self):
        super().__init__()

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        # train the score wrapper
        valset = context.datamodule.val_dataloader()
        wrapper = ScoreWrapper(context.model, threshold=0.8, threshold_deviation=3)
        wrapper.train_wrapper(valset, verbose=True)

        # export finished model as ONNX file
        # TODO

        return context