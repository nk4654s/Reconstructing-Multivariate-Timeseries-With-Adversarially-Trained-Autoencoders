"""This module contains the entry point to post-process a dataset"""
from S3Prep.cli import PostProcessingCLI
from S3Prep.diskio import FileWorkerFactory, FileWorkerPool
from S3Prep.helper import Metadata
from S3Prep.modules.postprocess import TrainNormalize, ApplyNormalize

# define module classes to be available here
module_classes = [
    TrainNormalize,
    ApplyNormalize
]

if __name__ == "__main__":
    # initialize custom CLIArgumentParser
    parser = PostProcessingCLI(module_classes)
    modules = parser.get_modules()

    # create multi-threaded file-hanlder
    factory = FileWorkerFactory(parser.args.folder, parser.args.num_workers)
    pool: FileWorkerPool = factory.initialize_workers(modules)

    # start processing files and provided file-handler
    pool.ensure_module_dependency(modules)
    metadata: Metadata = pool.start_processing()
    metadata.save(parser.args.folder)
