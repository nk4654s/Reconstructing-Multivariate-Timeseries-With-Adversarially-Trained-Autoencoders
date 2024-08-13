"""This module post-processes the downloaded and pre-processed dataset.
The `modules` container defines the order of the post-processing modules to
be executed on the downloaded DataFrames."""
from Dataflux.cli import PostProcessingCLI
from Dataflux.diskio import FileWorkerFactory, FileWorkerPool
from Dataflux.helper import Metadata
from Dataflux.modules.postprocess import NormalizeFit, NormalizeTransform, WeatherDataset, MonoFile, RemoveColumns

# define module classes to be available here
modules = [
    WeatherDataset,
    RemoveColumns,
    NormalizeFit,
    NormalizeTransform,
    MonoFile
]

def main():
    """
    The main entry point for post-processing operations. This function
    initializes the custom CLIArgumentParser with the module classes and
    handles command-line interface (CLI) arguments.
    """
    # initialize custom CLIArgumentParser
    parser = PostProcessingCLI(modules)
    prep_modules = parser.get_modules()

    # create multi-threaded file-hanlder
    factory = FileWorkerFactory(parser.args.folder, parser.args.num_workers)
    pool: FileWorkerPool = factory.initialize_workers(prep_modules)

    # start processing files and provided file-handler
    pool.ensure_module_dependency(prep_modules)
    metadata: Metadata = pool.start_processing()
    metadata.save(parser.args.folder)

if __name__ == "__main__":
    main()
