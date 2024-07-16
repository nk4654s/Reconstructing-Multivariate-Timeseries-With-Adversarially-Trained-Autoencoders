"""This module contains the entry point to post-process a dataset"""
from S3Prep.cli import PostProcessingCLI
from S3Prep.diskio import FileWorkerFactory, FileWorkerPool

# define module classes to be available here
module_classes = []


if __name__ == "__main__":
    # initialize custom CLIArgumentParser
    parser = PostProcessingCLI(module_classes)
    modules = parser.get_modules()

    # create multi-threaded file-hanlder
    factory = FileWorkerFactory(parser.args.path, parser.args.num_workers)
    pool: FileWorkerPool = factory.initialize_workers()

    # start processing files and provided file-handler
    pool.start_processing(modules)
