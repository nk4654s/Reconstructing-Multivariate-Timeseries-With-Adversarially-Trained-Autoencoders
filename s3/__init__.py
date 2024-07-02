import csv
from .utils import StorageMode, logger
from .tsvio import TSVDialect, TsvIO, DictMode, ReadMode, TSVInputType, splice_streams
from .s3io import S3

# Register TSV dialect
csv.register_dialect("tsv", TSVDialect)
