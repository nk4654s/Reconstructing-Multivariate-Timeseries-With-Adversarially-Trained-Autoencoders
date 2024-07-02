"""This module handles all the tsv-file input/output and defines the tsv dialect."""
import itertools
import os
import csv
from csv import DictWriter
from enum import Enum
from io import StringIO, TextIOBase, TextIOWrapper
from pathlib import Path
from tempfile import SpooledTemporaryFile
import shutil
from typing import Literal, Any, Optional, TextIO, Iterable
import pandas as pd

from .utils import StorageMode, logger

GiB = 1024 * 1024 * 1024


class TSVDialect(csv.Dialect):
    """The TSVDialect defines how the tsv file is formatted and how it can be processed.
    """
    delimiter = "\t"
    doublequote = True
    lineterminator = "\n"
    quotechar = '"'
    quoting = csv.QUOTE_NONE


class TSVInputType(Enum):
    STR = "str"
    TEXTIO = "TextIO"
    DICT = "dict"


class ReadMode(Enum):
    ROW = "row"  # Read as dictionary by row
    STR = "str"  # Read as string


class DictMode(Enum):
    ROW = "row"
    COL = "col"


class TsvIO(TextIOBase):
    """
    Provides a wrapper around a TSV file that allows for reading and writing in-memory or on-disk.

    The TSV file can be provided as a string, a TextIO stream or a dictionary.

    Can be read like a TextIO stream, or as a dictionary of rows or columns. The ReadMode can be
    used to specify how the
    file should be read when iterated.
    """

    def __init__(
            self,
            tsv: TextIO | str | dict[Any, dict[Any, Any]],
            storage_mode: StorageMode | Literal["in_memory", "on_disk"] = StorageMode.IN_MEMORY,
            *,
            file_path: str = None,
            read_mode: ReadMode = ReadMode.STR,
            dict_mode: DictMode | Literal["row", "col"] = DictMode.ROW,
            mask: dict[str, int] = None,
            **kwargs,
    ):
        """
        Initializes the VsvIO class and defines how to handle the s3 input stream.

        Args:
            tsv (TextIO | str | dict[Any, dict[Any, Any]]): The TSV file as a TextIO buffer.
            Assumed to have a timestamp as first column, followed by columns of data.
            storage_mode (StorageMode): Whether to store the TSV in memory or on disk.
            Defaults to `StorageMode.IN_MEMORY`.\n
            file_path (str): The path to the file to read from or write to. Only relevant
            for mode "on_disk". Defaults to None.\n
            read_mode (ReadMode): Row: Read the TSV as a dictionary of timestamp rows. More
            specifically a tuple ts, sensor -> val. Skips the header column automatically.
            Str: Read the TSV as a string directly from the underlying stream. Defaults to
            `ReadMode.STR`.\n
            dict_mode (DictMode): Row: A dictionary mapping the first column as keys for each
            complete row. I.e. ts -> sensor -> val. Col: A dictionary mapping every but the
            first column as the key for their entire column with timestamps. I.e. sensor ->
            ts -> val. Defaults to `DictMode.ROW`.\n
            mask (dict): A dict of column names to mask values. If None, no masking is done.
            The mask is interpreted as exclusive if any value is -1. Otherwise, it is
            interpreted as inclusive. For now: Only has effect when dictionary like data is
            returned. Defaults to None.\n
            kwargs: Additional arguments to pass to internals.
        """

        # Determine tsv input type
        match tsv:
            case str():
                self.tsv_input_type = TSVInputType.STR
            case TextIOWrapper() | TextIOBase() | StringIO():
                self.tsv_input_type = TSVInputType.TEXTIO
            case dict():
                self.tsv_input_type = TSVInputType.DICT
            case _:
                raise TypeError("tsv must be of type str, TextIO, or dict")

        self._stream_delegate = TsvIO._create_stream_delegate(
            storage_mode, tsv, file_path=file_path, dict_mode=dict_mode, **kwargs
        )

        self._read_mode = read_mode
        self._storage_mode = storage_mode

        self._stream_delegate.seek(0)
        self._header = self._stream_delegate.readline().strip("\n").split("\t")[1:]
        self._stream_delegate.seek(0)
        self._mask = self._normalize_mask(mask)  # Requires initialized header

    @property
    def read_mode(self) -> ReadMode:
        """Returns the read mode of the stream.
        """
        return self._read_mode

    @read_mode.setter
    def read_mode(self, read_mode: ReadMode):
        """Sets the read mode.
        """
        if read_mode not in ReadMode:
            raise ValueError(f"read_mode must be one of {ReadMode}")
        self._read_mode = read_mode

    @property
    def storage_mode(self) -> StorageMode:
        """Returns the storage mode.
        """
        return self._storage_mode

    @property
    def header(self) -> list[str]:
        """The header (only sensor column ids) of the TSV file as a list.
        """
        return self._header

    @header.getter
    def header(self) -> list[str]:
        return self._header

    @property
    def mask(self) -> dict | None:
        """The column mask to use for every read action.
        """
        return self.mask

    @mask.setter
    def mask(self, mask: list[str] | None):
        """Sets the column mask to use for every read action.
        """
        self._mask = self._normalize_mask(mask)

    @mask.getter
    def mask(self) -> dict | None:
        """The column inclusive mask to use for every read action.
        """
        if self._mask is None:
            return None
        else:
            return {col: self._mask[i] for i, col in enumerate(self.header)}

    @property
    def masked_header(self) -> list[str]:
        """The header with the mask applied.
        """
        if self._mask is None:
            return self.header
        else:
            return [col for i, col in enumerate(self.header) if self._mask[i] == 1]

    def get_full_header_string(self) -> str:
        """Returns the full header string of the TSV file without trailing newline.
        """
        return "ts\t{}".format("\t".join(self.header)) + "\n"

    def get_masked_full_header_string(self) -> str:
        """Returns the full header string of the TSV file without trailing newline.
        """
        return "ts\t{}".format("\t".join(self.masked_header)) + "\n"

    @staticmethod
    def _write_row_dict_to_stream(stream: TextIO, d: dict[Any, dict[Any, Any]]) -> None:
        """Writes a complete nested row dictionary to a stream.

        Args:
            stream (TextIO): The stream to write to.\n
            d (dict): The nested dictionary to write. Can only be nested one level deep. The 
            first level keys are the timestamps, the second level keys are the column names.
            All keys should be present for every timestamp as the first row is used to infer
            the column names.
        """

        # Infer column names from first row
        keys = [col for col in next(iter(d.values())).keys()]
        keys.sort()
        timestamp_header = "ts"
        fieldnames = [timestamp_header, *keys]

        dict_writer = DictWriter(stream, dialect="tsv", fieldnames=fieldnames)
        dict_writer.writeheader()
        for ts, row in d.items():
            row = {timestamp_header: ts, **row}
            dict_writer.writerow(row)

    @staticmethod
    def _create_stream_delegate(
            mode: Literal["in_memory", "on_disk"] | StorageMode,
            tsv: TextIO | str | dict[Any, dict[Any, Any]],
            *,
            file_path: Optional[str] = None,
            dict_mode: Literal["row", "col"] | DictMode = DictMode.ROW,
            spooling_size: int = 0.5 * GiB,
            buffering: int = -1,
            **kwargs,
    ):
        """Handles creating a stream delegate for the tsv in the given mode according to
        the given parameters.

        Args:
            mode (StorageMode): Whether to store the data in memory or on disk. If on disk,
            a temporary file is created.\n
            tsv (TextIO): The TSV data to write. Can be a string, a TextIO, or a dictionary
            of timestamp rows or sensor columns.\n
            file_path (str): The file path to write to. If not specified, a temporary file
            will be used. Only relevant for mode "on_disk".\n
            dict_mode (DictMode): Whether to interpret the TSV as a dictionary of timestamp
            rows or sensor columns. Defaults to `DictMode.ROW`.\n
            spooling_size (int): The size in bytes for which the data is spooled in memory.
            Only relevant for mode "on_disk". Defaults to 0.5 GiB.\n
            buffering (int): How to handle buffering. Defaults to -1, which means line buffering.
            Only relevant for mode "on_disk".
        """

        stream = None
        match mode:
            case StorageMode.IN_MEMORY:
                match tsv:
                    case str():
                        stream = StringIO(tsv, **kwargs)
                    case TextIOWrapper() | TextIOBase() | StringIO():
                        stream = StringIO(**kwargs)
                        for line in tsv:
                            stream.write(line)
                    case dict():
                        stream = StringIO(**kwargs)
                        match dict_mode:
                            case DictMode.ROW | "row":
                                TsvIO._write_row_dict_to_stream(stream, tsv)
                            case DictMode.COL | "col":
                                raise NotImplementedError
                            case _:
                                raise ValueError(f"Unknown DictMode: {dict_mode}")

            case StorageMode.ON_DISK:
                if file_path:
                    # stream = open(file_path, "x", buffering=buffering, **kwargs)
                    if os.path.exists(file_path):
                        raise FileExistsError(f"The file '{file_path}' already exists.")
                    else:
                        stream = open(file_path, "x+", encoding="UTF-8")
                else:
                    stream = SpooledTemporaryFile(
                        max_size=spooling_size, mode="w+", buffering=buffering, **kwargs
                    )

                match tsv:
                    case str():
                        stream.write(tsv)
                    case TextIOWrapper() | TextIOBase() | StringIO():
                        for line in tsv:
                            stream.write(line)
                    case dict():
                        match dict_mode:
                            case DictMode.ROW | "row":
                                TsvIO._write_row_dict_to_stream(stream, tsv)
                            case DictMode.COL | "col":
                                raise NotImplementedError
                            case _:
                                raise ValueError(f"Unknown DictMode: {dict_mode}")

        stream.seek(0)
        return stream

    def _normalize_mask(self, mask: dict[str, int] | None) -> list[int] | None:
        """Normalize the mask to an inclusive mask which consists of an index list of
        columns to include.

        Args:
            mask (dict): The mask to normalize.
        
        Returns:
            list[int]: The normalized mask.
        """
        if mask is None:
            return None
        else:
            if any([v == -1 for v in mask.values()]):
                # Exclusive mask
                excluded_columns = [k for k, v in mask.items() if v == -1]
                return [1 if k not in excluded_columns else 0 for k in self._header]
            else:
                # Inclusive mask
                return [1 if mask.get(col, 0) == 1 else 0 for col in self._header]

    #####################
    # Iterator Protocol #
    #####################
    def __iter__(self):
        return self

    def __next__(self) -> str | tuple[int, dict[str, float | None]]:
        match self.read_mode:
            case ReadMode.STR:
                return self.readline()
            case ReadMode.ROW:
                return self.readrow()
            case _:
                raise ValueError("Invalid read mode")

    def readrow(self) -> tuple[int, dict[str, float | None]]:
        """Read the single next line and return it as a tuple of ts and dictionary sensor -> val.
        """
        # Skip header if we are at the beginning of the file
        if self.tell() == 0:
            self.readline()

        row = self.readline()

        if not row:
            raise StopIteration

        return TsvIO.row_to_dict(row, self._header, mask=self._mask)

    def readrows(self, hint: int = -1) -> dict[int, dict[str, float | None]]:
        """ Read the next hint number of lines and return it as a tuple of ts and
        dictionary sensor -> val.
        """

        # Skip header if we are at the beginning of the file
        if self._delegate.tell() == 0:
            self._delegate.readline()

        rows_dict = {}

        if hint == -1:
            # Iterate over the entire file directly with delegate
            for row in self._delegate:
                ts, row_dict = TsvIO.row_to_dict(row, self._header, mask=self._mask)
                rows_dict.update({ts: row_dict})

        else:
            for _ in range(hint):
                ts, row_dict = self.readrow()
                rows_dict.update({ts: row_dict})

        return rows_dict

    def writerow(
            self, ts: int, row: dict[str, float | None], *, force_append: bool = True
    ):
        """Write a single row to the TSV file. Iterates over self.header to ensure the correct
        order. It has to be ensured by the caller that the reader is at the correct position to
        not overwrite data. The timestamp is also not scrutinized in any way.

        Args:
            ts (int): The timestamp.\n
            row (dict): The row to write as dictionary sensor -> val.\n
            force_append (bool): Whether to force appending the row to the very end of the file.
            Defaults to True.\n
        """

        if force_append:
            self._delegate.seek(0, os.SEEK_END)

        self._delegate.write(f"{ts}\t")

        vals = []
        for col in self.header:
            val = row.get(col, "")
            vals.append(str(val) if val is not None else "")
        self._delegate.write("\t".join(vals))

        self._delegate.write("\n")

    @staticmethod
    def row_to_dict(
            row: str, header: list[str], *, mask: list[int] = None
    ) -> tuple[int, dict[str, float | None]]:
        """Convert a single row of a TSV file to a dictionary of ts -> sensor -> val with
        a given header. Handles empty values and appended newlines.

        Args:
            row (str): The row to convert.\n
            header (List[str]): The header to use. Is mapped onto the row without any checks.
            Does not contain the timestamp. Needs to be in the same order as the row and 
            be of the same length.\n
            mask List[int]: An index mask of columns to include. If None, all columns are
            included. Has to be of the same length as the header.
        
        Return:
            row (Tuple[int, dict[str, float | None]]): The row as (ts, sensor -> val)
        """

        ts, *row = row.rstrip("\n").split("\t")
        # ts -> sensor -> val as float and None if empty
        if mask is None:
            row_dict = {
                header[i]: float(row[i]) if row[i] else None for i in range(len(header))
            }
        else:
            if len(mask) != len(header):
                raise ValueError(
                    f"Mask length ({len(mask)}) does not match header length ({len(header)})"
                )
            # Extract column indices to include
            indices = [i for i, v in enumerate(mask) if v == 1]
            row_dict = {
                header[idx]: float(row[i]) if row[i] else None
                for i, idx in enumerate(indices)  # `i` iterates over the row and `idx` over the header
            }

        return int(ts), row_dict

    #######################
    # Write Methods #
    #######################
    def append_row(self, ts: int, row: dict[str, float | None], filler: float = ""):
        """Appends a single row in a "smart" way to the TSV file fitting it to the header of
        this file. The proper order of the rows needs to be enforced by the caller.

        Args:
            ts (int): The timestamp.
            row (dict): The row as dictionary sensor -> val.
            filler (float): With what to fill missing values. Defaults to "".
        """

        self._delegate.write(f"{ts}\t")
        vals = []
        for sensor in self.header:
            val = row.get(sensor, filler)
            val = val if val is not None else filler
            vals.append(str(val))
        self._delegate.write("\t".join(vals))
        self._delegate.write("\n")

    #################################
    # Delegated text stream methods #
    #################################
    @property
    def _delegate(self):
        return self._stream_delegate

    def __getattr__(self, attr):
        return getattr(self._delegate, attr)

    def read(self, size=-1):
        return self._delegate.read(size)

    def readinto(self, b):
        return self._delegate.readinto(b)

    def readline(self, **kwargs):
        line = self._delegate.readline()
        if not line:
            raise StopIteration
        # Filter using mask
        if self._mask is None:
            return line
        else:
            row = line.rstrip("\n").split("\t")
            return row[0] + "\t" + "\t".join(
                [
                    val
                    for i, val in enumerate(row[1:])
                    if self._mask[i] == 1
                ]
            )

    def readlines(self, hint=-1):
        return self._delegate.readlines(hint)

    def write(self, s):
        return self._delegate.write(s)

    def writelines(self, lines):
        return self._delegate.writelines(lines)

    def truncate(self, size=None):
        return self._delegate.truncate(size)

    def seek(self, offset, whence=0):
        return self._delegate.seek(offset, whence)

    def tell(self):
        return self._delegate.tell()

    def flush(self):
        return self._delegate.flush()

    def isatty(self):
        return self._delegate.isatty()

    # Not directly part of the TextIOBase interface but useful for pytorch as well
    def reset(self):
        """Reset the stream to the beginning.
        """
        self._delegate.seek(0)

    ############################

    ###############
    # I/O Methods #
    ###############

    def write_to_file(self, file_path: str | Path, *, overwrite: bool = False):
        """Writes the TSV to a given file.
        
        Args:
            file_path (str): The file path to write to.
            overwrite (bool): Whether to overwrite the file if it already exists.
            Defaults to False.
        """
        if os.path.exists(file_path):
            if overwrite:
                os.remove(file_path)
            else:
                raise FileExistsError(f"The file '{file_path}' already exists.")

        with open(file_path, "w", encoding="UTF-8") as f:
            self.reset()
            shutil.copyfileobj(self, f)

        logger.info(f"Wrote TSV to '{file_path}'")

    def to_dataframe(self, *, ignore_mask: bool = False, **kwargs) -> pd.DataFrame:
        """Converts the TSV to a pandas DataFrame.

        Args:
            ignore_mask (bool): Whether to ignore the mask of this object and return all
            columns. Defaults to False.
            kwargs (dict): Keyword arguments passed to pandas.read_csv.
        """
        self.reset()
        df = pd.read_csv(self, sep="\t", index_col="ts", **kwargs)
        if self.mask is not None and not ignore_mask:
            df = df.loc[:, self.masked_header]

        return df


def splice_streams(
        streams: Iterable[TextIO | None],
        *,
        header: list[str] = None,
        storage_mode: StorageMode = StorageMode.ON_DISK,
        drop_additional_columns: bool = False,
        default_fill_value: float | str | None = "",
        **tsv_io_kwargs,
) -> TsvIO:
    """Splices the given TextIO streams in the given order. By comparing the headers of
    each TextIO this method for now throws an error when the headers do not match. It is
    planned to in the future be able to splice headers (and therefore columns) as well. This
    can be circumvented by getting a header that encompasses all columns of the given
    streams ahead of time.

    Args:
        streams (Iterable[TextIO]): TextIOs with matching headers.
        header (list[str]): The header to use. If None, the header of the first stream is
        used. Can be used to splice TSVs with different headers. Defaults to None.
        storage_mode (StorageMode): Defines the output mode. Defaults to StorageMode.ON_DISK.
        drop_additional_columns (bool): Whether to drop columns which are not present in the
        given header. Defaults to False.
        default_fill_value (float): The value to fill missing values with. If None, a
        ValueError is raised when a column is missing relative to the given header. Defaults
        to "".
        tsv_io_kwargs (dict): Keyword arguments passed to the TsvIO constructor.

    Return:        
        (TsvIO): The spliced stream wrapped in a TsvIO.
    """
    # Create iterator of streams in case it is not already
    streams = iter(streams)
    # Initialized with the first (not None) TextIO
    first_stream = next(streams)
    while first_stream is None:
        first_stream = next(streams)
    first_stream.seek(0)

    # Get header
    if header is not None:
        # Transform given header list to full header string
        header_str = "ts\t" + "\t".join(header) + "\n"
        # Still need to skip header of the first stream
        first_stream.readline()
    # Use header of first stream if none is given
    else:
        header_str = first_stream.readline()
        header = header_str.rstrip("\n").split("\t")[1:]
    # Write header
    output_stream = TsvIO(header_str, storage_mode=storage_mode, **tsv_io_kwargs)
    # Ensure to have the writing head after the header, so it won't be overwritten
    output_stream.seek(len(header_str))

    # Splice streams
    streams_with_additional = []  # The streams for which columns where dropped
    streams_with_missing = []  # The streams for which columns where inserted
    stream_count = 0
    empty_stream_count = 0
    for i, stream in enumerate(itertools.chain([first_stream], streams)):
        stream_count += 1
        if stream is None:
            empty_stream_count += 1
            continue
        # Reset
        stream.seek(0)
        # Check header is a subset of the overall header and create filter
        current_header = stream.readline().rstrip("\n").split("\t")[1:]

        # Any column missing in overall header (current_header > header) - aka. additional columns
        header_filter = [True] + [True] * len(current_header)
        if any(col not in header for col in current_header):
            additional_columns = [col for col in current_header if col not in header]
            logger.debug(
                f"The given header of stream {i} contains additional columns {additional_columns}."
            )
            streams_with_additional.append(i)

            if not drop_additional_columns:
                raise ValueError(
                    f"The given header of stream {i} contains additional columns {additional_columns}."
                    " To drop additional columns, set `drop_additional_columns=True`."
                )

            logger.debug(f"Dropping additional columns {additional_columns}.")
            # Get filter list of boolean "keep" values
            header_filter = [True] + [col in header for col in current_header]

        # Any column missing in current stream (current_header < header) aka. missing columns
        insertion_indices = (
            []
        )  # Where to insert the default value into the row (after additional columns are dropped)
        if any(col not in current_header for col in header):
            missing_columns = [
                col for col in header if col not in current_header
            ]  # missing from current stream
            logger.debug(
                f"The header of stream {i} is missing columns {missing_columns}."
            )
            streams_with_missing.append(i)

            if default_fill_value is None:
                raise ValueError(
                    f"The given header of stream {i} contains additional columns {missing_columns}."
                    " To drop missing columns, set `default_fill_value` to something other than `None`."
                )

            logger.debug(
                f"Filling missing values of columns {missing_columns} with {default_fill_value}."
            )
            insertion_indices = [header.index(col) for col in missing_columns]

        for line in stream:
            line = line.strip("\n")
            # Filter line
            filtered_line = [
                val for (flag, val) in zip(header_filter, line.split("\t")) if flag
            ]
            for idx in insertion_indices:
                filtered_line.insert(
                    idx + 1, str(default_fill_value)
                )  # +1 to account for timestamp column
            output_stream.write("\t".join(filtered_line) + "\n")

    if streams_with_additional:
        logger.warning(f"Streams with additional columns: {streams_with_additional}")
    if streams_with_missing:
        logger.warning(f"Streams with missing columns: {streams_with_missing}")
    if empty_stream_count > 0:
        logger.info(f"{empty_stream_count}/{stream_count} stream were empty {(empty_stream_count / stream_count) * 100}%")

    output_stream.flush()
    output_stream.seek(0)

    return output_stream
