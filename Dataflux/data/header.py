"""This module contains the header class"""


from typing import List


class Header:
    """This class defines the header of a file."""

    values: List[str] = []
    drop: List[str] = []

    def relative_complement(self, other: "Header", fail_safe: bool = True) -> None:
        """Calculates the relative complement of the header values of
        this class and another header class.

        Args:
            other (Header): Another header to compare with this one.
            fail_safe (bool): Whether to raise an error if the relative
            complement is not a subset of other header.
        """
        self.drop = list(set(self.values) - set(other.values))

        if fail_safe:
            assert self.length - len(self.drop) == len(
                other.values
            ), "The relative complement is not a subset of the other header."

    @property
    def length(self) -> int:
        """Returns length of header."""
        return len(self.values)
