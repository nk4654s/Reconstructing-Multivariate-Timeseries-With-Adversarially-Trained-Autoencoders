"""This module contains a search tree to find the file to load corresponding
to the sampling index."""
import sys
from typing import Dict, List, Tuple

from Dataflux.helper import Meta, Metadata
from Dataflux import logger

sys.setrecursionlimit(1500)


class Node:
    """
    A node in an index (binary) tree. Each node represents a day and contains references
    to its children nodes.
    
    Attr:
        day (str): The day represented by this node.
        size (int): The number of samples for the day represented by this node.
        parent (Node): The parent node of this node.
        smaller (Node): The left child node, which represents a day with fewer samples.
        bigger (Node): The right child node, which represents a day with more samples.
    """
    def __init__(self, day: str, index: int, size: int, parent: "Node | None" = None) -> None:
        self.smaller: Node | None = None
        self.bigger: Node | None = None
        self.parent = parent
        self._day = day

        self._index = index
        self._size = size

    @property
    def day(self) -> str:
        """The day string of this node.

        Returns:
            str: The day as string (YYYY-MM-DD)
        """
        return self._day

    @property
    def index(self) -> int:
        """The start index of this node in the whole tree. The most left
        node should have index 0. The most right node should equal the length
        of this tree. The middle should be roughly the root node.

        Returns:
            int: Start index of this node.
        """
        return self._index

    @property
    def size(self) -> int:
        """Number of samples, the day of this node offers.

        Returns:
            int: Number of samples, the day of this node offers.
        """
        return self._size

    def __eq__(self, other: int) -> bool: # type: ignore
        return self.size == other

    def __ge__(self, other: "Node") -> bool:
        return self.index >= other.index

    def __le__(self, other: "Node") -> bool:
        return self.index <= other.index


class IndexTree:
    """
    Represents an index tree data structure. This class manages nodes in an
    index tree and provides methods for traversing and searching the tree.
    
    Args:
        metadata (Metadata): The metadata object that contains information about the recordings.
        window_size (int): The size of the window used to calculate cumulative recordings.
        skip (int): The number of timestamps to skip. Defaults to 1 (do not skip any timestamps).
    """
    def __init__(self, metadata: Metadata, window_size: int = 1, skip: int = 1) -> None:
        logger.debug("Initialize indexing")
        self._root: Node | None = None
        self._win = window_size
        self._skip = skip
        logger.debug("Skip rows: %s", skip)

        self._size = 0
        self._build_tree(metadata[Meta.FILES])
        logger.debug("Finished indexing")

    def __len__(self) -> int:
        return self._size

    def _build_tree(self, node_list: List[Dict]) -> None:
        # define middle node as root, sorted by 'recordings'
        node_weighted_list = self._get_cumulative_size(node_list)
        self._root = node_weighted_list.pop(len(node_weighted_list) // 2)

        # insert alternating the highes, lowest node
        i = 0
        while len(node_weighted_list) != 0:
            node = node_weighted_list.pop(i)
            self._sorted_insert(self._root, node)
            i = 0 if i == len(node_weighted_list) - 1 else len(node_weighted_list) - 1

    def _get_cumulative_size(self, node_list: List[Dict]) -> List[Node]:
        # sort the node_list first, as this should help building
        # a balanced binary search tree
        node_list = sorted(node_list, key=lambda _: _[Meta.DAY.name])
        cum_weights = 0
        weights = []

        # log some stats
        samples_dropped = 0
        samples_included = 0

        for node in node_list:
            # create a new node
            node_size = node["recordings"] // self._skip
            node = Node(node[Meta.DAY.name], cum_weights, node_size)

            # skip dataframes which are smaller than the window size
            if node_size < self._win:
                samples_dropped += node_size
                continue

            # count the tree's size (subtract window size)
            samples_included += node_size
            cum_weights += node_size - self._win
            weights.append(node)

        logger.debug("Cumulative size after building Index Tree: %s", samples_included)
        logger.debug("Dropped samples from Index Tree: %s", samples_dropped)
        logger.debug("Total samples (with skip=%s): %s", self._skip,
                     samples_dropped + samples_included)

        self._size = cum_weights
        return weights

    def _sorted_insert(self, root: Node, node: Node) -> None:
        if root is None:
            return node
        if node <= root:
            if root.smaller is None:
                node.parent = root
                root.smaller = node
            else:
                self._sorted_insert(root.smaller, node)
        elif node >= root:
            if root.bigger is None:
                root.bigger = node
                node.parent = root
            else:
                self._sorted_insert(root.bigger, node)

    def _recursive_search(self, index: int, root: Node | None) -> Tuple[str, int]:
        if root is None:
            return "ERROR", index

        # Node too large
        if index < root.index:
            return self._recursive_search(index, root.smaller)

        # Node too small
        if index + self._win >= root.index + root.size:
            return self._recursive_search(index, root.bigger)

        # Node contains index
        return root.day, index - root.index

    def search(self, index: int) -> Tuple[str, int]:
        """Searches the index in the tree and returns the day (required to load
        the correct DataFrame) and index for that file.

        Args:
            index (int): The index to search the day for.
        
        Returns:
            Tuple[str, int]: The day to load with the index in that file.
        """
        return self._recursive_search(index, root=self._root)
