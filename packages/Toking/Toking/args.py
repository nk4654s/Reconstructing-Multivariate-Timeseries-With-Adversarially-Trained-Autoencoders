"""This module contains loading, parsing and using the application's argument."""
from argparse import ArgumentParser
from typing import Dict

from Toking.resource import DEFAULTS


def parse_resource(d: Dict) -> Dict:
    """
    This function parses a dictionary defining the ArgumentParser. When 'type' is defined, the value will be parsed
    using eval. Furthermore, if 'default' is set, the default-value will be parsed using the class of 'type'.
    Args:
        d: The input dictionary.

    Returns: The parsed dictionary such that it can be unfolded into (multiple) ArgumentParser.add_argument()-calls.
    """
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = parse_resource(val)

        if key == "type":
            d[key] = eval(val)

            if "default" in d.keys():
                default = d[key](d["default"])
                d["default"] = default

                help = d.get("help", "") + f" Defaults to '{default}'."
                d["help"] = help

    return d


class TokingCLI(ArgumentParser):
    """
    This class build the command line interface (CLI) for the Tokenier application.
    """
    def __init__(self):
        resource = parse_resource(DEFAULTS)
        super().__init__(description=resource["prog_descr"])

        for arg_name, definition in resource["args"].items():
            self.add_argument(
                f"--{arg_name}",
                f"-{arg_name[0]}",
                **definition
            )
