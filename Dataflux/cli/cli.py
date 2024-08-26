"""This module handles arguments and generation of those for pre-processing modules."""

import inspect
import re
from argparse import ArgumentParser, Namespace, Action
from typing import List

from Dataflux.modules import BasePreProcessor


class GroupedAction(Action):
    """This class groups arguments into a single namespace. For example, the class `TestModule` has
    the attributes `A` and `B`, then this class groups those attributes given the namespace
    `TestModule.A` and `TestModule.B`.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        group, dest = self.dest.split(".", 2)
        groupspace = getattr(namespace, group, Namespace())
        setattr(groupspace, dest, values)
        setattr(namespace, group, groupspace)


class CLIArgumentParser(ArgumentParser):
    """This class handles argument parsing and generation for pre-processing modules."""

    def __init__(
        self, modules: List[BasePreProcessor] | BasePreProcessor, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # init additional arguments from child classes
        self.prepare_arguments()

        # store all processing modules as classes and inspect their attributes
        # each attribute required by the classes constructer will be added as argument to
        # the parser. The namespace of the argument will be `class.attribute_name`
        self._preps = modules if isinstance(modules, list) else [modules]
        for prep in self._preps:
            prefix = re.findall("[A-Z][^A-Z]*", prep.__name__)[0]
            prefix = prefix.lower()

            # add arguments from pre-processor
            signature = inspect.signature(prep.__init__).parameters
            for attr in signature.keys():
                if attr == "self":
                    continue
                
                attr_arg = attr.replace("_", "-")
                self.add_argument(
                    f"--{prefix}-{attr_arg}",
                    type=signature[attr].annotation,
                    default=signature[attr].default,
                    action=GroupedAction,
                    dest=f"{prep.__name__.lower()}.{signature[attr].name}",
                    help=f"{signature[attr].name.capitalize()} of {prep.__name__}",
                )

        # parse arguments to have the namespace ready
        self._args = self.parse_args()

    def get_modules(self) -> List[BasePreProcessor]:
        """The method instatiates the processing modules. To call this method, arguments must
        define attributes of processing modules, if a default is not defined.
        """
        # initiate the each processing module with its arguments
        modules = []
        for prep in self._preps:
            # get the module's name
            name = prep.__name__.lower()
            kwargs = {}

            # get the attributes set by the user in the command line interface
            namespace = self.args.__dict__.get(name, None)
            if namespace is not None:
                kwargs = kwargs | namespace.__dict__

            # add defaults to kwargs, without overwriting the user's input
            if namespace is None:
                for key, val in self.args.__dict__.items():
                    arg_split = key.split(".")
                    if name == arg_split[0] and len(arg_split) == 2:
                        if arg_split[1] not in kwargs.keys():
                            kwargs[arg_split[1]] = val

            # instantiate the class
            modules.append(prep(**kwargs))

        return modules

    def prepare_arguments(self):
        """Adds the arguments to the parser, before creating the namespace. Overwrite this
        function in your sub-class and add `self.add_argument` calls for each argument you
        want to add.
        """
        return

    @property
    def args(self):
        """Returns the parsed arguments as a Namespace."""
        return self._args
