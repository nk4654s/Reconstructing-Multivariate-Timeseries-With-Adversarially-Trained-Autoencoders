"""This module handles arguments and generation of those for pre-processing modules."""
import inspect
from argparse import ArgumentParser, Namespace, Action
from typing import List

from S3Prep.modules import BasePreProcessor


class GroupedAction(Action):
    """This class groups arguments into a single namespace.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        group, dest = self.dest.split(".", 2)
        groupspace = getattr(namespace, group, Namespace())
        setattr(groupspace, dest, values)
        setattr(namespace, group, groupspace)


class CLIArgumentParser(ArgumentParser):
    """This class handles argument parsing and generation for pre-processing modules.
    """
    def __init__(
        self, preps: List[BasePreProcessor] | BasePreProcessor, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # init arguments from child classews
        self.prepare_arguments()

        # store all pre-processor instances and instantiate their arguments
        self._preps = preps if isinstance(preps, list) else [preps]
        for prep in self._preps:
            prefix = prep.__name__.lower()
            prefix = prefix if len(prefix) < 7 else prefix[:7]

            # add arguments from pre-processor
            signature = inspect.signature(prep.__init__).parameters
            for attr in signature.keys():
                if attr == "self":
                    continue

                self.add_argument(
                    f"--{prefix}-{attr}",
                    type=signature[attr].annotation,
                    default=signature[attr].default,
                    action=GroupedAction,
                    dest=f"{prep.__name__.lower()}.{signature[attr].name}",
                    help=f"{signature[attr].name.capitalize()} of {prep.__name__}",
                )

        # parse arguments to have the namespace ready
        self._args = self.parse_args()

    def get_pre_processors(self) -> List[BasePreProcessor]:
        """The method instatiates the pre-processor classes.
        """
        # initiate the pre-processors
        modules = []
        for prep in self._preps:
            name = prep.__name__.lower()
            kwargs = {}

            # check if class name is in arguments and add the arg to the kwargs
            namespace = self.args.__dict__.get(name, None)
            if namespace is not None:
                kwargs = kwargs | namespace.__dict__

            # add defaults to kwargs
            if namespace is None:
                for key, val in self.args.__dict__.items():
                    arg_split = key.split(".")
                    if name == arg_split[0] and len(arg_split) == 2:
                        if arg_split[1] not in kwargs.keys():
                            kwargs[arg_split[1]] = val

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
        """Returns the parsed arguments as a Namespace.
        """
        return self._args
