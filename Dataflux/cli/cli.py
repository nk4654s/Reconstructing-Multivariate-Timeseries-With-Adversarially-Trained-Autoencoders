"""This module handles arguments and generation of those for pre-processing modules."""

import inspect
import re
from argparse import ArgumentParser, Namespace, Action
from typing import Dict, List

from Dataflux.helper import fullname, class_for_name
from Dataflux.modules import BasePreProcessor

from Dataflux.cli.config_helper import build_config, expand_module_list, get_module_config_args
from Dataflux.cli.defaults import DEFAULT_ARGUMENTS


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
        self,
        modules_pre: List[BasePreProcessor] | BasePreProcessor,
        modules_post: List[BasePreProcessor] | BasePreProcessor,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # init additional arguments from child classes
        self.add_argument("--config", "-c", type=str, help="Path to pipeline config yaml.")
        self.prepare_arguments()

        # pre-parse args to load config if given
        self._args = self.parse_args()
        self.args.config = build_config(self.args.config)
        for key, value in DEFAULT_ARGUMENTS.items():
            value = self.args.config.get(key, value)
            arg_name = key.replace("_", "-")
            self.add_argument(f"--{arg_name}", default=value, dest=key, help=f"{key} generated from config.")

        # store all processing modules as classes and inspect their attributes
        # each attribute required by the classes constructer will be added as argument to
        # the parser. The namespace of the argument will be `class.attribute_name`
        self._modules = {
            "preprocessing": modules_pre if isinstance(modules_pre, list) else [modules_pre], 
            "postprocessing": modules_post if isinstance(modules_post, list) else [modules_post]
        }
        self._modules = expand_module_list(self._modules, self.args.config)

        for key, modules in self._modules.items():
            for module in modules:
                prefix = re.findall("[A-Z][^A-Z]*", module.__name__)[0]
                prefix = prefix.lower()

                init_args = get_module_config_args(self.args.config, key, module)
                assert isinstance(init_args, dict), f"{key}.{module.__name__}.init_args has to be dictionary in the config file."

                # add arguments from pre-processor
                signature = inspect.signature(module.__init__).parameters
                for attr in signature.keys():
                    if attr == "self":
                        continue
                    
                    attr_arg = attr.replace("_", "-")
                    self.add_argument(
                        f"--{prefix}-{attr_arg}",
                        type=signature[attr].annotation,
                        default=init_args.get(signature[attr].name, signature[attr].default),
                        action=GroupedAction,
                        dest=f"{key}.{module.__name__.lower()}.{signature[attr].name}",
                        help=f"{signature[attr].name.capitalize()} of {module.__name__}",
                    )

        # parse arguments to have the namespace ready
        self._args = self.parse_args()

        # build the config dictionary holding modules and init_args
        self._config = {"preprocessing": [], "postprocessing": []}
        self._load_modules_from_args()
    
    def _load_modules_from_args(self) -> None:
        for key, modules in self._modules.items():
            for i, module in enumerate(modules):
                # get the module's name
                name = module.__name__.lower()
                kwargs = {}

                # get the attributes set by the user in the command line interface
                namespace = self.args.__dict__.get(name, None)
                if namespace is not None:
                    kwargs = kwargs | namespace.__dict__

                # add defaults to kwargs, without overwriting the user's input
                if namespace is None:
                    for arg_key, val in self.args.__dict__.items():
                        arg_split = arg_key.split(".")
                        if name == arg_split[0] and len(arg_split) == 2:
                            if arg_split[1] not in kwargs.keys():
                                kwargs[arg_split[1]] = val

                self._config[key].append({"class_path": fullname(module), "order": i, "init_args": kwargs})

    def get_modules(self) -> Dict[str, List[BasePreProcessor]]:
        """The method instatiates the processing modules. To call this method, arguments must
        define attributes of processing modules, if a default is not defined.
        """
        # initiate the each processing module with its arguments
        modules = {"preprocessing": [], "postprocessing": []}
        for key, config_modules in self._config.items():
            for data in config_modules:
                # instantiate the class
                class_path, order, init_args = data["class_path"], data["order"], data["init_args"]
                module = class_for_name(*class_path.rsplit(".", maxsplit=1))(**init_args)
                modules[key].append((module, order))

            modules[key] = sorted(modules[key], key=lambda x: x[1])
            modules[key] = [module[0] for module in modules[key]]

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
