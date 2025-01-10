"""This module handles arguments and generation of those for pre-processing modules."""

from argparse import ArgumentParser, Namespace, Action
from typing import List

from Dataflux.cli.config_helper import build_config, expand_module_list
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
        modules: List,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # init additional arguments from child classes
        self.add_argument("--config", "-c", type=str, help="Path to pipeline config yaml.")
        self.add_argument("--latest", action="store_true", help="Whether to use the last hour of data or not.")
        self.add_argument("--folder", help="The output folder name")
        self.prepare_arguments()
        self._args = self.parse_args()

        # pre-parse args to load config if given
        config_file = self.args.config
        self.args.config = build_config(self.args.config)
        for key, value in DEFAULT_ARGUMENTS.items():
            value = self.args.config.get(key, value)

            # check if an arg was explicitly defined, then keep its value
            if key in vars(self.args).keys() and getattr(self.args, key):
                continue

            setattr(self.args, key, value)

        # store all processing modules as classes and inspect their attributes
        # each attribute required by the classes constructor will be added as argument to
        # the parser. The namespace of the argument will be `class.attribute_name`
        self._modules = modules if isinstance(modules, list) else [modules]
        self._modules = expand_module_list(self._modules, self.args.config)
        self.args.config = config_file

        # build the config dictionary holding modules and init_args
        self._config = []
        self._load_modules_from_args()
    
    def _load_modules_from_args(self) -> None:
        for i, module in enumerate(self._modules):
            # get the module's name
            name = module["class_path"].__name__.lower() # type: ignore
            kwargs = module["init_args"]

            # add defaults to kwargs, without overwriting the user's input
            for arg_key, val in self.args.__dict__.items():
                arg_split = arg_key.split(".")
                if len(arg_split) == 3 and "modules" == arg_split[0] and name == arg_split[1]:
                    if arg_split[2] not in kwargs.keys():
                        kwargs[arg_split[2]] = val

            self._config.append({"class_path": module["class_path"], "order": i, "init_args": kwargs})

    def get_modules(self) -> List:
        """The method instatiates the processing modules. To call this method, arguments must
        define attributes of processing modules, if a default is not defined.
        """
        # initiate the each processing module with its arguments
        modules = []
        for data in self._config:
            # instantiate the class
            class_path, order, init_args = data["class_path"], data["order"], data["init_args"]
            module = class_path(**init_args)
            modules.append((module, order))

        modules = sorted(modules, key=lambda x: x[1])
        modules = [module[0] for module in modules]

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
