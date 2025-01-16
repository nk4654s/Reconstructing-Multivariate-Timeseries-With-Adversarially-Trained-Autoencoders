"""This module handles arguments and generation of those for pre-processing modules."""

from argparse import ArgumentParser, Namespace
from typing import List

from Dataflux.cli.config_helper import build_config, expand_module_list
from Dataflux.cli.defaults import DEFAULT_ARGUMENTS


class FluxConfigManager:
    def __init__(self, config_path: str, default_modules: List = None) -> None:
        # initialize the namespace and add the config to it
        self.args = Namespace()
        setattr(self.args, "config", build_config(config_path))
        setattr(self.args, "config_path", config_path)

        # build the namespace based on the config's content
        for key, value in DEFAULT_ARGUMENTS.items():
            value = self.args.config.get(key, value)

            # check if an arg was explicitly defined, then keep its value
            if key in vars(self.args).keys() and getattr(self.args, key):
                continue

            setattr(self.args, key, value)

        # store all processing modules as classes, expand the modules by those configured in the config file
        self._modules = default_modules if default_modules else []
        self._modules = expand_module_list(self._modules, self.args.config)

        # build the config dictionary holding modules and init_args
        self._config = []
        self._load_modules_from_args()

    def set_latest(self, value: bool) -> None:
        if value is None:
            return None

        setattr(self.args, "latest", value)

    def set_folder(self, value: str) -> None:
        if value is None:
            return None

        setattr(self.args, "folder", value)


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
        """The method instantiates the processing modules. To call this method, arguments must
        define attributes of processing modules, if a default is not defined.
        """
        # initiate each processing module with its arguments
        modules = []
        for data in self._config:
            # instantiate the class
            class_path, order, init_args = data["class_path"], data["order"], data["init_args"]
            module = class_path(**init_args)
            modules.append((module, order))

        modules = sorted(modules, key=lambda x: x[1])
        modules = [module[0] for module in modules]

        return modules


class CLIArgumentParser(ArgumentParser):
    """This class handles argument parsing and generation for pre-processing modules."""

    def __init__(
        self,
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

        # build the configuration manager
        config_file = self._args.config
        self._fcm = FluxConfigManager(config_file)

        self._fcm.set_latest(self._args.latest)
        self._fcm.set_folder(self._args.folder)

    def get_flux(self) -> FluxConfigManager:
        """The method instantiates the processing modules. To call this method, arguments must
        define attributes of processing modules, if a default is not defined.
        """
        return self._fcm

    def prepare_arguments(self):
        """Adds the arguments to the parser, before creating the namespace. Overwrite this
        function in your subclass and add `self.add_argument` calls for each argument you
        want to add.
        """
        return

    @property
    def args(self):
        """Returns the parsed arguments as a Namespace."""
        return self._fcm.args
