from typing import Any

DEFAULTS = {
    "prog_descr": "This program provides access to a valid API access token.",
    "args": {
        "pool_id": {
            "type": "str",
            "default": "***_N7tUglZOg",
            "help": "The Pool ID, where the secrets are stored.",
        },
        "client_id": {
            "type": "str",
            "default": "3692mmbdag4us7es7q9kmu3vbq",
            "help": "The Client ID.",
        },
        "region_name": {
            "type": "str",
            "default": "eu-central-1",
            "help": "The pool's user region.",
        },
        "secret_name": {
            "type": "str",
            "default": "xplore-ai-cas/jwt-token",
            "help": "The secret's name/key.",
        },
        "username": {
            "type": "str",
            "default": "",
            "help": "The default username. If empty, the username stored in AWS secret manager will be used.",
        }
    },
}


def get_arg_default(by_name: str) -> Any:
    d = DEFAULTS["args"][by_name]
    val = d["default"]

    if "type" in d.keys():
        return eval(d["type"])(val)

    return val
