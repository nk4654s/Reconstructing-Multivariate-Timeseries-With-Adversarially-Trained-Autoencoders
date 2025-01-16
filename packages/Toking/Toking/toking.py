import datetime
from typing import Tuple, Dict
from getpass import getpass

from pycognito.aws_srp import AWSSRP

import boto3
from botocore.exceptions import ClientError

from Toking.args import TokingCLI
from Toking.resource import get_arg_default


class Toking:
    """Toking handles token generation and caching to access IAM.

    Args:
        region_name (str): The region name (e.g. 'eu-central-1').
        secret_name (str): The secret's key value.
    """

    # borg pattern
    __shared_state: Dict = None

    def __init__(
        self,
        region_name: str = None,
        secret_name: str = None,
        username: str = None,
        password: str = None,
    ) -> None:
        self._region_name = region_name or get_arg_default("region_name")
        self._secret_name = secret_name or get_arg_default("secret_name")

        # use the borg pattern
        if not Toking.__shared_state:
            # set or get credentials required to get a token
            self._credentials = (username, password)
            if not username:
                self._credentials = self.get_cognito_credentials()

            # token and expiration datetime
            self._last_token: Tuple[str | None, datetime.datetime | None] = (None, None)
            Toking.__shared_state = self.__dict__

        else:
            self.__dict__ = Toking.__shared_state

    def get_cognito_credentials(self) -> Tuple[str, str]:
        """This function read the secret value given a region and secret key name value.

        Returns:
            Tuple[str, str]: The secret value, usually a username and password.
        """
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name="secretsmanager", region_name=self._region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=self._secret_name
            )
        except ClientError as e:
            # For a list of exceptions thrown, see
            # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
            raise e

        result = eval(get_secret_value_response["SecretString"])
        return result.get("username", None), result.get("password", None)

    def get_access_token(self) -> str:
        """This function gets the access token for IAM portal using the `DEFAULTS` values defined in
        `Tokenier.resource`.

        Returns:
            str: The token

        """
        # check if there is a valid token in this class
        last_token, expiration = self._last_token
        if last_token and expiration > datetime.datetime.now():
            return last_token

        # no token found or expired, get a new one
        username, password = self._credentials

        aws = AWSSRP(
            username=username,
            password=password,
            pool_id=get_arg_default("pool_id"),
            client_id=get_arg_default("client_id"),
        )

        result = aws.authenticate_user()
        if result:
            # calculate when the token will expire
            expiration = result["AuthenticationResult"]["ExpiresIn"]
            expiration = datetime.datetime.now() + datetime.timedelta(
                seconds=expiration
            )

            # extract the token
            token = result["AuthenticationResult"]["AccessToken"].split(".")[0] + "."

            # cache the token again and make it accessible to the borg pattern
            self._last_token = (token, expiration)
            return token


def main():
    cli = TokingCLI()
    args = cli.parse_args()

    # get the password required for Cognito
    if not args.username:
        toking = Toking(region_name=args.region_name, secret_name=args.secret_name)

    else:
        password = getpass(f"Password for user '{args.username}': ")
        username = args.username

        toking = Toking(
            region_name=args.region_name,
            secret_name=args.secret_name,
            username=username,
            password=password,
        )

    print(toking.get_access_token())


if __name__ == "__main__":
    #main()
    Toking().get_access_token()
    Toking().get_access_token()
