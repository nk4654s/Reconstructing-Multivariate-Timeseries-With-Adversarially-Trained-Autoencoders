import os
from typing import Tuple
from getpass import getpass

from Toking.args import TokingCLI
from pycognito.aws_srp import AWSSRP

import boto3
from botocore.exceptions import ClientError

from Toking.resource import get_arg_default


def get_cognito_credentials(region_name: str, secret_name: str) -> Tuple[str, str]:
    """This function read the secret value given a region and secret key name value.

    Args:
        region_name (str): The region name (e.g. 'eu-central-1').
        secret_name (str): The secret's key value.

    Returns:
        Tuple[str, str]: The secret value, usually a username and password.
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    result = eval(get_secret_value_response['SecretString'])
    return result.get("username", None), result.get("password", None)


def get_access_token(secret_name: str = None) -> str:
    """This function gets the access token for IAM portal using the `DEFAULTS` values defined in
    `Tokenier.resource`.
    Args:
        secret_name (str): The secret name.

    Returns:
        str: The token

    """
    secret_name = secret_name or get_arg_default("secret_name")
    username, password = get_cognito_credentials(get_arg_default("region_name"), secret_name)

    aws = AWSSRP(
        username=username,
        password=password,
        pool_id=get_arg_default("pool_id"),
        client_id=get_arg_default("client_id")
    )

    result = aws.authenticate_user()
    if result:
        token = result['AuthenticationResult']['AccessToken'].split(".")[0] + "."
        return token


def main():
    cli = TokingCLI()
    args = cli.parse_args()

    # get the password required for Cognito
    if not args.username:
        credentials = get_cognito_credentials(
            region_name=args.region_name,
            secret_name=args.secret_name)

        username, password = credentials

    else:
        password = getpass(f"Password for user '{args.username}': ")
        username = args.username

    aws = AWSSRP(
        username=username,
        password=password,
        pool_id=args.pool_id,
        client_id=args.client_id,
    )

    result = aws.authenticate_user()
    if result:
        token = result['AuthenticationResult']['AccessToken'].split(".")[0] + "."
        os.system(f"export DB_SECRET={token}")
        print(token)


if __name__ == "__main__":
    main()
