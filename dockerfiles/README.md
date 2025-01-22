# Docker Configuration

The `.env` file holds environment variables that are crucial for the application's functioning, such as AWS access keys or CAS debug mode settings. This file is saved at `dockerfiles/.env` and is typically excluded from version control with a `.gitignore` file to prevent exposure of sensitive data.

To configure Docker build process using this `.env` file, you need to follow these steps:

1. Define all required environment variables in the `.env` file at the root directory of your project as `KEY=VALUE` pairs on separate lines. For instance:
    ```bash
    # .env
    AWS_ACCESS_KEY_ID=your-aws-access-key
    AWS_SECRET_ACCESS_KEY=your-secret-access-key
    CAS_DEBUG_MODE=0 # 1 or a machine's UUID
    ```
2. To build your Docker containers, use this command: 
    ```bash
    docker compose -f dockerfiles/docker-compose.yaml build
    ```

3. After building, you can start them with: 
    ```bash
    docker compose -f dockerfiles/docker-compose.yaml up
    ```
   
The project uses `docker compose` instead of plain docker for its container management tasks. The environment variables defined in the `.env` file are passed to docker's environment variables through the `docker-compose.yaml` file.