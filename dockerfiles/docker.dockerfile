FROM ubuntu:20.04 as cas-base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update
RUN apt-get install -y curl wget ca-certificates sudo git bzip2 libx11-6
RUN rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash nea-cas
RUN chown -R nea-cas:nea-cas /app
RUN echo "nea-cas ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER nea-cas

# All users can use /home/nea-cas as their home directory
ENV HOME=/home/nea-cas
RUN chmod 777 /home/nea-cas

# copy code to the docker
COPY --chown=nea-cas:nea-cas packages/Comprisk/ packages/Comprisk/
COPY --chown=nea-cas:nea-cas packages/Dataflux/ packages/Dataflux/
COPY --chown=nea-cas:nea-cas packages/Toking/ packages/Toking/
COPY --chown=nea-cas:nea-cas scripts/ scripts/
COPY --chown=nea-cas:nea-cas pyproject.toml pyproject.toml
COPY --chown=nea-cas:nea-cas environment/production/models models/

# install uv project
RUN uv sync

FROM cas-base
ENTRYPOINT sh scripts/upload.sh
