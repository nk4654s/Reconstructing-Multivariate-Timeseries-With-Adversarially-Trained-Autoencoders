FROM ubuntu:20.04 AS cas-base

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update
RUN apt-get install -y curl wget ca-certificates sudo git bzip2 libx11-6
RUN rm -rf /var/lib/apt/lists/*

# prepare conda
RUN mkdir /home/.conda
RUN mkdir /home/.cache

# Create a working directory
RUN mkdir /app
WORKDIR /app

# copy code to the docker
COPY packages/Comprisk/ packages/Comprisk/
COPY packages/Dataflux/ packages/Dataflux/
COPY scripts/ scripts/
COPY requirements.yml requirements.yml

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN chown -R user:user /app
RUN chown -R user:user /home/.conda
RUN chown -R user:user /home/.cache
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=true
ENV PATH=/home/user/.miniconda/bin:$PATH
ARG PATH=/home/user/.miniconda/bin:$PATH

# Install Miniconda and build comprisk
RUN sh scripts/setup.sh user cpu
RUN mkdir model

# automatically activate comprisk
RUN echo "source activate cas" > ~/.bashrc
ENV PATH=/home/user/.miniconda/envs/cas/bin:$PATH

FROM cas-base
USER user
WORKDIR /app

ARG MACHINE_UUID="0b3426b3-94af-4b8c-89c1-da57493c099a"
ARG TAG="AutoEncoder"
ARG MODEL_VERSION=0

# copy the model into this docker container
COPY environment/production/models/$TAG/$MACHINE_UUID/version_$MODEL_VERSION/ model/

# run the score generation
ENTRYPOINT flux-pipeline -c model/metadata/dataset_config.yaml --folder upload --latest && comprisk-generator -c model/config.yaml --dataset upload && /bin/bash
#RUN comprisk-generator -h