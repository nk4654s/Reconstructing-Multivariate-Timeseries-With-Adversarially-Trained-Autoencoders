FROM ubuntu:20.04 AS comprisk-base

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update
RUN apt-get install -y curl wget ca-certificates sudo git bzip2 libx11-6
RUN rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# copy code to the docker
COPY packages/Comprisk/ packages/
COPY packages/Dataflux/ packages/
COPY scripts/ scripts/

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda3/bin:$PATH
ARG PATH=/home/user/miniconda3/bin:$PATH

# Install Miniconda and build comprisk
RUN sh scripts/build.sh cpu user
RUN mkdir model

# automatically activate comprisk
RUN echo "source activate comprisk" > ~/.bashrc
ENV PATH=/home/user/miniconda3/envs/comprisk/bin:$PATH

FROM comprisk-base
ARG MACHINE_UUID="0b3426b3-94af-4b8c-89c1-da57493c099a"
ARG TAG="AutoEncoder"
ARG MODEL_VERSION=0

# copy the model into this docker container
WORKDIR /app
COPY environment/production/models/$TAG/$MACHINE_UUID/version_$MODEL_VERSION/ model/

# run the score generation
#RUN comprisk-generate -c model/config.yaml --latest
RUN comprisk-generate -h