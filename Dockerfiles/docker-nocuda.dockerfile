FROM ubuntu:20.04

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
COPY . .
RUN sh /app/scripts/clean.sh

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
RUN sh /app/scripts/build.sh

# add folders as volume mount points
RUN mkdir -p models configs logs dataset