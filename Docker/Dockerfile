FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# ENVIRONMENT STUFF FOR CUDA
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDA_ROOT /usr/local/cuda

# recommended from nvidia to use the cuda devices
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y vim \
    apt-utils \
    build-essential \
    psmisc \
    vim-gtk \
    tmux \
    git \
    wget \
    lsb-release \
    lsb-core \
    cmake \
    curl \
    ninja-build \
    libboost-all-dev \
    libtbb-dev \
    gfortran  && apt autoremove --purge -y && \
               apt clean -y && \
               rm -rf /var/lib/apt/lists/*

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

FROM python:3.7

LABEL maintainer="bharadwaj1098 <bharadwaj.reddy98@gmail.com>"

# Working directory is / by default. We explictly state it here for posterity
# WORKDIR /
#install python and upgrade pip
# RUN apt-get install -y python3-pip

# Move the requirements file into the image
COPY requirements.txt /tmp/

# Install the python requirements on the image
RUN pip3 install --trusted-host pypi.python.org -r /tmp/requirements.txt

# Remove the requirements file - this is no longer needed
RUN rm /tmp/requirements.txt

COPY ROMS /tmp/
RUN python3 -m atari_py.import_roms /tmp/ROMS/
RUN rm -rf /tmp/ROMS

# Create a directory in which we can do our work
# RUN mkdir /home

# Set it as the working directory
WORKDIR /

# Copy over the start-up script
ADD startup_script.sh /usr/local/bin/startup_script.sh

# Give permissions to execute
RUN chmod 777 /usr/local/bin/startup_script.sh

# Set the display when we run the container. This allows us to record without the user needing to type anything explicitly
# This code snippet was taken from https://github.com/duckietown/gym-duckietown/issues/123
ENTRYPOINT ["/usr/local/bin/startup_script.sh"]

