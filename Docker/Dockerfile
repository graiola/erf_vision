FROM dustynv/ros:noetic-pytorch-l4t-r35.3.1
USER root

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    build-essential \
    make \
    cmake \
    cmake-curses-gui \
    unzip \
    git \
    g++ \ 
    pkg-config \ 
    curl \
    python3 \
    python3-dev \
    python3-testresources \
    python3-pip \
    python3-numpy \
    python3-tk \
    python3-setuptools \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa-dev \
    libglfw3-dev \
    freeglut3-dev \
    mesa-common-dev \
    mesa-utils \
    libcurl4-openssl-dev \
    libssl-dev \
    libusb-1.0-0-dev \
    libgtk-3-dev \
    && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN ln -svf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade --force pip

# Ultralytics guide installation

RUN apt-get install -y libfreetype6-dev
COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt

CMD /bin/bash
