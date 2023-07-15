ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5+PTX"
##############################################

# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# # Install dependencies
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update
# RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
#     xterm xauth openssh-server tmux wget mate-desktop-environment-core

# RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     rm -rf /var/lib/apt/lists/*

# RUN apt-get clean
# RUN rm -rf /var/lib/apt/lists/*
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install --upgrade pip
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /src
