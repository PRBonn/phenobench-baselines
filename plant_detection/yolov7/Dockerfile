ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5+PTX"
##############################################

# # Install dependencies
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 zip htop screen libgl1-mesa-glx git -y

RUN pip install --upgrade pip
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

WORKDIR /src
USER user
