# taken from https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda/
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV FORCE_CUDA=True
ENV TORCH_CUDA_ARCH_LIST=Turing

WORKDIR /workspace

RUN pip install -U opencv-python

# FIXME: update nvidia keys?

RUN apt-get update ; \
  apt-get install -y build-essential \
  git \
  vim

# under your working directory
RUN pip install git+https://github.com/facebookresearch/detectron2.git@v0.6 && \
  pip install git+https://github.com/cocodataset/panopticapi.git && \
  pip install git+https://github.com/mcordts/cityscapesScripts.git

COPY src/Mask2Former/requirements.txt .

RUN pip install -r requirements.txt

COPY src/Mask2Former/mask2former/modeling/pixel_decoder/ops ops
RUN cd ops && sh make.sh

# COPY src/Mask2Former Mask2Former

# RUN cd Mask2Former && \
#   pip install -r requirements.txt && \
#   cd mask2former/modeling/pixel_decoder/ops && \
#   sh make.sh

RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install cog
# RUN curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
# RUN chmod +x /usr/local/bin/cog
RUN export DETECTRON2_DATASETS=/data
WORKDIR /src/Mask2Former
