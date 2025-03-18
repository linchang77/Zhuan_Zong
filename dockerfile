# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 设置非交互模式，避免安装软件时要求手动输入
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并安装必要工具
RUN apt-get update && apt-get install -y \
    wget curl git build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------
# 2. 手动下载安装 SUMO 1.15.0
# ---------------------------------

# 下载sumo安装包
WORKDIR /opt
RUN wget https://sumo.dlr.de/releases/1.15.0/sumo-src-1.15.0.tar.gz 
RUN tar -xzf sumo-src-1.15.0.tar.gz  

WORKDIR /opt/sumo-1.15.0
# 安装依赖
RUN apt update && \
    apt install -y \
        build-essential \
        autoconf \
        automake \
        libtool \
        libxerces-c-dev \
        libproj-dev \
        libgdal-dev \
        libgl2ps-dev \
        swig \
        libfox-1.6-dev \
        libxml2-dev \
        python3-dev \
        python3-pip \
        python3-tk \
        wget \
        unzip \
        git\
        cmake
# 编译sumo
RUN rm -rf /var/lib/apt/lists/* && \ 
    rm -rf build&&\
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd /opt && rm -rf /opt/sumo-1.15.0

# ---------------------------------
# 1. 安装 Miniconda（管理 Python 版本）
# ---------------------------------
#WORKDIR /opt
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#    bash miniconda.sh -b -p /opt/miniconda && \
#    rm miniconda.sh

# 配置 Conda 环境变量
#ENV PATH="/opt/miniconda/bin:$PATH"

# 创建 Conda 虚拟环境，并安装 Python 3.11
#RUN conda create -n airfogsim_env python=3.11 -y
# SHELL ["conda", "run", "-n", "airfogsim_env", "/bin/bash", "-c"]

# ---------------------------------
# 3. 复制代码并安装 Python 依赖
# ---------------------------------
    WORKDIR /app
    COPY . /app      
    RUN pip install --no-cache-dir -r requirements.txt

