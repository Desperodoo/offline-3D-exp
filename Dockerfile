# EPIC (Exploring on Point Clouds) Docker Image
# 一个支持LiDAR自主无人机探索的完整容器环境
# 包含ROS Noetic, C++17, Python3.9(Miniconda), PyTorch, NVIDIA GPU支持，并实现网络隔离

FROM ubuntu:20.04

LABEL maintainer="EPIC Development Team"
LABEL description="EPIC: A Lightweight LiDAR-Based AAV Exploration Framework for Large-Scale Scenarios"

# 配置时区和环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
# ENV HTTP_PROXY=http://172.18.196.129:7890
# ENV HTTPS_PROXY=http://172.18.196.129:7890
# ENV ALL_PROXY=http://172.18.196.129:7890
# ENV http_proxy=http://172.18.196.129:7890
# ENV https_proxy=http://172.18.196.129:7890
# ENV all_proxy=http://172.18.196.129:7890

# 设置系统环境变量
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Miniconda相关环境变量
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 配置APT镜像源使用阿里云镜像
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list

# 更新系统并安装基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    vim \
    nano \
    build-essential \
    software-properties-common \
    lsb-release \
    gnupg2 \
    ca-certificates \
    tzdata \
    locales \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libglu1-mesa \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装Miniconda
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=${CONDA_DIR}/bin:\${PATH}" >> ~/.bashrc

# 配置conda镜像源
RUN ${CONDA_DIR}/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    ${CONDA_DIR}/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    ${CONDA_DIR}/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    ${CONDA_DIR}/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    ${CONDA_DIR}/bin/conda config --set show_channel_urls yes

# 创建Python 3.9虚拟环境
RUN ${CONDA_DIR}/bin/conda create -n epic python=3.9 -y && \
    ${CONDA_DIR}/bin/conda clean -a -y

# 激活环境并设置为默认
ENV CONDA_DEFAULT_ENV=epic
ENV CONDA_PREFIX=${CONDA_DIR}/envs/epic
ENV PATH=${CONDA_PREFIX}/bin:${PATH}
RUN echo "conda activate epic" >> ~/.bashrc

# 添加ROS Noetic APT仓库（使用中科大镜像源）
RUN sh -c 'echo "deb http://mirrors.ustc.edu.cn/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# 更新package列表
RUN apt-get update

# 分步安装ROS Noetic，提高成功率
RUN apt-get install -y --no-install-recommends \
    ros-noetic-ros-core \
    ros-noetic-ros-base \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ros-noetic-catkin \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# 初始化rosdep
RUN rosdep init || true
RUN rosdep update || true

# 安装C++开发依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-9 \
    g++-9 \
    cmake \
    make \
    libeigen3-dev \
    libpcl-dev \
    libopencv-dev \
    libomp-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置C++17为默认编译器
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# 在conda环境中安装PyTorch及相关包
RUN ${CONDA_DIR}/envs/epic/bin/pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 在conda环境中安装PyTorch Geometric
RUN ${CONDA_DIR}/envs/epic/bin/pip install torch_geometric

# 在conda环境中安装PyTorch Geometric可选依赖
RUN ${CONDA_DIR}/envs/epic/bin/pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# 在conda环境中使用pip安装Python机器学习依赖
RUN ${CONDA_DIR}/envs/epic/bin/pip install \
    numpy==1.26.4 \
    scipy==1.11.4 \
    matplotlib==3.9.2 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    networkx==3.2.1 \
    pillow==10.4.0 \
    requests==2.32.3 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    seaborn==0.13.2 \
    opencv-python==4.10.0.84

# 使用pip在conda环境中安装额外的Python包
RUN ${CONDA_DIR}/envs/epic/bin/pip install \
    plotly==5.23.0 \
    jupyterlab==4.2.2 \
    lxml==5.2.2 \
    beautifulsoup4==4.12.3 \
    Flask==3.0.3 \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    websockets==12.0 \
    aiofiles==23.2.1 \
    httpx==0.27.0 \
    tensorboard==2.17.0 \
    wandb==0.17.3 \
    einops==0.8.0 \
    transformers==4.43.1 \
    datasets==2.20.0 \
    accelerate==0.32.1 \
    lightning==2.3.3 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    rich==13.7.1 \
    typer==0.12.3 \
    click==8.1.7

# 安装ROS相关Python包到conda环境
RUN ${CONDA_DIR}/envs/epic/bin/pip install \
    rospkg \
    catkin_pkg \
    empy \
    defusedxml

# 创建catkin工作空间
RUN mkdir -p /root/catkin_ws/src

# 设置ROS环境和网络配置（使用网络隔离配置）
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash || true" >> /root/.bashrc && \
    echo "conda activate epic" >> /root/.bashrc && \
    echo "export ROS_HOSTNAME=172.20.0.10" >> /root/.bashrc && \
    echo "export ROS_MASTER_URI=http://172.20.0.10:11311" >> /root/.bashrc && \
    echo "export ROS_IP=172.20.0.10" >> /root/.bashrc

# 创建EPIC项目目录
WORKDIR /root/catkin_ws

# 复制源代码
COPY src/ /root/catkin_ws/src/

# 创建必要的目录结构
RUN mkdir -p /root/catkin_ws/src/MARSIM/map_generator/resource && \
    mkdir -p /root/catkin_ws/build && \
    mkdir -p /root/catkin_ws/devel && \
    mkdir -p /root/catkin_ws/logs && \
    mkdir -p /root/catkin_ws/collected_data && \
    mkdir -p /root/catkin_ws/datasets

# 安装项目特定的ROS依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-cv-bridge \
    ros-noetic-message-filters \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-geometry-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-rviz \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-ackermann-msgs \
    ros-noetic-geometry2 \
    ros-noetic-hector-trajectory-server \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    ros-noetic-joy \
    ros-noetic-octomap-msgs \
    ros-noetic-ompl \
    ros-noetic-control-toolbox \
    ros-noetic-realtime-tools \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-slam-karto \
    ros-noetic-slam-gmapping \
    ros-noetic-robot-localization \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-effort-controllers \
    ros-noetic-position-controllers \
    ros-noetic-joint-trajectory-controller \
    ros-noetic-moveit \
    ros-noetic-moveit-commander \
    ros-noetic-moveit-planners-ompl \
    ros-noetic-industrial-trajectory-filters \
    ros-noetic-teb-local-planner \
    ros-noetic-base-local-planner \
    ros-noetic-move-base-msgs \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# 修复empy版本兼容性问题
RUN ${CONDA_DIR}/envs/epic/bin/pip install --upgrade empy==3.3.4

# 设置CUDA环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置OpenGL环境变量（支持GPU渲染）
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

# 初始化conda以支持conda activate命令
RUN ${CONDA_DIR}/bin/conda init bash

# 创建entrypoint脚本（模拟宿主机运行流程）
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '# 初始化conda' >> /entrypoint.sh && \
    echo 'source ~/.bashrc' >> /entrypoint.sh && \
    echo '# 先source ROS环境' >> /entrypoint.sh && \
    echo 'source /opt/ros/noetic/setup.bash' >> /entrypoint.sh && \
    echo 'source /root/catkin_ws/devel/setup.bash || true' >> /entrypoint.sh && \
    echo '# 然后激活conda环境' >> /entrypoint.sh && \
    echo 'conda activate epic' >> /entrypoint.sh && \
    echo '# 设置网络配置' >> /entrypoint.sh && \
    echo 'export ROS_HOSTNAME=172.20.0.10' >> /entrypoint.sh && \
    echo 'export ROS_MASTER_URI=http://172.20.0.10:11311' >> /entrypoint.sh && \
    echo 'export ROS_IP=172.20.0.10' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# 设置工作目录
WORKDIR /root/catkin_ws

# 暴露ROS端口和其他服务端口
EXPOSE 11311 8080 8888 5000

# 清理
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    ${CONDA_DIR}/bin/conda clean -a -y

# 设置入口点
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# 最终设置信息
RUN echo "EPIC Docker环境构建完成！" && \
    echo "包含以下主要组件：" && \
    echo "- ROS Noetic" && \
    echo "- C++17开发环境" && \
    echo "- Miniconda Python 3.9 虚拟环境" && \
    echo "- PyTorch + PyTorch Geometric (CUDA支持)" && \
    echo "- PCL, OpenCV, Eigen3等C++库" && \
    echo "- NVIDIA GPU支持" && \
    echo "- EPIC探索框架" && \
    echo "- 网络隔离配置 (172.20.0.10)"
