From registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.1-kuae1.3.0-s4000-py38

ENV MTHREADS_VISIBLE_DEVICES=all
ENV MTHREADS_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# set conda env
ENV CONDA_ENV_NAME=py38
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH
ENV PATH=/opt/conda/bin:$PATH
# Optional check (debug):
RUN which python && python --version

# set vllm_mtt env
ENV PYTHONPATH=/home/workspace/vllm_mtt
# set musa env
ENV MUSA_INSTALL_PATH=/usr/local/musa
ENV PATH=$MUSA_INSTALL_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$MUSA_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# 安装基础系统工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 JupyterHub 和 Notebook 到 py38 环境
RUN conda install -y jupyterhub notebook
# RUN conda clean --all --yes

# 设置默认工作目录
WORKDIR /srv/jupyterhub

# 安装常用 Python 库
RUN pip install --no-cache-dir \
   -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy matplotlib pandas jieba scikit-learn tqdm

# 暴露 JupyterHub 默认端口
EXPOSE 8000

COPY keep_alive.sh /workspace/keep_alive.sh
RUN chmod +x /workspace/keep_alive.sh
ENTRYPOINT ["/workspace/keep_alive.sh"]
