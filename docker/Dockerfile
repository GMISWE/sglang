ARG CUDA_VERSION=12.6.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/ \
    NVSHMEM_DIR=/sgl-workspace/nvshmem/install

# Set timezone and install all packages
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
 && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
 && apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common netcat-openbsd kmod unzip openssh-server \
    curl wget lsof zsh ccache tmux htop git-lfs tree \
    python3 python3-pip python3-dev libpython3-dev \
    build-essential cmake \
    libopenmpi-dev libnuma1 libnuma-dev \
    libibverbs-dev libibverbs1 libibumad3 \
    librdmacm1 libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
    ibverbs-providers infiniband-diags perftest \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libunwind-dev \
    libboost-all-dev libssl-dev \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev libcurl4-openssl-dev \
    libczmq4 libczmq-dev \
    libfabric-dev \
    patchelf \
    nvidia-dkms-550 \
    devscripts debhelper fakeroot dkms check libsubunit0 libsubunit-dev \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
 && git clone https://github.com/NVIDIA/gdrcopy.git -b v2.4.4 \
 && cd gdrcopy/packages \
 && CUDA=/usr/local/cuda ./build-deb-packages.sh \
 && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
 && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# Clone and install SGLang
WORKDIR /sgl-workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel html5lib six
RUN mkdir sglang

COPY . sglang/
RUN cd sglang \
 && case "$CUDA_VERSION" in \
      12.6.1) CUINDEX=126 ;; \
      12.8.1) CUINDEX=128 ;; \
      *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
 && python3 -m pip install --no-cache-dir -e "python[${BUILD_TYPE}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
 && if [ "$CUDA_VERSION" = "12.8.1" ]; then \
      python3 -m pip install --no-cache-dir nvidia-nccl-cu12==2.27.5 --force-reinstall --no-deps ; \
      python3 -m pip install --no-cache-dir https://github.com/sgl-project/whl/releases/download/v0.2.5/sgl_kernel-0.2.5+cu128-cp39-abi3-manylinux2014_x86_64.whl --force-reinstall --no-deps ; \
    fi

# Build and install NVSHMEM + DeepEP
RUN wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz \
 && git clone https://github.com/deepseek-ai/DeepEP.git \
 && tar -xf nvshmem_src_3.2.5-1.txz && mv nvshmem_src nvshmem \
 && cd nvshmem \
 && git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch \
 && sed -i '1i#include <unistd.h>' examples/moe_shuffle.cu \
 && rm -f /sgl-workspace/nvshmem_src_3.2.5-1.txz \
 && NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_DIR} -DCMAKE_CUDA_ARCHITECTURES=90 \
 && cmake --build build --target install -j \
 && cd /sgl-workspace/DeepEP \
 && NVSHMEM_DIR=${NVSHMEM_DIR} pip install .

# Python tools
RUN python3 -m pip install --no-cache-dir \
    datamodel_code_generator \
    mooncake_transfer_engine==0.3.4.post2 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core

# Install development tools and utilities
RUN apt-get update && apt-get install -y \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    wget \
    curl \
    locales \
    lsof \
    git \
    git-lfs \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    unzip \
    pkg-config \
    libssl-dev \
    bear \
    ccache \
    less \
    && apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt update -y \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64 /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt update -y \
    && apt install nsight-systems-cli -y

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install minimal Python packages
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pytest \
    black \
    isort \
    icdiff \
    scikit_build_core \
    uv \
    pre-commit \
    pandas \
    matplotlib \
    tabulate

# Install diff-so-fancy
RUN curl -LSso /usr/local/bin/diff-so-fancy https://github.com/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /usr/local/bin/diff-so-fancy

# Install clang-format
RUN curl -LSso /usr/local/bin/clang-format https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /usr/local/bin/clang-format

# Install clangd
RUN curl -L https://github.com/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip -o clangd.zip \
    && unzip clangd.zip \
    && cp -r clangd_18.1.3/bin/* /usr/local/bin/ \
    && cp -r clangd_18.1.3/lib/* /usr/local/lib/ \
    && rm -rf clangd_18.1.3 clangd.zip

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.1/cmake-3.31.1-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.31.1-linux-x86_64.tar.gz \
    && cp -r cmake-3.31.1-linux-x86_64/bin/* /usr/local/bin/ \
    && cp -r cmake-3.31.1-linux-x86_64/share/* /usr/local/share/ \
    && rm -rf cmake-3.31.1-linux-x86_64 cmake-3.31.1-linux-x86_64.tar.gz

# Add yank script
COPY docker/yank.sh /usr/local/bin/yank
RUN chmod +x /usr/local/bin/yank

# Install oh-my-zsh and plugins
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure Vim
COPY docker/.vimrc /root/.vimrc

# Configure tmux
COPY docker/.tmux.conf /root/.tmux.conf

# Configure Git
RUN git config --global core.editor "vim" \
    && git config --global core.whitespace "fix,-indent-with-non-tab,trailing-space,cr-at-eol" \
    && git config --global core.pager "diff-so-fancy | less --tabs=4 -RFX" \
    && git config --global color.ui true \
    && git config --global color."diff-highlight".oldNormal "red bold" \
    && git config --global color."diff-highlight".oldHighlight "red bold 52" \
    && git config --global color."diff-highlight".newNormal "green bold" \
    && git config --global color."diff-highlight".newHighlight "green bold 22" \
    && git config --global color.diff.meta "11" \
    && git config --global color.diff.frag "magenta bold" \
    && git config --global color.diff.commit "yellow bold" \
    && git config --global color.diff.old "red bold" \
    && git config --global color.diff.new "green bold" \
    && git config --global color.diff.whitespace "red reverse" \
    && git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset - %s %Cgreen(%cr) %C(bold blue)<%an>%Creset%C(auto)%d%Creset' --abbrev-commit --" \
    && git config --global http.sslVerify false \
    && git config --global pull.rebase true

# Configure zsh
COPY docker/.zshrc /root/.zshrc

RUN set -euxo ; \
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Set workspace directory
WORKDIR /sgl-workspace/sglang
