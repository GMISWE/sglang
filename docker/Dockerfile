ARG CUDA_VERSION=12.6.1
FROM us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/sglang:v0.4.9.post2-fix-health-9

ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/ \
    NVSHMEM_DIR=/sgl-workspace/nvshmem/install

# Clone and install SGLang
WORKDIR /sgl-workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel html5lib six
RUN mv sglang sglang.old
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

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

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

# Set workspace directory
WORKDIR /sgl-workspace/sglang
