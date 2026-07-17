ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev2

FROM ${BASE_IMAGE}

USER root

ARG WORKDIR="/workspaces/pollenating-demo"
ENV WORKDIR=${WORKDIR}
WORKDIR "${WORKDIR}"

ENV ARENA_PATH=${WORKDIR}/submodules/IsaacLab-Arena
ENV ISAACLAB_PATH=${ARENA_PATH}/submodules/IsaacLab

# Hide conflicting Vulkan files, if needed.
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

# Install apt packages
RUN apt-get update && apt-get install -y \
  git \
  git-lfs \
  cmake \
  ffmpeg \
  sudo \
  jq \
  python3-pip

################################
# Install Isaac Lab
################################
COPY submodules/IsaacLab-Arena/submodules/IsaacLab ${ISAACLAB_PATH}
ENV TERM=xterm
RUN ln -s /isaac-sim/ ${ISAACLAB_PATH}/_isaac_sim
RUN for DIR in ${ISAACLAB_PATH}/source/isaaclab*/; do /isaac-sim/python.sh -m pip install --no-deps -e "$DIR"; done
RUN chmod 777 -R /isaac-sim/kit/
RUN chmod a+x /isaac-sim
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${ISAACLAB_PATH}/source/isaaclab_visualizers
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${ISAACLAB_PATH}/source/isaaclab_teleop
RUN ${ISAACLAB_PATH}/isaaclab.sh -i

# DAQP version required for latest qpsolvers 4.12.0. Backport of IsaacLab #5556.
RUN /isaac-sim/python.sh -m pip install --force-reinstall daqp==0.8.5

# Install Isaac Teleop Python APIs (retargeters, device I/O, OpenXR bindings)
RUN /isaac-sim/python.sh -m pip install isaacteleop~=1.1.0 --extra-index-url https://pypi.nvidia.com

# Install isaaclab_newton's [all] extra so its pinned mujoco / mujoco-warp / newton
# versions take effect.
RUN /isaac-sim/python.sh -m pip install -e "${ISAACLAB_PATH}/source/isaaclab_newton[all]"

RUN /isaac-sim/python.sh -m pip install --upgrade pip
RUN /isaac-sim/python.sh -m pip install av

# cuRobo (motion planner used by isaaclab_mimic's CuroboPlanner) compiles CUDA
# extensions from source, so it needs the CUDA toolkit (nvcc + headers). The Isaac
# Sim base image ships torch built against CUDA 12.8 but no toolkit, so install
# cuda-toolkit-12-8 from NVIDIA's apt repo (mirrors Arena's docker/setup/install_cuda.sh).
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg ca-certificates && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# Point the build (and runtime) at the CUDA toolkit. TORCH_CUDA_ARCH_LIST keeps the
# cuRobo compile fast while staying forward-compatible via PTX JIT.
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
ENV TORCH_CUDA_ARCH_LIST="8.0+PTX"

# Install cuRobo from source at the commit IsaacLab pins/tests against.
# --no-build-isolation so it compiles against the already-installed torch.
RUN /isaac-sim/python.sh -m pip install --no-build-isolation \
    "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8"

ENV LW_API_ENDPOINT="https://api-dev.lightwheel.net"

RUN apt-get update && apt-get install -y pipx && \
    PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin \
    pipx install \
    "huggingface-hub[cli]" \
    pre-commit

RUN (type -p wget >/dev/null || (apt update && apt install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings && \
    out=$(mktemp) && wget -nv -O"$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg && \
    tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null < "$out" && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    mkdir -p -m 755 /etc/apt/sources.list.d && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt update && \
    apt install -y gh

################################
# Install IsaacLab Arena
################################
COPY submodules/IsaacLab-Arena/setup.py ${ARENA_PATH}/
COPY submodules/IsaacLab-Arena/pyproject.toml ${ARENA_PATH}/
COPY submodules/IsaacLab-Arena/isaaclab_arena ${ARENA_PATH}/isaaclab_arena
COPY submodules/IsaacLab-Arena/isaaclab_arena_g1 ${ARENA_PATH}/isaaclab_arena_g1
COPY submodules/IsaacLab-Arena/isaaclab_arena_gr00t ${ARENA_PATH}/isaaclab_arena_gr00t
COPY submodules/IsaacLab-Arena/isaaclab_arena_openpi ${ARENA_PATH}/isaaclab_arena_openpi
COPY submodules/IsaacLab-Arena/isaaclab_arena_environments ${ARENA_PATH}/isaaclab_arena_environments
COPY submodules/IsaacLab-Arena/isaaclab_arena_examples ${ARENA_PATH}/isaaclab_arena_examples

RUN /isaac-sim/python.sh -m pip install -e "${ARENA_PATH}/[dev]"

################################
# Install pollenating-demo package
################################
COPY arena_envs ${WORKDIR}/arena_envs
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/arena_envs"

COPY so101_arena ${WORKDIR}/so101_arena
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/so101_arena"

RUN echo "alias python='/isaac-sim/python.sh'" >> /etc/bash.bashrc
RUN echo "alias pip3='/isaac-sim/python.sh -m pip'" >> /etc/bash.bashrc
RUN echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/bash.bashrc
RUN echo "alias debugpy='python -Xfrozen_modules=off -m debugpy --listen localhost:5678 --wait-for-client'" >> /etc/bash.bashrc
RUN echo "PS1='[pollenating-demo] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '" >> /etc/bash.bashrc
RUN echo "alias ll='ls -alF --color=auto'" >> /etc/bash.bashrc
RUN echo "alias ..='cd ..'" >> /etc/bash.bashrc
RUN cp /etc/bash.bashrc /root/.bashrc

COPY docker/setup/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
