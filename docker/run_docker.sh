#!/bin/bash
set -e
DOCKER_IMAGE_NAME='pollenating-demo'
DOCKER_VERSION_TAG='latest'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "$SCRIPT_DIR/.." && pwd )

WORKDIR="/workspaces/pollenating-demo"

DATASETS_HOST_MOUNT_DIRECTORY="$HOME/datasets"
MODELS_HOST_MOUNT_DIRECTORY="$HOME/models"
EVAL_HOST_MOUNT_DIRECTORY="$HOME/eval"
FORCE_REBUILD=false
CONTAINER_SUFFIX=""
CONTAINER_SUFFIX_EXPLICIT=false

while getopts ":d:m:e:hn:rn:Rn:vn:s:" OPTION; do
    case $OPTION in

        d)
            DATASETS_HOST_MOUNT_DIRECTORY=$OPTARG
            ;;
        m)
            MODELS_HOST_MOUNT_DIRECTORY=$OPTARG
            ;;
        e)
            EVAL_HOST_MOUNT_DIRECTORY=$OPTARG
            ;;
        n)
            DOCKER_IMAGE_NAME=${OPTARG}
            ;;
        r)
            FORCE_REBUILD=true
            ;;

        R)
            FORCE_REBUILD=true
            NO_CACHE="--no-cache"
            ;;
        v)
            set -x
            ;;
        s)
            CONTAINER_SUFFIX="-${OPTARG}"
            CONTAINER_SUFFIX_EXPLICIT=true
            ;;
        h)
            script_name=$(basename "$0")
            echo "Helper script to build and run the pollenating-demo Docker environment."
            echo ""
            echo "Usage:"
            echo "$script_name [options]"
            echo ""
            echo "Options:"
            echo "  -v (Verbose output)"
            echo "  -d <datasets directory> (Path to datasets on the host. Default is \"$DATASETS_HOST_MOUNT_DIRECTORY\".)"
            echo "  -m <models directory> (Path to models on the host. Default is \"$MODELS_HOST_MOUNT_DIRECTORY\".)"
            echo "  -e <evaluation directory> (Path to evaluation data on the host. Default is \"$EVAL_HOST_MOUNT_DIRECTORY\".)"
            echo "  -n <docker name> (Name of the docker image that will be built or used. Default is \"$DOCKER_IMAGE_NAME\".)"
            echo "  -r (Force rebuilding of the docker image.)"
            echo "  -R (Force rebuilding of the docker image, without cache.)"
            echo "  -s <suffix> (Suffix appended to the container name, allowing multiple containers to run simultaneously."
            echo "      Defaults to the repo directory name.)"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

if [ "$CONTAINER_SUFFIX_EXPLICIT" = false ]; then
    repo_dir=$(basename "$REPO_ROOT")
    [ -n "$repo_dir" ] && CONTAINER_SUFFIX="-${repo_dir}"
fi

echo "Using Docker image: $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG"

if [ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG 2> /dev/null)" ] && \
    [ "$FORCE_REBUILD" = false ]; then
    echo "Docker image $DOCKER_IMAGE_NAME:$DOCKER_VERSION_TAG already exists. Not rebuilding."
    echo "Use -r option to force the rebuild."
else
    docker build --pull \
        $NO_CACHE \
        --progress=plain \
        --build-arg WORKDIR="${WORKDIR}" \
        -t ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} \
        --file $SCRIPT_DIR/Dockerfile \
        $REPO_ROOT
fi

if [ "$(docker ps -a --quiet --filter status=exited --filter "name=^${DOCKER_IMAGE_NAME}-${DOCKER_VERSION_TAG}${CONTAINER_SUFFIX}$")" ]; then
    docker rm $DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG$CONTAINER_SUFFIX > /dev/null
fi

add_volume_if_it_exists() {
    local src="$1"
    local dst="$2"
    [ -d "$src" ] && echo "-v $src:$dst"
}

SSH_DOCKER_ARGS=()
if [ -S "$SSH_AUTH_SOCK" ]; then
    SSH_DOCKER_ARGS+=("-v" "$SSH_AUTH_SOCK:/ssh-agent")
    SSH_DOCKER_ARGS+=("--env" "SSH_AUTH_SOCK=/ssh-agent")
fi

if [ "$( docker container inspect -f '{{.State.Running}}' $DOCKER_IMAGE_NAME'-'$DOCKER_VERSION_TAG$CONTAINER_SUFFIX 2>/dev/null)" = "true" ]; then
  echo "Container already running. Attaching."
  docker exec -it $DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG$CONTAINER_SUFFIX su $(id -un)
else
    DOCKER_RUN_ARGS=("--name" "$DOCKER_IMAGE_NAME-$DOCKER_VERSION_TAG$CONTAINER_SUFFIX"
                    "--privileged"
                    "--ulimit" "memlock=-1"
                    "--ulimit" "stack=-1"
                    "--ipc=host"
                    "--net=host"
                    "--runtime=nvidia"
                    "--gpus=all"
                    "-v" "${REPO_ROOT}:${WORKDIR}"
                    $(add_volume_if_it_exists $DATASETS_HOST_MOUNT_DIRECTORY /datasets)
                    $(add_volume_if_it_exists $MODELS_HOST_MOUNT_DIRECTORY /models)
                    $(add_volume_if_it_exists $EVAL_HOST_MOUNT_DIRECTORY /eval)
                    "-v" "$HOME/.bash_history:/home/$(id -un)/.bash_history"
                    "-v" "$HOME/.config/osmo:/home/$(id -un)/.config/osmo"
                    "-v" "$HOME/.config/gh:/home/$(id -un)/.config/gh"
                    "-v" "$HOME/.cache:/home/$(id -un)/.cache"
                    # # Persist Isaac Sim shader and CUDA compute caches across container runs.
                    # # Without these, shader recompilation adds several minutes to every cold start.
                    # "-v" "$HOME/.isaac-sim-kit-cache:/isaac-sim/kit/cache"
                    # "-v" "$HOME/.nv:/home/$(id -un)/.nv"
                    "-v" "/tmp:/tmp"
                    "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
                    "-v" "/var/run/docker.sock:/var/run/docker.sock"
                    "-v" "$HOME/.Xauthority:/root/.Xauthority"
                    "${SSH_DOCKER_ARGS[@]}"
                    "-v" "/etc/ssl/certs:/etc/ssl/certs:ro"
                    "--env" "DISPLAY"
                    "--env" "ACCEPT_EULA=Y"
                    "--env" "PRIVACY_CONSENT=Y"
                    "--env" "DOCKER_RUN_USER_ID=$(id -u)"
                    "--env" "DOCKER_RUN_USER_NAME=$(id -un)"
                    "--env" "DOCKER_RUN_GROUP_ID=$(id -g)"
                    "--env" "DOCKER_RUN_GROUP_NAME=$(id -gn)"
                    "-v" "${CXR_HOST_VOLUME_PATH:-$HOME/.cloudxr}:/cloudxr"
                    "--env" "XR_RUNTIME_JSON=/cloudxr/openxr_cloudxr.json"
                    "--env" "NV_CXR_RUNTIME_DIR=/cloudxr/run"
                    "--env" "ISAACLAB_PATH=${WORKDIR}/submodules/IsaacLab-Arena/submodules/IsaacLab"
                    "--env" "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
                    )

    if [ -n "$OMNI_PASS" ]; then
        DOCKER_RUN_ARGS+=("--env" "OMNI_USER=\$omni-api-token")
        DOCKER_RUN_ARGS+=("--env" "OMNI_PASS=$OMNI_PASS")
    else
        if [ -d "$HOME/.nvidia-omniverse" ]; then
            DOCKER_RUN_ARGS+=("-v" "$HOME/.nvidia-omniverse:/home/$(id -un)/.nvidia-omniverse")
        fi
    fi

    if [ -n "$NV_API_KEY" ]; then
        DOCKER_RUN_ARGS+=("--env" "NV_API_KEY")
    fi

    xhost +local:docker > /dev/null 2>&1 || true

    cd "$REPO_ROOT"
    docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm --tty ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG} "${@}"
fi
