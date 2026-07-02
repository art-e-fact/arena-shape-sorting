#!/bin/bash

# This script is used as entrypoint for the docker container.
# It will setup an user account for the host user inside the docker
# s.t. created files will have correct ownership.

set -euo pipefail

ldconfig

userdel "$DOCKER_RUN_USER_NAME" 2>/dev/null || true
userdel ubuntu 2>/dev/null || true

groupadd --force --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"

useradd --no-log-init \
        --create-home \
        --uid "$DOCKER_RUN_USER_ID" \
        --gid "$DOCKER_RUN_GROUP_NAME" \
        --groups sudo,isaac-sim \
        --shell /bin/bash \
        $DOCKER_RUN_USER_NAME
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME /home/$DOCKER_RUN_USER_NAME
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME $WORKDIR

echo 'root:root' | chpasswd
echo "$DOCKER_RUN_USER_NAME:root" | chpasswd

echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

touch /home/$DOCKER_RUN_USER_NAME/.sudo_as_admin_successful

cp /etc/bash.bashrc /home/$DOCKER_RUN_USER_NAME/.bashrc
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME /home/$DOCKER_RUN_USER_NAME/.bashrc

mkdir -p /datasets /models /eval
chown $DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME /datasets /models /eval

ISAACLAB_PATH="${WORKDIR}/submodules/IsaacLab-Arena/submodules/IsaacLab"
if [ ! -e "${ISAACLAB_PATH}/_isaac_sim" ]; then
    ln -s /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
fi

[ -f /etc/profile.d/groot_deps.sh ] && set -a && source /etc/profile.d/groot_deps.sh && set +a

if [ $# -ge 1 ]; then
    echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/aliasess.bashrc
    exec sudo --preserve-env -u $DOCKER_RUN_USER_NAME \
        -- env HOME=/home/$DOCKER_RUN_USER_NAME bash -ic "$@"
else
    su $DOCKER_RUN_USER_NAME
fi

exit
