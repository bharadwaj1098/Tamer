# user="1000:10"
# version="torch"
# path=$PWD

# while getopts u:v:p: option
# do
# case "${option}"
# in
# u) user=${OPTARG};;
# v) version=${OPTARG};;
# p) path=${OPTARG};;

# esac
# done
# # Nvidia display: https://www.pugetsystems.com/labs/hpc/NVIDIA-Docker2-with-OpenGL-and-X-Display-Output-1527/

# alias tamer-make="docker run \
#     --volume=$path:/home/dev/mnt \
#     --volume=/tmp/.X11-unix:/tmp/.X11-unix \
#     --volume=/etc/localtime:/etc/localtime:ro \
#     --name tamer-$version \
#     --user $user \
#     --gpus 1 \
#     --shm-size 8G \
#     -dit \
#     -e DISPLAY \
#     -e XAUTHORITY \
#     -e NVIDIA_DRIVER_CAPABILITIES=all \
#     -p 8888:8888 \
#     -p 6006:6006 \
#     -p 6886:6886 \
#     bpoole908/mlenv-gpu:$version"

# alias tamer-attach="docker run \
#     --volume=$path:/home/dev/mnt \
#     --volume=/tmp/.X11-unix:/tmp/.X11-unix \
#     --volume=/etc/localtime:/etc/localtime:ro \
#     --rm \
#     --name tamer-${version}-tmp \
#     --user $user \
#     --gpus all \
#     --shm-size 8G \
#     -dit \
#     -e DISPLAY \
#     -e XAUTHORITY \
#     -e NVIDIA_DRIVER_CAPABILITIES=all \
#     -p 8888:8888 \
#     -p 6006:6006 \
#     -p 6886:6886 \
#     bpoole908/mlenv-gpu:$version \
#     && docker attach tamer-${version}-tmp"


#!/bin/bash
xhost +

XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
  xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
  if [ ! -z "$xauth_list" ]; then
    echo $xauth_list | xauth -f $XAUTH nmerge -
  else
    touch $XAUTH
  fi
  chmod a+r $XAUTH
fi

# ./docker_clean.bash

no_gui=$1

# export REGISTRY_SRC_IMAGE=registry.gitlab.com/ghostusers/ghost_gazebo
export REGISTRY_SRC_IMAGE=bpoole908/mlenv-gpu
#docker pull ${REGISTRY_SRC_IMAGE}:release

# updated as of 9/15/20
# docker pull ${REGISTRY_SRC_IMAGE}:release_v3
docker pull ${REGISTRY_SRC_IMAGE}:torch
docker run -t -d --name="mlenv" \
  --gpus all \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --env="XAUTHORITY=$XAUTH" \
  --network host \
  --privileged \
  ${REGISTRY_SRC_IMAGE}:torch

sleep 2

docker exec -it mlenv bash
#\
#   sh -c "/run_sim.sh $no_gui"
