user="1000:10"
version="latest"
path=$PWD

while getopts u:v:p: option
do
case "${option}"
in
u) user=${OPTARG};;
v) version=${OPTARG};;
p) path=${OPTARG};;

esac
done
# Nvidia display: https://www.pugetsystems.com/labs/hpc/NVIDIA-Docker2-with-OpenGL-and-X-Display-Output-1527/

alias tamer-make="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/etc/localtime:/etc/localtime:ro \
    --name tamer-$version \
    --user $user \
    --gpus 1 \
    --shm-size 8G \
    -dit \
    -e DISPLAY \
    -e XAUTHORITY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 6886:6886 \
    bpoole908/mlenv-gpu:$version"

alias tamer-attach="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/etc/localtime:/etc/localtime:ro \
    --rm \
    --name tamer-${version}-tmp \
    --user $user \
    --gpus 1 \
    --shm-size 8G \
    -dit \
    -e DISPLAY \
    -e XAUTHORITY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 6886:6886 \
    bpoole908/mlenv-gpu:$version \
    && docker attach tamer-${version}-tmp"