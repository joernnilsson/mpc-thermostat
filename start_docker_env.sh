IMAGE_NAME=house_mpc
CONTAINER_NAME=$IMAGE_NAME"_container"

docker build \
    --build-arg USERNAME=$USER \
    --build-arg USER_SHELL=$SHELL \
    --build-arg REPO_ROOT=$PROJECT_ROOT \
    . -t $IMAGE_NAME || exit


docker run -it \
    --rm \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --volume="/dev:/dev" \
    --volume="/var/run/docker.sock:/var/run/docker.sock" \
    --volume="/var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket" \
    --volume="/usr/local/bin/sw:/usr/local/bin/sw" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --env ROS_MASTER_URI=http://$(hostname):11311 \
    --env ROS_DISTRO=$ROS_DISTRO \
    -v $HOME/.zshrc:$HOME/.zshrc \
    -v $HOME/.zsh:$HOME/.zsh \
    -v $HOME/.z:$HOME/.z \
    -v $HOME/.fzf.zsh:$HOME/.fzf.zsh \
    -v $HOME/.vim:$HOME/.vim \
    -v $HOME/.vimrc:$HOME/.vimrc \
    -v $HOME/dev/home/fzf:$HOME/dev/home/fzf \
    -v $HOME/.zsh_history:$HOME/.zsh_history \
    -v $HOME/.python_history:$HOME/.python_history \
    -v $HOME/.shared_history:$HOME/.shared_history \
    -v $HOME/.bash_history:$HOME/.bash_history \
    -v $HOME/.baktus:$HOME/.baktus \
    -v /snap/clion:/snap/clion \
    -v $HOME/.config:$HOME/.config \
    -v $HOME/.java:$HOME/.java \
    -v $(pwd):/ws \
    -w /ws \
    --user=$(id -u $USER):$(id -g $USER) \
    --net=host \
    --privileged \
    --name $CONTAINER_NAME \
    --env HOSTNAME=localhost \
    $IMAGE_NAME \
    /bin/zsh

