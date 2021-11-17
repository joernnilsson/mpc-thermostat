FROM ubuntu:21.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y vim python3 python3-pip zsh sudo git

# Adapt container to local env
ARG USERNAME
ARG USER_SHELL
ARG REPO_ROOT

RUN sed -i -e "s/\/bin\/bash/"$(echo $USER_SHELL | sed 's/\//\\\//g')"/" /etc/passwd

RUN groupadd $USERNAME -g 1000
RUN useradd $USERNAME -u 1000 -g 1000 -m -s $USER_SHELL

ENV SHELL $USER_SHELL

WORKDIR /ws

RUN usermod -aG sudo $USERNAME

RUN mkdir -p /tmp_dep
COPY modelling/requirements.txt /tmp_dep
RUN cd /tmp_dep && pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y python3-tk

