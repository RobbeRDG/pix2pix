# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install --no-install-recommends python3.10 python3-pip

# Install pip packages
RUN mkdir /init
COPY ./.devcontainer/requirements.txt /init/requirements.txt
RUN pip -q install pip --upgrade
RUN pip install -r /init/requirements.txt



