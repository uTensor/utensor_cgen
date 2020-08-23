FROM ubuntu:latest

LABEL maintainer="uTensor Team <utensor@utensor.ai>"

ENV PIPENV_VENV_IN_PROJECT=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y git \
    python3-dev \
    python3-pip && \
    python3 -m pip install pipenv && \
    mkdir /sciwork-sprint-2020
WORKDIR /sciwork-sprint-2020
RUN git clone --branch=sciwork --checkout https://github.com/uTensor/utensor_cgen.git && \
    git clone --branch=sciworks_tanh --checkout https://github.com/uTensor/uTensor.git && \
    cd utensor_cgen && \
    pipenv install
RUN rm -rf /var/lib/apt/lists/*

CMD [ "/bin/bash" ]