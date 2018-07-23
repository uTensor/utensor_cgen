FROM ubuntu
LABEL maintainer="Dboy Liao <qmalliao@gmail.com>"

ENV UTENSOR_CGEN_BRANCH=develop \
    PIPENV_VENV_IN_PROJECT=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y \
    cmake \
    python3-pip \
    git && \
    cd /root && \
    git clone https://github.com/uTensor/utensor_cgen.git && \
    cd utensor_cgen && \
    git checkout ${UTENSOR_CGEN_BRANCH} && \
    pip3 install -e .[dev]

WORKDIR /root

CMD [ "/bin/bash" ]
