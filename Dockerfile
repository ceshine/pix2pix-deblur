FROM ceshine/cuda-pytorch:0.2.0

MAINTAINER CeShine <ceshine@ceshine.net>

COPY . /home/docker

RUN pip install tqdm==4.15.0

CMD /bin/bash
