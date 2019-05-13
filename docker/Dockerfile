FROM ubuntu:latest

LABEL maintainer="haisam.ido@gmail.com"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get install -y git curl zip unzip

WORKDIR /khundman

RUN cd /khundman

RUN git clone https://github.com/khundman/telemanom.git

# TBD: It'd be better to mount this as a volume, on the host, so updates can be made
RUN cd /khundman/telemanom && \
curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

RUN cd /khundman/telemanom && \
pip install -r requirements.txt

WORKDIR /khundman/telemanom
