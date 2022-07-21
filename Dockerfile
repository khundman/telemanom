FROM ubuntu:18.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get install -y git curl zip unzip

WORKDIR /app

RUN cd /app

COPY . /app/telemanom

# TBD: It'd be better to mount this as a volume, on the host, so updates can be made
RUN cd /app/telemanom && \
curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

RUN cd /app/telemanom && \
pip install -r requirements.txt

WORKDIR /app/telemanom

ENTRYPOINT ["python", "example.py"]

LABEL maintainer_dockerfile="haisam.ido@gmail.com"
LABEL maintainer_code=https://github.com/khundman/telemanom

