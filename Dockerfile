FROM ubuntu:20.04
MAINTAINER Anna Semenova
RUN apt-get upgrade -y
RUN apt-get update -y
COPY . /opt/final_predictor
WORKDIR /opt/final_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 final_app.py
