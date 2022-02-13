FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /mnist
COPY . .

RUN apt-get update && apt-get install -y git
RUN git checkout docker
RUN pip install -r requirements.txt