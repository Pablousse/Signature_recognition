# Start from NVIDIA container matching tensorflow version
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN mkdir /app
WORKDIR /app
COPY . /app

# Update system
RUN apt update
RUN apt upgrade -y

#  Install python 3.6
RUN DEBIAN_FRONTEND="noninteractive" apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.6 python3-pip 

# Install pythons tools you need
RUN python3.6 -m pip install --upgrade setuptools pip distlib

# Install tensorflow and Keras
RUN python3.6 -m pip install -r requirements.txt
# RUN python3.6 -m pip install tensorflow==2.6.0 keras==2.6.0 scipy==1.5.4

# Run your python file
# CMD ["python3.6", "train_model.py"]
# CMD ["python3.6", "predict.py"]
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y