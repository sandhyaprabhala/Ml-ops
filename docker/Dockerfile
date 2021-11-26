FROM ubuntu:18.04
COPY mnist /exp/mnist
COPY requirements.txt /exp/requirements.txt
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp/mnist/api
CMD ["flask","run","--host=0.0.0.0","-p","8000"]