FROM ubuntu:20.04
COPY mnist /exp/mnist-example
COPY requirements.txt /exp/requirements.txt
COPY tests /exp/tests
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt

RUN mkdir /exp/models
WORKDIR /exp
CMD ["export FLASK_APP=", "./mnist-example/api/hello.py"]
CMD ["export FLASK_APP=", "./mnist-example/api/hello.py"]