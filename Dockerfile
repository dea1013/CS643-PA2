FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ADD model ./model

RUN apt update -y \
&& apt upgrade -y \
&& apt-get install default-jdk -y

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

RUN export JAVA_HOME

ENTRYPOINT ["python3", "app.py"]
