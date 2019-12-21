FROM debian

RUN apt-get update
RUN apt-get install -y python3 \
                       python3-pip

RUN pip3 install numpy \
                 pandas \
                 matplotlib

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY pd1.py .

CMD [ "python3", "pd1.py" ]
