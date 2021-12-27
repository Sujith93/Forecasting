FROM ubuntu:18.04

RUN apt-get update && \
  apt-get install -y software-properties-common vim

RUN apt-get update -y
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv && \
        apt-get install -y git

RUN apt-get update && apt-get install -y python3-pip

RUN apt-get install -y libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev
RUN apt-get install libssl-dev
RUN pip3 install --upgrade pip

RUN apt-get install -y emacs

RUN apt-get install -y wget bzip2

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
RUN conda --version
RUN conda config --set ssl_verify False

WORKDIR /ML

COPY  . .

RUN conda config --append channels conda-forge
RUN conda config --append channels saravji
RUN conda config --append channels anaconda

RUN conda install --yes --file requirements.txt

CMD ["python3","forecast_script.py"]