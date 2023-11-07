FROM nvidia/cuda:11.4.0-base-ubuntu20.04
WORKDIR /app

RUN apt update
RUN apt-get install -y python3 python3-pip

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY . .

CMD ["uvicorn", "--port", "80", "main:app"]
