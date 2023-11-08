FROM python:3.10
WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r ./requirements.txt

COPY . .

CMD ["uvicorn", "--port", "80", "--host", "0.0.0.0", "main:app"]
