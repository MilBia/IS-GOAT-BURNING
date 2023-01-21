FROM python:3.11

WORKDIR /code
COPY . /code

RUN apt-get update && apt-get install libgl1-mesa-glx  -y
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["python", "./burning_goat_detection.py"]