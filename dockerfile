FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

WORKDIR /code

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean autoclean && \
    apt-get autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "./burning_goat_detection.py"]