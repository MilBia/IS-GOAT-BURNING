FROM python:3.13-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean autoclean && \
    apt-get autoremove --yes && \
    pip install --upgrade pip

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

USER nobody:nogroup

CMD ["python", "./burning_goat_detection.py"]