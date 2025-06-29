FROM python:3.13-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libgl1-mesa-glx libglib2.0-0 gosu && \
    apt-get autoremove --yes && \
    apt-get clean && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Create the recordings directory. The RUN chown from your original fix is no longer needed here.
RUN mkdir -p /app/recordings

COPY . .

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint to our script
ENTRYPOINT ["entrypoint.sh"]

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python3", "burning_goat_detection.py"]
