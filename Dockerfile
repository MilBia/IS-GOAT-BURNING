

# --- Base Stage ---
# Use a specific Python version for consistency.
FROM python:3.13-slim-bullseye AS base

# Set environment variables for Python and PIP.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/app/.cache \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory.
WORKDIR /app

# Install common system dependencies.
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libgl1-mesa-glx libglib2.0-0 gosu && \
    apt-get autoremove --yes && \
    apt-get clean && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Copy the entrypoint script and make it executable.
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create the recordings and cache directories and set permissions.
RUN mkdir -p /app/recordings && \
    mkdir -p /app/.cache && \
    chown -R nobody:nogroup /app/.cache

# Set the entrypoint.
ENTRYPOINT ["entrypoint.sh"]

# Set the Python path.
ENV PYTHONPATH="${PYTHONPATH:-}:/app"

# --- CPU Stage ---
# This stage is for the CPU-only image.
FROM base AS cpu

# Copy CPU-specific requirements and install them.
COPY requirements-cpu.txt .
RUN pip install -r requirements-cpu.txt setuptools==75.8.0

# Copy the rest of the application code.
COPY pyproject.toml burning_goat_detection.py ./
COPY is_goat_burning/ ./is_goat_burning/

# Default command to run the application.
CMD ["python3.13", "burning_goat_detection.py"]

# --- GPU Builder Stage ---
# This stage builds OpenCV with CUDA support.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS gpu_builder

# Set the working directory.
WORKDIR /app

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends  \
    software-properties-common &&  \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends  \
    python3.13-full python3.13-dev \
    build-essential cmake git pkg-config libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev \
    libx264-dev libgtk-3-dev wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

# CRITICAL FIX 1: Bootstrap pip and install Python build dependencies BEFORE cmake
# This prevents issues with cmake finding the correct python version
RUN python3.13 -m ensurepip --upgrade --default-pip && \
    python3.13 -m pip install --no-cache-dir --upgrade pip setuptools==75.8.0 numpy

# Download and build OpenCV from source.
ARG OPENCV_VERSION=4.11.0
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip && \
    unzip opencv.zip &&     mv opencv-${OPENCV_VERSION} opencv &&     rm opencv.zip

ARG OPENCV_CONTRIB_VERSION=4.11.0
RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_CONTRIB_VERSION}.zip -O opencv_contrib.zip &&     unzip opencv_contrib.zip &&     mv opencv_contrib-${OPENCV_CONTRIB_VERSION} opencv_contrib &&     rm opencv_contrib.zip

ARG CUDA_ARCH=8.6
RUN mkdir -p /app/opencv/build && \
    cd /app/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_CUDA=ON \
          -D CUDA_ARCH_BIN="${CUDA_ARCH}" \
          -D CUDA_ARCH_PTX="" \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_OPENGL=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_TBB=ON \
          -D BUILD_opencv_python3=ON \
          -D BUILD_opencv_python2=OFF \
          -D PYTHON_EXECUTABLE=$(which python3.13) \
          -D PYTHON3_LIBRARY=$(python3.13 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))") \
          -D PYTHON3_INCLUDE_DIR=$(python3.13 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
          -D OPENCV_PYTHON3_INSTALL_PATH=$(python3.13 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/app/opencv_contrib/modules \
          .. && \
    make -j$(nproc) && \
    make install && \
    apt-get purge -y --auto-remove wget unzip curl && \
    apt-get autoremove --yes && \
    apt-get clean && \
    ldconfig && \
    rm -rf /app/opencv && \
    rm -rf /app/opencv_contrib && \
    rm -rf /var/lib/apt/lists/*

# --- GPU Runtime Stage ---
# This stage is for the GPU-accelerated image.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

# Set the working directory.
WORKDIR /app

ENV TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/app/.cache

# Install runtime dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends  \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends  \
    python3.13-full gosu \
    libjpeg-turbo8 libpng16-16 libtiff5 libavcodec58 libavformat58 libswscale5 libgtk-3-0 libgl1 && \
    apt-get remove -y --autoremove software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy OpenCV from the builder stage.
COPY --from=gpu_builder /usr/local/lib/python3.13/dist-packages/cv2 /usr/local/lib/python3.13/dist-packages/cv2
COPY --from=gpu_builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=gpu_builder /usr/local/include/opencv4 /usr/local/include/opencv4
RUN ldconfig

# Copy GPU-specific requirements and install them.
COPY requirements.txt .
RUN python3.13 -m ensurepip --upgrade --default-pip && \
    python3.13 -m pip install --no-cache-dir -r requirements.txt setuptools==75.8.0


# Copy the entrypoint script and make it executable.
# This script is used to set the correct user permissions and execute the main application.
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create the recordings directory.
RUN mkdir -p /app/recordings && \
    mkdir -p /app/.cache && \
    chown -R nobody:nogroup /app/.cache
# Copy the application code.
# This includes all the python scripts and other resources needed to run the application.
COPY pyproject.toml burning_goat_detection.py ./
COPY is_goat_burning/ ./is_goat_burning/

# Set the entrypoint.
ENTRYPOINT ["entrypoint.sh"]

# Set the Python path.
# This ensures that the application can find its modules.
ENV PYTHONPATH="${PYTHONPATH:-}:/app"

# Default command to run the application.
# This starts the main application script.
CMD ["python3.13", "burning_goat_detection.py"]


# --- Test Stage ---
# This stage is for running the unit and integration test suite.
FROM cpu AS cpu-test

RUN mkdir -p /app/.pytest_cache && \
    chown -R nobody:nogroup /app/.pytest_cache

# Install development dependencies required for testing.
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY tests/ ./tests/
COPY .env.tests ./.env.tests

# Set the default command for this stage to run pytest.
# The PYTHONPATH is inherited from the base stage.
CMD ["pytest"]


# This stage is for running the unit and integration test suite.
FROM gpu AS gpu-test

RUN mkdir -p /app/.pytest_cache && \
    chown -R nobody:nogroup /app/.pytest_cache

# Install development dependencies required for testing.
COPY requirements-dev.txt .
RUN grep -v "^opencv-python" requirements-dev.txt > /tmp/reqs.txt && pip install -r /tmp/reqs.txt

COPY tests/ ./tests/
COPY .env.tests ./.env.tests

# Set the default command for this stage to run pytest.
# The PYTHONPATH is inherited from the base stage.
CMD ["pytest"]
