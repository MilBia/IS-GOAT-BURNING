# Define build arguments for version consistency
# IMPORTANT: These versions should be kept in sync with pyproject.toml.
ARG SETUPTOOLS_VERSION=80.9.0
ARG NUMPY_VERSION=2.3.3

# --- Base Stage ---
# Use a specific Ubuntu version and install Python for consistency.
FROM ubuntu:22.04 AS base

# Set environment variables for Python and PIP.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/app/.cache \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory early.
WORKDIR /app

# Copy scripts with execute permissions, install Python, set up the runtime,
# and clean up all in a single layer to optimize image size.
COPY --chmod=755 scripts/ /tmp/scripts/
RUN /tmp/scripts/install_python.sh python3.13 python3.13-venv libgl1-mesa-glx libglib2.0-0 gosu ffmpeg && \
    /tmp/scripts/setup_runtime.sh && \
    rm -rf /tmp/scripts

# Set the entrypoint.
ENTRYPOINT ["entrypoint.sh"]

# Set the Python path.
ENV PYTHONPATH="${PYTHONPATH:-}:/app"

# Default command to run the application.
CMD ["python3", "burning_goat_detection.py"]

# --- CPU Stage ---
# This stage is for the CPU-only image.
FROM base AS cpu

# Expose the setuptools version argument to this stage
ARG SETUPTOOLS_VERSION

# Copy CPU-specific requirements and install them.
COPY requirements-cpu.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-cpu.txt setuptools==${SETUPTOOLS_VERSION}

# Copy the rest of the application code.
COPY pyproject.toml burning_goat_detection.py ./
COPY is_goat_burning/ ./is_goat_burning/

# --- GPU Builder Stage ---
# This stage builds OpenCV with CUDA support.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS gpu_builder

# Expose build arguments to this stage
ARG SETUPTOOLS_VERSION
ARG NUMPY_VERSION

# Set the working directory.
WORKDIR /app

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Copy scripts with execute permissions and run the centralized GPU builder setup.
COPY --chmod=755 scripts/ /tmp/scripts/
RUN /tmp/scripts/setup_builder.sh --type gpu --setuptools-version ${SETUPTOOLS_VERSION} --numpy-version ${NUMPY_VERSION}

# Download and build OpenCV from source.
ARG OPENCV_VERSION=4.11.0
ARG OPENCV_CONTRIB_VERSION=4.11.0
ARG CUDA_ARCH=8.6

RUN /tmp/scripts/build_opencv.sh \
    --opencv-version ${OPENCV_VERSION} \
    --contrib-version ${OPENCV_CONTRIB_VERSION} \
    --cuda \
    --cuda-arch ${CUDA_ARCH} && \
    apt-get purge -y --auto-remove \
    build-essential cmake git pkg-config wget unzip curl \
    software-properties-common gnupg \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev \
    libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev && \
    apt-get clean && \
    rm -rf /tmp/scripts && \
    rm -rf /var/lib/apt/lists/*

# --- GPU Runtime Stage ---
# This stage is for the GPU-accelerated image.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

# Expose the setuptools version argument to this stage
ARG SETUPTOOLS_VERSION

# Set the working directory.
WORKDIR /app

ENV TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/app/.cache

# Copy scripts with execute permissions, install Python, set up the runtime,
# and clean up all in a single layer to optimize image size.
COPY --chmod=755 scripts/ /tmp/scripts/
RUN /tmp/scripts/install_python.sh \
    python3.13 python3.13-full python3.13-venv gosu \
    libjpeg-turbo8 libpng16-16 libtiff5 libavcodec58 libavformat58 libswscale5 libgtk-3-0 libgl1 && \
    /tmp/scripts/setup_runtime.sh && \
    rm -rf /tmp/scripts

# Copy OpenCV from the builder stage.
COPY --from=gpu_builder /usr/local/lib/python3.13/dist-packages/cv2 /usr/local/lib/python3.13/dist-packages/cv2
COPY --from=gpu_builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=gpu_builder /usr/local/include/opencv4 /usr/local/include/opencv4
RUN ldconfig

# Copy GPU-specific requirements and install them.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt setuptools==${SETUPTOOLS_VERSION}

# Copy the application code.
COPY pyproject.toml burning_goat_detection.py ./
COPY is_goat_burning/ ./is_goat_burning/

# Set the entrypoint.
ENTRYPOINT ["entrypoint.sh"]

# Set the Python path.
ENV PYTHONPATH="${PYTHONPATH:-}:/app"

# Default command to run the application.
CMD ["python3", "burning_goat_detection.py"]


# --- OpenCL Builder Stage ---
# This stage builds OpenCV with OpenCL support.
FROM ubuntu:22.04 AS opencl_builder

# Expose build arguments to this stage
ARG SETUPTOOLS_VERSION
ARG NUMPY_VERSION

# Set the working directory.
WORKDIR /app

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Copy scripts with execute permissions and run the centralized OpenCL builder setup.
COPY --chmod=755 scripts/ /tmp/scripts/
RUN /tmp/scripts/setup_builder.sh --type opencl --setuptools-version ${SETUPTOOLS_VERSION} --numpy-version ${NUMPY_VERSION}

# Download and build OpenCV from source.
ARG OPENCV_VERSION=4.11.0
ARG OPENCV_CONTRIB_VERSION=4.11.0

RUN /tmp/scripts/build_opencv.sh \
    --opencv-version ${OPENCV_VERSION} \
    --contrib-version ${OPENCV_CONTRIB_VERSION} \
    --opencl && \
    apt-get purge -y --auto-remove \
    build-essential cmake git pkg-config wget unzip \
    software-properties-common gnupg \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev \
    libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev && \
    apt-get clean && \
    rm -rf /tmp/scripts && \
    rm -rf /var/lib/apt/lists/*

# --- OpenCL Runtime Stage ---
# This stage is for the OpenCL-accelerated image.
FROM base AS opencl

# Expose the setuptools version argument to this stage
ARG SETUPTOOLS_VERSION

ARG TARGETARCH

# Install OpenCL runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 \
    clinfo \
    $( [ "$TARGETARCH" = "amd64" ] && echo "intel-opencl-icd" ) \
    mesa-opencl-icd \
    libjpeg-turbo8 libpng16-16 libtiff5 libavcodec58 libavformat58 libswscale5 libgtk-3-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy OpenCV from the builder stage.
COPY --from=opencl_builder /usr/local/lib/python3.13/dist-packages/cv2 /usr/local/lib/python3.13/dist-packages/cv2
COPY --from=opencl_builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=opencl_builder /usr/local/include/opencv4 /usr/local/include/opencv4
RUN ldconfig

# Copy CPU-specific requirements (OpenCL uses standard pip packages usually)
COPY requirements-cpu.txt .
# Exclude opencv-python* from requirements and install the rest
RUN grep -v "^opencv-python" requirements-cpu.txt | python3 -m pip install --no-cache-dir -r /dev/stdin setuptools==${SETUPTOOLS_VERSION}

# Copy the application code.
COPY pyproject.toml burning_goat_detection.py ./
COPY is_goat_burning/ ./is_goat_burning/


# --- Test Stage ---
# This stage is for running the unit and integration test suite.
FROM cpu AS cpu-test

RUN mkdir -p /app/.pytest_cache && \
    chown -R nobody:nogroup /app/.pytest_cache

# Install development dependencies required for testing.
COPY requirements-dev.txt .
# Exclude opencv-python from dev requirements and pipe the rest directly into pip.
RUN grep -v "^opencv-python" requirements-dev.txt | python3 -m pip install --no-cache-dir -r /dev/stdin setuptools==${SETUPTOOLS_VERSION}

# Copy test suite and configuration AFTER installing dependencies for better caching
COPY tests/ ./tests/
COPY .env.tests ./.env.tests

# Set the default command for this stage to run pytest.
CMD ["pytest"]


# --- GPU Test Stage ---
# This stage is for running the unit and integration test suite on a GPU.
FROM gpu AS gpu-test

RUN mkdir -p /app/.pytest_cache && \
    chown -R nobody:nogroup /app/.pytest_cache

# Install development dependencies required for testing.
COPY requirements-dev.txt .
# Exclude opencv-python from dev requirements and pipe the rest directly into pip.
RUN grep -v "^opencv-python" requirements-dev.txt | python3 -m pip install --no-cache-dir -r /dev/stdin setuptools==${SETUPTOOLS_VERSION}

# Copy test suite and configuration AFTER installing dependencies for better caching
COPY tests/ ./tests/
COPY .env.tests ./.env.tests

# Set the default command for this stage to run pytest.
CMD ["pytest"]


# --- OpenCL Test Stage ---
# This stage is for running the unit and integration test suite with OpenCL.
FROM opencl AS opencl-test

RUN mkdir -p /app/.pytest_cache && \
    chown -R nobody:nogroup /app/.pytest_cache

# Install development dependencies required for testing.
COPY requirements-dev.txt .
# Exclude opencv-python from dev requirements and pipe the rest directly into pip.
RUN grep -v "^opencv-python" requirements-dev.txt | python3 -m pip install --no-cache-dir -r /dev/stdin setuptools==${SETUPTOOLS_VERSION}

# Copy test suite and configuration AFTER installing dependencies for better caching
COPY tests/ ./tests/
COPY .env.tests ./.env.tests

# Set the default command for this stage to run pytest.
CMD ["pytest"]
