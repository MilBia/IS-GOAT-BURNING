# Define build arguments for version consistency
# IMPORTANT: These versions should be kept in sync with pyproject.toml.
ARG SETUPTOOLS_VERSION=80.9.0
ARG NUMPY_VERSION=2.5.1

# --- Base Stage ---
# Use a specific Ubuntu version and install Python for consistency.
FROM ubuntu:22.04 AS base
COPY --from=denoland/deno:bin-2.5.6 /deno /usr/local/bin/deno

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
RUN /tmp/scripts/install_python.sh python3.14 python3.14-venv libgl1-mesa-glx libglib2.0-0 gosu ffmpeg && \
    /tmp/scripts/setup_runtime.sh && \
    rm -rf /tmp/scripts

# The FFmpeg bundled in the opencv-python wheel does not find the system CA bundle (installed with ca-certificates
# above), so HTTPS captures fail with "certificate verify failed" (#226). Side effect: Python libraries that honour
# SSL_CERT_FILE (e.g. google-genai) use the system bundle instead of certifi.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

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
# Keep in sync with opencv-python in requirements-cpu.txt (major.minor; see CONTRIBUTING.md, Dependabot).
ARG OPENCV_VERSION=5.1.0
ARG OPENCV_CONTRIB_VERSION=5.1.0
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
    python3.14 python3.14-full python3.14-venv gosu \
    libjpeg-turbo8 libpng16-16 libtiff5 libavcodec58 libavformat58 libswscale5 libgtk-3-0 libgl1 && \
    /tmp/scripts/setup_runtime.sh && \
    rm -rf /tmp/scripts

# Copy OpenCV from the builder stage.
COPY --from=gpu_builder /usr/local/lib/python3.14/dist-packages/cv2 /usr/local/lib/python3.14/dist-packages/cv2
COPY --from=gpu_builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=gpu_builder /usr/local/include/opencv5 /usr/local/include/opencv5
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
# The opencl stages use Ubuntu 26.04: the Intel compute runtime (intel-opencl-icd) of 22.04 does not expose recent Intel
# iGPUs such as Alder Lake-N (#223). 26.04 packages python3.14 itself (no deadsnakes PPA; pip bootstrapped from the
# python3-pip-whl wheel, hence PIP_BREAK_SYSTEM_PACKAGES for PEP 668 in this single-purpose image).
FROM ubuntu:26.04 AS opencl_builder

# Expose build arguments to this stage
ARG SETUPTOOLS_VERSION
ARG NUMPY_VERSION

# Set the working directory.
WORKDIR /app

ENV TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Copy scripts with execute permissions and run the centralized OpenCL builder setup.
COPY --chmod=755 scripts/ /tmp/scripts/
RUN /tmp/scripts/setup_builder.sh --type opencl --setuptools-version ${SETUPTOOLS_VERSION} --numpy-version ${NUMPY_VERSION}

# Download and build OpenCV from source.
# Keep in sync with opencv-python in requirements-cpu.txt (major.minor; see CONTRIBUTING.md, Dependabot).
ARG OPENCV_VERSION=5.1.0
ARG OPENCV_CONTRIB_VERSION=5.1.0

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
# This stage is for the OpenCL-accelerated image (Ubuntu 26.04 like its builder, see above; not based on `base`).
# Ships Ubuntu's Intel compute runtime (intel-opencl-icd 26.05.x on amd64) and Mesa's rusticl OpenCL ICD.
FROM ubuntu:26.04 AS opencl
COPY --from=denoland/deno:bin-2.5.6 /deno /usr/local/bin/deno

# Expose the setuptools version argument to this stage
ARG SETUPTOOLS_VERSION

ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    XDG_CACHE_HOME=/app/.cache \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python, the OpenCL runtime and OpenCV's runtime libraries, and set up the runtime.
COPY --chmod=755 scripts/ /tmp/scripts/
RUN /tmp/scripts/install_python.sh python3.14 python3.14-venv libgl1 libglib2.0-0t64 gosu ffmpeg \
    ocl-icd-libopencl1 \
    clinfo \
    $( [ "$TARGETARCH" = "amd64" ] && echo "intel-opencl-icd" ) \
    mesa-opencl-icd \
    libjpeg-turbo8 libpng16-16t64 libtiff6 libwebpdemux2 libavcodec62 libavformat62 libswscale9 libgtk-3-0t64 && \
    /tmp/scripts/setup_runtime.sh && \
    rm -rf /tmp/scripts

# See the base stage (#226).
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

ENTRYPOINT ["entrypoint.sh"]
ENV PYTHONPATH="${PYTHONPATH:-}:/app"
CMD ["python3", "burning_goat_detection.py"]

# Copy OpenCV from the builder stage.
COPY --from=opencl_builder /usr/local/lib/python3.14/dist-packages/cv2 /usr/local/lib/python3.14/dist-packages/cv2
COPY --from=opencl_builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=opencl_builder /usr/local/include/opencv5 /usr/local/include/opencv5
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
COPY tools/ ./tools/
COPY scripts/ ./scripts/
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
COPY tools/ ./tools/
COPY scripts/ ./scripts/
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
COPY tools/ ./tools/
COPY scripts/ ./scripts/
COPY .env.tests ./.env.tests

# Set the default command for this stage to run pytest.
CMD ["pytest"]
