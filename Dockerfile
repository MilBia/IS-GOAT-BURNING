

# --- Base Stage ---
# Use a specific Python version for consistency.
FROM python:3.13-slim-bullseye AS base

# Set environment variables for Python and PIP.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Create the recordings directory.
RUN mkdir -p /app/recordings

# Set the entrypoint.
ENTRYPOINT ["entrypoint.sh"]

# Set the Python path.
ENV PYTHONPATH="${PYTHONPATH:-}:/app"

# --- CPU Stage ---
# This stage is for the CPU-only image.
FROM base AS cpu

# Copy CPU-specific requirements and install them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Default command to run the application.
CMD ["python3", "burning_goat_detection.py"]

# --- GPU Builder Stage ---
# This stage builds OpenCV with CUDA support.
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 AS gpu_builder

# Set the working directory.
WORKDIR /app

# Install system dependencies for building OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev \
    libx264-dev libgtk-3-dev python3-dev python3-pip wget unzip curl \
    python3-numpy && \
    rm -rf /var/lib/apt/lists/*

# Download and build OpenCV from source.
ARG OPENCV_VERSION=4.11.0
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip && \
    unzip opencv.zip && \
    mv opencv-${OPENCV_VERSION} opencv

ARG OPENCV_CONTRIB_VERSION=4.11.0
RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_CONTRIB_VERSION}.zip -O opencv_contrib.zip && \
    unzip opencv_contrib.zip && \
    mv opencv_contrib-${OPENCV_CONTRIB_VERSION} opencv_contrib

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
          -D PYTHON_EXECUTABLE=$(which python3) \
          -D PYTHON3_LIBRARY=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/site-packages \
          -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/app/opencv_contrib/modules \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# --- GPU Runtime Stage ---
# This stage is for the GPU-accelerated image.
FROM gpu_builder AS gpu

# Copy OpenCV from the builder stage.
COPY --from=gpu_builder /usr/local /usr/local

# Copy the application code.
COPY . .

# Install Python dependencies.
RUN pip3 install -r requirements_cuda.txt

# Uninstall conflicting OpenCV packages.
RUN pip3 uninstall -y opencv-python opencv-python-headless || true

# Default command to run the application.
CMD ["python3", "burning_goat_detection.py"]
