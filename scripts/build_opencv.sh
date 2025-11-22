#!/bin/bash
set -e

# This script builds and installs OpenCV with specified configuration.
# It handles downloading source, configuring with CMake, building, and installing.

# Default values
OPENCV_VERSION="4.11.0"
OPENCV_CONTRIB_VERSION="4.11.0"
BUILD_TYPE="RELEASE"
WITH_CUDA="OFF"
WITH_OPENCL="OFF"
CUDA_ARCH_BIN=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --opencv-version)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --opencv-version requires a value." >&2
                exit 1
            fi
            OPENCV_VERSION="$2"
            shift 2
            ;;
        --contrib-version)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --contrib-version requires a value." >&2
                exit 1
            fi
            OPENCV_CONTRIB_VERSION="$2"
            shift 2
            ;;
        --cuda)
            WITH_CUDA="ON"
            shift
            ;;
        --opencl)
            WITH_OPENCL="ON"
            shift
            ;;
        --cuda-arch)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --cuda-arch requires a value." >&2
                exit 1
            fi
            CUDA_ARCH_BIN="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1" >&2
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "${WITH_CUDA}" == "ON" && -z "${CUDA_ARCH_BIN}" ]]; then
    echo "Error: --cuda-arch is required when --cuda is specified." >&2
    exit 1
fi

echo "Building OpenCV ${OPENCV_VERSION} (Contrib: ${OPENCV_CONTRIB_VERSION})"
echo "CUDA: ${WITH_CUDA}, OpenCL: ${WITH_OPENCL}, CUDA Arch: ${CUDA_ARCH_BIN}"

# Download and extract OpenCV and Contrib
wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip
unzip -q opencv.zip
mv opencv-${OPENCV_VERSION} opencv
rm opencv.zip

wget -q https://github.com/opencv/opencv_contrib/archive/${OPENCV_CONTRIB_VERSION}.zip -O opencv_contrib.zip
unzip -q opencv_contrib.zip
mv opencv_contrib-${OPENCV_CONTRIB_VERSION} opencv_contrib
rm opencv_contrib.zip

# Create build directory
mkdir -p opencv/build
cd opencv/build

# Determine Python paths
PYTHON_EXEC=$(which python3)
PYTHON_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))")
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

# Configure CMake
CMAKE_ARGS=(
    "-D CMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-D CMAKE_INSTALL_PREFIX=/usr/local"
    "-D WITH_OPENGL=ON"
    "-D WITH_V4L=ON"
    "-D WITH_QT=OFF"
    "-D WITH_TBB=ON"
    "-D BUILD_opencv_python3=ON"
    "-D BUILD_opencv_python2=OFF"
    "-D PYTHON_EXECUTABLE=${PYTHON_EXEC}"
    "-D PYTHON3_LIBRARY=${PYTHON_LIB}"
    "-D PYTHON3_INCLUDE_DIR=${PYTHON_INC}"
    "-D OPENCV_PYTHON3_INSTALL_PATH=${PYTHON_PACKAGES}"
    "-D INSTALL_PYTHON_EXAMPLES=OFF"
    "-D BUILD_EXAMPLES=OFF"
    "-D OPENCV_ENABLE_NONFREE=ON"
    "-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules"
)

if [[ "${WITH_CUDA}" == "ON" ]]; then
    CMAKE_ARGS+=(
        "-D WITH_CUDA=ON"
        "-D CUDA_ARCH_BIN=${CUDA_ARCH_BIN}"
        "-D CUDA_ARCH_PTX="
        "-D WITH_CUDNN=ON"
        "-D OPENCV_DNN_CUDA=ON"
        "-D ENABLE_FAST_MATH=1"
        "-D CUDA_FAST_MATH=1"
    )
else
    CMAKE_ARGS+=("-D WITH_CUDA=OFF")
fi

CMAKE_ARGS+=("-D WITH_OPENCL=${WITH_OPENCL}")

cmake "${CMAKE_ARGS[@]}" ..

# Build and Install
make -j$(nproc)
make install
ldconfig

# Cleanup
cd ../..
rm -rf opencv opencv_contrib
