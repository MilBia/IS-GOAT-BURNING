# Is the Gävle Goat Burning?

This project monitors the [Gävle Goat webcam](https://youtu.be/vDFPpkp9krY) feed to detect if the goat is on fire. It uses computer vision techniques to analyze the video feed and sends email and/or Discord notifications if fire is detected.

## Project Overview

The Gävle Goat is a giant straw goat built annually in Gävle, Sweden. It has become a tradition for vandals to attempt to burn it down. This project aims to provide real-time monitoring of the goat's status.

## Features

- **Real-time Fire Detection:** Utilizes computer vision to analyze the live webcam feed for signs of fire.
- **Email Notifications:** Sends email alerts immediately upon detecting fire, notifying designated recipients.
- **Discord Notifications:** Sends Discord alerts immediately upon detecting fire, notifying designated recipients.
- **Configurable Monitoring:** Allows adjustment of monitoring frequency to control resource usage and sensitivity.
- **Dockerized Deployment:** Easily deployable with Docker, ensuring consistent performance across different environments.
- **CUDA Acceleration (Optional):** Supports CUDA for significantly faster processing on NVIDIA GPUs.

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA drivers installed (if using CUDA acceleration)
- Docker (optional, for Docker deployment)

## INSTALLATION

1.  Clone the repository:
    ```bash
    git clone https://github.com/MilBia/IS-GOAT-BURNING.git
    cd IS-GOAT-BURNING
    ```

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install dependencies:
    *   For a CPU-based installation:
        ```bash
        pip install .[cpu]
        ```
    *   For a GPU-based installation (requires a manual build of OpenCV with CUDA, see Docker instructions):
        ```bash
        pip install .
        ```

## Dependency Management

This project uses `pyproject.toml` as the single source of truth for all Python dependencies. The `requirements.txt`, `requirements-cpu.txt`, and `requirements-dev.txt` files are auto-generated from this file using `pip-tools`.

**Important: Do not edit the `requirements*.txt` files manually.**

If you need to add or change a dependency, edit `pyproject.toml` and then run the following commands from the root of the project to regenerate the files:

```bash
pip-compile --resolver=backtracking -o requirements.txt pyproject.toml
pip-compile --resolver=backtracking --extra=cpu -o requirements-cpu.txt pyproject.toml
pip-compile --resolver=backtracking --extra=dev,cpu -o requirements-dev.txt pyproject.toml
```

## CONFIGURATION

1.  Create a `.env` file by copying the example:

    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file and fill in your configuration details.  Here's a breakdown of each setting:

    ```
    SOURCE="https://youtu.be/vDFPpkp9krY"                            # URL of the webcam feed
    USE_EMAILS=true                                                  # Set to true if you want email notifications
    SENDER="your_email@example.com"                                  # Your email address
    SENDER_PASSWORD="your_email_password"                            # Your email password or an app password for Gmail
    RECIPIENTS="recipient1@example.com,recipient2@example.com"       # Comma-separated list of email addresses to notify
    EMAIL_HOST="smtp.gmail.com"                                      # Your email host (e.g., smtp.gmail.com)
    EMAIL_PORT=587                                                   # Your email port (e.g., 587 for Gmail)
    USE_DISCORD=true                                                 # Set to true if you want Discord notifications
    DISCORD_HOOKS="/webhooks/webhooks/{webhook.id}/{webhook.token}"  # Your discord webhook URL.  See Discord documentation for how to create one.
    LOGGING=true                                                     # Enable or disable logging
    VIDEO_OUTPUT=true                                                # Display detected video frames (true) or not (false)
    CHECKS_PER_SECOND=1.0                                            # How many times to check per second (adjust for performance)
    OPEN_CL=false                                                    # Enable or disable use of OpenCL for faster processing (experimental)
    CUDA=false                                                       # Enable or disable use of CUDA for faster processing
    SAVE_VIDEO_CHUNKS=false                                          # Set to true to save video chunks to disk
    VIDEO_OUTPUT_DIRECTORY="./recordings"                            # Directory to save the video files
    VIDEO_CHUNK_LENGTH_SECONDS=300                                   # Length of each video chunk in seconds (e.g., 300 = 5 minutes)
    MAX_VIDEO_CHUNKS=20                                              # Maximum number of video chunks to keep on disk. Set to 0 or less to disable.
    CHUNKS_TO_KEEP_AFTER_FIRE=10                                     # Number of additional chunks to save AFTER a fire is first detected.
    ```

**Important**:

-   **Email Configuration Notes:**
    -   For Gmail, you *must* generate an "App Password" in your Google account settings (Security -> App Passwords) and use that instead of your regular password.  Enable "Less secure app access" is usually *not* sufficient anymore and is a security risk.
    -   For other email providers, consult their documentation for the correct `EMAIL_HOST` and `EMAIL_PORT` settings.
-   OpenCL (`OPEN_CL=true`) for faster processing is experimental and requires `VIDEO_OUTPUT=false`.
-   When using Docker, setting `VIDEO_OUTPUT` to `false` is necessary if you are running in a headless environment (without a display).  Otherwise, you will need to configure X11 forwarding, which is beyond the scope of this README.
-   **CUDA (`CUDA=true`) provides significant performance improvements when using an NVIDIA GPU. It requires NVIDIA GPU with CUDA drivers installed and a CUDA-enabled build of OpenCV.**  See the Docker section below for detailed instructions on setting up CUDA in Docker.  If you set `CUDA=true` without proper CUDA setup, the application will likely crash or fail to detect fire.
-   Ensure your `CUDA_ARCH_BIN` is set to your appropriate compute capability.

## HOW TO RUN

### Usage Python Directly:

1.  Ensure you have CUDA-enabled OpenCV installed if `CUDA` is set to `true`.

    - To verify CUDA-enabled OpenCV: after installing opencv run this command:
      ```bash
      python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
      ```
      If it outputs a number greater than 0, CUDA is enabled.

2.  Run the project:

    ```bash
    python burning_goat_detection.py
    ```

### Use Docker:

Docker allows you to run the application in a consistent and isolated environment.  The following steps guide you through building and running the Docker image, with and without GPU support.

1.  **Install Docker:** If you haven't already, install Docker Desktop from [docker-desktop](https://www.docker.com/products/docker-desktop/). Follow the instructions for your operating system.

2.  **Install the NVIDIA Container Toolkit (REQUIRED for CUDA):**
    If you want to use CUDA acceleration within Docker, you *must* install the NVIDIA Container Toolkit on your host machine. This toolkit allows Docker containers to access your NVIDIA GPU.

    -   Follow the instructions for your operating system here: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    -   **Important:**  After installing the NVIDIA Container Toolkit, you may need to restart your Docker daemon or your entire system for the changes to take effect.

3.  **Determine your GPU's Compute Capability (REQUIRED for CUDA):**
    -  CUDA applications are compiled for specific GPU architectures, identified by their "compute capability".  You need to determine your GPU's compute capability and set the `CUDA_ARCH_BIN` build argument accordingly.
    -  You can find your GPU's compute capability on NVIDIA's website.
    -  You will use this value in the `docker build` command.

4.  **Build the Docker image:**

    -  **Base run (CPU only):** This build will run the application using the CPU. This is suitable if you don't have an NVIDIA GPU or don't want to use CUDA.
        ```bash
        docker build --target cpu -t burning_goat_detection .
        ```

    -  **With GPU support (CUDA):** This build will run the application using the GPU.
        ```bash
        docker build --target gpu --build-arg CUDA_ARCH=YOUR_GPU_COMPUTE_CAPABILITY -t burning_goat_detection .
        ```
        -   **Replace `YOUR_GPU_COMPUTE_CAPABILITY` with the compute capability you determined in the previous step.** For example, if your GPU's compute capability is 8.6, the command would be:
            ```bash
            docker build --target gpu --build-arg CUDA_ARCH=8.6 -t burning_goat_detection .
            ```
        -   The `--build-arg CUDA_ARCH` flag passes the GPU architecture to the Dockerfile, which uses it to optimize the OpenCV build for your specific GPU.  If you skip this step, the application might not run correctly or might not use the GPU effectively.

5.  **Run the Docker container:**

    -  **Base run (CPU only):**
        ```bash
        docker run --name burning_goat_detection_container -d burning_goat_detection
        ```
        -   To save video chunks to your local machine, you need to mount a volume using the `-v` flag. By default, this maps to `/app/recordings` inside the container.
            -   **Example:**
                ```bash
                docker run --name burning_goat_detection_container -v /path/to/your/local/recordings:/app/recordings -d burning_goat_detection
                ```
            -   In this example, `/path/to/your/local/recordings` is a directory on your computer where you want to save the videos. The `/app/recordings` part is the default path inside the container. If you need to change this, please see the **Using a Custom Recordings Directory** section below.

    -  **With GPU support (CUDA):**
        ```bash
        docker run --gpus all --name burning_goat_detection_container -d burning_goat_detection
        ```
        -   `--gpus all`:  This flag is **critical** for enabling CUDA acceleration. It tells Docker to make all available GPUs accessible to the container. If you only want to use specific GPUs, you can specify their IDs instead (e.g., `--gpus device=0,1`).
        -   To save video chunks, mount a volume as shown above.
            -   **Example:**
                ```bash
                docker run --gpus all --name burning_goat_detection_container -v /path/to/your/local/recordings:/app/recordings -d burning_goat_detection
                ```
            -   In this example, `/path/to/your/local/recordings` is a directory on your computer where you want to save the videos. The `/app/recordings` part is the default path inside the container. If you need to change this, please see the **Using a Custom Recordings Directory** section below.

    -  **Configuring with an `.env` file:**
        The recommended way to provide configuration (like email credentials or Discord webhooks) is to use an `.env` file with the `--env-file` flag. This securely injects all your settings into the container.

        - **Example:**
          ```bash
          docker run --name burning_goat_detection_container \
            --env-file .env \
            -v /path/to/your/local/recordings:/app/recordings \
            -d burning_goat_detection
          ```

    -  **Using a Custom Recordings Directory:**
        If you wish to change the directory where videos are saved inside the container, you must update both the volume mount (`-v`) and provide the `VIDEO_OUTPUT_DIRECTORY` environment variable (`-e`). **The path for the environment variable must match the container-side path of the volume mount.**

        -   **Example with a custom directory:**
            ```bash
            docker run --name burning_goat_detection_container \
              -v /path/to/your/local/custom_vids:/app/custom_vids \
              -e VIDEO_OUTPUT_DIRECTORY=/app/custom_vids \
              -d burning_goat_detection
            ```

6.  **Accessing Container Logs:** To view the application's output and check for errors, you can view the container's logs:
    ```bash
    docker logs burning_goat_detection_container
    ```

7.  **Stopping and Starting the Container:**
    ```bash
    docker stop burning_goat_detection_container
    docker start burning_goat_detection_container
    ```

## Troubleshooting Docker/CUDA Issues

*   **"CUDA driver version is insufficient for CUDA runtime version" Error:** This usually means that the CUDA drivers on your host machine are older than the CUDA version used in the Docker image. Update your NVIDIA drivers to the latest version.
*   **Application Runs on CPU Instead of GPU:**
    *   Ensure that you have correctly installed the NVIDIA Container Toolkit.
    *   Verify that you are passing the `--gpus all` flag to the `docker run` command.
    *   Check the application logs within the container for any CUDA-related errors.
*   **"Could not load cuDNN" Error:** This indicates that the cuDNN libraries are not correctly installed or configured in the Docker image.  Verify that the base image in your `Dockerfile_cuda` includes cuDNN (e.g., `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04`).

## SOURCES

-   [Gävle Goat](https://en.wikipedia.org/wiki/G%C3%A4vle_goat) - Information about the Gävle Goat.
-   [VidGear](https://pypi.org/project/vidgear/) - A video processing framework.
-   [yt-dlp](https://pypi.org/project/yt-dlp/) - A youtube-dl fork with additional features.
-   [Fire Detection System](https://github.com/gunarakulangunaretnam/fire-detection-system-in-python-opencv) - The core computer vision logic for fire detection.
-   [Python Email Examples](https://docs.python.org/3/library/email.examples.html) - Example on email sending using python

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## CODING STANDARD

We use `pre-commit` with `ruff` to ensure consistent code formatting and linting.

**Setup:**

1.  Install development dependencies:
    ```bash
    pip install -e .[dev,cpu]
    ```
2.  Activate hooks: `pre-commit install`
3.  Run checks: `pre-commit run --all-files`

**How to contribute**

-   Ensure you have set up your pre-commit hooks.

Let's keep the Gävle Goat on check! 🐐🔥
