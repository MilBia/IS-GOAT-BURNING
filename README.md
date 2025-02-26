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

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA drivers installed (if using CUDA acceleration)
- Docker (optional, for Docker deployment)

## INSTALLATION

1.  Clone the repository:
    ```bash
    git clone https://github.com/MilBia/IS-GOAT-BURNING.git
    cd burning-goat-detection
    ```

2.  Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

## CONFIGURATION

1.  Create a `.env` file by copying the example:

    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file and fill in your configuration details:

    ```
    SOURCE="https://youtu.be/vDFPpkp9krY"                            # URL of the webcam feed
    USE_EMAILS=true                                                  # Set to true if you want email notifications
    SENDER="your_email@example.com"                                  # Your email address
    SENDER_PASSWORD="your_email_password"                            # Your email password or an app password for Gmail
    RECIPIENTS="recipient1@example.com,recipient2@example.com"       # Comma-separated list of email addresses to notify
    EMAIL_HOST="smtp.gmail.com"                                      # Your email host (e.g., smtp.gmail.com)
    EMAIL_PORT=587                                                   # Your email port (e.g., 587 for Gmail)
    USE_DISCORD=true                                                 # Set to true if you want Discord notifications
    DISCORD_HOOKS="/webhooks/webhooks/{webhook.id}/{webhook.token}"  # Your discord webhook
    LOGGING=true                                                     # Enable or disable logging
    VIDEO_OUTPUT=true                                                # Display detected video frames (true) or not (false)
    CHECKS_PER_SECOND=1.0                                            # How many times to check per second (adjust for performance)
    OPEN_CL=false                                                    # Enable or disable use of OpenCL for faster processing (experimental)
    CUDA=false                                                       # Enable or disable use of CUDA for faster processing (experimental)
    ```

**Important**:

-   OpenCL (`OPEN_CL=true`) for faster processing is experimental and requires `VIDEO_OUTPUT=false`.
-   OpenCL (`OPEN_CL=true`) is currently not supported in Docker containers.
-   For Gmail, it's often necessary to generate an app password in your Google account settings, instead of your regular password.
-   When using Docker, setting `VIDEO_OUTPUT` to `false` is necessary if you are running in a headless environment.
-   **CUDA (`CUDA=true`) requires an NVIDIA GPU with CUDA drivers installed and a CUDA-enabled build of OpenCV.**  See the Docker section below for instructions on setting up CUDA in Docker.
-   Ensure your CUDA_ARCH_BIN is set to your appropriate compute capability.

## HOW TO RUN

### Usage Python Directly:

1.  Ensure you have CUDA-enabled OpenCV installed if `CUDA` is set to `true`.

2.  Run the project:

    ```bash
    python burning_goat_detection.py
    ```

### Use Docker:

1.  **Install the NVIDIA Container Toolkit:**
    If you want to use CUDA acceleration within Docker, you must install the NVIDIA Container Toolkit on your host machine. Follow the instructions for your operating system here: [container toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

2.  Build the Docker image:
   -  **base run:**

        ```bash
        docker build -f dockerfile -t burning_goat_detection .
        ```

   -  **with GPU support:**

        ```bash
        docker build -f dockerfile_cuda -t burning_goat_detection .
        ```

3.  Run the Docker container:
   -  **base run:**

       ```bash
       docker run --name burning_goat_detection_container -d burning_goat_detection
       ```

   -  **with GPU support:**

       ```bash
       docker run --gpus all --name burning_goat_detection_container -d burning_goat_detection
       ```

       *   `--gpus all`:  This flag is **critical** for enabling CUDA acceleration. It tells Docker to make all available GPUs accessible to the container. If you only want to use specific GPUs, you can specify their IDs instead (e.g., `--gpus device=0,1`).

This command will build the image, and run the container in the background, named `burning_goat_detection_container`.

4.  Start/Stop existing container

    ```bash
    docker start burning_goat_detection_container
    docker stop burning_goat_detection_container
    ```

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

1.  Install `pip install -r requirements_dev.txt`
2.  Activate hooks: `pre-commit install`
3.  Run checks: `pre-commit run --all-files`

**How to contribute**

-   Ensure you have set up your pre-commit hooks.

Let's keep the Gävle Goat on check! 🐐🔥
