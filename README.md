# Is the Gävle Goat Burning?

This project monitors the [Gävle Goat webcam](https://youtu.be/vDFPpkp9krY) feed to detect if the goat is on fire.  It uses computer vision techniques to analyze the video feed and sends email notifications if fire is detected.

## Project Overview

The Gävle Goat is a giant straw goat built annually in Gävle, Sweden. It has become a tradition for vandals to attempt to burn it down. This project aims to provide real-time monitoring of the goat's status.

## Features

- **Real-time Fire Detection:** Utilizes computer vision to analyze the live webcam feed for signs of fire.
- **Email Notifications:** Sends email alerts immediately upon detecting fire, notifying designated recipients.
- **Configurable Monitoring:** Allows adjustment of monitoring frequency to control resource usage and sensitivity.
- **Dockerized Deployment:** Easily deployable with Docker, ensuring consistent performance across different environments.

## Prerequisites

- Python 3.8 or higher
- Docker (optional, for Docker deployment)

## INSTALLATION

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd burning-goat-detection
   ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

## CONFIGURATION

1. Create a `.env` file by copying the example:

    ```bash
    cp .env.example .env
    ```

2. Edit the .env file and fill in your configuration details:
 
    ```
    SOURCE="https://youtu.be/vDFPpkp9krY"                       # URL of the webcam feed
    SENDER="your_email@example.com"                             # Your email address
    SENDER_PASSWORD="your_email_password"                       # Your email password or an app password for Gmail
    RECIPIENTS="recipient1@example.com,recipient2@example.com"  # Comma-separated list of email addresses to notify
    EMAIL_HOST="smtp.gmail.com"                                 # Your email host (e.g., smtp.gmail.com)
    EMAIL_PORT=587                                              # Your email port (e.g., 587 for Gmail)
    LOGGING=true                                                # Enable or disable logging
    VIDEO_OUTPUT=true                                           # Display detected video frames (true) or not (false)
    CHECKS_PER_SECOND=1.0                                       # How many times to check per second (adjust for performance)
    ```

**Important**:

- For Gmail, it's often necessary to generate an app password in your Google account settings, instead of your regular password.
- When using Docker, setting `VIDEO_OUTPUT` to `false` is necessary.


## HOW TO RUN
### Usage Python Directly:

1. Run the project

    ```bash
    python burning_goat_detection.py
    ```

### Use Docker:

1. Build the Docker image:
    ```bash
    docker build -t burning_goat_detection .
    ```

2. Run the Docker container:
    ```bash
    docker run --name burning_goat_detection_container -d burning_goat_detection
    ```
This command will build the image, and run the container in the background, named `burning_goat_detection_container`.

3. Start/Stop existing container 
    ```bash
    docker start burning_goat_detection_container
    docker stop burning_goat_detection_container
    ```

## SOURCES

* [Gävle Goat](https://en.wikipedia.org/wiki/G%C3%A4vle_goat) - Information about the Gävle Goat.
* [VidGear](https://pypi.org/project/vidgear/) - A video processing framework.
* [yt-dlp](https://pypi.org/project/yt-dlp/) - A youtube-dl fork with additional features.
* [Fire Detection System](https://github.com/gunarakulangunaretnam/fire-detection-system-in-python-opencv) - The core computer vision logic for fire detection.
* [Python Email Examples](https://docs.python.org/3/library/email.examples.html) - Example on email sending using python

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## CODING STANDARD

- **Black**: Use `black -l 128` . to format the code.
- **Flake8**: Use  `flake8 --exclude=venv --max-line-length=128` to check for linting errors.

Let's keep the Gävle Goat on check! 🐐🔥