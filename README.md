# IS GOAT BURNING?

We'll checking if [The Gävle Goat](https://www.youtube.com/watch?v=TqvguE5cKT0) is burning

## SOURCES

https://en.wikipedia.org/wiki/G%C3%A4vle_goat

https://pypi.org/project/vidgear/

https://pypi.org/project/yt-dlp/

https://github.com/gunarakulangunaretnam/fire-detection-system-in-python-opencv

https://docs.python.org/3/library/email.examples.html

## DOCKER

```commandline
docker build -t burning_goat_detection .
docker run -d --name burning_goat_detection_container burning_goat_detection
```

## INSTALLING

```commandline
pip install -r requirements.txt
```

## CONFIGURATION

fill ```.env``` file

example: 
```
SOURCE="https://youtu.be/TqvguE5cKT0"
SENDER="sender"
SENDER_PASSWORD="password"
RECIPIENTS="recipient1,recipient2"
EMAIL_HOST="smtp.gmail.com"
EMAIL_PORT=587
LOGGING=true
VIDEO_OUTPUT=true
CHECKS_PER_SECOND=1.0
```

## CODING STANDARD

black -l 128 .

flake8 --exclude=venv --max-line-length=128
