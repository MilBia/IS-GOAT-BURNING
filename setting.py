from environs import Env

env = Env()
env.read_env()

SOURCE = env("SOURCE")

USE_EMAILS = env.bool("USE_EMAILS")
SENDER = env("SENDER")
SENDER_PASSWORD = env("SENDER_PASSWORD")
RECIPIENTS = env.list("RECIPIENTS")
EMAIL_HOST = env("EMAIL_HOST")
EMAIL_PORT = env.int("EMAIL_PORT")

USE_DISCORD = env.bool("USE_DISCORD")
DISCORD_HOOKS = env.list("DISCORD_HOOKS")

LOGGING = env.bool("LOGGING")
VIDEO_OUTPUT = env.bool("VIDEO_OUTPUT")
CHECKS_PER_SECOND = env.float("CHECKS_PER_SECOND")
OPEN_CL = env.bool("OPEN_CL")
CUDA = env.bool("CUDA")
