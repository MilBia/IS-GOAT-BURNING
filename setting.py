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

SAVE_VIDEO_CHUNKS = env.bool("SAVE_VIDEO_CHUNKS")
VIDEO_OUTPUT_DIRECTORY = env("VIDEO_OUTPUT_DIRECTORY")
VIDEO_CHUNK_LENGTH_SECONDS = env.int("VIDEO_CHUNK_LENGTH_SECONDS")
MAX_VIDEO_CHUNKS = env.int("MAX_VIDEO_CHUNKS")
CHUNKS_TO_KEEP_AFTER_FIRE = env.int("CHUNKS_TO_KEEP_AFTER_FIRE")
