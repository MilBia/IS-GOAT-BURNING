from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class EmailSettings(BaseModel):
    use_emails: bool = Field(default=False)
    sender: str | None = Field(default=None)
    sender_password: str | None = Field(default=None)
    recipients: list[str] = Field(default_factory=list)
    email_host: str | None = Field(default=None)
    email_port: int | None = Field(default=None)
    subject: str = Field(default="GOAT ON FIRE!")
    message: str = Field(default="Dear friend... Its time... Its time to Fight Fire With Fire!")


class DiscordSettings(BaseModel):
    use_discord: bool = Field(default=False)
    hooks: list[str] = Field(default_factory=list)
    message: str = Field(default="Dear friend... Its time... Its time to Fight Fire With Fire!")


class VideoSettings(BaseModel):
    save_video_chunks: bool = Field(default=False)
    video_output_directory: str | None = Field(default=None)
    video_chunk_length_seconds: int = Field(default=60)
    max_video_chunks: int = Field(default=10)
    chunks_to_keep_after_fire: int = Field(default=5)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__")

    source: str = Field(validation_alias="SOURCE")
    logging: bool = Field(alias="LOGGING", default=True)
    video_output: bool = Field(alias="VIDEO_OUTPUT", default=False)
    checks_per_second: float = Field(alias="CHECKS_PER_SECOND", default=2.0)
    open_cl: bool = Field(alias="OPEN_CL", default=False)
    cuda: bool = Field(alias="CUDA", default=False)

    # Nested settings
    email: EmailSettings
    discord: DiscordSettings
    video: VideoSettings


# Single, validated instance to be used across the application
settings = Settings()
