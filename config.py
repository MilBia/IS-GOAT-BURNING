from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class EmailSettings(BaseModel):
    _REQUIRED_FIELDS = {
        "sender": "EMAIL__SENDER",
        "recipients": "EMAIL__RECIPIENTS",
        "email_host": "EMAIL__EMAIL_HOST",
        "email_port": "EMAIL__EMAIL_PORT",
    }

    use_emails: bool = Field(default=False)
    sender: str | None = Field(default=None)
    sender_password: SecretStr | None = Field(default=None)
    recipients: list[str] = Field(default_factory=list)
    email_host: str | None = Field(default=None)
    email_port: int | None = Field(default=None)
    subject: str = Field(default="GOAT ON FIRE!")
    message: str = Field(default="Dear friend... Its time... Its time to Fight Fire With Fire!")

    @model_validator(mode="after")
    def check_required_fields(self) -> EmailSettings:
        if self.use_emails:
            for field_name, env_var in self._REQUIRED_FIELDS.items():
                value = getattr(self, field_name)
                if value is None or (hasattr(value, "__len__") and not value):
                    raise ValueError(f"{env_var} must be set when USE_EMAILS is true")
            if self.sender_password is None or not self.sender_password.get_secret_value():
                raise ValueError("EMAIL__SENDER_PASSWORD must be set when USE_EMAILS is true")
        return self


class DiscordSettings(BaseModel):
    use_discord: bool = Field(default=False)
    hooks: list[str] = Field(default_factory=list)
    message: str = Field(default="Dear friend... Its time... Its time to Fight Fire With Fire!")

    @model_validator(mode="after")
    def check_required_fields(self) -> DiscordSettings:
        if self.use_discord and not self.hooks:
            raise ValueError("DISCORD__HOOKS must be set when use_discord is true")
        return self


class VideoSettings(BaseModel):
    save_video_chunks: bool = Field(default=False)
    video_output_directory: str | None = Field(default=None)
    video_chunk_length_seconds: int = Field(default=300)
    max_video_chunks: int = Field(default=20)
    chunks_to_keep_after_fire: int = Field(default=10)

    @model_validator(mode="after")
    def check_required_fields(self) -> VideoSettings:
        if self.save_video_chunks and not self.video_output_directory:
            raise ValueError("VIDEO__VIDEO_OUTPUT_DIRECTORY must be set when save_video_chunks is true")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__")

    source: str = Field(validation_alias="SOURCE")
    fire_detection_threshold: float = Field(validation_alias="FIRE_DETECTION_THRESHOLD", default=0.1)
    logging: bool = Field(validation_alias="LOGGING", default=True)
    video_output: bool = Field(validation_alias="VIDEO_OUTPUT", default=False)
    checks_per_second: float = Field(validation_alias="CHECKS_PER_SECOND", default=1.0)
    open_cl: bool = Field(validation_alias="OPEN_CL", default=False)
    cuda: bool = Field(validation_alias="CUDA", default=False)

    # Nested settings
    email: EmailSettings
    discord: DiscordSettings
    video: VideoSettings


# Single, validated instance to be used across the application
settings = Settings()
