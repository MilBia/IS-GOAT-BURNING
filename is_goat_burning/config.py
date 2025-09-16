"""Defines the application's configuration model using Pydantic.

This module uses `pydantic-settings` to define a hierarchical settings structure
that is loaded from environment variables and `.env` files. It includes custom
validators to ensure that the configuration is consistent and complete.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class EmailSettings(BaseModel):
    """Configuration specific to email notifications.

    Attributes:
        use_emails: If True, email notifications are enabled.
        sender: The sender's email address.
        sender_password: The sender's email account password or app password.
        recipients: A list of email addresses to send notifications to.
        email_host: The SMTP host for the email server.
        email_port: The SMTP port for the email server.
        subject: The subject line for notification emails.
        message: The plain text body for notification emails.
    """

    _REQUIRED_FIELDS: ClassVar[dict[str, str]] = {
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
        """Validates that required fields are set if emails are enabled."""
        if self.use_emails:
            for field_name, env_var in self._REQUIRED_FIELDS.items():
                value = getattr(self, field_name)
                if value is None or (hasattr(value, "__len__") and not value):
                    raise ValueError(f"{env_var} must be set when EMAIL__USE_EMAILS is true")
            if self.sender_password is None or not self.sender_password.get_secret_value():
                raise ValueError("EMAIL__SENDER_PASSWORD must be set when EMAIL__USE_EMAILS is true")
        return self


class DiscordSettings(BaseModel):
    """Configuration specific to Discord webhook notifications.

    Attributes:
        use_discord: If True, Discord notifications are enabled.
        hooks: A list of Discord webhook URLs.
        message: The message content to send to the webhooks.
    """

    use_discord: bool = Field(default=False)
    hooks: list[str] = Field(default_factory=list)
    message: str = Field(default="Dear friend... Its time... Its time to Fight Fire With Fire!")

    @model_validator(mode="after")
    def check_required_fields(self) -> DiscordSettings:
        """Validates that webhook hooks are provided if Discord is enabled."""
        if self.use_discord and not self.hooks:
            raise ValueError("DISCORD__HOOKS must be set when DISCORD__USE_DISCORD is true")
        return self


class VideoSettings(BaseModel):
    """Configuration specific to saving video chunks.

    Attributes:
        save_video_chunks: If True, saving video chunks to disk is enabled.
        video_output_directory: The directory to save video files in.
        video_chunk_length_seconds: The duration of each video chunk.
        max_video_chunks: The maximum number of chunks to keep on disk.
        chunks_to_keep_after_fire: The number of extra chunks to save after a
            fire is detected.
    """

    save_video_chunks: bool = Field(default=False)
    video_output_directory: str | None = Field(default=None)
    video_chunk_length_seconds: int = Field(default=300)
    max_video_chunks: int = Field(default=20)
    chunks_to_keep_after_fire: int = Field(default=10)

    @model_validator(mode="after")
    def check_required_fields(self) -> VideoSettings:
        """Validates that an output directory is set if saving is enabled."""
        if self.save_video_chunks and not self.video_output_directory:
            raise ValueError("VIDEO__VIDEO_OUTPUT_DIRECTORY must be set when VIDEO__SAVE_VIDEO_CHUNKS is true")
        return self


class Settings(BaseSettings):
    """The main application settings model.

        This class aggregates all other settings models and defines the global
        configuration options. It is configured to load from `.env` files and
    aggressively from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=(".env.tests", ".env"), env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    source: str = Field(validation_alias="SOURCE")
    fire_detection_threshold: float = Field(validation_alias="FIRE_DETECTION_THRESHOLD", default=0.1)
    logging: bool = Field(validation_alias="LOGGING", default=True)
    video_output: bool = Field(validation_alias="VIDEO_OUTPUT", default=False)
    checks_per_second: float = Field(validation_alias="CHECKS_PER_SECOND", default=1.0)
    open_cl: bool = Field(validation_alias="OPEN_CL", default=False)
    cuda: bool = Field(validation_alias="CUDA", default=False)

    # Nested settings
    email: EmailSettings = Field(default_factory=EmailSettings)
    discord: DiscordSettings = Field(default_factory=DiscordSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)


# A single, validated instance to be used across the application.
settings = Settings()
