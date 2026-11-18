"""Defines the application's configuration model using Pydantic.

This module uses `pydantic-settings` to define a hierarchical settings structure
that is loaded from environment variables and `.env` files. It includes custom
validators to ensure that the configuration is consistent and complete.
"""

from enum import StrEnum
import os
from pathlib import Path
from typing import Annotated
from typing import ClassVar
from typing import Literal

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


def _clean_message_str(v: str) -> str:
    """Cleans a message string by stripping quotes and unescaping newlines.

    Args:
        v: The message string to clean.

    Returns:
        The cleaned message string.
    """
    if v:
        v = v.strip()
        # Strip surrounding quotes
        if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
            v = v[1:-1]
        # Unescape newlines
        v = v.replace("\\n", "\n")
    return v


CleanedMessage = Annotated[str, AfterValidator(_clean_message_str)]


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
    message: CleanedMessage = Field(default="Dear friend... Its time... Its time to Fight Fire With Fire!")

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
        send_video_chunks: If True, saved video chunks are uploaded to the
            configured Discord webhooks as they are archived after a fire event.
        file_limit_mb: Largest video chunk uploaded to Discord, in MiB (1024 * 1024 bytes). Larger files, and files Discord
            rejects with HTTP 413, are replaced by a short text message with the file name and size.
        attach_frame: If True, the first alert carries a JPEG of the frame that triggered it.
        timeout_seconds: Per-request timeout for text alerts.
        max_attempts: Maximum attempts per webhook for text alerts.
        retry_base_delay_seconds: Delay before the first retry (doubles each retry).
        retry_max_delay_seconds: Cap for any wait between attempts, including
            server-requested `Retry-After` delays.
        service_hooks: Webhook URLs of a separate service channel (startup, heartbeat, stream outage,
            verifier health, shutdown). Empty disables service notifications. Independent of `use_discord`
            and must not repeat a URL from `hooks`, so service messages never mix with fire alerts.
        service_heartbeat_interval_seconds: Interval between heartbeat messages on the service channel.
        service_stream_down_warning_seconds: Stream outage length after which a warning is sent (once per
            outage); the recovery after such an outage is reported too.
    """

    # Webhook URLs are secrets: never echo the raw input in validation errors.
    model_config = ConfigDict(hide_input_in_errors=True)

    use_discord: bool = Field(default=False)
    hooks: list[str] = Field(default_factory=list)
    message: CleanedMessage = Field(default="Dear friend... Its time... Its time to Fight Fire With Fire!")
    send_video_chunks: bool = Field(default=False)
    file_limit_mb: float = Field(default=18.0, gt=0)
    attach_frame: bool = Field(default=True)
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_attempts: int = Field(default=5, ge=1)
    retry_base_delay_seconds: float = Field(default=1.0, ge=0)
    retry_max_delay_seconds: float = Field(default=30.0, ge=0)
    service_hooks: list[str] = Field(default_factory=list)
    service_heartbeat_interval_seconds: float = Field(default=86400.0, gt=0)
    service_stream_down_warning_seconds: float = Field(default=600.0, gt=0)

    @model_validator(mode="after")
    def check_required_fields(self) -> DiscordSettings:
        """Validates that webhook hooks are provided if Discord is enabled and that the two channels differ."""
        if self.use_discord and not self.hooks:
            raise ValueError("DISCORD__HOOKS must be set when DISCORD__USE_DISCORD is true")
        if set(self.hooks) & set(self.service_hooks):
            # Never echo the URL: it is a secret.
            raise ValueError("DISCORD__SERVICE_HOOKS must not contain a webhook from DISCORD__HOOKS (keep alerts separate)")
        return self


class VideoSettings(BaseModel):
    """Configuration specific to saving video chunks.

    Attributes:
        save_video_chunks: If True, saving video chunks to disk is enabled.
        video_output_directory: The directory to save video files in.
        video_chunk_length_seconds: The duration of each video chunk file when
            using the "disk" buffer mode.
        max_video_chunks: The maximum number of chunks to keep on disk.
        chunks_to_keep_after_fire: The number of chunk-lengths of video to save after a
            fire is detected.
        buffer_mode: The buffering strategy. "disk" saves chunks to disk
            continuously. "memory" holds frames in RAM and only saves to
            disk when a fire is detected.
            "segments" stream-copies the source with ffmpeg (no decode, no
            re-encode) into a ring of short segments in `segments_directory`
            and only concatenates them on a fire event.
        memory_buffer_seconds: The duration in seconds of pre-fire video to
            hold in RAM when using "memory" or "segments" buffer mode.
        segments_directory: Directory of the segment ring for "segments" mode.
            Should be a tmpfs (RAM) path; in Docker raise `--shm-size` for it.
        segment_seconds: Minimum duration of one ring segment (segments are cut
            on keyframes, so real segments can be longer).
        record_during_fire: If True, recording will continue for the entire
            duration of the fire event.
        flush_num_threads: The number of threads to use when flushing the
            memory buffer to disk. Lower values reduce CPU contention. A
            value of 0 will let OpenCV determine the number of threads
            automatically.
        flush_throttle_frame_interval: The number of frames to process before
            sleeping during a memory buffer flush.
        flush_throttle_seconds: The duration in seconds to sleep during the
            memory buffer flush throttle.
        flush_throttle_enabled: If True, enables throttling during memory buffer
            flush to reduce CPU/IO contention.
    """

    save_video_chunks: bool = Field(default=False)
    video_output_directory: str | None = Field(default=None)
    video_chunk_length_seconds: int = Field(default=300)
    max_video_chunks: int = Field(default=20)
    chunks_to_keep_after_fire: int = Field(default=10)
    buffer_mode: Literal["disk", "memory", "segments"] = Field(default="memory")
    memory_buffer_seconds: int = Field(default=60)
    segments_directory: str = Field(default="/dev/shm/goat-ring")  # noqa: S108 - tmpfs by design, keeps the ring in RAM
    segment_seconds: int = Field(default=2, ge=1)
    record_during_fire: bool = Field(default=False)
    flush_num_threads: int = Field(default=1)
    flush_throttle_frame_interval: int = Field(default=10)
    flush_throttle_seconds: float = Field(default=0.01)
    flush_throttle_enabled: bool = Field(default=False)

    @field_validator("flush_num_threads")
    @classmethod
    def check_threads_non_negative(cls, v: int) -> int:
        """Ensures that the number of threads is a non-negative integer.

        Args:
            v: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is negative.
        """
        if v < 0:
            raise ValueError("flush_num_threads must be a non-negative integer")
        return v

    @field_validator("flush_throttle_seconds")
    @classmethod
    def check_seconds_non_negative(cls, v: float) -> float:
        """Ensures that the throttle duration is a non-negative float.

        Args:
            v: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is negative.
        """
        if v < 0.0:
            raise ValueError("flush_throttle_seconds must be a non-negative float")
        return v

    @field_validator("flush_throttle_frame_interval")
    @classmethod
    def check_interval_positive(cls, v: int) -> int:
        """Ensures that the throttle interval is a positive integer.

        Args:
            v: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        if v <= 0:
            raise ValueError("flush_throttle_frame_interval must be a positive integer")
        return v

    @model_validator(mode="after")
    def check_required_fields(self) -> VideoSettings:
        """Validates that an output directory is set if saving is enabled."""
        if self.save_video_chunks and not self.video_output_directory:
            raise ValueError("VIDEO__VIDEO_OUTPUT_DIRECTORY must be set when VIDEO__SAVE_VIDEO_CHUNKS is true")
        return self


class GeminiSettings(BaseModel):
    """Configuration specific to Gemini fire detection.

    Attributes:
        api_key: The Google Gemini API key.
        model: The model to use (e.g., "gemini-3.5-flash-lite").
        prompt: The prompt to send to the model.
        timeout_seconds: Per-request timeout; a timed-out request counts as a verifier error.
    """

    api_key: SecretStr | None = Field(default=None)
    model: str = Field(default="gemini-3.5-flash-lite")
    prompt: str = Field(default="Analyze this image and determine if there is a visible fire. Ignore purely digital overlays.")
    timeout_seconds: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def check_required_fields(self) -> GeminiSettings:
        """Validates that API key is set if Gemini strategy is used."""
        # Note: We can't easily check the global strategy here without circular imports or context.
        # So we'll rely on the detector factory to check for the key.
        return self


class DecisionSettings(BaseModel):
    """Configuration for fire detection with the Gemini Decisions API.

    The API key is shared with :class:`GeminiSettings` (``GEMINI__API_KEY``).

    Attributes:
        model: The decision model to use (e.g., "gemini-decision-1").
        threshold: Minimum calibrated probability ``P(fire)`` in ``[0, 1]`` to report fire.
        instructions: The yes/no question asked about every analysed frame.
        timeout_seconds: Per-request timeout; a timed-out request counts as an error.
    """

    model: str = Field(default="gemini-decision-1")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    instructions: str = Field(default="Is there a visible fire in the image? Ignore purely digital overlays.")
    timeout_seconds: float = Field(default=10.0, gt=0.0)


class Accelerator(StrEnum):
    """Hardware backend used for local (edge) model inference.

    Attributes:
        AUTO: Detect the best available backend at runtime.
        CPU: Generic CPU execution (fallback for any host).
        CUDA: NVIDIA GPU acceleration.
        OPENCL: OpenCL acceleration (AMD / Intel integrated GPUs).
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    OPENCL = "opencl"


class EdgeSettings(BaseModel):
    """Configuration specific to the local ``hybrid_edge`` verification strategy.

    Attributes:
        model_path: Path to the local inference model (e.g. ``.onnx`` or
            ``.xml``). Required when ``DETECTION_STRATEGY=hybrid_edge``; its
            existence is validated by the top-level ``Settings`` model.
        confidence_threshold: Minimum confidence above which a detection is
            confirmed as fire.
        accelerator: The hardware backend to run inference on.
        input_size: Side length (pixels) of the square model input; frames are
            squash-resized to ``input_size x input_size``.
    """

    model_path: str | None = Field(default=None)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    accelerator: Accelerator = Field(default=Accelerator.AUTO)
    input_size: int = Field(default=224, gt=0)


class EscalationLogSettings(BaseModel):
    """Configuration of the escalation log (frames sent to the verifier, with verdicts).

    Attributes:
        enabled: Whether escalations are recorded (off by default).
        directory: Directory receiving the JPEG frames and the ``escalations.jsonl`` index.
        max_mb: Size limit of the directory in MB; the oldest JPEGs are deleted beyond it.
        jpeg_quality: JPEG quality (1-100) of the stored frames.
    """

    enabled: bool = Field(default=False)
    directory: str = Field(default="./escalation_log")
    max_mb: int = Field(default=2000, gt=0)
    jpeg_quality: int = Field(default=85, ge=1, le=100)


class Settings(BaseSettings):
    """The main application settings model.

    This class aggregates all other settings models and defines the global
    configuration options. It is configured to load from `.env` files and
    aggressively from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=(".env.tests", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        # Inputs include secrets (webhook URLs, API keys, passwords): never echo them in startup errors.
        hide_input_in_errors=True,
    )

    source: str = Field(validation_alias="SOURCE")
    detection_strategy: Literal["classic", "gemini", "hybrid_cloud", "hybrid_edge", "gemini_decision", "hybrid_decision"] = (
        Field(default="classic", validation_alias="DETECTION_STRATEGY")
    )
    fire_detection_threshold: float = Field(validation_alias="FIRE_DETECTION_THRESHOLD", default=0.1)
    fire_detection_mode: Literal["frame_fraction", "area_relative"] = Field(
        default="frame_fraction", validation_alias="FIRE_DETECTION_MODE"
    )
    area_relative_threshold: float = Field(default=0.23, gt=0.0, le=1.0, validation_alias="AREA_RELATIVE_THRESHOLD")
    verifier_cooldown_seconds: float = Field(default=60.0, ge=0.0, validation_alias="VERIFIER_COOLDOWN_SECONDS")
    verifier_max_calls_per_hour: int = Field(default=0, ge=0, validation_alias="VERIFIER_MAX_CALLS_PER_HOUR")
    verifier_failure_policy: Literal["alert_unverified", "suppress"] = Field(
        default="alert_unverified", validation_alias="VERIFIER_FAILURE_POLICY"
    )
    verifier_errors_before_policy: int = Field(default=3, ge=1, validation_alias="VERIFIER_ERRORS_BEFORE_POLICY")
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = Field(default="INFO", validation_alias="LOGGING")
    video_output: bool = Field(validation_alias="VIDEO_OUTPUT", default=False)
    checks_per_second: float = Field(validation_alias="CHECKS_PER_SECOND", default=1.0)
    default_framerate: float = Field(default=30.0, validation_alias="DEFAULT_FRAMERATE")
    ytdlp_format: str = Field(default="bestvideo/best", validation_alias="YTDLP_FORMAT")
    open_cl: bool = Field(validation_alias="OPEN_CL", default=False)
    cuda: bool = Field(validation_alias="CUDA", default=False)
    reconnect_delay_seconds: int = Field(default=5, validation_alias="RECONNECT_DELAY_SECONDS")
    reconnect_max_delay_seconds: int = Field(default=300, ge=1, validation_alias="RECONNECT_MAX_DELAY_SECONDS")
    stream_inactivity_timeout: int = Field(default=60, validation_alias="STREAM_INACTIVITY_TIMEOUT")
    fire_detected_debounce_seconds: float = Field(default=0.0, validation_alias="FIRE_DETECTED_DEBOUNCE_SECONDS")
    fire_extinguished_debounce_seconds: float = Field(default=5.0, validation_alias="FIRE_EXTINGUISHED_DEBOUNCE_SECONDS")
    alert_cooldown_seconds: float = Field(default=600.0, ge=0.0, validation_alias="ALERT_COOLDOWN_SECONDS")
    motion_detection_threshold: int = Field(default=25, validation_alias="MOTION_DETECTION_THRESHOLD")

    # Nested settings
    email: EmailSettings = Field(default_factory=EmailSettings)
    discord: DiscordSettings = Field(default_factory=DiscordSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    edge: EdgeSettings = Field(default_factory=EdgeSettings)
    decision: DecisionSettings = Field(default_factory=DecisionSettings)
    escalation_log: EscalationLogSettings = Field(default_factory=EscalationLogSettings)

    @field_validator("detection_strategy", mode="before")
    @classmethod
    def migrate_legacy_hybrid(cls, v: object) -> object:
        """Maps the deprecated ``hybrid`` strategy to ``hybrid_cloud``.

        This preserves backward compatibility for existing deployments that
        still set ``DETECTION_STRATEGY=hybrid``.

        Args:
            v: The raw ``detection_strategy`` value (after alias resolution).

        Returns:
            ``"hybrid_cloud"`` if the value was the legacy ``"hybrid"``,
            otherwise the value unchanged.
        """
        if v == "hybrid":
            return "hybrid_cloud"
        return v

    @model_validator(mode="after")
    def check_edge_model_path(self) -> Settings:
        """Validates the edge model path when the ``hybrid_edge`` strategy is active.

        Returns:
            The validated Settings instance.

        Raises:
            ValueError: If ``hybrid_edge`` is selected but ``EDGE__MODEL_PATH`` is
                unset or points to a file that does not exist.
        """
        if self.detection_strategy == "hybrid_edge":
            if not self.edge.model_path:
                raise ValueError("EDGE__MODEL_PATH must be set when DETECTION_STRATEGY is 'hybrid_edge'")
            if not Path(self.edge.model_path).is_file():
                raise ValueError(f"EDGE__MODEL_PATH points to a non-existent file: {self.edge.model_path}")
        return self

    @model_validator(mode="after")
    def check_reconnect_delays(self) -> Settings:
        """Validates that the reconnect backoff cap is not lower than the base delay.

        Returns:
            The validated Settings instance.

        Raises:
            ValueError: If ``RECONNECT_MAX_DELAY_SECONDS`` is lower than ``RECONNECT_DELAY_SECONDS``.
        """
        if self.reconnect_max_delay_seconds < self.reconnect_delay_seconds:
            raise ValueError("RECONNECT_MAX_DELAY_SECONDS must not be lower than RECONNECT_DELAY_SECONDS")
        return self


# A single, validated instance to be used across the application.
settings = Settings()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"timeout;{settings.stream_inactivity_timeout * 1_000_000}"
