"""Implements fire detection using Google's Gemini API."""

import asyncio
from typing import TYPE_CHECKING

import cv2
from google import genai
from google.genai import types
import numpy as np
from pydantic import BaseModel
from pydantic import Field

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.verifier_health import STANDALONE_POLICY
from is_goat_burning.fire_detection.verifier_health import VerifierError
from is_goat_burning.fire_detection.verifier_health import VerifierStatus
from is_goat_burning.fire_detection.verifier_health import publish_verifier_status
from is_goat_burning.logger import get_logger

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse

logger = get_logger("GeminiFireDetector")

# Number of consecutive failed requests after which a CRITICAL message is logged (once per outage).
CONSECUTIVE_ERRORS_BEFORE_CRITICAL = 5


class FireDetectionResponse(BaseModel):
    """Structured output model for Gemini fire detection."""

    is_fire: bool = Field(description="True if fire is visible in the image, False otherwise.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")


class GeminiFireDetector:
    """Detects fire using the Google Gemini API.

    ``verify()`` returns the verdict or raises ``VerifierError`` when no verdict could be obtained; ``detect()``
    wraps it for standalone use and never raises (errors become "no fire"). Failures are logged at ERROR and,
    after ``CONSECUTIVE_ERRORS_BEFORE_CRITICAL`` in a row, once at CRITICAL.
    """

    def __init__(self) -> None:
        """Initializes the GeminiFireDetector.

        Raises:
            ValueError: If the GEMINI__API_KEY is not set in the configuration.
        """
        if not settings.gemini.api_key:
            raise ValueError("GEMINI__API_KEY must be set when using the 'gemini' detection strategy.")

        self.client = genai.Client(api_key=settings.gemini.api_key.get_secret_value())
        self.model_name = settings.gemini.model
        self.prompt = settings.gemini.prompt
        self.timeout_seconds = settings.gemini.timeout_seconds
        self._consecutive_errors = 0

    def _preprocess_frame(self, frame: np.ndarray) -> bytes:
        """Converts an OpenCV frame to a JPEG byte string.

        Args:
            frame: The OpenCV image (numpy array) to convert.

        Returns:
            A byte string of the image encoded as a JPEG.

        Raises:
            ValueError: If the frame fails to encode to JPEG.
        """
        # Encode frame to JPEG
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Failed to encode frame to JPEG.")
        return buffer.tobytes()

    def _record_error(self) -> None:
        """Counts a failed request and escalates to CRITICAL once per outage."""
        self._consecutive_errors += 1
        if self._consecutive_errors == CONSECUTIVE_ERRORS_BEFORE_CRITICAL:
            logger.critical(
                "Gemini API failed %d times in a row; fire verification is unavailable until it recovers.",
                self._consecutive_errors,
            )

    async def verify(self, frame: np.ndarray) -> bool:
        """Asks Gemini whether the frame shows fire.

        Args:
            frame: The input video frame (BGR numpy array).

        Returns:
            The verdict: True if Gemini reports a visible fire.

        Raises:
            VerifierError: If no verdict could be obtained (API error, timeout after
                ``GEMINI__TIMEOUT_SECONDS``, unparseable answer, frame encoding failure).
        """
        try:
            image_bytes = await asyncio.to_thread(self._preprocess_frame, frame)
            response: GenerateContentResponse = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=[
                        self.prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=FireDetectionResponse,
                    ),
                ),
                timeout=self.timeout_seconds,
            )
            # With ``response_schema`` the SDK puts the parsed model instance in ``parsed``.
            detection_result: FireDetectionResponse | None = response.parsed
            if not detection_result:
                raise ValueError("Gemini response did not contain a valid parsed result.")
        except TimeoutError as e:
            logger.error("Gemini API request timed out after %.1f s.", self.timeout_seconds)
            self._record_error()
            raise VerifierError(f"Gemini request timed out after {self.timeout_seconds:.1f} s") from e
        except (genai.errors.APIError, ValueError) as e:
            logger.error("Gemini API error: %s", e)
            self._record_error()
            raise VerifierError(f"Gemini error: {type(e).__name__}") from e
        except Exception as e:
            logger.exception("Unexpected error during Gemini fire detection.")
            self._record_error()
            raise VerifierError(f"Unexpected Gemini error: {type(e).__name__}") from e

        if self._consecutive_errors:
            logger.info("Gemini API recovered after %d consecutive errors.", self._consecutive_errors)
            self._consecutive_errors = 0
        logger.debug(f"Gemini detection result: is_fire={detection_result.is_fire}, confidence={detection_result.confidence}")
        return detection_result.is_fire

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Analyzes a frame for fire using the Gemini API (standalone ``gemini`` strategy).

        Never raises: a failed request is reported as "no fire" (an exception would abort the detection loop).
        After ``CONSECUTIVE_ERRORS_BEFORE_CRITICAL`` consecutive errors the outage is published to
        ``verifier_health`` (policy ``STANDALONE_POLICY``), and the next verdict publishes the recovery.
        Under ``hybrid_cloud`` the hybrid detector calls :meth:`verify` instead, so errors reach the
        ``VERIFIER_FAILURE_POLICY``.

        Args:
            frame: The input video frame (BGR numpy array).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was detected (``False`` on any error).
            - The original frame (annotation is not yet implemented).
        """
        try:
            verdict = await self.verify(frame)
        except VerifierError:
            if self._consecutive_errors >= CONSECUTIVE_ERRORS_BEFORE_CRITICAL:
                publish_verifier_status(
                    VerifierStatus(
                        available=False,
                        verifier=type(self).__name__,
                        reason=f"{self._consecutive_errors} consecutive errors",
                        policy=STANDALONE_POLICY,
                    )
                )
            return False, frame
        publish_verifier_status(
            VerifierStatus(
                available=True, verifier=type(self).__name__, reason="Gemini API recovered", policy=STANDALONE_POLICY
            )
        )
        return verdict, frame
