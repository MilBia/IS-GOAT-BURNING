"""Implements fire detection using the Gemini Decisions API (``gemini-decision-1``)."""

import asyncio
import base64

import cv2
from google import genai
from google.genai import types
import numpy as np

from is_goat_burning.config import settings
from is_goat_burning.logger import get_logger

logger = get_logger("DecisionFireDetector")

_QUESTION_ID = "fire"
# Number of consecutive failed requests after which a CRITICAL message is logged (once per outage).
CONSECUTIVE_ERRORS_BEFORE_CRITICAL = 5


class DecisionFireDetector:
    """Detects fire with a calibrated yes/no ``noul`` question from the Decisions API.

    The model returns ``P(true)`` for the question instead of generated text, so it is
    cheap (input tokens only) and low-latency. A frame is reported as fire when
    ``P(fire) >= settings.decision.threshold``.

    API failures never raise (an exception would abort the detection loop); they are logged
    at ERROR level, treated as "no fire", and counted. After
    ``CONSECUTIVE_ERRORS_BEFORE_CRITICAL`` consecutive failures a CRITICAL message announces
    that fire detection is effectively disabled.
    """

    def __init__(self) -> None:
        """Initializes the DecisionFireDetector.

        Raises:
            ValueError: If the GEMINI__API_KEY is not set in the configuration.
        """
        if not settings.gemini.api_key:
            raise ValueError("GEMINI__API_KEY must be set when using the 'gemini_decision' or 'hybrid_decision' strategy.")

        # The Decisions API is only available on the v1beta API version.
        self.client = genai.Client(
            api_key=settings.gemini.api_key.get_secret_value(),
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.model_name = settings.decision.model
        self.threshold = settings.decision.threshold
        self.timeout_seconds = settings.decision.timeout_seconds
        self._question = {"type": "noul", "instructions": settings.decision.instructions}
        self._consecutive_errors = 0

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> list[dict[str, str]]:
        """Converts an OpenCV frame to a Decisions API image input block.

        Args:
            frame: The OpenCV image (numpy array) to convert.

        Returns:
            A single-element list with a base64-encoded JPEG ``image`` content block.

        Raises:
            ValueError: If the frame fails to encode to JPEG.
        """
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Failed to encode frame to JPEG.")
        data = base64.b64encode(buffer.tobytes()).decode("ascii")
        return [{"type": "image", "data": data, "mime_type": "image/jpeg"}]

    def _record_error(self) -> None:
        """Counts a failed request and escalates to CRITICAL once per outage."""
        self._consecutive_errors += 1
        if self._consecutive_errors == CONSECUTIVE_ERRORS_BEFORE_CRITICAL:
            logger.critical(
                "Decisions API failed %d times in a row; fire detection is effectively disabled until it recovers.",
                self._consecutive_errors,
            )

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Analyzes a frame for fire using the Decisions API.

        Args:
            frame: The input video frame (BGR numpy array).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was detected (``False`` on any API error).
            - The original frame (annotation is not implemented).
        """
        try:
            image_input = await asyncio.to_thread(self._preprocess_frame, frame)
            response = await asyncio.wait_for(
                self.client.aio.decisions.create(
                    model=self.model_name,
                    input=image_input,
                    questions={_QUESTION_ID: self._question},
                ),
                timeout=self.timeout_seconds,
            )
            probability = float(response.answers[_QUESTION_ID].noul)
        except TimeoutError:
            logger.error("Decisions API request timed out after %.1f s.", self.timeout_seconds)
            self._record_error()
            return False, frame
        except genai.errors.APIError as e:
            logger.error("Decisions API error: %s", e)
            self._record_error()
            return False, frame
        except Exception:
            logger.exception("Unexpected error during Decisions API fire detection.")
            self._record_error()
            return False, frame

        if self._consecutive_errors:
            logger.info("Decisions API recovered after %d consecutive errors.", self._consecutive_errors)
            self._consecutive_errors = 0

        is_fire = probability >= self.threshold
        logger.debug("Decision result: P(fire)=%.3f threshold=%.2f is_fire=%s", probability, self.threshold, is_fire)
        return is_fire, frame
