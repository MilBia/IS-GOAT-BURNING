"""Implements fire detection using Google's Gemini API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import cv2
from google import genai
from google.genai import types
import numpy as np
from pydantic import BaseModel
from pydantic import Field

from is_goat_burning.config import settings
from is_goat_burning.logger import get_logger

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse

logger = get_logger("GeminiFireDetector")


class FireDetectionResponse(BaseModel):
    """Structured output model for Gemini fire detection."""

    is_fire: bool = Field(description="True if fire is visible in the image, False otherwise.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")


class GeminiFireDetector:
    """Detects fire using the Google Gemini API."""

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

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Analyzes a frame for fire using the Gemini API.

        Args:
            frame: The input video frame (BGR numpy array).

        Returns:
            A tuple containing:
            - A boolean indicating if fire was detected.
            - The original frame (annotation is not yet implemented).
        """
        try:
            loop = asyncio.get_running_loop()
            image_bytes = await loop.run_in_executor(None, self._preprocess_frame, frame)

            response: GenerateContentResponse = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[
                    self.prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FireDetectionResponse,
                ),
            )

            # Parse the response
            # The SDK should return a parsed object if response_schema is provided,
            # but let's be safe and check how to access it.
            # Based on docs, `parsed` attribute should contain the model instance.
            detection_result: FireDetectionResponse | None = response.parsed

            if detection_result:
                is_fire = detection_result.is_fire
                logger.debug(f"Gemini detection result: is_fire={is_fire}, confidence={detection_result.confidence}")
                return is_fire, frame

            logger.warning("Gemini response did not contain a valid parsed result.")
            return False, frame

        except (genai.errors.ClientError, ValueError) as e:
            logger.warning(f"Gemini API client error: {e}")
            return False, frame
        except Exception:
            logger.exception("Unexpected error during Gemini fire detection.")
            return False, frame
