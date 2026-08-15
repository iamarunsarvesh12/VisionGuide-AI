from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import logging

logger = logging.getLogger("TTSEngine")


class TTSEngineInterface(ABC):
    """
    Abstract interface for Offline Text-to-Speech (TTS) Engines.
    Allows swappable backends (pyttsx3, SAPI5, MockTTSEngine, Android TTS).
    """

    @abstractmethod
    def initialize(self, volume: float = 1.0, speech_rate: int = 170, voice_name: str = "") -> bool:
        """Initialize TTS hardware engine resources."""
        pass

    @abstractmethod
    def speak(self, text: str) -> bool:
        """Speak the given text string synchronously."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Interrupt active speech synthesis."""
        pass

    @abstractmethod
    def set_volume(self, volume: float) -> bool:
        """Set output volume (0.0 to 1.0)."""
        pass

    @abstractmethod
    def set_rate(self, rate: int) -> bool:
        """Set speech rate (words per minute)."""
        pass

    @abstractmethod
    def get_voices(self) -> List[Dict[str, Any]]:
        """Return available voice profiles."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying engine resources."""
        pass


class Pyttsx3TTSEngine(TTSEngineInterface):
    """
    Offline TTS Engine implementation using pyttsx3 (SAPI5 on Windows).
    Ensures safe COM thread initialization and exception handling.
    """

    def __init__(self, backend: str = "sapi5"):
        self.backend = backend
        self._engine = None
        self._is_initialized = False
        self.volume = 1.0
        self.speech_rate = 170
        self.voice_name = ""

    def initialize(self, volume: float = 1.0, speech_rate: int = 170, voice_name: str = "") -> bool:
        """Initialize pyttsx3 engine and configure properties."""
        try:
            # Ensure COM is initialized for multithreaded Windows environment if pythoncom is available
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            import pyttsx3
            self._engine = pyttsx3.init(driverName=self.backend)
            self.set_volume(volume)
            self.set_rate(speech_rate)

            # Voice selection
            if voice_name:
                voices = self.get_voices()
                matched = False
                for v in voices:
                    if voice_name.lower() in v["name"].lower():
                        try:
                            self._engine.setProperty("voice", v["id"])
                            self.voice_name = v["name"]
                            matched = True
                            logger.info(f"Selected TTS voice: {v['name']}")
                            break
                        except Exception as ve:
                            logger.warning(f"Failed to set voice {voice_name}: {ve}")
                if not matched and voices:
                    self.voice_name = voices[0]["name"]
                    logger.info(f"Fallback to default TTS voice: {self.voice_name}")
            else:
                voices = self.get_voices()
                if voices:
                    self.voice_name = voices[0]["name"]

            self._is_initialized = True
            logger.info("Pyttsx3TTSEngine (SAPI5) initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Pyttsx3TTSEngine: {e}")
            self._is_initialized = False
            return False

    def speak(self, text: str) -> bool:
        """Speak text using pyttsx3."""
        if not self._is_initialized or not self._engine:
            logger.warning("Pyttsx3TTSEngine not initialized; skipping speech.")
            return False
        try:
            # Multi-threading COM check
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error during pyttsx3 speech synthesis: {e}")
            return False

    def stop(self) -> None:
        """Interrupt pyttsx3 speech."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception as e:
                logger.warning(f"Error stopping pyttsx3 engine: {e}")

    def set_volume(self, volume: float) -> bool:
        """Set speech volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        if self._engine:
            try:
                self._engine.setProperty("volume", self.volume)
                return True
            except Exception as e:
                logger.warning(f"Error setting volume: {e}")
        return False

    def set_rate(self, rate: int) -> bool:
        """Set speech rate (words per minute)."""
        self.speech_rate = max(50, min(400, rate))
        if self._engine:
            try:
                self._engine.setProperty("rate", self.speech_rate)
                return True
            except Exception as e:
                logger.warning(f"Error setting rate: {e}")
        return False

    def get_voices(self) -> List[Dict[str, Any]]:
        """Return available voices."""
        if not self._engine:
            return []
        try:
            voices = self._engine.getProperty("voices")
            return [
                {
                    "id": getattr(v, "id", str(i)),
                    "name": getattr(v, "name", f"Voice_{i}"),
                    "languages": getattr(v, "languages", []),
                }
                for i, v in enumerate(voices)
            ]
        except Exception as e:
            logger.warning(f"Error retrieving voices: {e}")
            return []

    def close(self) -> None:
        """Release engine resources."""
        self.stop()
        self._engine = None
        self._is_initialized = False


class MockTTSEngine(TTSEngineInterface):
    """
    Mock TTS Engine for fast, silent, deterministic unit testing and benchmarking.
    Records all generated messages in `spoken_messages` array without invoking audio hardware.
    """

    def __init__(self, simulate_latency_ms: float = 0.0):
        self.simulate_latency_ms = simulate_latency_ms
        self._is_initialized = False
        self.volume = 1.0
        self.speech_rate = 170
        self.voice_name = "Mock Voice"
        self.spoken_messages: List[str] = []
        self.spoken_texts: List[str] = []
        self.spoken_timestamps: List[float] = []

    def initialize(self, volume: float = 1.0, speech_rate: int = 170, voice_name: str = "") -> bool:
        self.volume = volume
        self.speech_rate = speech_rate
        self.voice_name = voice_name if voice_name else "Mock Voice"
        self._is_initialized = True
        self.spoken_messages.clear()
        self.spoken_texts.clear()
        self.spoken_timestamps.clear()
        return True

    def speak(self, text: str) -> bool:
        if not self._is_initialized:
            return False
        if self.simulate_latency_ms > 0:
            time.sleep(self.simulate_latency_ms / 1000.0)
        now = time.time()
        self.spoken_messages.append(text)
        self.spoken_texts.append(text)
        self.spoken_timestamps.append(now)
        return True

    def stop(self) -> None:
        pass

    def set_volume(self, volume: float) -> bool:
        self.volume = max(0.0, min(1.0, volume))
        return True

    def set_rate(self, rate: int) -> bool:
        self.speech_rate = max(50, min(400, rate))
        return True

    def get_voices(self) -> List[Dict[str, Any]]:
        return [{"id": "mock_voice_1", "name": "Mock SAPI5 Voice", "languages": ["en-US"]}]

    def close(self) -> None:
        self._is_initialized = False


# Alias FakeTTSEngine for backward compatibility
FakeTTSEngine = MockTTSEngine
