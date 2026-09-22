from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import sys
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


class NativeSAPI5TTSEngine(TTSEngineInterface):
    """
    High-performance native Windows SAPI5 TTS Engine using direct Win32 COM dispatch.
    Provides sub-millisecond dispatch, accurate speech rate & volume scaling,
    cancellable asynchronous speech loops, and instant STOP interrupt purges
    without the event-loop swallowing bugs present in third-party wrappers.
    """

    def __init__(self):
        self._speaker = None
        self._stop_event = None
        self._is_initialized = False
        self.volume = 1.0
        self.speech_rate = 170
        self.voice_name = ""
        self._cached_voices: List[Dict[str, Any]] = []

    def initialize(self, volume: float = 1.0, speech_rate: int = 170, voice_name: str = "") -> bool:
        """Initialize native SAPI5 COM object and discover voices."""
        import threading
        self._stop_event = threading.Event()
        self.volume = max(0.0, min(1.0, volume))
        self.speech_rate = max(50, min(400, speech_rate))
        self.voice_name = voice_name

        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            import win32com.client
            self._speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.set_volume(self.volume)
            self.set_rate(self.speech_rate)

            # Discover installed voices
            self._cached_voices = []
            voices = self._speaker.GetVoices()
            for i in range(voices.Count):
                v = voices.Item(i)
                self._cached_voices.append({
                    "id": str(i),
                    "name": v.GetDescription(),
                    "languages": ["en-US"],
                })

            if self.voice_name:
                for i, v in enumerate(self._cached_voices):
                    if self.voice_name.lower() in v["name"].lower():
                        self._speaker.Voice = voices.Item(i)
                        self.voice_name = v["name"]
                        break

            if not self.voice_name and self._cached_voices:
                self.voice_name = self._cached_voices[0]["name"]

            self._is_initialized = True
            logger.info("NativeSAPI5TTSEngine (Windows Direct SAPI5) initialized successfully.")
            return True
        except Exception as e:
            logger.warning(f"NativeSAPI5TTSEngine initialization failed: {e}")
            self._is_initialized = False
            return False

    def speak(self, text: str) -> bool:
        """Speak text with fine-grained interrupt checking."""
        if not self._is_initialized or not self._speaker:
            return False
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            if self._stop_event:
                self._stop_event.clear()

            # Ensure volume & rate are applied
            self.set_volume(self.volume)
            self.set_rate(self.speech_rate)

            # SVSFlagsAsync = 1: Speak asynchronously and wait in poll loop so we can interrupt instantly
            self._speaker.Speak(text, 1)

            # Poll until speech finishes or stop is requested
            while self._stop_event and not self._stop_event.is_set():
                if self._speaker.WaitUntilDone(30):
                    return True

            # Interrupted / Purged
            try:
                # SVSFPurgeBeforeSpeak = 2, SVSFlagsAsync = 1
                self._speaker.Speak("", 1 | 2)
            except Exception:
                pass
            return True

        except Exception as e:
            logger.error(f"Native SAPI5 speech synthesis exception: {e}")
            return False

    def stop(self) -> None:
        """Interrupt active speech synthesis immediately."""
        if self._stop_event:
            self._stop_event.set()
        if self._speaker:
            try:
                # SVSFPurgeBeforeSpeak = 2, SVSFlagsAsync = 1
                self._speaker.Speak("", 1 | 2)
            except Exception as e:
                logger.warning(f"Error purging SAPI5 speech: {e}")

    def set_volume(self, volume: float) -> bool:
        """Set output volume (0.0 to 1.0 mapped to 0-100)."""
        self.volume = max(0.0, min(1.0, volume))
        if self._speaker:
            try:
                self._speaker.Volume = int(self.volume * 100)
            except Exception as e:
                logger.warning(f"Error setting SAPI5 volume: {e}")
        return True

    def set_rate(self, rate: int) -> bool:
        """Set speech rate (words per minute mapped to SAPI5 -10 to +10)."""
        self.speech_rate = max(50, min(400, rate))
        if self._speaker:
            try:
                # Standard conversion: 170 WPM is ~0. -10 is ~50 WPM, +10 is ~400 WPM.
                sapi_rate = int((self.speech_rate - 170) / 23.0)
                self._speaker.Rate = max(-10, min(10, sapi_rate))
            except Exception as e:
                logger.warning(f"Error setting SAPI5 rate: {e}")
        return True

    def get_voices(self) -> List[Dict[str, Any]]:
        """Return cached voice profiles."""
        return self._cached_voices

    def close(self) -> None:
        """Release underlying engine resources."""
        self.stop()
        self._speaker = None
        self._is_initialized = False


class Pyttsx3TTSEngine(TTSEngineInterface):
    """
    Offline TTS Engine implementation using pyttsx3 with automatic native SAPI5 dispatch
    on Windows and cross-platform fallback handling.
    """

    def __init__(self, backend: str = "sapi5"):
        import threading
        self.backend = backend
        self._is_initialized = False
        self.volume = 1.0
        self.speech_rate = 170
        self.voice_name = ""
        self._cached_voices: List[Dict[str, Any]] = []
        self._native_engine: Optional[NativeSAPI5TTSEngine] = None
        self._local = threading.local()

    def initialize(self, volume: float = 1.0, speech_rate: int = 170, voice_name: str = "") -> bool:
        """Initialize TTS backend, preferring native SAPI5 on Windows for maximum reliability."""
        self.volume = max(0.0, min(1.0, volume))
        self.speech_rate = max(50, min(400, speech_rate))
        self.voice_name = voice_name

        # On Windows, try native SAPI5 first for flawless multi-command delivery
        if sys.platform == "win32" or self.backend.lower() in ["sapi5", "windows"]:
            native_eng = NativeSAPI5TTSEngine()
            if native_eng.initialize(volume=self.volume, speech_rate=self.speech_rate, voice_name=self.voice_name):
                self._native_engine = native_eng
                self._cached_voices = native_eng.get_voices()
                self._is_initialized = True
                logger.info("Pyttsx3TTSEngine initialized via Native Windows SAPI5 backend.")
                return True

        # Fallback to standard pyttsx3
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            import pyttsx3
            probe = pyttsx3.init(driverName=self.backend)
            self._cached_voices = []
            try:
                raw_voices = probe.getProperty("voices")
                for i, v in enumerate(raw_voices):
                    self._cached_voices.append({
                        "id": getattr(v, "id", str(i)),
                        "name": getattr(v, "name", f"Voice_{i}"),
                        "languages": getattr(v, "languages", []),
                    })
            except Exception as ve:
                logger.warning(f"Error querying voices during probe: {ve}")

            if voice_name:
                for v in self._cached_voices:
                    if voice_name.lower() in v["name"].lower():
                        self.voice_name = v["name"]
                        break

            if not self.voice_name and self._cached_voices:
                self.voice_name = self._cached_voices[0]["name"]

            try:
                probe.stop()
            except Exception:
                pass
            del probe

            self._is_initialized = True
            logger.info("Pyttsx3TTSEngine initialized with driver backend.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Pyttsx3TTSEngine: {e}")
            self._is_initialized = False
            return False

    def speak(self, text: str) -> bool:
        """Speak text synchronously or through native backend."""
        if not self._is_initialized:
            logger.warning("Pyttsx3TTSEngine not initialized; skipping speech.")
            return False

        if self._native_engine:
            return self._native_engine.speak(text)

        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass
            import pyttsx3
            eng = pyttsx3.init(driverName=self.backend)
            eng.setProperty("volume", self.volume)
            eng.setProperty("rate", self.speech_rate)
            if self.voice_name:
                for v in self._cached_voices:
                    if self.voice_name.lower() in v["name"].lower():
                        eng.setProperty("voice", v["id"])
                        break
            eng.say(text)
            eng.runAndWait()
            try:
                eng.stop()
            except Exception:
                pass
            del eng
            return True
        except Exception as e:
            logger.error(f"Error during pyttsx3 speech synthesis: {e}")
            return False

    def stop(self) -> None:
        """Interrupt active speech."""
        if self._native_engine:
            self._native_engine.stop()
            return
        if hasattr(self._local, "engine") and self._local.engine:
            try:
                self._local.engine.stop()
            except Exception as e:
                logger.warning(f"Error stopping pyttsx3 engine: {e}")

    def set_volume(self, volume: float) -> bool:
        """Set speech volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        if self._native_engine:
            return self._native_engine.set_volume(self.volume)
        return True

    def set_rate(self, rate: int) -> bool:
        """Set speech rate (words per minute)."""
        self.speech_rate = max(50, min(400, rate))
        if self._native_engine:
            return self._native_engine.set_rate(self.speech_rate)
        return True

    def get_voices(self) -> List[Dict[str, Any]]:
        """Return available voices."""
        if self._native_engine:
            return self._native_engine.get_voices()
        return self._cached_voices

    def close(self) -> None:
        """Release engine resources."""
        self.stop()
        if self._native_engine:
            self._native_engine.close()
            self._native_engine = None
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
