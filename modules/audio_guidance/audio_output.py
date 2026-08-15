import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("AudioOutputDevice")


class AudioOutputDevice:
    """
    Audio Output Device Manager for Module 10 — Offline Audio Guidance.
    
    Detects active Windows default audio output devices (Laptop Speakers, Bluetooth Headphones,
    Bluetooth Earbuds), tracks availability, and handles device disconnects gracefully without
    crashing the perception or decision pipeline.
    """

    def __init__(self, use_system_default: bool = True, preferred_device_name: Optional[str] = None):
        self.use_system_default = use_system_default
        self.preferred_device_name = preferred_device_name
        self.device_name = "System Default Audio Output"
        self.is_bluetooth = False
        self.is_available = True
        self.error_message: Optional[str] = None

    def initialize(self) -> bool:
        """
        Detect active audio output hardware endpoint.
        """
        try:
            self.device_name, self.is_bluetooth = self._detect_output_device()
            self.is_available = True
            self.error_message = None
            logger.info(f"Audio Output Device initialized: {self.device_name} (Bluetooth: {self.is_bluetooth})")
            return True
        except Exception as e:
            logger.warning(f"Audio Output Device query failed, falling back to default: {e}")
            self.device_name = "Default Audio Output"
            self.is_bluetooth = False
            self.is_available = True
            self.error_message = str(e)
            return True  # Fail-safe mode

    def _detect_output_device(self) -> tuple[str, bool]:
        """
        Attempt to query the OS default render endpoint via sounddevice or PyAudio or Windows SAPI5/Win32 APIs.
        """
        # Try sounddevice if available
        try:
            import sounddevice as sd
            default_device_info = sd.query_devices(kind='output')
            name = default_device_info.get('name', 'Default Audio Output')
            is_bt = any(keyword in name.lower() for keyword in ['bluetooth', 'hands-free', 'bth', 'airpods', 'buds', 'headset', 'wireless'])
            return name, is_bt
        except Exception:
            pass

        # Try pyttsx3/SAPI5 voice output property query as secondary heuristic
        try:
            import pyttsx3
            engine = pyttsx3.init('sapi5')
            # Check default voice
            voice = engine.getProperty('voice')
            engine.stop()
            if voice:
                return f"SAPI5 Audio Endpoint ({voice.split('\\')[-1]})", False
        except Exception:
            pass

        return "Default System Speakers / Headphones", False

    def get_status(self) -> Dict[str, Any]:
        """
        Return active device telemetry.
        """
        return {
            "selected_output_device": self.device_name,
            "is_bluetooth": self.is_bluetooth,
            "is_available": self.is_available,
            "error_message": self.error_message,
        }

    def check_health(self) -> bool:
        """
        Verify output endpoint is still operational.
        """
        # Re-check device status
        try:
            name, is_bt = self._detect_output_device()
            self.device_name = name
            self.is_bluetooth = is_bt
            self.is_available = True
            return True
        except Exception as e:
            self.is_available = False
            self.error_message = f"Audio output disconnect detected: {e}"
            logger.error(self.error_message)
            return False
