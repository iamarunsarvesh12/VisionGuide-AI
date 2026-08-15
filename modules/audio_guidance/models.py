from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class NavigationAudioMessage:
    """
    Structured Audio Message model representing a candidate spoken audio command.
    """
    command: str  # "LEFT", "RIGHT", "FORWARD", "STOP"
    text: str     # "Left", "Right", "Forward", "Stop"
    priority: int # 100 (STOP), 80 (LEFT/RIGHT), 50 (FORWARD)
    timestamp: float
    repeated: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert navigation audio message to a serializable dictionary."""
        return {
            "command": self.command,
            "text": self.text,
            "priority": self.priority,
            "timestamp": round(float(self.timestamp), 3),
            "repeated": self.repeated,
            "reason": self.reason,
        }


@dataclass
class AudioResult:
    """
    Structured Output Model returned after processing, queueing, or delivering audio.
    """
    success: bool
    command: Optional[str] = None
    message: Optional[str] = None  # Spoken string, e.g. "Left", "Stop"
    output_device: str = "Default Audio Output"
    latency_ms: float = 0.0
    timestamp: float = 0.0
    error: Optional[str] = None

    @property
    def text(self) -> Optional[str]:
        """Alias for backward compatibility with message."""
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert audio result instance to a serializable dictionary."""
        return {
            "success": self.success,
            "command": self.command,
            "message": self.message,
            "output_device": self.output_device,
            "latency_ms": round(float(self.latency_ms), 3),
            "timestamp": round(float(self.timestamp), 3),
            "error": self.error,
        }


@dataclass
class AudioConfiguration:
    """
    Type-safe configuration structure for Module 10 Offline Audio Guidance.
    """
    enabled: bool = True
    speech_rate: int = 170
    volume: float = 1.0
    repetition_interval: Dict[str, float] = field(
        default_factory=lambda: {
            "FORWARD": 2.0,
            "LEFT": 1.5,
            "RIGHT": 1.5,
            "STOP": 0.8,
        }
    )
    stop_repeat_interval: float = 0.8
    bluetooth_output_enabled: bool = True
    voice_name: str = ""
    language: str = "en-US"

    def to_dict(self) -> Dict[str, Any]:
        """Convert audio configuration to a serializable dictionary."""
        return {
            "enabled": self.enabled,
            "speech_rate": self.speech_rate,
            "volume": round(float(self.volume), 2),
            "repetition_interval": self.repetition_interval,
            "stop_repeat_interval": self.stop_repeat_interval,
            "bluetooth_output_enabled": self.bluetooth_output_enabled,
            "voice_name": self.voice_name,
            "language": self.language,
        }


@dataclass
class AudioCommand:
    """
    Backward-compatible command structure.
    """
    command: str
    text: str
    priority: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "text": self.text,
            "priority": self.priority,
            "timestamp": round(float(self.timestamp), 3),
        }


@dataclass
class AudioStatus:
    """
    Current operational status snapshot of the Audio Guidance engine.
    """
    is_initialized: bool = False
    is_speaking: bool = False
    current_command: Optional[str] = None
    current_text: Optional[str] = None
    queue_size: int = 0
    volume: float = 1.0
    speech_rate: int = 170
    last_spoken_timestamp: float = 0.0
    error_state: Optional[str] = None
    output_device: str = "Default Audio Output"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_initialized": self.is_initialized,
            "is_speaking": self.is_speaking,
            "current_command": self.current_command,
            "current_text": self.current_text,
            "queue_size": self.queue_size,
            "volume": round(float(self.volume), 2),
            "speech_rate": self.speech_rate,
            "last_spoken_timestamp": round(float(self.last_spoken_timestamp), 3),
            "error_state": self.error_state,
            "output_device": self.output_device,
        }
