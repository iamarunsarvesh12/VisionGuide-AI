"""
Backward-compatibility wrapper for AudioController pointing to OfflineAudioGuidance.
"""
from typing import Dict, Any, Optional
from modules.audio_guidance.guidance import OfflineAudioGuidance
from modules.audio_guidance.models import AudioResult
from modules.audio_guidance.tts_engine import TTSEngineInterface


class AudioController(OfflineAudioGuidance):
    """
    Legacy AudioController wrapper maintaining backwards compatibility while utilizing
    the new OfflineAudioGuidance core architecture.
    """

    def __init__(self, config_path: str = "config/config.yaml", tts_engine_override: Optional[TTSEngineInterface] = None):
        super().__init__(config_path=config_path, tts_engine_override=tts_engine_override)

    def speak(self, text: str, priority: int = 50) -> AudioResult:
        """Forward speak requests to speak_message."""
        return self.speak_message(message=text, priority=priority)

    def _purge_stale_directional_commands(self) -> None:
        """Legacy helper compatibility stub."""
        pass
