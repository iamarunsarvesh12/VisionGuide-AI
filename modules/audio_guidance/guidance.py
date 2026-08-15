import time
import os
import sys
import logging
import threading
import queue
import yaml
from typing import Dict, Any, Optional, Union, List

from modules.audio_guidance.models import (
    NavigationAudioMessage,
    AudioResult,
    AudioConfiguration,
    AudioStatus,
)
from modules.audio_guidance.interface import AudioGuidanceInterface
from modules.audio_guidance.tts_engine import (
    TTSEngineInterface,
    Pyttsx3TTSEngine,
    MockTTSEngine,
)
from modules.audio_guidance.audio_output import AudioOutputDevice

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Module Logger setup
logger = logging.getLogger("OfflineAudioGuidance")
logger.setLevel(logging.INFO)
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler("logs/audio_guidance.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s"))
    logger.addHandler(f_handler)


VALID_COMMANDS = {"LEFT", "RIGHT", "FORWARD", "STOP"}

COMMAND_TEXT_MAP = {
    "LEFT": "Left",
    "RIGHT": "Right",
    "FORWARD": "Forward",
    "STOP": "Stop",
}

DEFAULT_PRIORITY_MAP = {
    "STOP": 100,
    "LEFT": 80,
    "RIGHT": 80,
    "FORWARD": 50,
}

PRIORITY_LEVEL_MAP = {
    "CRITICAL": 100,
    "HIGH": 80,
    "NORMAL": 50,
    "LOW": 30,
}


class OfflineAudioGuidance(AudioGuidanceInterface):
    """
    Module 10 — Offline Audio Guidance for VisionGuide AI.
    
    Acts as the deterministic spoken audio output adapter for Module 09 Decision Engine outputs.
    Enforces safety priority arbitration, immediate STOP safety overrides, command repetition
    suppression, non-blocking asynchronous TTS scheduling, and Bluetooth audio device status.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        tts_engine_override: Optional[TTSEngineInterface] = None,
        audio_output_override: Optional[AudioOutputDevice] = None,
    ):
        self.config_path = config_path
        self.config = AudioConfiguration()
        self.raw_config: Dict[str, Any] = {}
        self._is_initialized = False

        # Sub-components
        self.tts_engine: Optional[TTSEngineInterface] = tts_engine_override
        self.audio_output: Optional[AudioOutputDevice] = audio_output_override

        # Multi-threading & Queues
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=10)
        self._lock = threading.Lock()
        self._item_counter = 0

        # State tracking
        self.is_speaking = False
        self.current_command: Optional[str] = None
        self.last_command: Optional[str] = None
        self.last_audio_result: Optional[AudioResult] = None
        self.last_spoken_time_per_command: Dict[str, float] = {}

        # Telemetry
        self.total_commands_received = 0
        self.total_commands_spoken = 0
        self.total_suppressed = 0
        self.total_stop_overrides = 0
        self.total_errors = 0
        self.error_state: Optional[str] = None

    def initialize(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize parameters, TTS backend, audio output, and worker thread."""
        try:
            cfg_to_apply = {}
            if config_dict is not None:
                cfg_to_apply = config_dict
            elif os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    full_cfg = yaml.safe_load(f)
                    if full_cfg and "audio_guidance" in full_cfg:
                        cfg_to_apply = full_cfg["audio_guidance"]

            self._parse_config(cfg_to_apply)

            # Initialize Audio Output Device
            if self.audio_output is None:
                use_sys_def = self.raw_config.get("output", {}).get("use_system_default", True)
                self.audio_output = AudioOutputDevice(use_system_default=use_sys_def)
            self.audio_output.initialize()

            # Initialize TTS Engine
            if self.tts_engine is None:
                tts_cfg = self.raw_config.get("tts", {})
                engine_name = tts_cfg.get("engine", "sapi5").lower()
                legacy_type = str(self.raw_config.get("tts_engine", "pyttsx3")).lower()

                if engine_name == "mock" or legacy_type == "mock" or legacy_type == "fake":
                    self.tts_engine = MockTTSEngine()
                else:
                    backend_str = "sapi5" if engine_name in ["sapi5", "pyttsx3"] else engine_name
                    self.tts_engine = Pyttsx3TTSEngine(backend=backend_str)

            tts_ok = self.tts_engine.initialize(
                volume=self.config.volume,
                speech_rate=self.config.speech_rate,
                voice_name=self.config.voice_name,
            )

            if not tts_ok:
                self.error_state = "TTS initialization failed"
                logger.error("TTS Engine initialization failed; running in degraded silent mode.")

            # Start worker thread
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

            self._is_initialized = True
            logger.info(
                f"OfflineAudioGuidance initialized successfully (Output Device: {self.audio_output.device_name}, "
                f"Bluetooth: {self.audio_output.is_bluetooth})."
            )
            return True
        except Exception as e:
            logger.error(f"OfflineAudioGuidance initialization failed: {e}")
            self.error_state = str(e)
            self._is_initialized = False
            return False

    def _parse_config(self, cfg: Dict[str, Any]) -> None:
        """Parse structured config dict into AudioConfiguration."""
        self.raw_config = cfg
        self.config.enabled = cfg.get("enabled", True)

        tts_cfg = cfg.get("tts", {})
        self.config.speech_rate = tts_cfg.get("rate", cfg.get("speech_rate", 170))
        self.config.volume = float(tts_cfg.get("volume", cfg.get("volume", 1.0)))
        self.config.voice_name = tts_cfg.get("voice", "")

        rep_cfg = cfg.get("repetition", {})
        if rep_cfg:
            intervals = {
                "FORWARD": float(rep_cfg.get("forward_interval_sec", 2.0)),
                "LEFT": float(rep_cfg.get("left_interval_sec", 1.5)),
                "RIGHT": float(rep_cfg.get("right_interval_sec", 1.5)),
                "STOP": float(rep_cfg.get("stop_interval_sec", 0.8)),
            }
            self.config.repetition_interval = intervals
            self.config.stop_repeat_interval = float(rep_cfg.get("stop_interval_sec", 0.8))

        prio_cfg = cfg.get("priority", cfg.get("priorities", {}))
        if prio_cfg:
            for k, v in prio_cfg.items():
                k_upper = str(k).upper()
                if k_upper in DEFAULT_PRIORITY_MAP:
                    DEFAULT_PRIORITY_MAP[k_upper] = int(v)

    def is_available(self) -> bool:
        """Return True if initialized and audio engine is operational."""
        return self._is_initialized and (self.audio_output is None or self.audio_output.is_available)

    def speak_command(self, command: Union[str, Any], context: Optional[Dict[str, Any]] = None) -> AudioResult:
        """
        Primary adapter method for Module 09 DecisionResult / NavigationCommand.
        Extracts command, validates, applies priority & repetition rules, dispatches speech.
        """
        t0 = time.perf_counter()
        now = time.time()

        if not self._is_initialized:
            self.initialize()

        with self._lock:
            self.total_commands_received += 1

        # Step 1: Extract Command string & Decision Metadata
        cmd_str = ""
        reason = None

        if hasattr(command, "command"):  # DecisionResult object
            cmd_str = str(getattr(command, "command", "")).upper()
            reason = getattr(command, "reason", None)
        elif isinstance(command, str):
            cmd_str = command.strip().upper()
            if context and isinstance(context, dict):
                reason = context.get("reason")
        elif hasattr(command, "value"):  # NavigationCommand Enum
            cmd_str = str(command.value).upper()
        else:
            cmd_str = str(command).upper()

        # Step 2: Validate Command
        if cmd_str not in VALID_COMMANDS:
            err_msg = f"Invalid command '{cmd_str}'. Must be one of {sorted(list(VALID_COMMANDS))}."
            logger.warning(f"Audio Guidance Command Validation Failed: {err_msg}")
            with self._lock:
                self.total_errors += 1
            res = AudioResult(
                success=False,
                command=cmd_str,
                message=None,
                output_device=self.audio_output.device_name if self.audio_output else "Unknown",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                timestamp=now,
                error=err_msg,
            )
            self.last_audio_result = res
            return res

        spoken_text = COMMAND_TEXT_MAP[cmd_str]
        priority = DEFAULT_PRIORITY_MAP.get(cmd_str, 50)
        rep_interval = self.config.repetition_interval.get(cmd_str, 1.5)

        # Step 3: Check Repetition Suppression vs Command Change
        is_command_change = (self.last_command is not None and self.last_command != cmd_str)
        time_since_last_spoken = now - self.last_spoken_time_per_command.get(cmd_str, 0.0)

        if not is_command_change and time_since_last_spoken < rep_interval:
            with self._lock:
                self.total_suppressed += 1
            logger.debug(f"Suppressed repeated audio command '{cmd_str}' ({time_since_last_spoken:.2f}s < {rep_interval:.2f}s)")
            res = AudioResult(
                success=True,
                command=cmd_str,
                message=spoken_text,
                output_device=self.audio_output.device_name if self.audio_output else "Unknown",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                timestamp=now,
                error="Suppressed (repetition interval)",
            )
            self.last_audio_result = res
            return res

        # Create NavigationAudioMessage model
        msg_model = NavigationAudioMessage(
            command=cmd_str,
            text=spoken_text,
            priority=priority,
            timestamp=now,
            repeated=not is_command_change,
            reason=reason,
        )

        # Step 4: STOP Safety Override Rule
        if cmd_str == "STOP":
            self._execute_stop_override()

        # Step 5: Enqueue audio message
        self.last_command = cmd_str
        self.last_spoken_time_per_command[cmd_str] = now

        res = self._enqueue_message(msg_model, t0)
        self.last_audio_result = res
        return res

    def speak_message(self, message: str, priority: Union[str, int] = "NORMAL") -> AudioResult:
        """Speak arbitrary text message with given priority."""
        t0 = time.perf_counter()
        now = time.time()

        if isinstance(priority, str):
            prio_int = PRIORITY_LEVEL_MAP.get(priority.upper(), 50)
        else:
            prio_int = int(priority)

        msg_model = NavigationAudioMessage(
            command="CUSTOM",
            text=message,
            priority=prio_int,
            timestamp=now,
            repeated=False,
            reason="Arbitrary message request",
        )

        res = self._enqueue_message(msg_model, t0)
        self.last_audio_result = res
        return res

    def _execute_stop_override(self) -> None:
        """Safety Priority Rule: Immediately interrupt active speech and purge pending non-STOP queues."""
        with self._lock:
            self.total_stop_overrides += 1

        logger.info("[STOP PRIORITY] Emergency STOP received. Interrupting lower priority audio & clearing queue.")

        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                logger.warning(f"Error interrupting TTS engine for STOP override: {e}")

        # Purge pending queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _enqueue_message(self, msg: NavigationAudioMessage, start_perf_time: float) -> AudioResult:
        """Enqueue message to priority queue for background worker execution."""
        try:
            with self._lock:
                self._item_counter += 1
                cnt = self._item_counter

            # Invert priority (-priority) so higher priority executes first in min-heap
            item = (-msg.priority, msg.timestamp, cnt, msg)

            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Discard oldest and put new
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._queue.put_nowait(item)
                except Exception:
                    pass

            lat_ms = (time.perf_counter() - start_perf_time) * 1000.0
            return AudioResult(
                success=True,
                command=msg.command,
                message=msg.text,
                output_device=self.audio_output.device_name if self.audio_output else "Default Audio Output",
                latency_ms=lat_ms,
                timestamp=msg.timestamp,
            )
        except Exception as e:
            logger.error(f"Failed to enqueue audio message: {e}")
            with self._lock:
                self.total_errors += 1
            return AudioResult(
                success=False,
                command=msg.command,
                message=msg.text,
                output_device=self.audio_output.device_name if self.audio_output else "Unknown",
                latency_ms=(time.perf_counter() - start_perf_time) * 1000.0,
                timestamp=msg.timestamp,
                error=str(e),
            )

    def _worker_loop(self) -> None:
        """Worker thread loop consuming queue and dispatching to TTS engine."""
        while not self._stop_event.is_set():
            try:
                try:
                    prio, ts, cnt, msg = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self._deliver_speech(msg)
                self._queue.task_done()
            except Exception as e:
                logger.error(f"Unhandled error in audio worker loop: {e}")
                time.sleep(0.05)

    def _deliver_speech(self, msg: NavigationAudioMessage) -> bool:
        """Synthesize and speak audio message."""
        if not self.tts_engine:
            return False

        with self._lock:
            self.is_speaking = True
            self.current_command = msg.command

        logger.info(f"AUDIO OUT -> Command: {msg.command} | Spoken Text: \"{msg.text}\" | Priority: {msg.priority}")
        t0 = time.perf_counter()
        ok = False
        try:
            ok = self.tts_engine.speak(msg.text)
        except Exception as e:
            logger.error(f"TTS Engine speech exception: {e}")
            with self._lock:
                self.total_errors += 1
            ok = False

        duration_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self.is_speaking = False
            self.current_command = None
            if ok:
                self.total_commands_spoken += 1

        logger.info(f"AUDIO COMPLETED -> Text: \"{msg.text}\" | Success: {ok} | Duration: {duration_ms:.1f}ms")
        return ok

    def stop(self) -> None:
        """Interrupt active speech and clear queues."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                logger.warning(f"Error stopping TTS engine: {e}")

    def reset(self) -> None:
        """Reset internal telemetry, queues, and command history."""
        self.stop()
        with self._lock:
            self.current_command = None
            self.last_command = None
            self.last_audio_result = None
            self.last_spoken_time_per_command.clear()
            self.total_commands_received = 0
            self.total_commands_spoken = 0
            self.total_suppressed = 0
            self.total_stop_overrides = 0
            self.total_errors = 0
            self.error_state = None
        logger.info("OfflineAudioGuidance state reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """Return telemetry counters and system metrics."""
        with self._lock:
            return {
                "is_initialized": self._is_initialized,
                "is_speaking": self.is_speaking,
                "total_commands_received": self.total_commands_received,
                "total_commands_spoken": self.total_commands_spoken,
                "total_suppressed": self.total_suppressed,
                "total_stop_overrides": self.total_stop_overrides,
                "total_errors": self.total_errors,
                "queue_size": self._queue.qsize(),
                "output_device": self.audio_output.device_name if self.audio_output else "Unknown",
                "is_bluetooth": self.audio_output.is_bluetooth if self.audio_output else False,
            }

    def get_last_audio(self) -> Optional[AudioResult]:
        """Return the most recent AudioResult generated."""
        return self.last_audio_result

    def close(self) -> None:
        """Clean shutdown of background threads and hardware resources."""
        self._stop_event.set()
        self.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self.tts_engine:
            self.tts_engine.close()
        self._is_initialized = False
        logger.info("OfflineAudioGuidance shut down cleanly.")
