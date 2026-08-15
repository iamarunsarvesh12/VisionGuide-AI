from typing import Dict, Tuple, Optional
from modules.decision_engine.models import NavigationCommand


class CommandMapper:
    """
    Translates structured navigation commands (LEFT, RIGHT, FORWARD, STOP)
    into spoken text strings and priority values. Rejects arbitrary or unverified commands.
    """

    DEFAULT_MAPPINGS: Dict[str, Tuple[str, int]] = {
        NavigationCommand.LEFT.value: ("Left", 50),
        NavigationCommand.RIGHT.value: ("Right", 50),
        NavigationCommand.FORWARD.value: ("Forward", 40),
        NavigationCommand.STOP.value: ("Stop", 100),
    }

    def __init__(self, custom_priorities: Optional[Dict[str, int]] = None):
        self.mappings = {}
        for cmd, (txt, default_prio) in self.DEFAULT_MAPPINGS.items():
            prio = custom_priorities.get(cmd, default_prio) if custom_priorities else default_prio
            self.mappings[cmd] = (txt, prio)

    def is_valid_command(self, command_str: str) -> bool:
        """Check if command string is a recognized navigation command."""
        if not command_str or not isinstance(command_str, str):
            return False
        return command_str.upper() in self.mappings

    def map_command(self, command_str: str) -> Tuple[str, int]:
        """
        Map command string to (speech_text, priority).
        Raises ValueError if command is invalid.
        """
        if not command_str or not isinstance(command_str, str):
            raise ValueError(f"Invalid command type or empty command: {command_str}")

        cmd_upper = command_str.upper()
        if cmd_upper not in self.mappings:
            raise ValueError(f"Unsupported navigation command: '{command_str}'. Supported commands: {list(self.mappings.keys())}")

        return self.mappings[cmd_upper]
