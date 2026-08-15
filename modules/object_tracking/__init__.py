"""
VisionGuide AI - Module 04: Object Tracking Subsystem
"""

from modules.object_tracking.interface import ObjectTrackerInterface, Track
from modules.object_tracking.tracker import BoTSORTTracker

__all__ = ["ObjectTrackerInterface", "Track", "BoTSORTTracker"]
