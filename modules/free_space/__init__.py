"""
VisionGuide AI - Module 08: Free-Space Analysis Subsystem
"""

from modules.free_space.models import RegionOccupancy, FreeSpaceAnalysisResult
from modules.free_space.interface import FreeSpaceAnalyzerInterface
from modules.free_space.analyzer import ImageSpaceFreeSpaceAnalyzer

__all__ = [
    "RegionOccupancy",
    "FreeSpaceAnalysisResult",
    "FreeSpaceAnalyzerInterface",
    "ImageSpaceFreeSpaceAnalyzer",
]
