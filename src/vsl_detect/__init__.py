"""
Vietnamese Sign Language Detection System

A real-time sign language recognition system using deep learning and computer vision
techniques, specialized for Vietnamese sign language.
"""

__version__ = "1.0.0"
__author__ = "Vietnamese Sign Language Team"
__email__ = "letho1608@example.com"

from .core.detector import SignLanguageDetector
from .core.model import VSLModel
from .utils.config import Config

__all__ = [
    "SignLanguageDetector",
    "VSLModel", 
    "Config",
    "__version__",
]