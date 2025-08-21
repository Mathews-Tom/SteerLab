"""
SteerLab: Modern Python laboratory for activation steering research and development.

Build, test, and deploy steerable LLMs with preference-based personalization.
"""

__version__ = "0.1.0"
__author__ = "SteerLab Team"

from .core.model import SteerableModel
from .core.vectors import SteeringVectorManager

__all__ = ["SteerableModel", "SteeringVectorManager"]