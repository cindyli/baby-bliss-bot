"""
Bliss Engine - A language-independent rule-based module for Blissymbolics composition.

This module provides the core functionality for analyzing and composing Bliss symbols
according to Blissymbolics composition rules at the word level.
"""

from .bliss_engine import BlissEngine
from .symbol_classifier import SymbolClassifier
from .composer import BlissComposer
from .analyzer import BlissAnalyzer

__all__ = [
    'BlissEngine',
    'SymbolClassifier',
    'BlissComposer',
    'BlissAnalyzer',
]

__version__ = '1.0.0'
