"""
Main Bliss Engine module - unified interface for Blissymbolics operations.

Provides integrated functionality for:
- Use Case 1: Look up a symbol ID in the Bliss dictionary
- Use Case 2: Look up a word composition in the Bliss dictionary
- Use Case 3: Compose new Bliss words from semantic specifications
"""

from typing import List, Dict, Union
from .analyzer import BlissAnalyzer
from .composer import BlissComposer
from .symbol_classifier import SymbolClassifier


class BlissEngine:
    """
    Main engine for Blissymbolics composition and analysis.

    Supports three primary use cases:
    1. Look up a symbol ID in the Bliss dictionary
    2. Look up a word composition in the Bliss dictionary
    3. Compose new Bliss words from semantic specifications
    """

    def __init__(self, bliss_dict: Dict):
        """
        Initialize the Bliss Engine with a Bliss dictionary.

        Args:
            bliss_dict: Dict of Bliss symbol definitions (typically loaded from bliss_dict_multi_langs.json)

        Raises:
            TypeError: If bliss_dict is not a dictionary
        """
        if not isinstance(bliss_dict, dict):
            raise TypeError("bliss_dict must be a dictionary")

        self.bliss_dict = bliss_dict
        self.analyzer = BlissAnalyzer(bliss_dict)
        self.composer = BlissComposer(bliss_dict)
        self.classifier = SymbolClassifier(bliss_dict)

    # ============================================================================
    # USE CASE 1: Look up a symbol ID in the Bliss dictionary
    # ============================================================================

    def get_symbol_glosses(self, symbol_id: int, language: str = "en") -> Dict:
        """
        Look up a symbol ID and return its glosses and explanation.

        Use Case 1: Given a symbol ID, look up in the Bliss dictionary to find
        the symbol and return its glosses and explanation. If not found, returns an error.

        Args:
            symbol_id: The ID of the Bliss symbol (integer)
            language: ISO 639-1 language code (default: "en" for English)

        Returns:
            Dict containing:
            - id: The symbol ID
            - glosses: List of glosses in the requested language
            - explanation: Explanation of the symbol
            - isCharacter: Whether this is a Bliss character or composed word
            - error: Error message if symbol not found
        """
        return self.analyzer.get_symbol_glosses(symbol_id, language)

    # ============================================================================
    # USE CASE 2: Analyze new compositions and extract semantic information
    # ============================================================================

    def lookup_composition(self, composition: Union[List[str], List[int]],
                           language: str = "en") -> Dict:
        """
        Look up a word composition in the Bliss dictionary.

        Use Case 2: Given a word composition, first look up in the Bliss dictionary
        to find if the composition is already in the dictionary by checking the
        "composition" values.

        If it exists, returns the symbol ID with its glosses and explanation.
        If it doesn't exist, returns its semantic meaning.

        Rendering elements such as "/" and ";" are automatically ignored when
        comparing compositions.

        Example output for existing composition:
        {
            "composition": [14647, 14905, 9011],
            "is_existing_symbol": true,
            "symbol_id": 12345,
            "glosses": ["many buildings"],
            "explanation": "..."
        }

        Example output for new composition:
        {
            "composition": [14647, 14905, 24920, 9011],
            "is_existing_symbol": false,
            "classifier": 14905,
            "classifier_info": "building",
            "specifiers": [24920],
            "specifier_info": ["medicine"],
            "indicators": [9011],
            "modifiers": [14647],
            "semantics": {"NUMBER": "plural", "QUANTIFIER": "many"}
        }

        Args:
            composition: List of symbol IDs (may include rendering elements)
            language: ISO 639-1 language code for glosses

        Returns:
            Dict with is_existing_symbol flag and appropriate data
        """
        return self.analyzer.lookup_composition(composition, language)

    def analyze_composition(self, composition: Union[List[str], List[int]],
                            language: str = "en") -> Dict:
        """
        Analyze a new Bliss word composition and extract semantic meaning.

        Use Case 2: Analyzes component IDs and returns their combined semantic
        information including classifier, specifiers, indicators, and modifiers.

        Example output for "many hospitals":
        {
            "classifier": 14905,
            "specifiers": [24920],
            "indicators": [9011],
            "modifiers": [14647],
            "classifier_info": "building",
            "specifier_info": ["medicine"],
            "semantics": {
                "NUMBER": "plural",
                "QUANTIFIER": "many"
            }
        }

        Args:
            composition: List of symbol IDs in composition order
            language: ISO 639-1 language code for glosses

        Returns:
            Dict with semantic analysis including:
            - classifier: The main semantic category
            - specifiers: Refining symbols
            - indicators: Grammatical information
            - modifiers: Meaning modifiers (prefixes/suffixes)
            - semantics: Combined semantic properties
        """
        return self.analyzer.analyze_composition(composition, language)

    # ============================================================================
    # USE CASE 3: Compose new Bliss words from semantic specifications
    # ============================================================================

    def compose_from_semantic(self, semantic_spec: Dict) -> Dict:
        """
        Compose a new Bliss word from a semantic specification.

        Use Case 3: Takes a semantic JSON and returns a Bliss composition.

        Input example:
        {
            "classifier": "building",
            "specifiers": ["medicine"],
            "semantics": {
                "NUMBER": "plural",
                "QUANTIFIER": "many"
            }
        }

        Output example:
        {
            "composition": ["14647", "14905", "24920", "9011"],
            "original_spec": {...}
        }

        Args:
            semantic_spec: Dict with:
                - classifier: str (gloss for the semantic category)
                - specifiers: List[str] (glosses refining the meaning, optional)
                - semantics: Dict (semantic modifications, optional)
                  Format: {"TYPE": "value", "TYPE2": "value2", ...}

        Returns:
            Dict with:
            - composition: List of symbol IDs
            - errors: Any composition errors
            - warnings: Non-critical issues
        """
        return self.composer.compose_from_semantic_spec(semantic_spec)

    # ============================================================================
    # Utility methods
    # ============================================================================

    def get_symbol_info(self, symbol_id: str) -> Dict:
        """
        Get comprehensive information about a symbol.

        Args:
            symbol_id: The symbol ID

        Returns:
            Dict with symbol details: type, glosses, explanation, semantics
        """
        return self.classifier.get_symbol_info(symbol_id)

    def classify_symbols(self, symbol_ids: List[str]) -> Dict:
        """
        Classify symbols by their functional roles.

        Args:
            symbol_ids: List of symbol IDs

        Returns:
            Dict with roles: classifier, specifiers, indicators, modifiers
        """
        return self.classifier.classify_composition(symbol_ids)

    def is_classifier(self, symbol_id: str) -> bool:
        """Check if a symbol can be a classifier."""
        return self.classifier.is_classifier(symbol_id)

    def is_modifier(self, symbol_id: str) -> bool:
        """Check if a symbol is a modifier."""
        return self.classifier.is_modifier(symbol_id)

    def is_indicator(self, symbol_id: str) -> bool:
        """Check if a symbol is an indicator."""
        return self.classifier.is_indicator(symbol_id)

    def get_bliss_dict_info(self) -> Dict:
        """
        Get statistics about the Bliss dictionary.

        Returns:
            Dict with dictionary statistics
        """
        return {
            "symbols": len(self.bliss_dict),
            "characters": sum(1 for s in self.bliss_dict.values() if s.get("isCharacter", False)),
            "composed_words": sum(1 for s in self.bliss_dict.values() if not s.get("isCharacter", False)),
            "description": "Blissymbolics symbol dictionary"
        }
