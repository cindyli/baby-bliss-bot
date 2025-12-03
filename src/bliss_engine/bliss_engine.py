"""
Main Bliss Engine module - unified interface for Blissymbolics operations.

Provides integrated functionality for:
- Use Case 1: Get glosses for existing Bliss IDs or compositions
- Use Case 2: Analyze new compositions and extract semantic meaning
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
    1. Retrieve glosses and explanations for existing Bliss words
    2. Analyze new compositions and extract combined semantic information
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
    # USE CASE 1: Get glosses and explanations for existing Bliss words
    # ============================================================================

    def get_symbol_glosses(self, symbol_id: str, language: str = "en") -> Dict:
        """
        Get glosses for a single Bliss symbol.

        Use Case 1: Retrieve the gloss(es) and explanation for a Bliss ID.

        Args:
            symbol_id: The ID of the Bliss symbol
            language: ISO 639-1 language code (default: "en" for English)

        Returns:
            Dict containing:
            - id: The symbol ID
            - glosses: List of glosses in the requested language
            - explanation: Explanation of the symbol
            - isCharacter: Whether this is a Bliss character or composed word
        """
        return self.analyzer.get_symbol_glosses(symbol_id, language)

    def get_composition_glosses(self, composition: Union[List[str], List[int]],
                                language: str = "en") -> Dict:
        """
        Get glosses for an existing Bliss composition/word.

        Use Case 1: If the input is an existing composition with an ID,
        returns its glosses and explanation.

        Args:
            composition: List of symbol IDs composing the Bliss word
            language: ISO 639-1 language code

        Returns:
            Dict with composition info and glosses for each component
        """
        composition = [str(c) for c in composition]

        result = {
            "composition": composition,
            "components": []
        }

        for symbol_id in composition:
            if symbol_id.isdigit():
                gloss_info = self.analyzer.get_symbol_glosses(symbol_id, language)
                result["components"].append(gloss_info)

        return result

    # ============================================================================
    # USE CASE 2: Analyze new compositions and extract semantic information
    # ============================================================================

    def analyze_composition(self, composition: Union[List[str], List[int]],
                            language: str = "en") -> Dict:
        """
        Analyze a new Bliss word composition and extract semantic meaning.

        Use Case 2: Analyzes component IDs and returns their combined semantic
        information including classifier, specifiers, indicators, and modifiers.

        Example output for "many hospitals":
        {
            "original_composition": [14647, 14905, 24920, 9011],
            "classifier": "14905",
            "classifier_info": {"gloss": ["building"]},
            "specifiers": ["24920"],
            "specifier_info": [{"gloss": ["medicine"]}],
            "semantics": [
                {"symbol_id": "9011", "indicator": {"NUMBER": "plural"}},
                {"symbol_id": "14647", "modifier": {"QUANTIFIER": "many"}}
            ],
            "indicators": ["9011"],
            "modifiers": ["14647"]
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

    def get_composition_structure(self, composition: Union[List[str], List[int]]) -> Dict:
        """
        Get the structural breakdown of a composition.

        Args:
            composition: List of symbol IDs

        Returns:
            Dict with structural information and role classification
        """
        return self.analyzer.get_composition_structure(composition)

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
            "semantics": [
                {"NUMBER": "plural"},
                {"QUANTIFIER": "many"}
            ]
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
                - semantics: List[Dict] (semantic modifications, optional)
                  Each semantic dict should have format: {"TYPE": "value"}

        Returns:
            Dict with:
            - composition: List of symbol IDs
            - errors: Any composition errors
            - warnings: Non-critical issues
        """
        return self.composer.compose_from_semantic_spec(semantic_spec)

    def compose_with_ids(self, classifier_id: str, specifier_ids: List[str] = None,
                         modifier_ids: List[str] = None,
                         indicator_ids: List[str] = None) -> Dict:
        """
        Compose a Bliss word using symbol IDs directly.

        Args:
            classifier_id: ID of the classifier symbol
            specifier_ids: List of specifier symbol IDs (optional)
            modifier_ids: List of modifier symbol IDs (optional)
            indicator_ids: List of indicator symbol IDs (optional)

        Returns:
            Dict with composed sequence and validation information
        """
        return self.composer.compose_with_modifiers(
            classifier_id,
            specifier_ids,
            modifier_ids,
            indicator_ids
        )

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

    def get_knowledge_graph_info(self) -> Dict:
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
