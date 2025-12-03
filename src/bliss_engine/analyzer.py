"""
Analyzer for extracting semantic meaning from existing Bliss compositions.

Handles Use Case 1 and 2:
- Use Case 1: Return glosses and explanations for existing Bliss words
- Use Case 2: Analyze new compositions and extract their combined semantic meaning
"""

from typing import List, Dict, Optional, Union

# Import from src.data for project-root compatibility
try:
    from src.data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
except ImportError:
    try:
        from data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
    except ImportError:
        from ..data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
from .symbol_classifier import SymbolClassifier


class BlissAnalyzer:
    """Analyzes Bliss symbols and compositions to extract semantic meaning."""

    def __init__(self, bliss_dict):
        """
        Initialize the analyzer with a Bliss dictionary.

        Args:
            bliss_dict: Dict of Bliss symbol definitions
        """
        self.bliss_dict = bliss_dict
        self.classifier = SymbolClassifier(bliss_dict)

    def get_symbol_glosses(self, symbol_id: str, language: str = "en") -> Dict:
        """
        Get the glosses for a single Bliss symbol.

        Args:
            symbol_id: The ID of the Bliss symbol
            language: ISO 639-1 language code (default: "en")

        Returns:
            Dict with glosses, explanation, and symbol information
        """
        if symbol_id not in self.bliss_dict:
            return {"error": f"Symbol {symbol_id} not found"}

        node = self.bliss_dict[symbol_id]
        glosses = node.get("glosses", {})

        result = {
            "id": symbol_id,
            "glosses": glosses.get(language, glosses.get("en", [])),
            "explanation": node.get("explanation", ""),
            "isCharacter": node.get("isCharacter", False),
        }

        return result

    def analyze_composition(self, composition: Union[List[str], List[int]],
                            language: str = "en") -> Dict:
        """
        Analyze a new Bliss composition and extract semantic meaning.

        Use Case 2: Analyzes new compositions to extract combined semantic information.

        Composition may contain both symbol IDs (numbers) and rendering markers
        (like "/" for spacing and ";" for separator). Only numeric IDs are
        analyzed; rendering markers are automatically filtered out.

        Args:
            composition: List of symbol IDs (and optional rendering markers) composing the Bliss word
            language: ISO 639-1 language code for glosses

        Returns:
            Dict with classifier, specifiers, indicators, modifiers, and semantics
        """
        # Convert to strings
        composition = [str(c) for c in composition]

        # Classify the composition
        classification = self.classifier.classify_composition(composition)

        if classification["errors"]:
            return {"error": classification["errors"][0], "details": classification}

        result = {
            "original_composition": composition,
            "classifier": None,
            "classifier_info": None,
            "specifiers": [],
            "specifier_info": [],
            "semantics": [],
            "indicators": [],
            "modifiers": [],
        }

        # Get classifier information
        if classification["classifier"]:
            classifier_id = classification["classifier"]
            result["classifier"] = classifier_id
            result["classifier_info"] = self._get_symbol_glosses(classifier_id, language)

        # Get specifier information
        for specifier_id in classification["specifiers"]:
            result["specifiers"].append(specifier_id)
            result["specifier_info"].append(self._get_symbol_glosses(specifier_id, language))

        # Extract semantics from indicators and modifiers
        for indicator_id in classification["indicators"]:
            semantics = self._extract_semantics(indicator_id, "indicator")
            if semantics:
                result["semantics"].append(semantics)
            result["indicators"].append(indicator_id)

        for modifier_id in classification["modifiers"]:
            semantics = self._extract_semantics(modifier_id, "modifier")
            if semantics:
                result["semantics"].append(semantics)
            result["modifiers"].append(modifier_id)

        return result

    def _get_symbol_glosses(self, symbol_id: str, language: str = "en") -> Dict:
        """Helper to get glosses for a symbol."""
        if symbol_id not in self.bliss_dict:
            return {"id": symbol_id, "error": "not found"}

        node = self.bliss_dict[symbol_id]
        glosses = node.get("glosses", {})

        return {
            "id": symbol_id,
            "gloss": glosses.get(language, glosses.get("en", ["(unknown)"])),
            "isCharacter": node.get("isCharacter", False),
        }

    def _extract_semantics(self, symbol_id: str, symbol_type: str) -> Optional[Dict]:
        """
        Extract semantic meaning from an indicator or modifier.

        Args:
            symbol_id: The symbol ID
            symbol_type: "indicator" or "modifier"

        Returns:
            Dict with semantic information, or None if no semantics found
        """
        semantics_map = INDICATOR_SEMANTICS if symbol_type == "indicator" else MODIFIER_SEMANTICS

        if symbol_id not in semantics_map:
            return None

        semantic_info = semantics_map[symbol_id]

        # Handle "or" semantics (alternative interpretations)
        if "or" in semantic_info:
            return {
                "symbol_id": symbol_id,
                "type": symbol_type,
                "alternatives": semantic_info["or"]
            }

        # Handle "and" semantics (multiple properties)
        if "and" in semantic_info:
            return {
                "symbol_id": symbol_id,
                "type": symbol_type,
                "combined": semantic_info["and"]
            }

        # Handle simple semantics
        if "type" in semantic_info and "value" in semantic_info:
            return {
                "symbol_id": symbol_id,
                symbol_type: {
                    semantic_info["type"]: semantic_info["value"]
                }
            }

        return None

    def analyze_symbol_with_context(self, symbol_id: str, context_ids: List[str] = None,
                                    language: str = "en") -> Dict:
        """
        Analyze a symbol with optional context from surrounding symbols.

        Args:
            symbol_id: The main symbol to analyze
            context_ids: Optional list of surrounding symbols for context
            language: ISO 639-1 language code

        Returns:
            Dict with symbol information and contextual analysis
        """
        result = self.get_symbol_glosses(symbol_id, language)
        result["type"] = self.classifier.get_symbol_info(symbol_id).get("type", "unknown")

        if context_ids:
            context_classification = self.classifier.classify_composition(context_ids)
            result["context_classification"] = context_classification

        return result

    def get_composition_structure(self, composition: Union[List[str], List[int]]) -> Dict:
        """
        Get the structural breakdown of a composition.

        Args:
            composition: List of symbol IDs

        Returns:
            Dict with detailed structural information
        """
        composition = [str(c) for c in composition]
        classification = self.classifier.classify_composition(composition)

        return {
            "original_composition": composition,
            "structure": {
                "classifier": classification["classifier"],
                "specifiers": classification["specifiers"],
                "indicators": classification["indicators"],
                "modifiers": classification["modifiers"],
            },
            "interpretation": {
                "classifier_glosses": (self._get_symbol_glosses(classification["classifier"])
                                       if classification["classifier"] else None),
                "specifier_glosses": [self._get_symbol_glosses(s)
                                      for s in classification["specifiers"]],
                "indicator_count": len(classification["indicators"]),
                "modifier_count": len(classification["modifiers"]),
            }
        }
