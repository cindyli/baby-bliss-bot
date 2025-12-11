"""
Analyzer for extracting semantic meaning from existing Bliss compositions.

Handles Use Case 1 and 2:
- Use Case 1: Look up a symbol ID and return its glosses and explanation
- Use Case 2: Look up a word composition and return its symbol ID (if exists) or semantic meaning
"""

from typing import List, Dict, Optional, Union

# Import from src.data for project-root compatibility
from src.data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
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

    def get_symbol_glosses(self, symbol_id: int, language: str = "en") -> Dict:
        """
        Get the glosses for a single Bliss symbol.

        Args:
            symbol_id: The ID of the Bliss symbol (integer)
            language: ISO 639-1 language code (default: "en")

        Returns:
            Dict with glosses, explanation, and symbol information
        """
        symbol_id_str = str(symbol_id)
        if symbol_id_str not in self.bliss_dict:
            return {"error": f"Symbol {symbol_id} not found"}

        node = self.bliss_dict[symbol_id_str]
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
            "classifier": None,
            "classifier_info": None,
            "specifiers": [],
            "specifier_info": [],
            "semantics": {},
            "indicators": [],
            "modifiers": [],
        }

        # Get classifier information
        if classification["classifier"]:
            classifier_id = classification["classifier"]
            result["classifier"] = int(classifier_id)
            result["classifier_info"] = self._get_first_symbol_gloss(classifier_id, language)

        # Get specifier information
        for specifier_id in classification["specifiers"]:
            result["specifiers"].append(int(specifier_id))
            result["specifier_info"].append(self._get_first_symbol_gloss(specifier_id, language))

        # Extract semantics from indicators and modifiers
        for indicator_id in classification["indicators"]:
            semantics = self._extract_semantics(indicator_id, "indicator")
            if semantics:
                result["semantics"].update(semantics)
            result["indicators"].append(int(indicator_id))

        for modifier_id in classification["modifiers"]:
            semantics = self._extract_semantics(modifier_id, "modifier")
            if semantics:
                result["semantics"].update(semantics)
            result["modifiers"].append(int(modifier_id))

        return result

    def _get_symbol_glosses(self, symbol_id: str, language: str = "en") -> Dict:
        """Helper to get glosses for a symbol (returns dict format)."""
        if symbol_id not in self.bliss_dict:
            return {"id": symbol_id, "error": "not found"}

        node = self.bliss_dict[symbol_id]
        glosses = node.get("glosses", {})

        return {
            "id": symbol_id,
            "gloss": glosses.get(language, glosses.get("en", ["(unknown)"])),
            "isCharacter": node.get("isCharacter", False),
        }

    def _get_first_symbol_gloss(self, symbol_id: str, language: str = "en") -> str:
        """Helper to get the first gloss for a symbol as a string."""
        if symbol_id not in self.bliss_dict:
            return "(unknown)"

        node = self.bliss_dict[symbol_id]
        glosses = node.get("glosses", {})
        gloss_list = glosses.get(language, glosses.get("en", ["(unknown)"]))

        # Return first gloss if it's a list, otherwise return as string
        if isinstance(gloss_list, list) and gloss_list:
            return gloss_list[0]
        return gloss_list if isinstance(gloss_list, str) else "(unknown)"

    def _extract_semantics(self, symbol_id: str, symbol_type: str) -> Optional[Dict]:
        """
        Extract semantic meaning from an indicator or modifier.

        Args:
            symbol_id: The symbol ID
            symbol_type: "indicator" or "modifier"

        Returns:
            Dict with semantic information in format {"TYPE": "value"}, or None if no semantics found
        """
        semantics_map = INDICATOR_SEMANTICS if symbol_type == "indicator" else MODIFIER_SEMANTICS

        if symbol_id not in semantics_map:
            return None

        semantic_info = semantics_map[symbol_id]

        # Handle "or" semantics (alternative interpretations) - return first alternative
        if "or" in semantic_info:
            first_alternative = semantic_info["or"][0]
            if "type" in first_alternative and "value" in first_alternative:
                return {first_alternative["type"]: first_alternative["value"]}
            return None

        # Handle "and" semantics (multiple properties) - return first property
        if "and" in semantic_info:
            first_prop = semantic_info["and"][0]
            if "type" in first_prop and "value" in first_prop:
                return {first_prop["type"]: first_prop["value"]}
            return None

        # Handle simple semantics
        if "type" in semantic_info and "value" in semantic_info:
            return {semantic_info["type"]: semantic_info["value"]}

        return None

    def _normalize_composition(self, composition: Union[List[str], List[int]]) -> List[str]:
        """
        Normalize composition by removing rendering elements.

        Removes elements like "/" and ";" which are rendering elements,
        keeping only numeric symbol IDs.

        Args:
            composition: List of symbol IDs and optional rendering elements

        Returns:
            List of normalized numeric symbol IDs as strings
        """
        # Rendering elements to filter out
        rendering_elements = {"/", ";"}

        normalized = []
        for item in composition:
            item_str = str(item)
            # Only keep items that are numeric IDs (not rendering elements)
            if item_str not in rendering_elements and item_str.isdigit():
                normalized.append(item_str)

        return normalized

    def lookup_composition(self, composition: Union[List[str], List[int]],
                           language: str = "en") -> Dict:
        """
        Look up a word composition in the Bliss dictionary.

        First looks up in the Bliss dictionary to see if the given composition
        already exists by checking the "composition" values.

        If it exists, returns the symbol ID with its glosses and explanation.
        If it doesn't exist, returns its semantic meaning.
        Note: Rendering elements such as "/" and ";" are ignored when comparing compositions.

        Args:
            composition: List of symbol IDs (may include rendering elements like "/" and ";")
            language: language code for glosses

        Returns:
            Dict with:
            - composition: The normalized input composition (rendering elements removed, symbol IDs as integers)
            - is_existing_symbol: Boolean indicating if composition exists in dictionary
            - If is_existing_symbol is True:
                - symbol_id: The ID of the existing symbol
                - glosses: List of glosses for the symbol
                - explanation: Explanation of the symbol
            - If is_existing_symbol is False:
                - classifier: The main semantic category
                - specifiers: Refining symbols
                - indicators: Grammatical information
                - modifiers: Meaning modifiers
                - semantics: Combined semantic properties
        """
        result = {"composition": composition}

        # Normalize composition (remove rendering elements)
        normalized_composition = self._normalize_composition(composition)

        # Look for existing symbol by checking composition values in dictionary
        for symbol_id, symbol_data in self.bliss_dict.items():
            if "composition" in symbol_data:
                # Get the composition from the dictionary and normalize it
                dict_composition = symbol_data["composition"]
                # Convert to strings and normalize
                dict_composition_normalized = self._normalize_composition(dict_composition)

                # Compare normalized compositions
                if dict_composition_normalized == normalized_composition:
                    # Found existing symbol
                    glosses = symbol_data.get("glosses", {})
                    result.update({
                        "is_existing_symbol": True,
                        "symbol_id": int(symbol_id),
                        "glosses": glosses.get(language, glosses.get("en", [])),
                        "explanation": symbol_data.get("explanation", ""),
                    })

                    return result

        # Composition not found in dictionary, analyze semantic meaning
        analysis = self.analyze_composition(normalized_composition, language)

        result["is_existing_symbol"] = False
        result.update(analysis)

        return result
