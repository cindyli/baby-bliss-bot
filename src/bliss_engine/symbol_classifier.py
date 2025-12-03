"""
Symbol classifier for identifying symbol types in Blissymbolics compositions.

Classifies symbols into:
- Classifiers: Set the semantic category (POS: YELLOW, RED, GREEN, BLUE)
- Specifiers: Refine the meaning of the classifier
- Indicators: Denote grammatical information (POS: GREY or WHITE)
- Modifiers: Prefix/suffix symbols that modify meaning (POS: GREY or WHITE)
"""

from typing import List, Dict

# Import from src.data for project-root compatibility
try:
    from src.data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
except ImportError:
    try:
        from data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
    except ImportError:
        from ..data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS


class SymbolClassifier:
    """Classifies Bliss symbols based on their properties and context."""

    # Color-coded part-of-speech categories
    CLASSIFIER_POS = {"YELLOW", "RED", "GREEN", "BLUE"}
    MODIFIER_POS = {"GREY", "WHITE"}
    INDICATOR_POS = {"GREY", "WHITE"}

    def __init__(self, bliss_dict):
        """
        Initialize the symbol classifier with a Bliss dictionary.

        Args:
            bliss_dict: Dict of Bliss symbol definitions
        """
        self.bliss_dict = bliss_dict
        self.modifiers = set(MODIFIER_SEMANTICS.keys())
        self.indicators = set(INDICATOR_SEMANTICS.keys())

    def is_classifier(self, symbol_id: str) -> bool:
        """Check if a symbol can be a classifier."""
        if symbol_id not in self.bliss_dict:
            return False

        pos = self.bliss_dict[symbol_id].get("pos", "")
        return pos in self.CLASSIFIER_POS

    def is_modifier(self, symbol_id: str) -> bool:
        """Check if a symbol is a modifier."""
        return symbol_id in self.modifiers

    def is_indicator(self, symbol_id: str) -> bool:
        """Check if a symbol is an indicator."""
        return symbol_id in self.indicators

    def is_specifier(self, symbol_id: str) -> bool:
        """Check if a symbol can be a specifier (any symbol that's not explicitly classified)."""
        if symbol_id not in self.bliss_dict:
            return False
        return True

    def classify_composition(self, symbol_ids: List[str]) -> Dict:
        """
        Classify symbols in a composition into their functional roles.

        Composition values may contain both symbol IDs (numbers) and rendering
        markers (like "/" for spacing and ";" for separator). Only numeric IDs
        are processed; rendering markers are filtered out.

        Typical composition order: [modifier, classifier, indicator, specifier1, specifier2...]

        Classification rules (in priority order):
        1. **Indicator-Based**: If composition contains indicators (found in INDICATOR_SEMANTICS),
           the classifier is the symbol immediately before the first indicator.
           All symbols before the classifier are modifiers/prefixes.
           All symbols after the indicator are specifiers (or additional modifiers if they're in MODIFIER_SEMANTICS).
        2. **POS-Based**: Symbols with pos in {YELLOW, RED, GREEN, BLUE} are classifiers.
        3. **First Symbol**: If all symbols are GREY/WHITE (modifiers/indicators only),
           the first symbol is the classifier.
        4. **Default**: Any remaining symbols are specifiers.

        Args:
            symbol_ids: List of symbol IDs in composition order (may include "/" and ";")

        Returns:
            Dict with keys: classifier, specifiers, indicators, modifiers, errors
        """
        result = {
            "classifier": None,
            "specifiers": [],
            "indicators": [],
            "modifiers": [],
            "errors": []
        }

        # Filter out non-digit symbols (like "/" and ";" which are rendering markers)
        # Only keep numeric symbol IDs
        valid_ids = [s for s in symbol_ids if str(s).isdigit()]

        if not valid_ids:
            result["errors"].append("No valid symbol IDs found in composition")
            return result

        # Rule 1: Find classifier before first indicator
        first_indicator_idx = None
        for i, symbol_id in enumerate(valid_ids):
            if self.is_indicator(symbol_id):
                first_indicator_idx = i
                break

        # If there's an indicator, the classifier is the symbol immediately before it
        if first_indicator_idx is not None:
            if first_indicator_idx > 0:
                # The symbol immediately before the first indicator is the classifier
                result["classifier"] = valid_ids[first_indicator_idx - 1]

                # Everything before the classifier are modifiers/prefixes
                for i in range(first_indicator_idx - 1):
                    result["modifiers"].append(valid_ids[i])

                # Indicators and following symbols (after the first indicator)
                for i in range(first_indicator_idx, len(valid_ids)):
                    symbol_id = valid_ids[i]
                    if self.is_indicator(symbol_id):
                        result["indicators"].append(symbol_id)
                    elif self.is_modifier(symbol_id):
                        # Modifiers after indicator are suffixes
                        result["modifiers"].append(symbol_id)
                    else:
                        # Symbols after indicator are specifiers
                        result["specifiers"].append(symbol_id)
            else:
                # First symbol is an indicator (unusual but possible)
                result["errors"].append("First symbol is an indicator; no classifier found before it")

            return result

        # Rule 2: No indicators - use POS-based classification
        has_classifier = False

        for i, symbol_id in enumerate(valid_ids):
            # Check type in order of precedence
            if self.is_modifier(symbol_id):
                result["modifiers"].append(symbol_id)
            elif self.is_classifier(symbol_id):
                if not has_classifier:
                    result["classifier"] = symbol_id
                    has_classifier = True
                else:
                    # Second classifier becomes a specifier
                    result["specifiers"].append(symbol_id)
            else:
                # Default to specifier
                result["specifiers"].append(symbol_id)

        # Rule 3: Special case - if no classifier found but all symbols are GREY/WHITE
        if not has_classifier and valid_ids:
            first_id = str(valid_ids[0])
            if first_id in self.bliss_dict:
                pos = self.bliss_dict[first_id].get("pos", "")
                if pos in self.MODIFIER_POS:
                    result["classifier"] = first_id
                    # Move specifiers list to include what was previously classified
                    result["specifiers"] = [s for s in result["specifiers"] if s != first_id]
                else:
                    result["errors"].append("No classifier found in composition")
            else:
                result["errors"].append(f"Symbol {first_id} not found in knowledge graph")

        return result

    def get_symbol_info(self, symbol_id: str) -> Dict:
        """Get detailed information about a symbol."""
        if symbol_id not in self.bliss_dict:
            return {"error": f"Symbol {symbol_id} not found"}

        node = self.bliss_dict[symbol_id]
        info = {
            "id": symbol_id,
            "pos": node.get("pos", "unknown"),
            "glosses": node.get("glosses", {}),
            "isCharacter": node.get("isCharacter", False),
            "explanation": node.get("explanation", ""),
        }

        # Add semantic/indicator effects if present
        if "symbolSemantics" in node:
            info["symbolSemantics"] = node["symbolSemantics"]

        if symbol_id in self.modifiers:
            info["type"] = "modifier"
            info["semantics"] = MODIFIER_SEMANTICS.get(symbol_id, {})
        elif symbol_id in self.indicators:
            info["type"] = "indicator"
            info["semantics"] = INDICATOR_SEMANTICS.get(symbol_id, {})
        else:
            info["type"] = "character_or_word"

        return info
