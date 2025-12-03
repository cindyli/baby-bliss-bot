"""
Composer for generating Bliss compositions from semantic specifications.

Handles Use Case 3: Compose new Bliss words from semantic specifications.
Takes a semantic JSON specification and returns a Bliss composition.
"""

from typing import Dict, List, Optional

# Import from src.data for project-root compatibility
try:
    from src.data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
except ImportError:
    try:
        from data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
    except ImportError:
        from ..data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
from .symbol_classifier import SymbolClassifier


class BlissComposer:
    """Composes Bliss words from semantic specifications."""

    def __init__(self, bliss_dict):
        """
        Initialize the composer with a Bliss dictionary.

        Args:
            bliss_dict: Dict of Bliss symbol definitions
        """
        self.bliss_dict = bliss_dict
        self.classifier = SymbolClassifier(bliss_dict)
        self._build_reverse_lookups()

    def _build_reverse_lookups(self):
        """Build reverse lookup maps for efficient composition."""
        # Map from gloss to symbol ID (prioritize characters)
        self.gloss_to_id = {}

        # Map from semantic effect to symbol ID
        self.semantic_to_id = {}

        # Map from semantic path (type:value) to symbol ID
        self.semantic_path_to_id = {}

        for node_id in self.bliss_dict:
            node = self.bliss_dict[node_id]

            # Index glosses
            glosses = node.get("glosses", {})
            if glosses:
                # Use English gloss as primary key
                en_glosses = glosses.get("en", [])
                if isinstance(en_glosses, list):
                    for gloss in en_glosses:
                        if gloss not in self.gloss_to_id:
                            self.gloss_to_id[gloss] = node_id
                        else:
                            # Prefer characters over composed words
                            existing_id = self.gloss_to_id[gloss]
                            if (node.get("isCharacter", False) and
                                    not self.bliss_dict[existing_id].get("isCharacter", False)):
                                self.gloss_to_id[gloss] = node_id

            # Index semantic effects
            if "semantics" in node:
                effect = node["semantics"]
                self.semantic_to_id[node_id] = effect

                if "type" in effect and "value" in effect:
                    path = f"{effect['type']}:{effect['value']}"
                    if path not in self.semantic_path_to_id:
                        self.semantic_path_to_id[path] = node_id

    def compose_from_semantic_spec(self, semantic_spec: Dict) -> Dict:
        """
        Compose a Bliss word from a semantic specification.

        Use Case 3: Takes a semantic JSON and returns a Bliss composition.

        Args:
            semantic_spec: Dict with keys:
                - classifier: str (gloss for the classifier)
                - specifiers: List[str] (glosses for specifiers, optional)
                - semantics: List[Dict] (semantic modifications, optional)

        Returns:
            Dict with composition and metadata, or error info
        """
        if "classifier" not in semantic_spec:
            return {"error": "Missing required field: classifier"}

        result = {
            "original_spec": semantic_spec,
            "composition": [],
            "errors": [],
            "warnings": []
        }

        # Find classifier
        classifier_id = self._find_symbol_by_gloss(semantic_spec["classifier"])
        if not classifier_id:
            return {
                "error": f"Classifier not found: {semantic_spec['classifier']}",
                "details": result
            }
        result["composition"].append(classifier_id)

        # Add specifiers
        specifiers = semantic_spec.get("specifiers", [])
        for specifier_gloss in specifiers:
            specifier_id = self._find_symbol_by_gloss(specifier_gloss)
            if specifier_id:
                result["composition"].append(specifier_id)
            else:
                result["warnings"].append(f"Specifier not found: {specifier_gloss}")

        # Add semantic modifiers and indicators
        semantics = semantic_spec.get("semantics", [])
        semantic_ids = self._find_symbols_for_semantics(semantics)
        if "error" in semantic_ids:
            return {
                "error": f"Error processing semantics: {semantic_ids['error']}",
                "details": result
            }

        result["composition"].extend(semantic_ids["ids"])
        if semantic_ids.get("warnings"):
            result["warnings"].extend(semantic_ids["warnings"])

        return result

    def _find_symbol_by_gloss(self, gloss: str, prefer_character: bool = True) -> Optional[str]:
        """
        Find a symbol ID by its gloss.

        Args:
            gloss: The gloss text to search for
            prefer_character: If True, prefer character over composed words

        Returns:
            Symbol ID if found, None otherwise
        """
        # Exact match in gloss index
        if gloss in self.gloss_to_id:
            return self.gloss_to_id[gloss]

        # Search glosses in all languages (case-insensitive)
        gloss_lower = gloss.lower()
        for node_id in self.bliss_dict:
            node = self.bliss_dict[node_id]
            glosses = node.get("glosses", {})

            for lang, gloss_list in glosses.items():
                if isinstance(gloss_list, list):
                    for g in gloss_list:
                        if g.lower() == gloss_lower:
                            return node_id

        return None

    def _find_symbols_for_semantics(self, semantics: List[Dict]) -> Dict:
        """
        Find symbol IDs for semantic specifications.

        Args:
            semantics: List of semantic dicts, each with a semantic attribute.
                      Rendering markers (like "/" and ";") are ignored.

        Returns:
            Dict with keys "ids" (list of symbol IDs) and "warnings"
        """
        result = {"ids": [], "warnings": []}

        for semantic_item in semantics:
            # Handle simple semantic: {"ATTRIBUTE": "value"}
            if len(semantic_item) == 1:
                attr_type, attr_value = list(semantic_item.items())[0]
                symbol_id = self._find_semantic_symbol(attr_type, attr_value)
                if symbol_id:
                    result["ids"].append(symbol_id)
                else:
                    result["warnings"].append(
                        f"No symbol found for semantic {attr_type}:{attr_value}"
                    )
            else:
                result["warnings"].append(f"Complex semantic spec not fully supported: {semantic_item}")

        return result

    def _find_semantic_symbol(self, semantic_type: str, semantic_value: str) -> Optional[str]:
        """
        Find a symbol ID that represents a specific semantic.

        Args:
            semantic_type: Type of semantic (e.g., "NUMBER", "QUANTIFIER")
            semantic_value: Value of semantic (e.g., "plural", "many")

        Returns:
            Symbol ID if found, None otherwise
        """
        # Normalize value (case variations)
        value_lower = semantic_value.lower()

        # Search in both indicators and modifiers
        for symbol_id in INDICATOR_SEMANTICS:
            semantic_info = INDICATOR_SEMANTICS[symbol_id]
            if self._matches_semantic(semantic_info, semantic_type, value_lower):
                return symbol_id

        for symbol_id in MODIFIER_SEMANTICS:
            semantic_info = MODIFIER_SEMANTICS[symbol_id]
            if self._matches_semantic(semantic_info, semantic_type, value_lower):
                return symbol_id

        return None

    def _matches_semantic(self, semantic_info: Dict, sem_type: str, sem_value_lower: str) -> bool:
        """
        Check if semantic info matches the requested type and value.

        Args:
            semantic_info: Semantic info dict from INDICATOR_SEMANTICS or MODIFIER_SEMANTICS
            sem_type: Requested semantic type
            sem_value_lower: Requested semantic value (lowercase)

        Returns:
            True if matches, False otherwise
        """
        # Direct match
        if (semantic_info.get("type") == sem_type and
                semantic_info.get("value", "").lower() == sem_value_lower):
            return True

        # Check "or" alternatives
        if "or" in semantic_info:
            for alt in semantic_info["or"]:
                if (alt.get("type") == sem_type and
                        alt.get("value", "").lower() == sem_value_lower):
                    return True

        # Check "and" combinations
        if "and" in semantic_info:
            for item in semantic_info["and"]:
                if (item.get("type") == sem_type and
                        item.get("value", "").lower() == sem_value_lower):
                    return True

        return False

    def compose_with_modifiers(self, classifier_id: str, specifier_ids: List[str] = None,
                               modifier_ids: List[str] = None,
                               indicator_ids: List[str] = None) -> Dict:
        """
        Compose a Bliss word by directly providing symbol IDs.

        Args:
            classifier_id: ID of the classifier symbol
            specifier_ids: List of specifier symbol IDs (optional)
            modifier_ids: List of modifier symbol IDs (optional)
            indicator_ids: List of indicator symbol IDs (optional)

        Returns:
            Dict with composed sequence and validation info
        """
        result = {
            "composition": [],
            "errors": [],
            "warnings": []
        }

        # Add modifiers first (they're typically prefixes)
        if modifier_ids:
            for m_id in modifier_ids:
                if m_id in self.bliss_dict:
                    result["composition"].append(m_id)
                else:
                    result["warnings"].append(f"Modifier {m_id} not found")

        # Add classifier
        if classifier_id not in self.bliss_dict:
            result["error"] = f"Classifier {classifier_id} not found"
            return result
        result["composition"].append(classifier_id)

        # Add specifiers
        if specifier_ids:
            for s_id in specifier_ids:
                if s_id in self.bliss_dict:
                    result["composition"].append(s_id)
                else:
                    result["warnings"].append(f"Specifier {s_id} not found")

        # Add indicators (they're typically suffixes)
        if indicator_ids:
            for i_id in indicator_ids:
                if i_id in self.bliss_dict:
                    result["composition"].append(i_id)
                else:
                    result["warnings"].append(f"Indicator {i_id} not found")

        return result
