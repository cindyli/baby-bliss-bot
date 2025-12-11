# Run from project root: python -m src.bliss_engine.tests

"""
Unit tests for the Bliss Engine module.

Tests all three use cases and utility functions.
"""

import json
import unittest

# Run from project root
from src.bliss_engine import BlissEngine
from src.bliss_engine.symbol_classifier import SymbolClassifier
from src.bliss_engine.analyzer import BlissAnalyzer
from src.bliss_engine.composer import BlissComposer

dict_path = "src/data/bliss_dict/bliss_dict_multi_langs.json"

with open(dict_path, 'r', encoding='utf-8') as f:
    bliss_dict = json.load(f)


class TestBlissEngineInitialization(unittest.TestCase):
    """Test engine initialization and setup."""

    def setUp(self):
        """Create a mock Bliss dictionary for testing."""
        self.bliss_dict = bliss_dict

    def test_engine_initialization_with_valid_dict(self):
        """Test engine initializes with valid dictionary."""
        engine = BlissEngine(self.bliss_dict)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.bliss_dict, self.bliss_dict)

    def test_engine_initialization_with_invalid_graph(self):
        """Test engine raises error with invalid input."""
        with self.assertRaises(TypeError):
            BlissEngine("not a dict")

        with self.assertRaises(TypeError):
            BlissEngine([])


class TestBlissEngineUseCases(unittest.TestCase):
    """Test the three primary use cases."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = bliss_dict

        self.engine = BlissEngine(self.bliss_dict)

    def test_use_case_1_get_symbol_glosses(self):
        """Test Use Case 1: Look up a symbol ID and get its glosses and explanation."""
        expected = {
            "id": 14905,
            "glosses": [
                "house",
                "building",
                "dwelling",
                "residence"
            ],
            "explanation": "(foundation + protection: pictograph of the outline of a house.The symbol can also be explained as: combination of enclosure and protection.)  - Character (superimposed)",
            "isCharacter": True
        }
        result = self.engine.get_symbol_glosses(14905, language="en")
        self.assertEqual(expected, result)

    def test_use_case_1_nonexistent_symbol(self):
        """Test Use Case 1: Look up non-existent symbol returns error."""
        result = self.engine.get_symbol_glosses(99999, language="en")

        self.assertIn("error", result)

    def test_use_case_2_lookup_existing_composition(self):
        """Test Use Case 2: Look up existing composition returns symbol ID."""
        # Test with an existing composition (if available in dictionary)
        expected = {
            "composition": [14133, 8998, 17717, 23599],
            "is_existing_symbol": True,
            "symbol_id": 24924,
            "explanation": "(oval,ellipse + description indicator)",
            "glosses": ["oval", "elliptic", "elliptical"],
        }
        result = self.engine.lookup_composition([14133, 8998, 17717, 23599], language="en")
        self.assertEqual(expected, result)

    def test_use_case_2_lookup_existing_composition_with_rendering_components(self):
        """Test Use Case 2: Look up existing composition returns symbol ID."""
        # Test with an existing composition (if available in dictionary)
        expected = {
            "composition": [14133, ";", 8998, "/", 17717, "/", 23599],
            "is_existing_symbol": True,
            "symbol_id": 24924,
            "explanation": "(oval,ellipse + description indicator)",
            "glosses": ["oval", "elliptic", "elliptical"],
        }
        result = self.engine.lookup_composition([14133, ";", 8998, "/", 17717, "/", 23599], language="en")
        self.assertEqual(expected, result)

    def test_use_case_2_lookup_new_composition(self):
        """Test Use Case 2: Look up new composition returns semantic meaning."""
        # Test with a composition unlikely to exist
        expected = {
            "composition": [14647, 14905, 9011, 24920],
            "is_existing_symbol": False,
            "classifier": 14905,
            "classifier_info": "house",
            "specifiers": [24920],
            "specifier_info": ["medicine"],
            "semantics": {
                "NUMBER": "plural",
                "QUANTIFIER": "many"
            },
            "indicators": [9011],
            "modifiers": [14647]
        }
        result = self.engine.lookup_composition([14647, 14905, 9011, 24920], language="en")
        self.assertEqual(expected, result)

    def test_use_case_2_lookup_with_rendering_elements(self):
        """Test Use Case 2: Lookup ignores rendering elements like / and ;."""
        # Composition with rendering elements
        result = self.engine.lookup_composition([14905, "/", 9011, ";"], language="en")
        expected = {
            "composition": [14905, "/", 9011, ";"],
            "is_existing_symbol": False,
            "classifier": 14905,
            "classifier_info": "house",
            "specifiers": [],
            "specifier_info": [],
            "semantics": {"NUMBER": "plural"},
            "indicators": [9011],
            "modifiers": []
        }

        self.assertEqual(expected, result)

    def test_use_case_3_compose_from_semantic(self):
        """Test Use Case 3: Compose from semantic specification."""
        semantic_spec = {
            "classifier": "building",
            "specifiers": ["medicine"],
            "semantics": {"NUMBER": "plural"}
        }

        expected = {
            "original_spec": {
                "classifier": "building",
                "specifiers": ["medicine"],
                "semantics": {"NUMBER": "plural"}
            },
            "composition": ["14905", "24920", "9011"],
            "errors": [],
            "warnings": []
        }
        result = self.engine.compose_from_semantic(semantic_spec)
        self.assertEqual(expected, result)


class TestSymbolClassifier(unittest.TestCase):
    """Test the SymbolClassifier module."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = bliss_dict

        self.classifier = SymbolClassifier(self.bliss_dict)

    def test_is_classifier(self):
        """Test classifier identification."""
        self.assertTrue(self.classifier.is_classifier("14905"))
        self.assertTrue(self.classifier.is_classifier("24920"))
        self.assertTrue(self.classifier.is_classifier("14647"))  # 14647 has POS=YELLOW, so it IS a classifier

    def test_is_modifier(self):
        """Test modifier identification."""
        self.assertTrue(self.classifier.is_modifier("14647"))
        self.assertFalse(self.classifier.is_modifier("14905"))

    def test_is_indicator(self):
        """Test indicator identification."""
        self.assertTrue(self.classifier.is_indicator("9011"))
        self.assertFalse(self.classifier.is_indicator("14905"))

    def test_classify_composition(self):
        """Test composition classification."""
        # Composition: [14647 (modifier), 14905 (classifier), 9011 (indicator), 24920 (specifier)]
        expected = {
            "classifier": "14905",
            "specifiers": ["24920"],
            "indicators": ["9011"],
            "modifiers": ["14647"],
            "errors": []
        }
        result = self.classifier.classify_composition(["14647", "14905", "9011", "24920"])
        self.assertEqual(expected, result)

    def test_classify_composition_with_rendering_markers(self):
        """Test composition classification with rendering markers like '/' and ';'."""
        # Composition may contain "/" and ";" which are rendering markers and should be filtered
        # After filtering: [14647, 14905, 9011, 24920]
        expected = {
            "classifier": "14905",
            "specifiers": ["24920"],
            "indicators": ["9011"],
            "modifiers": ["14647"],
            "errors": []
        }
        result = self.classifier.classify_composition(["14647", "/", "14905", ";", "9011", "/", "24920"])
        self.assertEqual(expected, result)

    def test_classify_composition_indicator_based_rule(self):
        """Test the indicator-based classifier rule: classifier is before first indicator."""
        # When composition has indicators, the symbol immediately before the first
        # indicator should be identified as the classifier
        # Composition: [14905 (classifier), 9011 (indicator), 24920 (specifier)]
        expected = {
            "classifier": "14905",
            "specifiers": ["24920"],
            "indicators": ["9011"],
            "modifiers": [],
            "errors": []
        }
        result = self.classifier.classify_composition(["14905", "9011", "24920"])
        self.assertEqual(expected, result)


class TestBlissAnalyzer(unittest.TestCase):
    """Test the BlissAnalyzer module."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = bliss_dict

        self.analyzer = BlissAnalyzer(self.bliss_dict)

    def test_get_symbol_glosses(self):
        """Test getting symbol glosses."""
        expected = {
            "id": 14905,
            "glosses": [
                "house",
                "building",
                "dwelling",
                "residence"
            ],
            "explanation": "(foundation + protection: pictograph of the outline of a house.The symbol can also be explained as: combination of enclosure and protection.)  - Character (superimposed)",
            "isCharacter": True
        }
        result = self.analyzer.get_symbol_glosses(14905)
        self.assertEqual(expected, result)

    def test_analyze_composition_with_single_element(self):
        """Test composition analysis."""
        expected = {
            "classifier": 14905,
            "classifier_info": "house",
            "specifiers": [],
            "specifier_info": [],
            "semantics": {},
            "indicators": [],
            "modifiers": []
        }
        result = self.analyzer.analyze_composition([14905])
        self.assertEqual(expected, result)

    def test_analyze_composition_with_all_elements(self):
        """Test composition analysis with a composition of all elements present."""
        # Composition for "many hospitals": [14647 (modifier), 14905 (classifier), 24920 (specifier), 9011 (indicator)]
        expected = {
            "classifier": 14905,
            "classifier_info": "house",
            "specifiers": [24920],
            "specifier_info": ["medicine"],
            "modifiers": [14647],
            "indicators": [9011],
            "semantics": {"NUMBER": "plural", "QUANTIFIER": "many"},
        }
        result = self.analyzer.analyze_composition([14647, "/", 14905, ";", 9011, "/", 24920])
        self.assertEqual(expected, result)

    def test_analyze_composition_with_all_modifiers(self):
        """Test composition analysis with a composition of all modifiers."""
        # Composition for "no": [15474 (minus, no), 14947 (intensify), 14947 (intensify)]
        expected = {
            "classifier": 15474,
            "classifier_info": "minus",
            "modifiers": [14947, 14947],
            "indicators": [],
            "specifiers": [],
            "specifier_info": [],
            "semantics": {"INTENSIFIER": "high"},
        }
        result = self.analyzer.analyze_composition([15474, "/", 14947, ";", 14947])
        self.assertEqual(expected, result)

    def test_lookup_composition_existing(self):
        """Test lookup_composition for existing symbol."""
        # Lookup a composition that exist in dictionary
        expected = {
            "composition": [14133, 8998, 17717, 23599],
            "is_existing_symbol": True,
            "symbol_id": 24924,
            "explanation": "(oval,ellipse + description indicator)",
            "glosses": ["oval", "elliptic", "elliptical"],
        }
        result = self.analyzer.lookup_composition([14133, 8998, 17717, 23599], language="en")
        self.assertEqual(expected, result)

    def test_lookup_composition_new(self):
        """Test lookup_composition for new composition."""
        # Try to lookup a composition unlikely to exist
        expected = {
            "composition": [14647, 14905, 9011, 24920],
            "is_existing_symbol": False,
            "classifier": 14905,
            "classifier_info": "house",
            "specifiers": [24920],
            "specifier_info": ["medicine"],
            "semantics": {
                "NUMBER": "plural",
                "QUANTIFIER": "many"
            },
            "indicators": [9011],
            "modifiers": [14647]
        }
        result = self.analyzer.lookup_composition([14647, 14905, 9011, 24920], language="en")
        self.assertEqual(expected, result)

    def test_lookup_composition_with_rendering_elements(self):
        """Test lookup_composition filters rendering elements."""
        # Composition with rendering markers
        expected = {
            "composition": [14905, "/", 9011, ";"],
            "is_existing_symbol": False,
            "classifier": 14905,
            "classifier_info": "house",
            "specifiers": [],
            "specifier_info": [],
            "semantics": {"NUMBER": "plural"},
            "indicators": [9011],
            "modifiers": []
        }

        result = self.analyzer.lookup_composition([14905, "/", 9011, ";"], language="en")
        self.assertEqual(expected, result)

    def test_normalize_composition(self):
        """Test normalization of compositions with rendering elements."""
        # Test the normalize_composition helper method
        expected = ["14905", "9011", "24920"]
        result = self.analyzer._normalize_composition([14905, "/", 9011, ";", 24920])
        self.assertEqual(expected, result)


class TestBlissComposer(unittest.TestCase):
    """Test the BlissComposer module."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = bliss_dict

        self.composer = BlissComposer(self.bliss_dict)

    def test_find_symbol_by_gloss(self):
        """Test finding symbol by gloss."""
        expected = "14905"
        result = self.composer._find_symbol_by_gloss("building")
        self.assertEqual(expected, result)

    def test_find_symbol_by_gloss_alternative(self):
        """Test finding symbol by alternative gloss."""
        expected = "14905"
        result = self.composer._find_symbol_by_gloss("house")
        self.assertEqual(expected, result)

    def test_find_semantic_symbol(self):
        """Test finding symbol by semantic."""
        expected = "14647"
        result = self.composer._find_semantic_symbol("QUANTIFIER", "many")
        self.assertEqual(expected, result)

    def test_compose_from_semantic_spec(self):
        """Test composition from semantic spec."""
        semantic_spec = {
            "classifier": "building",
            "specifiers": ["medicine"],
            "semantics": {"QUANTIFIER": "many"}
        }

        expected = {
            "original_spec": {
                "classifier": "building",
                "specifiers": ["medicine"],
                "semantics": {"QUANTIFIER": "many"}
            },
            "composition": ["14905", "24920", "14647"],
            "errors": [],
            "warnings": []
        }
        result = self.composer.compose_from_semantic_spec(semantic_spec)
        self.assertEqual(expected, result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = {
            "1": {"pos": "YELLOW"}
        }
        self.engine = BlissEngine(self.bliss_dict)

    def test_use_case_1_nonexistent_symbol(self):
        """Test Use Case 1: getting a non-existent symbol returns error."""
        result = self.engine.get_symbol_glosses(99999)

        self.assertIn("error", result)

    def test_use_case_2_lookup_with_only_rendering_elements(self):
        """Test Use Case 2: lookup with only rendering elements."""
        expected = {
            "composition": ["/", ";", "/"],
            "is_existing_symbol": False,
            "error": "No valid symbol IDs found in composition",
            "details": {
                "classifier": None,
                "specifiers": [],
                "indicators": [],
                "modifiers": [],
                "errors": ["No valid symbol IDs found in composition"]
            }
        }
        result = self.engine.lookup_composition(["/", ";", "/"], language="en")
        self.assertEqual(expected, result)

    def test_compose_missing_classifier(self):
        """Test composition with missing classifier."""
        semantic_spec = {
            "specifiers": ["medicine"]
        }

        result = self.engine.compose_from_semantic(semantic_spec)

        self.assertIn("error", result)

    def test_empty_composition(self):
        """Test analyzing empty composition."""
        result = self.engine.analyze_composition([])

        # Should handle gracefully
        self.assertIsNotNone(result)


class TestLanguageSupport(unittest.TestCase):
    """Test multi-language support."""

    def setUp(self):
        """Create test Bliss dictionary with multiple languages."""
        self.bliss_dict = bliss_dict

        self.engine = BlissEngine(self.bliss_dict)

    def test_get_glosses_english(self):
        """Test getting glosses in English."""
        expected = {
            "id": 14905,
            "glosses": [
                "house",
                "building",
                "dwelling",
                "residence"
            ],
            "explanation": "(foundation + protection: pictograph of the outline of a house.The symbol can also be explained as: combination of enclosure and protection.)  - Character (superimposed)",
            "isCharacter": True
        }
        result = self.engine.get_symbol_glosses(14905, language="en")
        self.assertEqual(expected, result)

    def test_get_glosses_swedish(self):
        """Test getting glosses in Swedish."""
        expected = {
            "id": 14905,
            "glosses": [
                "hus",
                "byggnad"
            ],
            "explanation": "(foundation + protection: pictograph of the outline of a house.The symbol can also be explained as: combination of enclosure and protection.)  - Character (superimposed)",
            "isCharacter": True
        }
        result = self.engine.get_symbol_glosses(14905, language="sv")
        self.assertEqual(expected, result)

    def test_get_glosses_french(self):
        """Test getting glosses in French."""
        expected = {
            "id": 14905,
            "glosses": [
                "maison",
                "bâtiment"
            ],
            "explanation": "(foundation + protection: pictograph of the outline of a house.The symbol can also be explained as: combination of enclosure and protection.)  - Character (superimposed)",
            "isCharacter": True
        }
        result = self.engine.get_symbol_glosses(14905, language="fr")
        self.assertEqual(expected, result)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
