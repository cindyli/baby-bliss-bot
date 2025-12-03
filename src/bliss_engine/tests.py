"""
Unit tests for the Bliss Engine module.

Tests all three use cases and utility functions.
"""

import unittest

# Support running from project root
try:
    from src.bliss_engine import BlissEngine
    from src.bliss_engine.symbol_classifier import SymbolClassifier
    from src.bliss_engine.analyzer import BlissAnalyzer
    from src.bliss_engine.composer import BlissComposer
except ImportError:
    from bliss_engine import BlissEngine
    from bliss_engine.symbol_classifier import SymbolClassifier
    from bliss_engine.analyzer import BlissAnalyzer
    from bliss_engine.composer import BlissComposer


class TestBlissEngineInitialization(unittest.TestCase):
    """Test engine initialization and setup."""

    def setUp(self):
        """Create a mock Bliss dictionary for testing."""
        self.bliss_dict = {
            "8484": {
                "pos": "YELLOW",
                "isCharacter": True,
                "glosses": {"en": ["%"], "sv": ["procent"]}
            },
            "14905": {
                "pos": "YELLOW",
                "isCharacter": True,
                "glosses": {"en": ["building"], "sv": ["byggnad"]}
            },
            "24920": {
                "pos": "BLUE",
                "isCharacter": True,
                "glosses": {"en": ["medicine"]}
            },
            "14647": {
                "pos": "WHITE",
                "isCharacter": False,
                "glosses": {"en": ["many"]},
                "semantics": {
                    "type": "QUANTIFIER",
                    "value": "many"
                }
            },
            "9011": {
                "pos": "WHITE",
                "isCharacter": False,
                "glosses": {"en": ["plural"]},
                "semantics": {
                    "type": "NUMBER",
                    "value": "plural"
                }
            }
        }

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
        self.bliss_dict = {
            "14905": {
                "pos": "YELLOW",
                "isCharacter": True,
                "glosses": {"en": ["building"], "sv": ["byggnad"]},
                "explanation": "A structure"
            },
            "24920": {
                "pos": "BLUE",
                "isCharacter": True,
                "glosses": {"en": ["medicine"]}
            },
            "14647": {
                "pos": "WHITE",
                "isCharacter": False,
                "glosses": {"en": ["many"]},
                "semantics": {"type": "QUANTIFIER", "value": "many"}
            },
            "9011": {
                "pos": "WHITE",
                "isCharacter": False,
                "glosses": {"en": ["plural"]},
                "semantics": {"type": "NUMBER", "value": "plural"}
            },
            "8998": {
                "pos": "WHITE",
                "isCharacter": False,
                "glosses": {"en": ["adjective"]},
                "semantics": {"type": "POS", "value": "adjective"}
            }
        }

        self.engine = BlissEngine(self.bliss_dict)

    def test_use_case_1_get_symbol_glosses(self):
        """Test Use Case 1: Get glosses for a symbol."""
        result = self.engine.get_symbol_glosses("14905", language="en")

        self.assertEqual(result["id"], "14905")
        self.assertIn("building", result["glosses"])

    def test_use_case_1_get_composition_glosses(self):
        """Test Use Case 1: Get glosses for a composition."""
        result = self.engine.get_composition_glosses([14647, 14905], language="en")

        self.assertEqual(result["composition"], ["14647", "14905"])
        self.assertEqual(len(result["components"]), 2)

    def test_use_case_2_analyze_composition(self):
        """Test Use Case 2: Analyze a composition."""
        result = self.engine.analyze_composition([14647, 14905, 24920, 9011])

        self.assertEqual(result["classifier"], "14905")
        self.assertIn("24920", result["specifiers"])
        self.assertIn("9011", result["indicators"])
        self.assertIn("14647", result["modifiers"])

    def test_use_case_2_get_composition_structure(self):
        """Test Use Case 2: Get composition structure."""
        result = self.engine.get_composition_structure([14905, 24920])

        self.assertEqual(result["structure"]["classifier"], "14905")
        self.assertIn("24920", result["structure"]["specifiers"])

    def test_use_case_3_compose_from_semantic(self):
        """Test Use Case 3: Compose from semantic specification."""
        semantic_spec = {
            "classifier": "building",
            "specifiers": ["medicine"],
            "semantics": [{"NUMBER": "plural"}]
        }

        result = self.engine.compose_from_semantic(semantic_spec)

        self.assertIn("composition", result)
        self.assertIn("14905", result["composition"])  # building


class TestSymbolClassifier(unittest.TestCase):
    """Test the SymbolClassifier module."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = {
            "14905": {"pos": "YELLOW"},
            "24920": {"pos": "BLUE"},
            "14647": {"pos": "WHITE"},
            "9011": {"pos": "WHITE"}
        }

        self.classifier = SymbolClassifier(self.bliss_dict)

    def test_is_classifier(self):
        """Test classifier identification."""
        self.assertTrue(self.classifier.is_classifier("14905"))
        self.assertTrue(self.classifier.is_classifier("24920"))
        self.assertFalse(self.classifier.is_classifier("14647"))

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
        result = self.classifier.classify_composition(["14647", "14905", "24920", "9011"])

        self.assertEqual(result["classifier"], "14647")
        self.assertIn("14905", result["specifiers"])
        self.assertIn("9011", result["indicators"])

    def test_classify_composition_with_rendering_markers(self):
        """Test composition classification with rendering markers like '/' and ';'."""
        # Composition may contain "/" and ";" which are rendering markers and should be filtered
        result = self.classifier.classify_composition(["14647", "/", "14905", ";", "24920", "/", "9011"])

        # Should only process numeric IDs
        self.assertEqual(result["classifier"], "14647")
        self.assertIn("14905", result["specifiers"])
        self.assertIn("9011", result["indicators"])
        self.assertEqual(len(result["errors"]), 0)

    def test_classify_composition_indicator_based_rule(self):
        """Test the indicator-based classifier rule: classifier is before first indicator."""
        # When composition has indicators, the symbol immediately before the first
        # indicator should be identified as the classifier
        # Composition: [14905 (classifier), 24920 (specifier), 9011 (indicator)]
        result = self.classifier.classify_composition(["14905", "24920", "9011"])

        # 14905 should be classifier (symbol before first indicator 9011)
        self.assertEqual(result["classifier"], "14905")
        self.assertIn("24920", result["specifiers"])
        self.assertIn("9011", result["indicators"])


class TestBlissAnalyzer(unittest.TestCase):
    """Test the BlissAnalyzer module."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = {
            "14905": {
                "pos": "YELLOW",
                "isCharacter": True,
                "glosses": {"en": ["building"]}
            },
            "14647": {
                "pos": "WHITE",
                "glosses": {"en": ["many"]},
                "semantics": {"type": "QUANTIFIER", "value": "many"}
            }
        }

        self.analyzer = BlissAnalyzer(self.bliss_dict)

    def test_get_symbol_glosses(self):
        """Test getting symbol glosses."""
        result = self.analyzer.get_symbol_glosses("14905")

        self.assertEqual(result["id"], "14905")
        self.assertIn("building", result["glosses"])

    def test_analyze_composition(self):
        """Test composition analysis."""
        result = self.analyzer.analyze_composition(["14905"])

        self.assertIsNotNone(result["classifier"])
        self.assertIn("glosses", str(result))

    def test_analyze_composition_with_rendering_markers(self):
        """Test composition analysis ignores rendering markers."""
        # Composition with rendering markers "/" and ";"
        result = self.analyzer.analyze_composition(["14905", "/", "24920", ";", "9011"])

        # Should only process numeric IDs
        self.assertIsNotNone(result["classifier"])
        self.assertIn("24920", result["specifiers"])
        self.assertIn("9011", result["indicators"])


class TestBlissComposer(unittest.TestCase):
    """Test the BlissComposer module."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = {
            "14905": {
                "pos": "YELLOW",
                "isCharacter": True,
                "glosses": {"en": ["building", "house"]}
            },
            "24920": {
                "pos": "BLUE",
                "isCharacter": True,
                "glosses": {"en": ["medicine"]}
            },
            "14647": {
                "pos": "WHITE",
                "glosses": {"en": ["many"]},
                "semantics": {"type": "QUANTIFIER", "value": "many"}
            },
            "9011": {
                "pos": "WHITE",
                "glosses": {"en": ["plural"]},
                "semantics": {"type": "NUMBER", "value": "plural"}
            }
        }

        self.composer = BlissComposer(self.bliss_dict)

    def test_find_symbol_by_gloss(self):
        """Test finding symbol by gloss."""
        result = self.composer._find_symbol_by_gloss("building")

        self.assertEqual(result, "14905")

    def test_find_symbol_by_gloss_alternative(self):
        """Test finding symbol by alternative gloss."""
        result = self.composer._find_symbol_by_gloss("house")

        self.assertEqual(result, "14905")

    def test_find_semantic_symbol(self):
        """Test finding symbol by semantic."""
        result = self.composer._find_semantic_symbol("QUANTIFIER", "many")

        self.assertEqual(result, "14647")

    def test_compose_from_semantic_spec(self):
        """Test composition from semantic spec."""
        semantic_spec = {
            "classifier": "building",
            "specifiers": ["medicine"],
            "semantics": [{"QUANTIFIER": "many"}]
        }

        result = self.composer.compose_from_semantic_spec(semantic_spec)

        self.assertIn("composition", result)
        self.assertIn("14905", result["composition"])

    def test_compose_with_ids(self):
        """Test composition with direct IDs."""
        result = self.composer.compose_with_modifiers(
            classifier_id="14905",
            specifier_ids=["24920"],
            modifier_ids=["14647"]
        )

        self.assertEqual(result["composition"][0], "14647")  # modifier
        self.assertEqual(result["composition"][1], "14905")  # classifier
        self.assertEqual(result["composition"][2], "24920")  # specifier


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create test Bliss dictionary."""
        self.bliss_dict = {
            "1": {"pos": "YELLOW"}
        }
        self.engine = BlissEngine(self.bliss_dict)

    def test_get_nonexistent_symbol(self):
        """Test getting a non-existent symbol."""
        result = self.engine.get_symbol_glosses("99999")

        self.assertIn("error", result)

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
        self.bliss_dict = {
            "14905": {
                "pos": "YELLOW",
                "isCharacter": True,
                "glosses": {
                    "en": ["building"],
                    "sv": ["byggnad"],
                    "fr": ["bâtiment"]
                }
            }
        }

        self.engine = BlissEngine(self.bliss_dict)

    def test_get_glosses_english(self):
        """Test getting glosses in English."""
        result = self.engine.get_symbol_glosses("14905", language="en")

        self.assertIn("building", result["glosses"])

    def test_get_glosses_swedish(self):
        """Test getting glosses in Swedish."""
        result = self.engine.get_symbol_glosses("14905", language="sv")

        self.assertIn("byggnad", result["glosses"])

    def test_get_glosses_french(self):
        """Test getting glosses in French."""
        result = self.engine.get_symbol_glosses("14905", language="fr")

        self.assertIn("bâtiment", result["glosses"])


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
