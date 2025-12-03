"""
Example usage and tests for the Bliss Engine.

Demonstrates all three primary use cases:
1. Retrieve glosses for existing Bliss IDs
2. Analyze new compositions and extract semantic meaning
3. Compose new Bliss words from semantic specifications
"""

import json

# Support running from project root
try:
    from src.bliss_engine import BlissEngine
except ImportError:
    try:
        from bliss_engine import BlissEngine
    except ImportError:
        from . import BlissEngine


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def example_use_case_1(engine):
    """
    USE CASE 1: Get glosses and explanations for existing Bliss symbols.

    If the input is a Bliss ID or an existing composition, return the glosses
    and its explanation based on the requested language code.
    """
    print_section("USE CASE 1: Retrieve Glosses for Existing Bliss Words")

    # Example 1a: Get gloss for a single symbol
    print("Example 1a: Get glosses for symbol ID 14905 (building)")
    print("-" * 70)
    result = engine.get_symbol_glosses("14905", language="en")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n\nExample 1b: Get gloss in another language (Swedish)")
    print("-" * 70)
    result = engine.get_symbol_glosses("14905", language="sv")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example 1c: Get glosses for a composition
    print("\n\nExample 1c: Get glosses for a composition [14647, 14905, 9011]")
    print("(which represents 'many buildings')")
    print("-" * 70)
    result = engine.get_composition_glosses([14647, 14905, 9011], language="en")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def example_use_case_2(engine):
    """
    USE CASE 2: Analyze new compositions and extract semantic meaning.

    If the input is a new Bliss-word composition, analyze its component
    Bliss IDs and return their combined semantic information.
    """
    print_section("USE CASE 2: Analyze New Compositions")

    # Example 2a: Analyze "many hospitals"
    # Composition: [14647, 14905, 24920, 9011]
    # - 14647: "many" (QUANTIFIER modifier)
    # - 14905: "building" (classifier)
    # - 24920: "medicine" (specifier)
    # - 9011: "plural" (NUMBER indicator)

    print("Example 2a: Analyze composition for 'many hospitals'")
    print("Composition: [14647, 14905, 24920, 9011]")
    print("  - 14647: QUANTIFIER modifier (many)")
    print("  - 14905: Classifier (building)")
    print("  - 24920: Specifier (medicine)")
    print("  - 9011: Indicator (plural)")
    print("-" * 70)

    result = engine.analyze_composition([14647, 14905, 24920, 9011], language="en")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example 2b: Get composition structure
    print("\n\nExample 2b: Get structural breakdown of the composition")
    print("-" * 70)
    result = engine.get_composition_structure([14647, 14905, 24920, 9011])
    print(json.dumps(result, indent=2, ensure_ascii=False))


def example_use_case_3(engine):
    """
    USE CASE 3: Compose new Bliss words from semantic specifications.

    If the input is a semantic JSON, return the Bliss ID or a Bliss composition.
    """
    print_section("USE CASE 3: Compose from Semantic Specifications")

    # Example 3a: Compose "many hospitals" from semantic spec
    print("Example 3a: Compose 'many hospitals' from semantic specification")
    print("-" * 70)

    semantic_spec = {
        "classifier": "building",
        "specifiers": ["medicine"],
        "semantics": [
            {"NUMBER": "plural"},
            {"QUANTIFIER": "many"}
        ]
    }

    print("Input semantic specification:")
    print(json.dumps(semantic_spec, indent=2))
    print("\nComposing...")

    result = engine.compose_from_semantic(semantic_spec)
    print("\nOutput composition:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example 3b: Compose using symbol IDs directly
    print("\n\nExample 3b: Compose using symbol IDs directly")
    print("-" * 70)
    print("Composing with:")
    print("  - Classifier: 14905 (building)")
    print("  - Specifiers: [24920] (medicine)")
    print("  - Modifiers: [14647] (many)")
    print("  - Indicators: [9011] (plural)")

    result = engine.compose_with_ids(
        classifier_id="14905",
        specifier_ids=["24920"],
        modifier_ids=["14647"],
        indicator_ids=["9011"]
    )
    print("\nComposed sequence:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def example_utility_methods(engine):
    """Demonstrate utility methods."""
    print_section("Utility Methods")

    # Get detailed symbol information
    print("Example 1: Get detailed symbol information")
    print("-" * 70)
    result = engine.get_symbol_info("14905")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Classify symbols in a composition
    print("\n\nExample 2: Classify symbols in a composition")
    print("-" * 70)
    result = engine.classify_symbols(["14647", "14905", "24920", "9011"])
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Check symbol types
    print("\n\nExample 3: Check symbol types")
    print("-" * 70)
    symbols = ["14905", "14647", "9011"]
    for sym_id in symbols:
        print(f"\nSymbol {sym_id}:")
        print(f"  - Is classifier: {engine.is_classifier(sym_id)}")
        print(f"  - Is modifier: {engine.is_modifier(sym_id)}")
        print(f"  - Is indicator: {engine.is_indicator(sym_id)}")

    # Get knowledge graph statistics
    print("\n\nExample 4: Knowledge graph information")
    print("-" * 70)
    result = engine.get_knowledge_graph_info()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    """
    Main function demonstrating Bliss Engine usage.

    This requires a knowledge graph to be loaded. The knowledge graph is typically
    created from the Bliss dictionary.
    """
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  BLISS ENGINE - Comprehensive Usage Examples".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    # Load Bliss dictionary
    print("\nLoading Bliss dictionary...")

    # Try multiple paths for the Bliss dictionary (from project root)
    dict_paths = [
        "src/data/bliss_dict/bliss_dict_multi_langs.json",
        "data/bliss_dict/bliss_dict_multi_langs.json",
        "./data/bliss_dict/bliss_dict_multi_langs.json",
        "../data/bliss_dict/bliss_dict_multi_langs.json",
    ]

    bliss_dict = None
    for path in dict_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                bliss_dict = json.load(f)
            # The Bliss dictionary is already a Python dict keyed by ID
            print(f"✓ Loaded Bliss dictionary from {path}")
            break
        except FileNotFoundError:
            continue

    if bliss_dict is None:
        print("ERROR: Could not load Bliss dictionary. Please ensure")
        print("  bliss_dict_multi_langs.json exists in the src/data/bliss_dict directory.")
        print("  When running from project root, use: python -m src.bliss_engine.examples")
        return

    # Initialize engine
    print("Initializing Bliss Engine...")
    engine = BlissEngine(bliss_dict)
    print("✓ Engine initialized successfully\n")

    # Run examples
    try:
        example_use_case_1(engine)
        example_use_case_2(engine)
        example_use_case_3(engine)
        example_utility_methods(engine)

        print("\n" + "="*70)
        print("  All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
