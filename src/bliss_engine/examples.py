# Run from project root: python -m src.bliss_engine.examples

"""
Example usage and tests for the Bliss Engine.

Demonstrates all three primary use cases:
1. Look up a symbol ID in the Bliss dictionary
2. Look up a word composition in the Bliss dictionary
3. Compose new Bliss words from semantic specifications
"""

import json

# Support running from project root
from src.bliss_engine import BlissEngine


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def example_use_case_1(engine):
    """
    USE CASE 1: Look up a symbol ID in the Bliss dictionary.

    Given a symbol ID, look up in the Bliss dictionary to find the symbol
    and return its glosses and explanation. If not found, return an error.
    """
    print_section("USE CASE 1: Look Up Symbol by ID")

    # Example 1a: Get gloss for a single symbol
    print("Example 1a: Look up symbol ID 14905 (building)")
    print("-" * 70)
    result = engine.get_symbol_glosses(14905, language="en")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n\nExample 1b: Look up symbol in another language (Swedish)")
    print("-" * 70)
    result = engine.get_symbol_glosses(14905, language="sv")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example 1c: Look up non-existent symbol
    print("\n\nExample 1c: Look up non-existent symbol (error case)")
    print("-" * 70)
    result = engine.get_symbol_glosses(99999, language="en")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def example_use_case_2(engine):
    """
    USE CASE 2: Look up a word composition in the Bliss dictionary.

    Given a word composition, first look up in the Bliss dictionary to find
    if the composition is already in the dictionary by looking up the "composition"
    values.

    If it exists, return the symbol ID with its glosses and explanation.
    If it doesn't exist, return its semantic meaning.

    Includes an "is_existing_symbol" flag to indicate if this composition
    already exists in the dictionary.

    Note: When comparing the input composition with the one in the dictionary,
    rendering elements such as "/" and ";" are ignored.
    """
    print_section("USE CASE 2: Look Up Word Composition")

    # Example 2a: Look up an existing composition
    print("Example 2a: Look up existing composition")
    print("-" * 70)
    result = engine.lookup_composition([14133, 8998, 17717, 23599], language="en")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example 2b: Look up a new composition (not in dictionary)
    # Composition: [14647, 14905, 24920, 9011]
    # This represents "many hospitals"
    print("\n\nExample 2b: Look up new composition [14647, 14905, 24920, 9011] ('many hospitals')")
    print("Composition structure:")
    print("  - 14647: QUANTIFIER modifier (many)")
    print("  - 14905: Classifier (building)")
    print("  - 24920: Specifier (medicine)")
    print("  - 9011: Indicator (plural)")
    print("-" * 70)
    result = engine.lookup_composition([14647, 14905, 24920, 9011], language="en")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example 2c: Look up composition with rendering elements
    print("\n\nExample 2c: Look up existing composition with rendering elements")
    print("(Rendering elements like '/' and ';' are ignored in comparison)")
    print("-" * 70)
    result = engine.lookup_composition([14133, ";", 8998, "/", 17717, "/", 23599], language="en")
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
        "semantics": {
            "NUMBER": "plural",
            "QUANTIFIER": "many"
        }
    }

    print("Input semantic specification:")
    print(json.dumps(semantic_spec, indent=2))
    print("\nComposing...")

    result = engine.compose_from_semantic(semantic_spec)
    print("\nOutput composition:")
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

    # Get Blissymbolics dictionary statistics
    print("\n\nExample 4: Blissymbolics dictionary information")
    print("-" * 70)
    result = engine.get_bliss_dict_info()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    """
    Main function demonstrating Bliss Engine usage.

    This requires a Blissymbolics dictionary to be loaded. The dictionary is typically
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
