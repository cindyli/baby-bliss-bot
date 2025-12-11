# Bliss Engine

A language-independent rule-based core module for Blissymbolics supporting composition rules at the word level.

## Overview

The Bliss Engine is an interface for working with Blissymbolics symbols and compositions. It provides three primary capabilities:

1. **Retrieve Glosses**: Look up glosses and explanations for Bliss symbols
2. **Analyze Compositions**: Analyze word compositions to find proper symbol or extract semantic meaning
3. **Compose Words**: Create new Bliss words from semantic specifications

## Quick Start

### Installation

```python
from bliss_engine import BlissEngine
import json

# Load the Bliss dictionary
with open('path/to/bliss_dict_multi_langs.json', 'r') as f:
    bliss_dict = json.load(f)

# Initialize the engine
engine = BlissEngine(bliss_dict)
```

### Basic Usage

When the `language` parameter is not provided, the default language is `en` (English).

```python
# Use Case 1: Get glosses for a symbol
result = engine.get_symbol_glosses(14905, language="en")
# Returns: {"id": 14905, "glosses": ["building"], "explanation": "...", "isCharacter": true}

# Use Case 2: Analyze a composition
result = engine.lookup_composition([14647, 14905, 24920, 9011])
# Returns: semantic breakdown or symbol ID if exists

# Use Case 3: Compose from semantic specification
semantic_spec = {
    "classifier": "building",
    "specifiers": ["medicine"],
    "semantics": {"NUMBER": "plural", "QUANTIFIER": "many"}
}
result = engine.compose_from_semantic(semantic_spec)
# Returns: {"composition": [14647, 14905, 24920, 9011], ...}
```

For detailed API reference, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Symbol Types

The Bliss Engine recognizes four types of symbols in compositions:

### Classifiers
- Set the semantic category of the composition
- Part-of-speech (pos): YELLOW, RED, GREEN, BLUE
- Typically the primary meaning-bearing element

### Specifiers
- Refine the meaning of the classifier
- Can be any symbol not classified as indicator or classifier
- Appear after the classifier in typical compositions

### Indicators
- Denote grammatical information (pos: GREY, WHITE)
- Examples: number, tense, voice, aspect, case
- Appear after the classifier in composition order

### Modifiers
- Modify the classifier's meaning (pos: GREY, WHITE)
- Can be prefixes (before classifier) or suffixes (after indicators)
- Examples: quantifiers ("many", "few"), intensifiers ("very"), operators ("opposite")

## Knowledge Base

The Bliss Engine uses the Bliss dictionary from [../data/bliss_dict/bliss_dict_multi_langs.json](../data/bliss_dict/bliss_dict_multi_langs.json). The dictionary is a Python dictionary keyed by symbol ID where each entry contains:

- `isCharacter`: Boolean indicating if it's a Bliss character (True) or composed word (False)
- `composition`: Component symbol IDs for composed words
- `glosses`: Multi-language glosses keyed by language code
- `pos`: Part-of-speech category (YELLOW, RED, GREEN, BLUE, GREY, WHITE)
- `explanation`: Description of the symbol
- `symbolSemantics`: Optional semantic disambiguation information

Indicator and modifier semantic mappings are defined in [../data/bliss_semantics.py](../data/bliss_semantics.py).

## Module Architecture

The engine consists of four core modules:

- **`BlissEngine`** (bliss_engine.py): Main unified interface
- **`SymbolClassifier`** (symbol_classifier.py): Identifies symbol roles in compositions
- **`BlissAnalyzer`** (analyzer.py): Analyzes compositions and extracts semantics
- **`BlissComposer`** (composer.py): Composes words from semantic specifications

Additional resources:
- **`API_DOCUMENTATION.md`**: Complete API reference with all methods and parameters
- **`examples.py`**: Usage examples for all three use cases
- **`tests.py`**: Comprehensive unit tests

## Multi-Language Support

The engine supports multiple languages defined in the Bliss dictionary glosses:

- `en` (English), `sv` (Swedish), `no` (Norwegian), `fi` (Finnish), `hu` (Hungarian)
- `de` (German), `nl` (Dutch), `af` (Afrikaans), `ru` (Russian), `lv` (Latvian)
- `po` (Polish), `fr` (French), `es` (Spanish)
- `pt` (Portuguese - draft), `it` (Italian - draft), `dk` (Danish - draft)

```python
# Swedish glosses
engine.get_symbol_glosses(14905, language="sv")  # "byggnad"

# French glosses
engine.get_symbol_glosses(14905, language="fr")  # "bâtiment"
```

## Running Examples and Tests

All scripts should be run from the project root directory:

```bash
# Run usage examples
python -m src.bliss_engine.examples

# Run all tests
python -m unittest src.bliss_engine.tests -v

# Run specific test class
python -m unittest src.bliss_engine.tests.TestBlissEngineInitialization -v
```

## Symbol Classification Rules

For detailed information about symbol classification rules, composition order, and semantic type definitions, refer to the [Symbol Classification Rules section in API_DOCUMENTATION.md](API_DOCUMENTATION.md#symbol-classification-rules).
