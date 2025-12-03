# Bliss Engine

A language-independent rule-based core module for Blissymbolics supporting composition rules at the word level.

## Overview

The Bliss Engine provides three primary capabilities:

1. **Retrieve Glosses**: Get glosses and explanations for existing Bliss symbols or compositions
2. **Analyze Compositions**: Analyze new compositions to extract combined semantic meaning
3. **Compose Words**: Create new Bliss words from semantic specifications

## Symbol Types

The engine recognizes and manages four types of Bliss symbols:

### Classifier and Specifier
- **Classifier**: The first symbol setting the semantic category (pos: YELLOW, RED, GREEN, BLUE)
- **Specifier**: Subsequent symbols refining the meaning

### Indicators
Small symbols denoting grammatical information:
- Part of speech (noun, verb, adjective)
- Tense (past, present, future)
- Number (singular, plural)
- Voice (active, passive)
- Other grammatical properties

Indicators have pos value: GREY or WHITE

### Modifiers
Prefixes and suffixes modifying word meaning:
- Quantifiers: "many", "few"
- Intensifiers: "very"
- Operators: "opposite", "part of"
- Comparatives: "more", "most"
- Temporal: "ago", "now"
- And more...

Modifiers have pos value: GREY or WHITE

## Knowledge Base

The Bliss Engine uses the Bliss dictionary from [../data/bliss_dict/bliss_dict_multi_langs.json](../data/bliss_dict/bliss_dict_multi_langs.json).

The dictionary is a **Python dictionary keyed by symbol ID**. Each key is a symbol ID (string), and each value is the symbol data.

Each symbol has:
- `isCharacter`: Boolean indicating if it's a Bliss-character (True) or composed word (False)
- `composition`: Component symbol IDs for composed words (renders "/" and ";" for display)
- `glosses`: Multi-language glosses keyed by language code
- `pos`: Part-of-speech category (YELLOW, RED, GREEN, BLUE, GREY, WHITE)
- `explanation`: Description of the symbol


### Symbol Semantics (disambiguation)

Each symbol entry in the dictionary may include an optional `symbolSemantics` object that provides machine-friendly disambiguation when multiple Bliss symbols share the same gloss text. This property is used by the engine to select the correct symbol for a given semantic intent.

Example symbol entry (schema):
```json
{
    "id": "14905",
    "glosses": {"en": ["building"]},
    "pos": "YELLOW",
    "isCharacter": true,
    "symbolSemantics": {
        "sense_id": "building.structure",
        "domain": "architecture",
        "register": "neutral",
        "features": {"countable": true}
    }
}
```

Key points:
- `symbolSemantics` is optional but recommended for symbols with ambiguous glosses.
- Typical fields:
    - `sense_id`: stable identifier for the specific sense (recommended)
    - `domain`: thematic domain (e.g., "architecture", "medicine")
    - `register`: style or usage (e.g., "formal", "colloquial", "neutral")
    - `features`: arbitrary key/value semantic features (e.g., {"countable": true})
- The engine treats `symbolSemantics` as authoritative for disambiguation when gloss-based lookups are ambiguous.

Behavioral changes and API notes
- Gloss resolution:
    - get_symbol_glosses / get_symbol_info now return `symbolSemantics` if present.
    - New internal lookup flow: gloss → candidate symbols → apply symbolSemantics filter (if provided) → select best match.
- Composition analysis:
    - analyze_composition includes `symbolSemantics` in each symbol's returned info.
- Composition from semantic:
    - compose_from_semantic accepts optional `symbolSemantics` hints inside `semantic_spec` to force a particular sense:
        ```python
        semantic_spec = {
            "classifier": "building",
            "symbolSemantics": {"domain":"architecture","sense_id":"building.structure"},
            ...
        }
        ```
    - If multiple matches remain after applying semantics, the engine falls back to pos/priorities and returns a warning listing candidates.
- New helper methods (exposed by engine):
    - `resolve_symbol_by_gloss(gloss, symbolSemantics=None, language='en')` — returns best matching symbol id(s).
    - `get_symbols_by_semantics(symbolSemantics)` — search dictionary by semantics attributes.

Implementation notes (scripts)
- Data:
    - Ensure ../data/bliss_dict/bliss_dict_multi_langs.json includes `symbolSemantics` where applicable.
- Loading:
    - Loader adds `symbolSemantics` to the reverse index and enables attribute-based filtering.
- Classification and composition modules:
    - SymbolClassifier, BlissAnalyzer, BlissComposer updated to consider `symbolSemantics` when:
        - Selecting classifier/specifier among candidates sharing glosses.
        - Mapping semantic_spec to concrete symbol ids.
- Tests:
    - Add unit tests for disambiguation: same gloss different domains/registers → correct symbol chosen with provided semantics.

Traceability
- When ambiguity is resolved using `symbolSemantics`, engine responses will include a `resolution` field explaining the decision and any fallen-back candidates to aid debugging.

This addition ensures reliable selection of the intended Bliss symbol sense when gloss text alone is insufficient.

********

Semantic mappings for indicators and modifiers are in [../data/bliss_semantics.py](../data/bliss_semantics.py).

## Use Cases

### Use Case 1: Retrieve Glosses

Get glosses and explanations for existing Bliss symbols or compositions.

```python
from bliss_engine import BlissEngine

# Get gloss for a single symbol
result = engine.get_symbol_glosses("14905", language="en")
# Returns glosses and explanation for "building"

# Get glosses for a composition
result = engine.get_composition_glosses([14647, 14905, 9011], language="en")
# Returns gloss information for each component
```

### Use Case 2: Analyze Compositions

Analyze new compositions to extract combined semantic information.

For example, "many hospitals" composition [14647, 14905, 24920, 9011]:
- 14647: Modifier (QUANTIFIER: "many")
- 14905: Classifier (concept: "building")
- 24920: Specifier (refinement: "medicine")
- 9011: Indicator (NUMBER: "plural")

```python
result = engine.analyze_composition([14647, 14905, 24920, 9011])
# Returns:
# {
#   "original_composition": ["14647", "14905", "24920", "9011"],
#   "classifier": "14905",
#   "classifier_info": {"gloss": ["building"]},
#   "specifiers": ["24920"],
#   "specifier_info": [{"gloss": ["medicine"]}],
#   "semantics": [
#     {"symbol_id": "14647", "modifier": {"QUANTIFIER": "many"}},
#     {"symbol_id": "9011", "indicator": {"NUMBER": "plural"}}
#   ],
#   "indicators": ["9011"],
#   "modifiers": ["14647"]
# }
```

### Use Case 3: Compose from Semantic

Create new Bliss compositions from semantic specifications.

```python
semantic_spec = {
    "classifier": "building",
    "specifiers": ["medicine"],
    "semantics": [
        {"NUMBER": "plural"},
        {"QUANTIFIER": "many"}
    ]
}

result = engine.compose_from_semantic(semantic_spec)
# Returns: {"composition": ["14647", "14905", "24920", "9011"]}
```

## Module Architecture

### Core Modules

- **`BlissEngine`** (bliss_engine.py): Main unified interface
- **`SymbolClassifier`** (symbol_classifier.py): Identifies symbol roles (classifier, specifier, indicator, modifier)
- **`BlissAnalyzer`** (analyzer.py): Analyzes compositions and extracts semantics
- **`BlissComposer`** (composer.py): Composes words from semantic specifications

### Supporting Files

- **`__init__.py`**: Package initialization and exports
- **`API_DOCUMENTATION.md`**: Complete API reference
- **`examples.py`**: Usage examples for all three use cases
- **`tests.py`**: Comprehensive unit tests

## Quick Start

### From Project Root

All scripts are designed to run from the project root directory:

```bash
# Run examples
python -m src.bliss_engine.examples

# Run tests
python -m unittest src.bliss_engine.tests -v

# Run specific test class
python -m unittest src.bliss_engine.tests.TestBlissEngineInitialization -v
```

### Python Code

```python
# When running from project root
from src.bliss_engine import BlissEngine
from src.data.bliss_semantics import MODIFIER_SEMANTICS, INDICATOR_SEMANTICS
import json

# Load Bliss dictionary (dict format keyed by ID)
with open('src/data/bliss_dict/bliss_dict_multi_langs.json', 'r') as f:
    bliss_dict = json.load(f)

# Initialize engine
engine = BlissEngine(bliss_dict)

# Use Case 1: Get glosses
glosses = engine.get_symbol_glosses("14905", language="en")

# Use Case 2: Analyze composition
analysis = engine.analyze_composition([14647, 14905, 24920, 9011])

# Use Case 3: Compose from semantic
composition = engine.compose_from_semantic({
    "classifier": "building",
    "specifiers": ["medicine"],
    "semantics": [{"NUMBER": "plural"}, {"QUANTIFIER": "many"}]
})
```

### Legacy Imports (from bliss_engine directory)

If running from the `src/bliss_engine` directory:

```python
from bliss_engine import BlissEngine
import json

with open('../data/bliss_dict/bliss_dict_multi_langs.json', 'r') as f:
    bliss_dict = json.load(f)

engine = BlissEngine(bliss_dict)
```

## Multi-Language Support

The engine supports any language defined in the Bliss dictionary:

```python
# English
engine.get_symbol_glosses("14905", language="en")  # "building"

# Swedish
engine.get_symbol_glosses("14905", language="sv")  # "byggnad"

# French
engine.get_symbol_glosses("14905", language="fr")  # "bâtiment"
```

## Symbol Classification Rules

### Composition Order

The standard Blissymbolics composition order is:
```
[modifier(s), classifier, indicator(s), specifier(s), ...]
```

Examples:
- `[14905, 9011, 24920]` → classifier=14905, indicator=9011, specifier=24920
- `[14647, 14905, 9011]` → modifier/classifier=14647, classifier=14905, indicator=9011

### Finding Classifiers (Priority Order)

The engine uses a multi-layered approach to identify classifiers:

**Rule 1: Indicator-Based (Highest Priority)**
If the composition contains any indicators (symbols in `INDICATOR_SEMANTICS`), the **classifier is the symbol immediately before the first indicator**. All symbols before the classifier are modifiers/prefixes.

Example: In `[14905, 9011, 24920]` where 9011 is an indicator, 14905 is the classifier.

**Rule 2: POS-Based**
Symbols with pos in: `{YELLOW, RED, GREEN, BLUE}` are classifiers.

**Rule 3: First Symbol Convention**
If all symbols have pos in `{GREY, WHITE}` (no standard classifiers or indicators), assume the first symbol is the classifier.

### Finding Specifiers
Symbols that appear after indicators (typically refining the classifier's meaning) or between the classifier and first indicator if no indicators exist.

### Finding Indicators
Symbols with pos in: `{GREY, WHITE}` found in `INDICATOR_SEMANTICS`. Indicators appear after the classifier in typical compositions.

### Finding Modifiers
Symbols with pos in: `{GREY, WHITE}` found in `MODIFIER_SEMANTICS`. Modifiers appear before the classifier (as prefixes) or after indicators (as suffixes).

## API Methods

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete method reference.

**Main Methods:**
- `get_symbol_glosses(symbol_id, language)` - Get glosses for a symbol
- `get_composition_glosses(composition, language)` - Get glosses for composition
- `analyze_composition(composition, language)` - Analyze semantic meaning
- `compose_from_semantic(semantic_spec)` - Create composition from semantics
- `compose_with_ids(...)` - Create composition from symbol IDs
- `classify_symbols(symbol_ids)` - Classify symbol roles
- `get_symbol_info(symbol_id)` - Get comprehensive symbol information

## Examples

Run examples demonstrating all use cases:

```bash
python -m bliss_engine.examples
```

## Testing

Run unit tests:

```bash
python -m pytest bliss_engine/tests.py -v
```

Or without pytest:

```bash
python bliss_engine/tests.py
```

## Integration

The Bliss Engine integrates with:
- **utils_knowledge_graph.py**: Load/save knowledge graphs
- **bliss_semantics.py**: Indicator and modifier semantic definitions
- **bliss_dict/**: Blissymbolics dictionary data

## Error Handling

The engine provides descriptive error messages:

```python
# Missing symbol
result = engine.get_symbol_glosses("invalid_id")
# Returns: {"error": "Symbol invalid_id not found"}

# Missing required field
result = engine.compose_from_semantic({"specifiers": ["medicine"]})
# Returns: {"error": "Missing required field: classifier"}
```

## Performance

- O(1) gloss lookups via reverse indexing
- Sequential composition processing
- Efficient symbol classification

Load knowledge graph once and reuse for multiple operations.
