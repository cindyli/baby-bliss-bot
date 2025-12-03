# Bliss Engine API Documentation

## Overview

The Bliss Engine is a language-independent rule-based module for Blissymbolics that supports composition rules at the word level. It provides functionality to:

1. **Retrieve glosses** for existing Bliss symbols and compositions
2. **Analyze new compositions** to extract combined semantic meaning
3. **Compose new Bliss words** from semantic specifications

## Installation

```python
from bliss_engine import BlissEngine
import json

# Load the Bliss dictionary (already in dict format keyed by ID)
with open('path/to/bliss_dict_multi_langs.json', 'r') as f:
    bliss_dict = json.load(f)

# Initialize the engine
engine = BlissEngine(bliss_dict)
```

## Module Architecture

### Core Modules

#### `BlissEngine` (bliss_engine.py)
Main unified interface providing access to all engine functionality.

#### `SymbolClassifier` (symbol_classifier.py)
Classifies symbols into their functional roles:
- **Classifiers**: Set the semantic category (POS: YELLOW, RED, GREEN, BLUE)
- **Specifiers**: Refine the meaning of the classifier
- **Indicators**: Denote grammatical information (POS: GREY, WHITE)
- **Modifiers**: Prefix/suffix symbols that modify meaning (POS: GREY, WHITE)

#### `BlissAnalyzer` (analyzer.py)
Analyzes Bliss symbols and compositions to extract semantic meaning.

#### `BlissComposer` (composer.py)
Composes new Bliss words from semantic specifications.

## API Reference

### BlissEngine Class

#### Initialization
```python
engine = BlissEngine(bliss_dict)
```

**Parameters:**
- `bliss_dict` (Dict): Python dictionary keyed by symbol ID, with symbol data as values

**Raises:**
- `TypeError`: If bliss_dict is not a dictionary

---

### Use Case 1: Retrieve Glosses

#### `get_symbol_glosses(symbol_id, language="en")`
Get glosses for a single Bliss symbol.

**Parameters:**
- `symbol_id` (str): The ID of the Bliss symbol
- `language` (str): ISO 639-1 language code (default: "en")

**Returns:** Dict with keys:
- `id`: The symbol ID
- `glosses`: List of glosses in requested language
- `explanation`: Explanation of the symbol
- `isCharacter`: Boolean indicating if Bliss character

**Example:**
```python
result = engine.get_symbol_glosses("14905", language="en")
# {
#   "id": "14905",
#   "glosses": ["building"],
#   "explanation": "A structure...",
#   "isCharacter": true
# }
```

#### `get_composition_glosses(composition, language="en")`
Get glosses for a Bliss composition/word.

**Parameters:**
- `composition` (List[str] or List[int]): Symbol IDs in composition order
- `language` (str): ISO 639-1 language code

**Returns:** Dict with:
- `composition`: Input composition
- `components`: List of gloss info for each component

**Example:**
```python
result = engine.get_composition_glosses([14647, 14905, 9011])
# Returns gloss info for "many buildings"
```

---

### Use Case 2: Analyze Compositions

#### `analyze_composition(composition, language="en")`
Analyze a new Bliss composition and extract semantic meaning.

**Parameters:**
- `composition` (List[str] or List[int]): Symbol IDs to analyze
- `language` (str): ISO 639-1 language code

**Returns:** Dict with:
- `original_composition`: Input composition
- `classifier`: ID of the classifier symbol
- `classifier_info`: Gloss info for classifier
- `specifiers`: List of specifier IDs
- `specifier_info`: Gloss info for each specifier
- `semantics`: Extracted semantic information
- `indicators`: List of indicator IDs
- `modifiers`: List of modifier IDs

**Example:**
```python
result = engine.analyze_composition([14647, 14905, 24920, 9011])
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

#### `get_composition_structure(composition)`
Get the structural breakdown of a composition.

**Parameters:**
- `composition` (List[str] or List[int]): Symbol IDs

**Returns:** Dict with structural and interpretative information

---

### Use Case 3: Compose from Semantic

#### `compose_from_semantic(semantic_spec)`
Compose a new Bliss word from semantic specification.

**Parameters:**
- `semantic_spec` (Dict): Specification with:
  - `classifier` (str): Gloss for semantic category
  - `specifiers` (List[str], optional): Glosses refining meaning
  - `semantics` (List[Dict], optional): Semantic modifications
    - Each entry format: `{"SEMANTIC_TYPE": "value"}`

**Returns:** Dict with:
- `composition`: List of symbol IDs
- `errors`: Any composition errors
- `warnings`: Non-critical issues

**Example:**
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
# {
#   "composition": ["14647", "14905", "24920", "9011"],
#   "original_spec": {...},
#   "errors": [],
#   "warnings": []
# }
```

#### `compose_with_ids(classifier_id, specifier_ids=None, modifier_ids=None, indicator_ids=None)`
Compose a Bliss word using symbol IDs directly.

**Parameters:**
- `classifier_id` (str): ID of classifier symbol
- `specifier_ids` (List[str], optional): Specifier symbol IDs
- `modifier_ids` (List[str], optional): Modifier symbol IDs
- `indicator_ids` (List[str], optional): Indicator symbol IDs

**Returns:** Dict with composition and validation info

**Example:**
```python
result = engine.compose_with_ids(
    classifier_id="14905",
    specifier_ids=["24920"],
    modifier_ids=["14647"],
    indicator_ids=["9011"]
)
# {
#   "composition": ["14647", "14905", "24920", "9011"],
#   "errors": [],
#   "warnings": []
# }
```

---

### Utility Methods

#### `get_symbol_info(symbol_id)`
Get comprehensive information about a symbol.

**Returns:** Dict with:
- `id`: Symbol ID
- `pos`: Part-of-speech category
- `glosses`: Glosses in all languages
- `isCharacter`: Boolean
- `explanation`: Symbol explanation
- `type`: Symbol type (modifier, indicator, character_or_word)
- `semantics`: Semantic information if applicable

#### `classify_symbols(symbol_ids)`
Classify symbols by their functional roles.

**Parameters:**
- `symbol_ids` (List[str]): Symbol IDs to classify

**Returns:** Dict with:
- `classifier`: Classifier symbol ID
- `specifiers`: List of specifier IDs
- `indicators`: List of indicator IDs
- `modifiers`: List of modifier IDs
- `errors`: Any classification errors

#### `is_classifier(symbol_id)`
Check if a symbol can be a classifier.

**Returns:** Boolean

#### `is_modifier(symbol_id)`
Check if a symbol is a modifier.

**Returns:** Boolean

#### `is_indicator(symbol_id)`
Check if a symbol is an indicator.

**Returns:** Boolean

#### `get_knowledge_graph_info()`
Get statistics about the knowledge graph.

**Returns:** Dict with:
- `nodes`: Number of nodes
- `edges`: Number of edges
- `description`: Graph description

---

## Symbol Classification Rules

### Composition Order

The standard Blissymbolics composition order is:
```
[modifier(s), classifier, indicator(s), specifier(s), ...]
```

Examples:
- `[14905, 9011, 24920]` → classifier=14905, indicator=9011, specifier=24920
- `[14647, 14905, 9011]` → modifier=14647, classifier=14905, indicator=9011

### Finding Classifiers (Priority Order)

The engine uses a multi-layered approach to identify classifiers:

**Rule 1: Indicator-Based (Highest Priority)**
If the composition contains any indicators (symbols in `INDICATOR_SEMANTICS`), the **classifier is the symbol immediately before the first indicator**. All symbols before the classifier are modifiers/prefixes.

Example: In `[14905, 9011, 24920]` where 9011 is an indicator, 14905 is the classifier.

**Rule 2: POS-Based**
Symbols with pos in: `{YELLOW, RED, GREEN, BLUE}` are classifiers.

**Rule 3: First Symbol Convention**
If all symbols have pos in `{GREY, WHITE}` (no standard classifiers or indicators), assume the first symbol is the classifier.

### Specifiers
Symbols that appear after indicators (typically refining the classifier's meaning) or between the classifier and first indicator if no indicators exist.

### Indicators
Symbols with pos in: `{GREY, WHITE}`

Found in `INDICATOR_SEMANTICS` in `bliss_semantics.py`

Denote grammatical information like:
- Part of speech (noun, verb, adjective)
- Tense (past, present, future)
- Number (singular, plural)
- Voice (active, passive)

Typically appear after the classifier in compositions.

### Modifiers
Symbols with pos in: `{GREY, WHITE}`

Found in `MODIFIER_SEMANTICS` in `bliss_semantics.py`

Used as prefixes and suffixes to modify meaning:
- Quantifiers: "many", "few"
- Intensifiers: "very"
- Operators: "opposite", "part of"
- Comparatives: "more", "most"

Typically appear before the classifier (prefixes) or after indicators (suffixes).

### Special Case
If all symbols in a composition have pos in `{GREY, WHITE}`, the first symbol is assumed to be the classifier.

---

## Semantic Types

### Indicator Semantics
- **POS**: Part of speech (noun, verb, adjective, adverb)
- **TENSE**: Temporal information (past, present, future)
- **NUMBER**: Quantity (singular, plural)
- **VOICE**: Active/passive
- **ASPECT**: Verb aspect (continuous)
- **GENDER**: Grammatical gender
- **PERSON**: Grammatical person
- **DEFINITENESS**: Definite/indefinite
- **COMPARISON**: Comparative/superlative

### Modifier Semantics
- **QUANTIFIER**: "many", "few"
- **INTENSIFIER**: "very", "high intensity"
- **NEGATION**: "without"
- **OPERATOR**: "opposite", "generalization", "part of"
- **TIME**: "ago", "now", "future"
- **COMPARISON**: "more", "most"
- **POSSESSION**: "belongs to"
- **NUMBER**: Specific numbers (0-9)
- **USAGE_NOTE**: "metaphor", "slang"

---

## Dictionary Structure

The Bliss dictionary is a **Python dictionary keyed by symbol ID**, where each key is a symbol ID (string) and each value contains the symbol data.

**Symbol Data:**
- `pos`: Part-of-speech category (YELLOW, RED, GREEN, BLUE, GREY, WHITE)
- `glosses`: Dict mapping language codes to gloss lists (e.g., `{"en": ["building"], "sv": ["byggnad"]}`)
- `isCharacter`: Boolean indicating if it's a Bliss character
- `explanation`: Text explanation of the symbol
- `composition`: List of component symbol IDs (for composed words)
- `symbolSemantics`: Optional semantic disambiguation information

**Example:**
```python
bliss_dict = {
    "14905": {
        "pos": "YELLOW",
        "glosses": {"en": ["building"], "sv": ["byggnad"]},
        "isCharacter": True,
        "explanation": "A structure...",
        "symbolSemantics": {...}  # optional
    },
    "9011": {
        "pos": "WHITE",
        "glosses": {"en": ["plural"]},
        "isCharacter": False,
        "semantics": {"type": "NUMBER", "value": "plural"}
    }
    # ... more symbols
}
```

---

## Error Handling

The engine provides graceful error handling with detailed error messages:

```python
result = engine.get_symbol_glosses("invalid_id")
# Returns: {"error": "Symbol invalid_id not found"}

result = engine.compose_from_semantic({"specifiers": ["medicine"]})
# Returns: {"error": "Missing required field: classifier", ...}
```

---

## Language Support

The engine supports any language defined in the Bliss dictionary glosses. Common language codes:

- `en`: English
- `sv`: Swedish
- `no`: Norwegian
- `fi`: Finnish
- `fr`: French
- `de`: German
- `es`: Spanish
- `ru`: Russian

Example:
```python
engine.get_symbol_glosses("14905", language="fr")
# Returns French glosses
```

---

## Performance Considerations

- The composer builds reverse lookup maps on initialization for O(1) gloss lookups
- Large compositions may be processed sequentially
- The knowledge graph should be loaded once and reused across calls

---

## Examples

See `examples.py` for comprehensive usage examples of all three use cases.

Run examples:
```bash
python bliss_engine/examples.py
```

---

## Testing

Unit tests are provided in `tests.py`:

```bash
python -m pytest bliss_engine/tests.py -v
```

Or:
```bash
python bliss_engine/tests.py
```

---

## Integration with Other Modules

The Bliss Engine is designed to work with:

- **utils_knowledge_graph.py**: For loading/saving the knowledge graph
- **bliss_semantics.py**: Source of truth for indicator/modifier semantics
- **bliss_dict**: The underlying Blissymbolics dictionary data
