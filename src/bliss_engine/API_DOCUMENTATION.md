# Bliss Engine API Documentation

## Overview

The Bliss Engine is a language-independent rule-based module for Blissymbolics supporting composition rules at the word level. For an overview and quick start guide, see [README.md](README.md).

This document provides complete API reference for all methods and their parameters.

## Installation

```python
from src.bliss_engine import BlissEngine, load_bliss_dict

# Load the Bliss dictionary, keyed by B-id
bliss_dict = load_bliss_dict('src/data/bliss_dict/bliss_symbol_explanations_english.json')

# Initialize the engine
engine = BlissEngine(bliss_dict)
```

## BlissEngine Class

The main unified interface providing access to all engine functionality.

### Initialization

```python
engine = BlissEngine(bliss_dict)
```

**Parameters:**
- `bliss_dict` (Dict): Python dictionary keyed by B-id (the "id" field, without the leading "B"), with symbol data as values

**Raises:**
- `TypeError`: If bliss_dict is not a dictionary
Look up a symbol ID in the Bliss dictionary and return its glosses and explanation.

**Parameters:**
- `symbol_id` (int): The ID of the Bliss symbol
- `language` (str): ISO 639-1 language code (default: "en")

**Returns:** Dict with keys:
- `id`: The symbol ID
- `glosses`: List of glosses in requested language
- `explanation`: Explanation of the symbol
- `isCharacter`: Boolean indicating if Bliss character
- `error` (if not found): Error message

**Example:**
```python
result = engine.get_symbol_glosses(392, language="en")
# {
#   "id": 392,
#   "glosses": ["building"],
#   "explanation": "A structure...",
#   "isCharacter": true
# }

# Error case:
result = engine.get_symbol_glosses(99999, language="en")
# {
#   "error": "Symbol 99999 not found"
# }
```

## API Methods

### get_symbol_glosses(symbol_id, language="en")
Look up a word composition in the Bliss dictionary. If the composition exists in the dictionary, returns the symbol ID with its glosses and explanation. If it doesn't exist, returns its semantic meaning.

Rendering elements such as "/" and ";" are automatically ignored when comparing compositions.

**Parameters:**
- `composition` (List[str] or List[int]): Symbol IDs composing the Bliss word (may include rendering elements)
- `language` (str): ISO 639-1 language code

**Returns:** Dict with:
- `composition`: Normalized input composition
- `is_existing_symbol`: Boolean indicating if composition exists in dictionary
- **If is_existing_symbol is True:**
  - `symbol_id`: The ID of the existing symbol
  - `glosses`: List of glosses for the symbol
  - `explanation`: Explanation of the symbol
- **If is_existing_symbol is False:**
  - `classifier`: ID of the classifier symbol
  - `classifier_info`: Gloss info for classifier
  - `specifiers`: List of specifier IDs
  - `specifier_info`: Gloss info for each specifier
  - `semantics`: Extracted semantic information
  - `indicators`: List of indicator IDs
  - `modifiers`: List of modifier IDs

**Example - Existing composition:**
```python
result = engine.lookup_composition([303, 86, 651, 842], language="en")
# {
#   "composition": [
#     303,
#     86,
#     651,
#     842
#   ],
#   "is_existing_symbol": true,
#   "symbol_id": 4809,
#   "glosses": [
#     "oval",
#     "elliptic",
#     "elliptical"
#   ],
#   "explanation": "(oval,ellipse + description indicator)"
# }
```

**Example - New composition:**
```python
result = engine.lookup_composition([368, 392, 958, 99], language="en")
# If new composition:
# {
#   "composition": [368, 392, 958, 99],
#   "is_existing_symbol": false,
#   "classifier": 392,
#   "classifier_info": "building",
#   "specifiers": [958],
#   "specifier_info": ["medicine"],
#   "indicators": [99],
#   "modifiers": [368],
#   "semantics": {
#     "NUMBER": "plural",
#     "QUANTIFIER": "many"
#   }
# }
```

**Example - Composition with rendering elements:**
```python
result = engine.lookup_composition([392, "/", 99, ";"], language="en")
# Rendering elements "/" and ";" are automatically filtered out
# {
#   "composition": [392, 99],
#   "is_existing_symbol": false|true,
#   ...
# }
```

#### `analyze_composition(composition, language="en")`
Analyze a composition and extract semantic meaning. This is used internally by `lookup_composition` for new compositions.

**Parameters:**
- `composition` (List[str] or List[int]): Symbol IDs to analyze
- `language` (str): ISO 639-1 language code

**Returns:** Dict with:
- `classifier`: ID of the classifier symbol
- `classifier_info`: Gloss info for classifier
- `specifiers`: List of specifier IDs
- `specifier_info`: Gloss info for each specifier
- `semantics`: Extracted semantic information
- `indicators`: List of indicator IDs
- `modifiers`: List of modifier IDs

---

### Use Case 3: Compose from Semantic

#### `compose_from_semantic(semantic_spec)`
Compose a new Bliss word from semantic specification.

**Parameters:**
- `semantic_spec` (Dict): Specification with:
  - `classifier` (str): Gloss for semantic category
  - `specifiers` (List[str], optional): Glosses refining meaning
  - `semantics` (Dict, optional): Semantic modifications
    - Format: `{"SEMANTIC_TYPE": "value", "SEMANTIC_TYPE2": "value2", ...}`

**Returns:** Dict with:
- `composition`: List of symbol IDs
- `errors`: Any composition errors
- `warnings`: Non-critical issues

**Example:**
```python
semantic_spec = {
    "classifier": "building",
    "specifiers": ["medicine"],
    "semantics": {
        "NUMBER": "plural",
        "QUANTIFIER": "many"
    }
}

result = engine.compose_from_semantic(semantic_spec)
# {
#   "composition": [368, 392, 958, 99],
#   "original_spec": {...},
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

#### `get_bliss_dict_info()`
Get statistics about the Blissymbolics dictionary.

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
- `[392, 99, 958]` → classifier=392, indicator=99, specifier=958
- `[368, 392, 99]` → modifier=368, classifier=392, indicator=99

### Finding Classifiers (Priority Order)

The engine uses a multi-layered approach to identify classifiers:

**Rule 1: Indicator-Based (Highest Priority)**
If the composition contains any indicators (symbols in `INDICATOR_SEMANTICS`), the **classifier is the symbol immediately before the first indicator**. All symbols before the classifier are modifiers/prefixes.

Example: In `[392, 99, 958]` where 99 is an indicator, 392 is the classifier.

**Rule 2: POS-Based**
Symbols with pos in: `{noun, action, person, description}` are classifiers.

**Rule 3: First Symbol Convention**
If all symbols have pos in `{expression, function}` (no standard classifiers or indicators), assume the first symbol is the classifier.

### Specifiers
Symbols that appear after indicators (typically refining the classifier's meaning) or between the classifier and first indicator if no indicators exist.

### Indicators
Symbols with pos in: `{expression, function}`

Found in `INDICATOR_SEMANTICS` in `bliss_semantics.py`

Denote grammatical information like:
- Part of speech (noun, verb, adjective)
- Tense (past, present, future)
- Number (singular, plural)
- Voice (active, passive)

Typically appear after the classifier in compositions.

### Modifiers
Symbols with pos in: `{expression, function}`

Found in `MODIFIER_SEMANTICS` in `bliss_semantics.py`

Used as prefixes and suffixes to modify meaning:
- Quantifiers: "many", "few"
- Intensifiers: "very"
- Operators: "opposite", "part of"
- Comparatives: "more", "most"

Typically appear before the classifier (prefixes) or after indicators (suffixes).

### Special Case
If all symbols in a composition have pos in `{expression, function}`, the first symbol is assumed to be the classifier.

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
- `pos`: Part-of-speech category (noun, action, person, description, expression, function)
- `glosses`: Dict mapping language codes to gloss lists (e.g., `{"en": ["building"], "sv": ["byggnad"]}`)
- `isCharacter`: Boolean indicating if it's a Bliss character
- `explanation`: Text explanation of the symbol
- `composition`: List of component symbol IDs (for composed words)
- `symbolSemantics`: Optional semantic disambiguation information

**Example:**
```python
bliss_dict = {
    "392": {
        "pos": "noun",
        "glosses": {"en": ["building"], "sv": ["byggnad"]},
        "isCharacter": True,
        "explanation": "A structure...",
        "symbolSemantics": {...}  # optional
    },
    "99": {
        "pos": "function",
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
result = engine.get_symbol_glosses(99999)
# Returns: {"error": "Symbol invalid_id not found"}

result = engine.compose_from_semantic({"specifiers": ["medicine"]})
# Returns: {"error": "Missing required field: classifier", ...}
```

---

## Language Support

The engine supports multiple languages defined in the Bliss dictionary glosses. Supported language codes:

- `en`: English
- `sv`: Swedish
- `no`: Norwegian
- `fi`: Finnish
- `hu`: Hungarian
- `de`: German
- `nl`: Dutch
- `af`: Afrikaans
- `ru`: Russian
- `lv`: Latvian
- `po`: Polish
- `fr`: French
- `es`: Spanish
- `pt`: Portugese - draft
- `it`: Italian - draft
- `dk`: Danish - draft

Note that The dictionary definitions for Portugese, Italian and Danish are in draft.

* Example:

```python
engine.get_symbol_glosses(392, language="fr")
# Returns French glosses
```

---

## Examples

Run examples demonstrating all use cases. Run commands from the project root directory:

```bash
python -m src.bliss_engine.examples
```

---

## Testing

Run unit tests:

```bash
python -m unittest src.bliss_engine.tests -v
```

---

## Integration with Other Modules

The Bliss Engine is designed to work with:

- **bliss_dict**: The underlying Blissymbolics dictionary data
- **bliss_semantics.py**: Source of truth for indicator/modifier semantics
