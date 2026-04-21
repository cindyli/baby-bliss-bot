"""
===============================================================================
README — Blissymbolics Linguistic Annotation Schema
===============================================================================
This file defines metadata and semantics for Blissymbolics indicators
and modifiers. It uses a structured annotation schema to represent
linguistic information such as part of speech, grammatical features,
semantic shifts, and usage notes.

-------------------------------------------------------------------------------
Core Attributes
-------------------------------------------------------------------------------

Each entry may contain the following attributes:

1. Type: Optional. Specifies the kind of annotation the symbol represents.
Valid values (exactly one):
     - POS          : Part of speech (e.g., noun, verb)
     - TYPE_SHIFT   : Transforms one POS into another (e.g., verb → noun)

2. Type Value: Optional. Identifies the specific value for the selected Type.
   - POS values: "noun", "verb", "adjective", "adverb" (Exactly one)
   - TYPE_SHIFT value: "concretization"

3. Category: Optional. Broad linguistic grouping. Valid values (one or more):
     - "grammatical"
     - "semantic"
     - "syntactical"

4. Features: Fine‑grained linguistic properties. Available features depend on
POS and may have single or multiple values, as specified below.

-------------------------------------------------------------------------------
Features by Part of Speech
-------------------------------------------------------------------------------
VERBS
-----
- tense: Locates an action in time.
Valid values: "null" | "past" | "present" | "future" (one)

- voice: Shows relationship between the subject and action.
Valid values: "null" | "active" | "passive" (one)

- mood: Expresses attitude or intent.
Valid values: "declarative" | "conditional" | "imperative" (one)
Note: mood may vary language to language on how its used. Declarative is
assumed unless question/exclamation markers are present.

- aspect: Indicates how an action occurs over time.
Valid values: "continuous"

- form: Variations of verbs.
Valid values: "inflected" | "infinitive" | "present-participle" | "past-participle-1" |
 "past-participle-2" (one)
Note: If tense, voice, aspect, and mood are all "null", the verb is treated as infinitive.

- negation: "without" | "not" | "opposite" (one)

NOUNS
-----
- number: "singular" | "plural" (one)

- definiteness: Identifies a specific or general thing.
Valid values: "indefinite" | "definite" (one)
Example: "an apple" (indefinite), "the apple" (definite)

- gender: "neutral" | "feminine" | "masculine" (one)

- person: "first-person" | "second-person" | "third-person" (one)

- size: "diminutive"

- possessive: "possessor"

- position: "pre" | "post" (one or more)
Syntax note:
    - pre: modifier before head (e.g., "colour of the car")
    - post : modifier after head (e.g., "car's colour")

- default-position  : "pre" | "post" (one)

- quantifier: "many, much" | "all" | "any" | "both" | "each, every" | "either" | "neither" | "half" | "quarter" | "one third" | "two thirds" | "three quarters" | "several" (one)

- link: "association" | "derivative" (one)
Example:
    - furniture ↔ chair (association)
    - province → country (derivative)

- time: "ago, then (past)" | "now" | "then_future, so, later" (one)
Note: Attaches to nouns but functions adverbially.

- numeric: "zero" → "nine" (one)

- negation: "without" | "not" | "opposite" (one)

ADJECTIVES & ADVERBS
-------------------
- modality: Represents whether something is possible or realized.
Valid values: "potential" | "completed" (one)

- degree: "intensity" | "more (comparative)" | "most (comparative)" (one)

- negation: "without" | "not" | "opposite" (one)

-------------------------------------------------------------------------------
Additional Metadata
-------------------------------------------------------------------------------

- equivalent_modifier / equivalent_indicator: References the ID of an equivalent
Blissymbolics indicator or modifier. (one)

- priority: Determines processing precedence. Represented as a list of IDs ordered
from highest to lowest priority.

Note: Action and description indicators are commonly used across users, while
present‑action and adverb indicators are more typical in full‑form usage.
"""

# Blissymbolics Indicators
INDICATOR_SEMANTICS = {
    # action indicators
    # infinitive verb or present tense verb; similar to ID: 24807 (includes tense as present), here is doesn't include tense
    "8993": {
        "POS": "verb",
        "category": "grammatical",
        "features": {
            "form": "infinitive"
        },
        "priority": ["8993", "24807"]
    },
    # active verb
    "8994": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "active", "mood": "declarative", "form": "inflected"}
    },
    # the equivalent of the English present conditional form
    "8995": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "active", "mood": "conditional", "form": "inflected"}
    },

    # description indicators
    # the equivalent of the English -ed or -en ending
    "8996": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "features": {"modality": "completed"}
    },
    # equivalent to English words ending in -able
    "8997": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "features": {"modality": "potential"}
    },
    # the equivalent of English adjectives/adverbs
    "8998": {
        "POS": ["adjective", "adverb"],
        "category": "semantic",
        "priority": ["8998", "24665"]
    },
    # back to action indicators
    # the equivalent of the English future tense
    "8999": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "active", "mood": "declarative", "form": "inflected"}
    },
    # the equivalent of the English future conditional form
    "9000": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "active", "mood": "conditional", "form": "inflected"}
    },
    # the equivalent of the English future passive form
    "9001": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "passive", "mood": "declarative", "form": "inflected"}
    },
    # the equivalent of the English future passive conditional form
    "9002": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "future", "voice": "passive", "mood": "conditional", "form": "inflected"}
    },
    # something is being acted upon
    "9003": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "passive", "mood": "declarative", "form": "inflected"}
    },
    # the equivalent of the English past tense
    "9004": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "active", "mood": "declarative", "form": "inflected"}
    },
    # the equivalent of the English past conditional form
    "9005": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "active", "mood": "conditional", "form": "inflected"}
    },
    # the equivalent of the English past passive conditional form
    "9006": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "passive", "mood": "conditional", "form": "inflected"}
    },
    # the equivalent of the English past passive form
    "9007": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "past", "voice": "passive", "mood": "declarative", "form": "inflected"}
    },
    # the equivalent of the English present passive conditional form
    "9008": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "passive", "mood": "conditional", "form": "inflected"}
    },

    # represent a concrete object
    "9009": {
        "and": [{
            "POS": "noun",
            "category": "grammatical"
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },

    # represent multiple concrete objects
    "9010": {
        "and": [{
            "POS": "noun",
            "category": "grammatical",
            "features": {"number": "plural"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },
    "9011": {
        "category": "grammatical",
        "features": {"number": "plural"}
    },
    "24667": {
        "category": "grammatical",
        "features": {"definiteness": "definite", "number": "singular"},
        "notes": "for teaching purposes"
    },
    # the female modifier (ID: 14166) is used more. Indicator is not used in communication
    "24668": {
        "category": "grammatical",
        "features": {"gender": "feminine", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "14166",
        "priority": ["14166", "24668"]
    },
    "12335": {
        "category": "grammatical",
        "features": {"gender": "masculine", "number": "singular"}
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24669": {
        "category": "grammatical",
        "features": {"person": "first-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8497",
        "priority": ["8497", "24669"]
    },
    # the past participle form
    "28044": {
        "category": "grammatical",
        "features": {"number": "plural", "definiteness": "definite"}
    },
    "28045": {
        "and": [{
            "category": "grammatical",
            "features": {"definiteness": "definite", "number": "singular"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },
    "28046": {
        "and": [{
            "POS": "noun",
            "category": "grammatical",
            "features": {"number": "plural", "definiteness": "definite"}
        }, {
            "TYPE_SHIFT": "concretization",
            "category": "semantic"
        }]
    },

    # indicator (adverb)
    "24665": {
        "POS": "adverb",
        "category": "grammatical",
        "notes": "for teaching purposes",
        "priority": ["8998", "24665"]
    },
    # similar to ID: 8993;
    "24807": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "null", "mood": "declarative", "aspect": "null", "form": "inflected"},
        "notes": "for teaching purposes",
        "priority": ["8993", "24807"]
    },
    # the diminutive modifier is used more. Indicator (ID: 28052) is not used
    "25458": {
        "category": "grammatical",
        "features": {"size": "diminutive", "form": "inflected"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "28052",
        "priority": ["28052", "25458"]
    },
    # imperative mood
    "24670": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": "inflected"}
    },
    # 3 participles
    "24674": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": "past-participle-1"},
        "notes": "for teaching purposes"
    },
    "24675": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": "past-participle-2"},
        "notes": "for teaching purposes"
    },
    "24677": {
        "POS": ["verb", "adjective"],
        "category": "grammatical",
        "features": {"form": "present-participle"},
        "notes": "for teaching purposes"
    },
    # back to nouns
    "24671": {
        "category": "grammatical",
        "features": {"definiteness": "indefinite", "number": "singular"},
        "notes": "for teaching purposes"
    },
    "24672": {
        "category": "grammatical",
        "features": {"gender": "neutral", "number": "singular"},
        "notes": "for teaching purposes"
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24678": {
        "category": "grammatical",
        "features": {"person": "second-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8498",
        "priority": ["8498", "24678"]
    },
    "24679": {
        "category": "grammatical",
        "features": {"person": "third-person", "number": "singular"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8499",
        "priority": ["8499", "24679"]
    },

    # possessive indicator; both indicator and modifier (ID: 12663) are used, but modifier is used more in English (opposite is true for Swedish).
    "24676": {
        "POS": "noun",
        "category": ["grammatical", "syntactical"],
        "features": {
            "grammatical": {"possessive": "possessor"},
            "syntactical": {
               "position": ["pre", "post"],
               "default-position": "post"
            },
        },
        "notes": "for teaching purposes",
        "equivalent_modifier": "12663",
        "priority": ["12663", "24676"]
    },
    # object form; can use object form with or without indicator - is an alternative, modifier (ID: 28057) has never been used
    "24673": {
        "POS": "noun",
        "category": "syntactical",
        "features": {"position": ["pre", "post"], "default-position": "post"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "28057",
        "priority": ["optional", "24673", "28057"]
    },
}

# Blissymbolics Modifiers
MODIFIER_SEMANTICS = {
    # "B314"
    "14166": {
        "features": {
           "gender": "feminine",
           "number": "singular",
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "equivalent_indicator": "24668",
        "priority": ["14166", "24668"]
    },
    # "B10"
    "8497": {
        "features": {
           "person": "first-person",
           "number": "singular",
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "equivalent_indicator": "24669",
        "priority": ["8497", "24669"],
    },
    # "B11"
    "8498": {
        "features": {
            "person": "second-person",
            "number": "singular",
            "position": "suffix",
            "middle-position": "suffix-first-part",
        },
        "equivalent_indicator": "24678",
        "priority": ["8498", "24678"],
    },
    # "B12"
    "8499": {
        "features": {
            "person": "third-person",
            "number": "singular",
            "position": "suffix",
            "middle-position": "suffix-first-part",
        },
        "equivalent_indicator": "24679",
        "priority": ["8499", "24679"],
    },
    # "B5999"
    "28052": {
        "features": {
            "size": "diminutive",
            "position": "suffix",
            "middle-position": "suffix-first-part",
        },
        "equivalent_indicator": "25458",
        "priority": ["28052", "25458"],
    },

    # "B112"
    "12352": {
        "features": {
           "time": "ago, then (past)",
           "position": "suffix",
    },

    # "B648"
    "17705": {
        "features": {
           "time": "then_future, so, later",
           "position": "suffix",
    },

    # "B474"
    "15736": {
        "features": {
           "time": "now",
           "position": "suffix",
    },
         
    # Structural markers
    # "B233"
    "13382": {
        "meaning": "combine marker",
        "notes": "special case (combine marker acts like quotation marks surrounding a set of symbols)"
    },

    # What
    # "B699"
    "18229": {
        "meaning": "what",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "notes": "interrogative when used as a prefix, otherwise a specifier"
    },

    # Scalar degree operators
    # "B401"
    "14947": {
        "features": {
           "degree": "intensity",
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "notes": "exclamatory when used as a prefix, otherwise a specifier"
    },
    # "B937", has different position for different context - need to discuss
    "24879": {
        "features": {
           "degree": "more (comparative)",
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "notes": "position is prefix if positive context"
    },
    # "B968", has different position for different context - need to discuss
    "24944": {
        "features": {
          "degree": "most (comparative)",
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "notes": "position is prefix if positive context"
    },

    # Identity-affecting operators
    # "B449/B401"
    "15733": {
        "features": {
          "negation": "not, negative, no, don't, doesn't",
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "priority": ["15474", "15733", "15927"]
    },
    # "B486"
    "15927": {
        "features": {
          "negation": "opposite",
           "position": "prefix",
           "middle-position": "prefix-second-part"
        },
        "priority": ["15474", "15733", "15927"]
    },
    # Concept-transforming operators
    # "B1060/B578"
    "16984": {
        "meaning": "similar to",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B1060/B578/B303"
    "16985": {
        "meaning": "look similar to",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B1060/B578/B608"
    "16986": {
        "meaning": "sound similar to",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B578/B608"
    "16714": {
        "meaning": "same sound",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B578/B303": "look same" but missing in the BCI-AV
    # "B348"
    "14430": {
        "meaning": "generalization",
        "features": {
           "link": "association",
           "position": "prefix",
           "middle-position": "prefix-second-part"
        }
    },
    # Relational operators
    # "B449"
    "15474": {
        "features": {
          "negation": "minus, no, without",
           "position": "prefix",
           "middle-position": "prefix-second-part"
        },
        "priority": ["15474", "15733", "15927"]
    },
    # "B578"
    "16713": {
        "meaning": "same, equal, equality",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        }
    },
    # "B502/B167"
    "12858": {
        "meaning": "blissymbol part",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        }
    },
    # "B502", has different position for different context - need to discuss
    "15972": {
        "meaning": "part of",
        "features": {
           "link": "derivative",
           "position": "prefix",
           "middle-position": "prefix-second-part"
        },
        "notes": "position is prefix when describing part of/component of X (e.g. tonsils are a part of the throat, gene is part of DNA). Position is suffix when describing X into parts, divided into/produces components (e.g. suit, jigsaw puzzle)"
    },
    # "B102"
    "12324": {
        "meaning": "about, concerning, regarding, in relation to",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B104", cannot identify middle-position due to lack of data
    "12333": {
        "meaning": "across",
        "features": {
           "position": "suffix"
        }
    },
    # "B109", exception: #15177, #21292
    "12348": {
        "meaning": "after, behind",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B111", has different position for different context - need to discuss
    "12351": {
        "meaning": "against, opposed to",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        },
        "notes": "Position is prefix (most cases), suffix (when specifying what type)"
    },
    # "B120/B120", cannot identify middle-position due to lack of data
    "12364": {
        "meaning": "along with",
        "features": {
           "position": "suffix",
        }
    },
    # "B162/B368"
    "25653": {
        "meaning": "among",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "notes": "Related meanings: between, to, inside"
    },
    # "B134"
    "12580": {
        "meaning": "around",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
    },
    # "B135"
    "12591": {
        "meaning": "at",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "notes": "position is suffix (inferred by related meanings: about (most related), on, around (time), of). Middle-position is suffix for the first part of the word (inferred by related meanings according to specificness: on and around (time))"
    },
    # "B158", exception: #16242, #25293, #13896
    "12656": {
        "meaning": "before, in front of, prior to",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
    },
    # "B162"
    "12669": {
        "meaning": "between",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
    },
    # "B195"
    "13100": {
        "meaning": "by, by means of, of",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "notes": "position is suffix (inferred by related meanings: about (most related), on, around (time), of"
    },
    # "B482"
    "15918": {
        "meaning": "on",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        }
    },
    # "B491", has different position for different context - need to discuss
    "15943": {
        "meaning": "out of (forward)",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        },
        "notes": "position is prefix when related to motion - something leaves. Position is suffix when related to direction - goes outward"
    },
    # "B492", cannot identify middle-position due to lack of data
    "15944": {
        "meaning": "out of (downward)",
        "features": {
           "position": "suffix"
        }
    },
    # "B977", cannot identify middle-position due to lack of data
    "25134": {
        "meaning": "out of (upward)",
        "features": {
           "position": "suffix"
        }
    },
    # "B976", cannot identify middle-position due to lack of data
    "25133": {
        "meaning": "out of (backward)",
        "features": {
           "position": "prefix"
        }
    },
    # "B402", has different position for different context - need to discuss
    "14952": {
        "meaning": "into (forward)",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        },
        "notes": "position is prefix when related to motion - something enters. Position is suffix when related to direction - goes inward"
    },
    # "B1124", cannot identify middle-position due to lack of data
    "25895": {
        "meaning": "into (downward)",
        "features": {
           "position": "suffix"
        }
    },
    # "B1125", cannot identify middle-position due to lack of data
    "25896": {
        "meaning": "into (upward)",
        "features": {
           "position": "suffix"
        }
    },
    # "B1123", cannot identify middle-position due to lack of data
    "25894": {
        "meaning": "into (backward)",
        "features": {
           "position": "suffix"
        }
    },
    # "B490", has different position for different context - need to discuss
    "15942": {
        "meaning": "outside",
        "features": {
           "position": "prefix"
        },
        "notes": "position is prefix if abstract. position is suffix if physical"
    },
    # "B398", has different position for different context - need to discuss
    "14932": {
        "meaning": "inside",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        },
        "notes": "Position is prefix if physical. Position is suffix if abtract"
    },
    # "B493", exception: #24325
    "15948": {
        "meaning": "over, above",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B676", exception: #24296
    "17969": {
        "meaning": "under, below",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B1102"
    "25628": {
        "meaning": "under (ground level)",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B331"
    "14381": {
        "meaning": "instead",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B332"
    "14382": {
        "meaning": "for the purpose of, in order to",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        }
    },
    # "B337"
    "14403": {
        "meaning": "from",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "notes": "position is suffix (inferred by related meaning: to)"
    },
    # "B657", exception: #29032
    "17739": {
        "meaning": "to, toward",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B653"
    "17724": {
        "meaning": "through",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        }
    },
    # "B677"
    "17982": {
        "meaning": "until",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "notes": "Position is suffix when it is the final state of something. Position is prefix when its related to the end of a cycle-related event. Related meaning:  end"
    },
    # "B160"
    "12663": {
        "meaning": "belongs to",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "equivalent_indicator": "24676",
        "priority":  ["12663", "24676"]
    },
    # Quantifiers
    # "B368"
    # prefix modifier
    "14647": {
        "features": {
           "quantifier": "many, much",
           "position": "prefix",
           "middle-position": "prefix-second-part"
        }
    },
    # pending: few (not yet in bliss-glyph-data.js)
    # "B117", exceptions: #14117, #29036, #12361, #22836, #24520.
    "12360": {
        "features": {
          "quantifier": "all",
           "position": "suffix",
           "middle-position": "suffix-first-part"
        }
    },
    # "B100", cannot identify middle-position due to lack of data
    "12321": {
        "features": {
           "quantifier": "any",
           "position": "prefix"
        }
    },
    # "B11/B117", cannot identify middle-position due to lack of data
    "12879": {
        "features": {
          "quantifier": "both",
           "position": "suffix"
        }
    },
    # "B10/B117", cannot identify middle-position due to lack of data
    "13893": {
        "features": {
          "quantifier": "each, every",
           "position": "suffix"
        },
        "notes": "position is suffix (inferred by related meanings: both (most related), all)"
    },
    # "B286", cannot identify middle-position due to lack of data
    "13914": {
        "features": {
          "quantifier": "either",
           "position": "suffix"
        }
    },
    # "B449/B286", cannot identify middle-position due to lack of data
    "15706": {
        "features": {
          "quantifier": "neither",
           "position": "suffix"
        },
        "notes": "position is suffix (inferred by related meaning: either)"
    },
    # "B951", cannot identify middle-position due to lack of data
    "24906": {
        "features": {
          "quantifier": "half",
           "position": "prefix"
        }
    },
    # "B962", cannot identify middle-position due to lack of data
    "24932": {
        "features": {
          "quantifier": "quarter",
           "position": "prefix"
        }
    },
    # "B1151", cannot identify middle-position due to lack of data
    "26064": {
        "features": {
          "quantifier": "one third",
           "position": "prefix"
        }
    },
    # "B1152", cannot identify middle-position due to lack of data
    "26065": {
        "features": {
           "quantifier": "two thirds",
           "position": "prefix"
        }
    },
    # "B1153", cannot identify middle-position due to lack of data
    "26066": {
        "features": {
          "quantifier": "three quarters",
           "position": "prefix"
        }
    },
    # "B559/B11", cannot identify middle-position due to lack of data
    "16762": {
        "features": {
          "quantifier": "several",
           "position": "prefix"
        },
        "notes": "position is prefix (inferred by related meaning: many/much)"
    },
    # "B9", cannot identify middle-position due to lack of data
    "8496": {
        "features": {
          "numeric": "zero",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B10", cannot identify middle-position due to lack of data
    "8497": {
        "features": {
          "numeric": "one",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B11", cannot identify middle-position due to lack of data
    "8498": {
        "features": {
          "numeric": "two",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B12", cannot identify middle-position due to lack of data
    "8499": {
        "features": {
          "numeric": "three",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B13", cannot identify middle-position due to lack of data
    "8500": {
        "features": {
          "numeric": "four",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B14", cannot identify middle-position due to lack of data
    "8501": {
        "features": {
          "numeric": "five",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B15", cannot identify middle-position due to lack of data
    "8502": {
        "features": {
          "numeric": "six",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B16", cannot identify middle-position due to lack of data
    "8503": {
        "features": {
          "numeric": "seven",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B17", cannot identify middle-position due to lack of data
    "8504": {
        "features": {
          "numeric": "eight",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B18", cannot identify middle-position due to lack of data
    "8505": {
        "features": {
          "numeric": "nine",
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    }
}
