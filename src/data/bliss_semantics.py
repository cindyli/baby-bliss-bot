"""
# README
Attributes
Type - Specifies the kind of annotation the data represents. Valid values are "POS", "TYPE_SHIFT" and "USAGE_NOTE". Values cannot be more than one. # POS value is the specific part of speech (e.g. noun, verb, etc). TYPE_SHIFT value transforms POS (e.g. verb to noun). USAGE_NOTE is for signalling modifiers.
Value -  Identifies type of POS, TYPE_SHIFT or USAGE_NOTE values. For POS, valid values are "noun", "verb" "adjective", and "adverb". For TYPE_SHIFT, valid value is "concretization". Values can be more than one value. For USAGE_NOTE, valid value is "signalling"
Category - Broad grouping of linguistic information. Valid values are "grammatical", "semantic", and "syntactical". Values can be more than one value.
Features - Specific properties of a word within its POS. Valid values are indicated below. Values can be more than one value.
    * Verbs:
        * tense - Locates an action in time. Valid values are "null", "past", "present", and "future". Values cannot be more than one.
        * voice - Shows relationship between the subject and action. Valid values are "null", "passive", and "active". Values cannot be more than one.
        * mood  - Expresses attitude or intent. Valid values are "null", "declarative", "conditional", and "imperative". Values cannot be more than one. # mood may vary language to language on how its used. For communication purposes, question/exclamation mark is used; without question/exclamation mark, its declarative.
        * aspect - Indicates how an action occurs over time. Valid values are "null" and "continuous". Values cannot be more than one. # aspect may vary language to language on how its used
        * form - Variations of verbs. Valid values are "inflected", "infinitive", "present-participle", "past-participle-1", and "past-participle-2". Values cannot be more than one. # simplifying finite (inflected) and infinite (infinitive and participles); when tense, voice, aspect, mood are null, its an infinitive
        * intensity: Valid value is "high"
        * negation: Valid values are "without", "not", and "opposite". Values cannot be more than one.
    * Nouns:
        * number: Valid values are "singular" and "plural". Values cannot be more than one.
        * definiteness - Identifies a specific or general thing. Valid values are "indefinite" and "definite". Values cannot be more than one. # indefinite: an apple; definite: the apple
        * gender: Valid values are "neutral", "feminine", and "masculine". Values cannot be more than one.
        * person: Valid values are "first-person", "second-person", and "third-person". Values cannot be more than one.
        * size: Valid value is "diminutive"
        * possessive: Valid values are "possessor" and "posessed". Values cannot be more than one.
        * position: Valid values are "pre" and "post". Values can be more than one. # syntax: if modifier comes before the head (classifier) is pre; e.g. colour of the car = colour + (MODIFIER + car). If modifier comes after the head (classifier) is post; e.g. car's colour = (car + MODIFIER) + colour.
        * default-position: Valid values are "pre" and "post". Values cannot be more than one. # syntax
        * quantifier: Valid value is "many"
        * link - Distingushes between grouped with something (association) versus part of something (derivative). Valid values are "association" and "derivative". Values cannot be more than one. # e.g. furniture is associated with chair and table versus province is derived of a country
        * time: Valid values are "ago", "now", "then_future". Values cannot be more than one. # attached to nouns but becomes adverb
        * numeric: Valid values are "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", and "nine". Values cannot be more than one.
        * negation: Valid values are "without", "not", and "opposite". Values cannot be more than one.
     * Adjectives + Adverbs:
        * modality - Semantic expression of possibility. Valid values are "null", "potential", and "completed". Values cannot be more than one. # modality is the state at which something is possible
        * intensity: Valid value is "high"
        * degree: Valid values are "comparative" and "superlative". Values cannot be more than one
        * negation: Valid values are "without", "not", and "opposite". Values cannot be more than one.
Priority (optional) - Indicates processing priority. Valid values are "1" or "2", where "1" is higher priority than "2". Values cannot be more than one. # action and description indicators are commonly used between different users, while present action and adverb indicators are used in full-form
"""

# Blissymbolics Indicators and Modifiers
INDICATOR_SEMANTICS = {
    # action indicators
    # infinitive verb or present tense verb; similar to ID: 24807 (includes tense as present), here is doesn't include tense
    "8993": {
        "POS": "verb",
        "category": "grammatical",
        "features": {
            "form": "infinitive"
        },
        "priority": "1"
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
        "priority": "1"
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
        "POS": "noun",
        "category": "grammatical",
        "features": {"number": "plural"}
    },
    "24667": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"definiteness": "definite"},
        "notes": "for teaching purposes"
    },
    # the female modifier (ID: 14166) is used more. Indicator is not used in communication
    "24668": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"gender": "feminine"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "14166",
        "priority": "2"
    },
    "14166": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"gender": "feminine"},
        "equivalent_indicator": "24668",
        "priority": "1"
    },
    "12335": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"gender": "masculine"},
        "priority": "1"
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24669": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"person": "first-person"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8497",
        "priority": "2"
    },
    # the past participle form
    "28044": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"number": "plural", "definiteness": "definite"}
    },
    "28045": {
        "and": [{
            "POS": "noun",
            "category": "grammatical",
            "features": {"definiteness": "definite"}
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
        "priority": "2"
    },
    # similar to ID: 8993;
    "24807": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"tense": "present", "voice": "null", "mood": "declarative", "aspect": "null", "form": "inflected"},
        "notes": "for teaching purposes",
        "priority": "2"
    },
    # the diminutive modifier is used more. Indicator (ID: 28052) is not used
    "25458": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"size": "diminutive", "form": "inflected"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "28052"
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
        "features": {"form": {"past-participle-1"}},
        "notes": "for teaching purposes"
    },
    "24675": {
        "POS": "verb",
        "category": "grammatical",
        "features": {"form": {"past-participle-2"}},
        "notes": "for teaching purposes"
    },
    "24677": {
        "POS": ["verb", "adjective"],
        "category": "grammatical",
        "features": {"form": {"present-participle"}},
        "notes": "for teaching purposes"
    },
    # back to nouns
    "24671": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"definiteness": "indefinite"},
        "notes": "for teaching purposes"
    },
    "24672": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"gender": "neutral"},
        "notes": "for teaching purposes"
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24678": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"person": "second-person"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8498",
        "priority": "2"
    },
    "24679": {
        "POS": "noun",
        "category": "grammatical",
        "features": {"person": "third-person"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8499",
        "priority": "2"
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
        "priority": "2"
    },
    # object form; can use object form with or without indicator - is an alternative, modifier (ID: 28057) has never been used
    "24673": {
        "POS": "noun",
        "category": "syntactical",
        "features": {"position": ["pre", "post"], "default-position": "post"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "28057",
        "priority": ["optional", "1"]
    },
}


MODIFIER_SEMANTICS = {
    "13382": {
        "meaning": "combine marker"
    },
    "24879": {
        "meaning": "more (comparative)"
    },
    "24944": {
        "meaning": "most (comparative)"
    },
    "14647": {
        "meaning": "many, much",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {"semantic": {"quantifier": "many"}, "syntactical": {"position": "pre", "default-position": "pre"}}
    },
    "16984": {
        "meaning": "similar to"
    },
    "16985": {
        "meaning": "look similar to"
    },
    "16986": {
        "meaning": "sound similar to"
    },
    "15474": {
        "meaning": "without",
        "POS": "noun",
        "category": "semantic",
        "features": {"negation": "without"},
        "notes": "negates existence or presence, expresses lacking/missing something",
        "priority": "1"
    },
    "15733": {
        "meaning": "not",
        "POS": ["verb", "adjective", "noun", "adverb"],
        "category": "semantic",
        "features": {"negation": "not"},
        "notes": "negates property or quality of something",
        "priority": "2"
    },
    "15927": {
        "meaning": "opposite",
        "POS": ["noun", "adjective"],
        "category": "semantic",
        "features": {"negation": "opposite"},
        "notes": "negates relationally or conceptually, can also be used in figurative/metaphorical contexts",
        "priority": "3"},
    "15972": {
        "meaning": "part of",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {"semantic": {"link": "derivative"}, "syntactical": {"position": "pre", "default-position": "pre"}}
    }
    # "B578/B303": looks like
    # "B578/B608": sounds like
}
