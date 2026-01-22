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
        * definiteness - Identifies a specific or general thing. Valid values are "indefinite" and "definite". Values cannot be more than one. # indefinite noun: an apple; definite noun: the apple
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
Equivalent indicators/modifiers - Valid values are its equivalent indicator/modifier ID. Values cannot be more than one.
Priority - Indicates processing priority. Valid values are presented as IDs and "optional" in a list from highest to least priority. Values cannot be more than one. # action and description indicators are commonly used between different users, while present action and adverb indicators are used in full-form
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
        "features": {"definiteness": "definite"},
        "notes": "for teaching purposes"
    },
    # the female modifier (ID: 14166) is used more. Indicator is not used in communication
    "24668": {
        "category": "grammatical",
        "features": {"gender": "feminine"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "14166",
        "priority": ["14166", "24668"]
    },
    "12335": {
        "category": "grammatical",
        "features": {"gender": "masculine"}
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24669": {
        "category": "grammatical",
        "features": {"person": "first-person"},
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
        "category": "grammatical",
        "features": {"definiteness": "indefinite"},
        "notes": "for teaching purposes"
    },
    "24672": {
        "category": "grammatical",
        "features": {"gender": "neutral"},
        "notes": "for teaching purposes"
    },
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24678": {
        "category": "grammatical",
        "features": {"person": "second-person"},
        "notes": "for teaching purposes",
        "equivalent_modifier": "8498",
        "priority": ["8498", "24678"]
    },
    "24679": {
        "category": "grammatical",
        "features": {"person": "third-person"},
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


MODIFIER_SEMANTICS = {
   "14166": {
        "category": "grammatical",
        "features": {"gender": "feminine"},
        "equivalent_indicator": "24668",
        "priority": ["14166", "24668"]
    },
   "8497": {
        "category": "grammatical",
        "features": {"person": "first-person"},
        "equivalent_indicator": "24669",
        "priority": ["8497", "24669"]
    },
   "8498": {
        "category": "grammatical",
        "features": {"person": "second-person"},
        "equivalent_indicator": "24678",
        "priority": ["8498", "24678"]
    },
    "8499": {
        "category": "grammatical",
        "features": {"person": "third-person"},
        "equivalent_indicator": "24679",
        "priority": ["8499", "24679"]
    },
   "28052": {
        "category": "grammatical",
        "features": {"size": "diminutive", "form": "inflected"},
        "equivalent_indicator": "25458",
       "priority": ["28052", "25458"]
    },
   "28057": {
        "POS": "noun",
        "category": "syntactical",
        "features": {"position": ["pre", "post"], "default-position": "post"},
        "notes": "never used",
        "equivalent_indicator": "24673",
        "priority": ["optional", "24673", "28057"]
    },
    # Structural markers
    # "B233"
    "13382": {
        "meaning": "combine marker"
    },

    # What
    # "B699" // interrogative when used as a prefix, otherwise a specifier
    "18229": {
        "meaning": "what"
    },

    # Scalar degree operators
    # "B401" // exclamatory when used as a prefix, otherwise a specifier
    "14947": {
        "meaning": "intensity",
        "POS": ["verb", "adjective", "adverb"],
        "features": {"semantic": {"intensity": "high"}, "syntactical": {"position": "post", "default-position": "post"}}
    },
    # "B937"; prefix modifier
    "24879": {
        "meaning": "more (comparative)"
    },
    # "B968"; mostly prefix modifier
    "24944": {
        "meaning": "most (comparative)"
    },

    # Identity-affecting operators
    # "B449/B401"; suffix modifier
    "15733": {
        "meaning": "not, negative, no, don't, doesn't",
        "POS": ["verb", "adjective", "noun", "adverb"],
        "category": "semantic",
        "features": {"negation": "not"},
        "notes": "negates property or quality of something",
        "priority": ["15474", "15733", "15927"]
    },
    # "B486"; prefix modifier
    "15927": {
        "meaning": "opposite",
        "POS": ["noun", "adjective"],
        "category": "semantic",
        "features": {"negation": "opposite"},
        "notes": "negates relationally or conceptually, can also be used in figurative/metaphorical contexts",
        "priority": ["15474", "15733", "15927"]
    },
   # prefix modifier
    "14647": {
        "meaning": "many, much",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {"semantic": {"quantifier": "many"}, "syntactical": {"position": "pre", "default-position": "pre"}}
    },

    # Concept-transforming operators
    # "B1060/B578"; prefix modifier
    "16984": {
        "meaning": "similar to"
    },
    # "B1060/B578/B303"; prefix modifier
    "16985": {
        "meaning": "look similar to"
    },
    # "B1060/B578/B608"; prefix modifier e.g. duck + sound similar to = goose (new meaning)
    "16986": {
        "meaning": "sound similar to"
    },
    # "B578/B608"; prefix modifier e.g. bird whistle has same sound to duck (new meaning)
    "16714": {
        "meaning": "same sound"
    },
    # "B578/B303": "look same" but missing in the BCI-AV
    # "B348"; prefix modifier
    "14430": {
        "meaning": "generalization",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {"semantic": {"link": "association"}, "syntactical": {"position": "pre", "default-position": "pre"}},
    },
---------------------- CONTINUE ------------------------------------------
    # Relational operators
    # "B449"; prefix and suffix modifier
    "15474": {
        "meaning": "minus, no, without",
        "POS": "noun",
        "category": "semantic",
        "features": {"negation": "without"},
        "notes": "negates existence or presence, expresses lacking/missing something",
        "priority": ["15474", "15733", "15927"]
    },
    # "B578"
    "16713": {
        "meaning": "same, equal, equality"
    },
    # "B502/B167"
    "12858": {
        "meaning": "blissymbol part"
    },
    # "B502", // part of
    "15972": {
        "meaning": "part of",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {"semantic": {"link": "derivative"}, "syntactical": {"position": "pre", "default-position": "pre"}}
    },
    # "B102", // about
    "12324": {
        "meaning": "about, concerning, regarding, in relation to"
    },
    # "B104", // across
    "12333": {
        "meaning": "across"
    },
    # "B109", // after
    "12348": {
        "meaning": "after, behind"
    },
    # "B111", // against
    "12351": {
        "meaning": "against, opposed to"
    },
    # "B120/B120", // along with
    "12364": {
        "meaning": "along with"
    },
    # "B162/B368", // among
    "25653": {
        "meaning": "among"
    },
    # "B134", // around
    "12580": {
        "meaning": "around"
    },
    # "B135", // at
    "12591": {
        "meaning": "at"
    },
    # "B158", // before
    "12656": {
        "meaning": "before, in front of, prior to"
    },
    # "B162", // between
    "12669": {
        "meaning": "between"
    },
    # "B195", // by
    "13100": {
        "meaning": "by, by means of, of"
    },
    # "B482", // on
    "15918": {
        "meaning": "on"
    },
    # "B491", // out of (forward)
    "15943": {
        "meaning": "out of (forward)"
    },
    # "B492", // out of (downward)
    "15944": {
        "meaning": "out of (downward)"
    },
    # "B977", // out of (upward)
    "25134": {
        "meaning": "out of (upward)"
    },
    # "B976", // out of (backward)
    "25133": {
        "meaning": "out of (backward)"
    },
    # "B402", // into (forward)
    "14952": {
        "meaning": "into (forward)"
    },
    # "B1124", // into (downward)
    "25895": {
        "meaning": "into (downward)"
    },
    # "B1125", // into (upward)
    "25896": {
        "meaning": "into (upward)"
    },
    # "B1123", // into (backward)
    "25894": {
        "meaning": "into (backward)"
    },
    # "B490", // outside
    "15942": {
        "meaning": "outside"
    },
    # "B398", // inside
    "14932": {
        "meaning": "inside"
    },
    # "B493", // over, above
    "15948": {
        "meaning": "over, above"
    },
    # "B676", // under, below
    "17969": {
        "meaning": "under, below"
    },
    # "B1102", // under (ground level)
    "25628": {
        "meaning": "under (ground level)"
    },
    # "B331", // instead of
    "14381": {
        "meaning": "instead"
    },
    # "B332", // for the purpose of
    "14382": {
        "meaning": "for the purpose of, in order to"
    },
    # "B337", // from
    "14403": {
        "meaning": "from"
    },
    # "B657", // to, toward
    "17739": {
        "meaning": "to, toward"
    },
    # "B653", // through
    "17724": {
        "meaning": "through"
    },
    # "B677", // until
    "17982": {
        "meaning": "until"
    },
    # "B160", // belongs to
    "12663": {
        "meaning": "belongs to",
        "POS": "noun",
        "category": ["grammatical", "syntactical"],
        "features": {
            "grammatical": {"possessive": "possessor"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "post"
            }
        },
        "equivalent_indicator": "24676",
        "priority":  ["12663", "24676"]
    },

    # Quantifiers
    # "B368", // many/much (lake, village)
    "14647": {
        "meaning": "many, much"
    },
    # pending: few (not yet in bliss-glyph-data.js)
    # "B117", // all
    "12360": {
        "meaning": "all"
    },
    # "B100", // any
    "12321": {
        "meaning": "any"
    },
    # "B11/B117", // both
    "12879": {
        "meaning": "both"
    },
    # "B10/B117", // each/every
    "13893": {
        "meaning": "each, every"
    },
    # "B286", // either
    "13914": {
        "meaning": "either"
    },
    # "B449/B286", // neither
    "15706": {
        "meaning": "neither"
    },
    # "B951", // half
    "24906": {
        "meaning": "half"
    },
    # "B962", // quarter
    "24932": {
        "meaning": "quarter"
    },
    # "B1151", // one third
    "26064": {
        "meaning": "one third"
    },
    # "B1152", // two thirds
    "26065": {
        "meaning": "two thirds"
    },
    # "B1153", // three quarters
    "26066": {
        "meaning": "three quarters"
    },
    # "B559/B11", // several
    "16762": {
        "meaning": "several"
    },
    # "B9", // zero
    "8496": {
        "meaning": "zero",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "zero"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B10", // one
    "8497": {
        "meaning": "one",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "one"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B11", // two
    "8498": {
        "meaning": "two",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "two"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B12", // three
    "8499": {
        "meaning": "three",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "three"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B13", // four
    "8500": {
        "meaning": "four",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "four"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B14", // five
    "8501": {
        "meaning": "five",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "five"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B15", // six
    "8502": {
        "meaning": "six",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "six"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },  
    # "B16", // seven
    "8503": {
        "meaning": "seven",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "seven"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B17", // eight
    "8504": {
        "meaning": "eight",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "eight"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B18", // nine
    "8505": {
        "meaning": "nine",
        "POS": "noun",
        "category": ["semantic", "syntactical"],
        "features": {
            "semantic": {"numeric": "nine"},
            "syntactical": {
                "position": ["pre", "post"],
                "default-position": "pre"
            }
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    }
}
