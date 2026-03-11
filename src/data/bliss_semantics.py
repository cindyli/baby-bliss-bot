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
   # "B314"
   "14166": {
        "features": {
           "gender": "feminine",
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
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "equivalent_indicator": "24669",
        "priority": ["8497", "24669"]
    },
   # "B11"
   "8498": {
        "features": {
           "person": "second-person",
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "equivalent_indicator": "24678",
        "priority": ["8498", "24678"]
    },
   # "B12"
    "8499": {
        "features": {
           "person": "third-person",
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "equivalent_indicator": "24679",
        "priority": ["8499", "24679"]
    },
   # "B5999"
   "28052": {
        "features": {
           "size": "diminutive",
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
        "equivalent_indicator": "25458",
       "priority": ["28052", "25458"]
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
        "meaning": "intensity",
        "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        },
       "notes": "exclamatory when used as a prefix, otherwise a specifier"
    },
    # "B937", has different position for different context - need to discuss
    "24879": {
        "meaning": "more (comparative)",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
       "notes": "position is prefix if positive context"
    },
    # "B968", has different position for different context - need to discuss
    "24944": {
        "meaning": "most (comparative)",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
       "notes": "position is prefix if positive context"
    },

    # Identity-affecting operators
    # "B449/B401"
    "15733": {
        "meaning": "not, negative, no, don't, doesn't",
        "features": {
           "position": "suffix",
           "middle-position": "prefix-second-part"
        },
        "priority": ["15474", "15733", "15927"]
    },
    # "B486"
    "15927": {
        "meaning": "opposite",
        "features": {
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
        "meaning": "minus, no, without",
        "features": {
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
        "meaning": "many, much",
        "features": {
           "position": "prefix",
           "middle-position": "prefix-second-part"
        }
    },
    # pending: few (not yet in bliss-glyph-data.js)
    # "B117", exceptions: #14117, #29036, #12361, #22836, #24520.
    "12360": {
        "meaning": "all",
       "features": {
           "position": "suffix",
           "middle-position": "suffix-first-part"
        }
    },
    # "B100", cannot identify middle-position due to lack of data
    "12321": {
        "meaning": "any",
       "features": {
           "position": "prefix"
        }
    },
    # "B11/B117", cannot identify middle-position due to lack of data
    "12879": {
        "meaning": "both",
       "features": {
           "position": "suffix"
        }
    },
    # "B10/B117", cannot identify middle-position due to lack of data
    "13893": {
        "meaning": "each, every"
       "features": {
           "position": "suffix"
        },
       "notes": "position is suffix (inferred by related meanings: both (most related), all)"
    },
    # "B286", cannot identify middle-position due to lack of data
    "13914": {
        "meaning": "either",
       "features": {
           "position": "suffix"
        }
    },
    # "B449/B286", cannot identify middle-position due to lack of data
    "15706": {
        "meaning": "neither",
       "features": {
           "position": "suffix"
        },
       "notes": "position is suffix (inferred by related meaning: either)"
    },
    # "B951", cannot identify middle-position due to lack of data
    "24906": {
        "meaning": "half",
       "features": {
           "position": "prefix"
        }
    },
    # "B962", cannot identify middle-position due to lack of data
    "24932": {
        "meaning": "quarter",
       "features": {
           "position": "prefix"
        }
    },
    # "B1151", cannot identify middle-position due to lack of data
    "26064": {
        "meaning": "one third",
       "features": {
           "position": "prefix"
        }
    },
    # "B1152", cannot identify middle-position due to lack of data
    "26065": {
        "meaning": "two thirds",
       "features": {
           "position": "prefix"
        }
    },
    # "B1153", cannot identify middle-position due to lack of data
    "26066": {
        "meaning": "three quarters",
       "features": {
           "position": "prefix"
        }
    },
    # "B559/B11", cannot identify middle-position due to lack of data
    "16762": {
        "meaning": "several",
       "features": {
           "position": "prefix"
        },
       "notes": "position is prefix (inferred by related meaning: many/much)"
    },
    # "B9", cannot identify middle-position due to lack of data
    "8496": {
        "meaning": "zero",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B10", cannot identify middle-position due to lack of data
    "8497": {
        "meaning": "one",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B11", cannot identify middle-position due to lack of data
    "8498": {
        "meaning": "two",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B12", cannot identify middle-position due to lack of data
    "8499": {
        "meaning": "three",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B13", cannot identify middle-position due to lack of data
    "8500": {
        "meaning": "four",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B14", cannot identify middle-position due to lack of data
    "8501": {
        "meaning": "five",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B15", cannot identify middle-position due to lack of data
    "8502": {
        "meaning": "six",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },  
    # "B16", cannot identify middle-position due to lack of data
    "8503": {
        "meaning": "seven",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B17", cannot identify middle-position due to lack of data
    "8504": {
        "meaning": "eight",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    },
    # "B18", cannot identify middle-position due to lack of data
    "8505": {
        "meaning": "nine",
        "features": {
           "position": "prefix"
        },
        "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"
    }
}
