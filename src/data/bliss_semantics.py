# READ ME - Data Section
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


# Blissymbolics Indicators and Modifiers
INDICATOR_SEMANTICS = {
    #UPDATED SECTION
    
    # action indicators
    # infinitive verb or present tense verb; similar to ID: 24807 (includes tense as present), here is doesn't include tense
    "8993": {"POS": "verb", "category": "grammatical", "features": {"tense": "null", "voice": "null", "mood": "null", "aspect" : "null", "form": "infinitive"}, "priority": "1"}
    # active verb
    "8994": {"POS": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "active", "mood": "declarative", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English present conditional form
    "8995": {"POS": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "active", "mood": "conditional", "aspect" : "null", "form": "inflected"}},

    
    # description indicators
    # the equivalent of the English -ed or -en ending
    "8996": {"POS": ["adjective", "adverb"], "category": "semantic", "features": {"modality": "completed"}},
    # equivalent to English words ending in -able
    "8997": {"POS": ["adjective", "adverb"],  "category": "semantic", "features": {"modality": "potential"}},
    # the equivalent of English adjectives/adverbs
    "8998": {"POS": ["adjective", "adverb"], "category": "semantic", "features": {"modality": "null"}}, "priority": "1"},


    # back to action indicators
    # the equivalent of the English future tense
    "8999": {"POS": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "active", "mood": "declarative", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English future conditional form
    "9000": {"POS": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "active", "mood": "conditional", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English future passive form
    "9001": {"POS": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "passive", "mood": "declarative", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English future passive conditional form
    "9002": {"POS": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "passive", "mood": "conditional", "aspect" : "null", "form": "inflected"}},
    # something is being acted upon
    "9003": {"POS": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "passive", "mood": "declarative", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English past tense
    "9004": {"POS": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "active", "mood": "declarative", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English past conditional form
    "9005": {"POS": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "active", "mood": "conditional", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English past passive conditional form
    "9006": {"POS": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "passive", "mood": "conditional", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English past passive form
    "9007": {"POS": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "passive", "mood": "declarative", "aspect" : "null", "form": "inflected"}},
    # the equivalent of the English present passive conditional form
    "9008": {"POS": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "passive", "mood": "conditional", "aspect" : "null", "form": "inflected"}},

       
    # represent a concrete object
    "9009": {
        "and": [
            {"POS": "noun", "category": "grammatical"},
            {"TYPE_SHIFT": "concretization", "category": "semantic"},
        ]
    },
    
    # represent multiple concrete objects
    "9010": {
        "and": [
            {"POS": "noun", "category": "grammatical", "features": {"number": "plural"}},
            {"TYPE_SHIFT": "concretization", "category": "semantic"},
        ]
    },
    "9011": {"POS": "noun", "category": "grammatical", "features": {"number": "plural"}},
    "24667": {"POS": "noun", "category": "grammatical", "features": {"definiteness": "definite"}, "notes": "for teaching purposes"},
   # the female modifier (ID: 14166) is used more. Indicator is not used in communication
   "24668": {"POS": "noun", "category": "grammatical", "features": {"gender": "feminine"}, "notes": "for teaching purposes", "equivalent_modifier": "14166", "priority":"2"},
   "14166": {"POS": "noun", "category": "grammatical", "features": {"gender": "feminine"}, "equivalent_indicator": "24668", "priority":"1"},
   "12335": {"POS": "noun", "category": "grammatical", "features": {"gender": "masculine"}, "priority":"1"},
   # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24669": {"POS": "noun", "category": "grammatical", "features": {"person": "first-person", "notes": "for teaching purposes", "equivalent_modifier": "8497", "priority":"2"},
    "28043": {"POS": "verb", "category": "grammatical", "features": {"tense": "null", "voice": "null", "mood": "null", "aspect": "continuous", "form": "inflected"}, "notes": "for teaching purposes"},
    "28044": {"POS": "noun", "category": "grammatical", "features": {"number": "plural", "definiteness": "definite"},
    "28045":
    {
        "and": [
            {"POS": "noun", "category": "grammatical", "features": {"definiteness": "definite"}},
            {"TYPE_SHIFT": "concretization", "category": "semantic"},
        ]
    },
    "28046": {
        "and": [
            {"POS": "noun", "category": "grammatical", "features": {"number": "plural", "definiteness": "definite"}},
            {"TYPE_SHIFT": "concretization", "category": "semantic"},
        ]
    },
    
    # indicator (adverb)
    "24665": {"POS": "adverb", "category": "grammatical", "notes": "for teaching purposes", "priority": "2"},
    # similar to ID: 8993; 
    "24807": {"POS": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "null", "mood": "declarative", "aspect" : "null", "form": "inflected"}, "notes": "for teaching purposes", "priority": "2"},
    # the diminutive modifier is used more. Indicator (ID: 28052) is not used
    "25458": {"POS": "noun", "category": "grammatical", "features": {"size": "diminutive", "form": "inflected"}, "notes": "for teaching purposes", "equivalent_modifier": "28052", "priority":"2"},

    # added more indicators from WinBliss
    # imperative mood
    "24670": {"POS": "verb", "category": "grammatical", "features": {"tense": "null", "voice": "null", "mood": "imperative", "aspect" : "null", "form": "inflected"}},
    # 3 participles
    "24674": {"POS": "verb", "category": "grammatical", "features": {"form": {"past-participle-1"},} "notes": "for teaching purposes"},
    "24675": {"POS": "verb", "category": "grammatical", "features": {"form": {"past-participle-2"}}, "notes": "for teaching purposes"},
    "24677": {"POS": ["verb", "adjective"], "category": "grammatical", "features": {"form": {"present-participle"}}, "notes": "for teaching purposes"},
    # back to nouns
    "24671": {"POS": "noun", "category": "grammatical", "features": {"definiteness": "indefinite"}, "notes": "for teaching purposes"},
    "24672": {"POS": "noun", "category": "grammatical", "features": {"gender": "neutral"}, "notes": "for teaching purposes"},
    # person indicators are only used for grammar teaching - not used in communication; modifiers (actually specifiers) are used for communication
    "24678": {"POS": "noun", "category": "grammatical", "features": {"person": "second-person"}, "notes": "for teaching purposes", "equivalent_modifier": "8498", "priority":"2"},
    "24679": {"POS": "noun", "category": "grammatical", "features": {"person": "third-person"}, "notes": "for teaching purposes", "equivalent_modifier": "8499", "priority":"2"},

   # possessive indicator; both indicator and modifier (ID: 12663) are used, but modifier is used more in English (opposite is true for Swedish).
   "24676": {"POS": "noun", "category": ["grammatical", "syntactical"], "features": {"grammatical": {"possessive": "possessor"}, "syntactical": {"position": ["pre", "post"], "default-position": "post"}, "notes": "for teaching purposes", "equivalent_modifier": "12663", "priority":"2"},
   # object form; can use object form with or without indicator - is an alternative, modifier (ID: 28057) has never been used
   "24673": {"POS": "noun", "category": "syntactical", "features": {"position": ["pre", "post"], "default-position": "post"}, "notes": "for teaching purposes", "equivalent_modifier": "28057", "priority":["optional", "1"]},
}


MODIFIER_SEMANTICS = {
   # Semantic Modifiers
   "14647": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"quantifier": "many"}, "syntactical": {"position": "pre", "default-position": "pre"}}},
   "14947": {"POS": ["verb", "adjective", "adverb"], "category": ["semantic", "syntactical"], "features": {"semantic": {"intensity": "high"}, "syntactical": {"position": "post", "default-position": "post"}}},
   # negation
   "15474": {"POS": "noun", "category": "semantic", "features": {"negation": "without"}, "notes": "negates existence or presence, expresses lacking/missing something", "priority": "1"},
   "15733": {"POS": ["verb", "adjective", "noun", "adverb"], "category": "semantic", "features": {"negation": "not"}, "notes": "negates property or quality of something", "priority": "2"},
   "15927": {"POS": ["noun", "adjective"], "category": "semantic", "features": {"negation": "opposite"}, "notes": "negates relationally or conceptually, can also be used in figurative/metaphorical contexts", "priority": "3"},
   # generalization modifier has a link of association; it is associated with something e.g. furniture is associated with chair and table
   "14430": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"link": "association"}, "syntactical": {"position": "pre", "default-position": "pre"}}},
   # constituence modifier has a link of derivative; it is a derivative of something e.g. province is a derivative of a country
   "15972": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"link": "derivative"}, "syntactical": {"position": "pre", "default-position": "pre"}}},
   ------------------ need to figure out these three -----------------------
   "12352": {"type": "POS", "value": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"time": "ago"}, "syntactical": {"position": "post", "default-position": "post"}}, "notes": "creates an adverb"},
   "15736": {"type": "POS", "value": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"time": "now"}, "syntactical": {"position": "post", "default-position": "post"}}, "notes": "creates an adverb"},
   "17705": {"type": "POS", "value": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"time": "then_future"}, "syntactical": {"position": "post", "default-position": "post"}}, "notes": "creates an adverb"},
   -----------------------------------------------------------------------
   
    # Grammatical Modifiers
    "15654": {"POS": ["adjective", "adverb"], "category": ["grammatical", "syntactical"], "features": {"grammatical": {"degree": "comparative"}, "syntactical": {"position": "pre", "default-position": "pre"}}},
    "15661": {"POS": ["adjective", "adverb"], "category": ["grammatical", "syntactical"], "features": {"grammatical": {"degree": "superlative"}, "syntactical": {"position": "pre", "default-position": "pre"}}},
    "12663": {"POS": "noun", "category": ["grammatical", "syntactical"], "features": {"grammatical": {"possessive": "possessor"}, "syntactical": {"position": ["pre", "post"], "default-position": "post"}}, "equivalent_indicator": "24676", "priority":"1"}},

    # Semantic Numerical Modifiers
    "8510": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "zero"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8511": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "one"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8512": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "two"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8513": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "three"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8514": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "four"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8515": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "five"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8516": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "six"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8517": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "seven"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8518": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "eight"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},
    "8519": {"POS": "noun", "category": ["semantic", "syntactical"], "features": {"semantic": {"numeric": "nine"}, "syntactical": {"position": ["pre", "post"], "default-position": "pre"}}, "notes": "when in default position (prefix), functions as a cardinal to indicate number of items. otherwise (suffixed), functions as an ordinal"},

    # Grammatical Numerical Modifiers
    "8497": {"POS": "noun", "category": ["grammatical", "syntactical"], "features": {"grammatical": {"person": "first-person"}, "syntactical": {"position": "post", "default-position": "post"}}, "equivalent_indicator": "24669", "priority":"1"},
    "8498": {"POS": "noun", "category": ["grammatical", "syntactical"], "features": {"grammatical": {"person": "second-person"}, "syntactical": {"position": "post", "default-position": "post"}}, "equivalent_indicator": "24678", "priority":"1"},
    "8499": {"POS": "noun", "category": ["grammatical", "syntactical"], "features": {"grammatical": {"person": "third-person"}, "syntactical": {"position": "post", "default-position": "post"}}, "equivalent_modifier": "24679", "priority":"1"},
    
    # Signalling Modifiers
    "15460": {"USAGE_NOTE": "signalling", "category": "syntactical", "features": {"position": "pre", "default-position": "pre"}}, "notes": "used as a metaphor"},
    "21624": {"USAGE_NOTE": "signalling", "category": "syntactical", "features": {"position": "pre", "default-position": "pre"}}, "notes": "used as a Blissname"},
    "24961": {"USAGE_NOTE": "signalling", "category": "syntactical", "features": {"position": "pre", "default-position": "pre"}}, "notes": "used as slang"},
    "24962": {"USAGE_NOTE": "signalling", "category": "syntactical", "features": {"position": "pre", "default-position": "pre"}}, "notes": "used as course slang"},,

   
# PREVIOUS SECTION - DIDNT DELETE.
    # # infinitive verb or present tense verb
    # "8993": {"type": "POS", "value": "verb", "category": "grammatical"},
    # # active verb
    # "8994": {"type": "VOICE", "value": "active", "category": "grammatical"},
    # # the equivalent of the English present conditional form
    # "8995": {"type": "POS", "value": "present_conditional", "category": "grammatical"},
    # # the equivalent of the English -ed or -en ending
    # "8996": {"type": "POS", "value": "past_participle", "category": "grammatical"},
    # # equivalent to English words ending in -able
    # "8997": {"type": "POS", "value": "able", "category": "grammatical"},
    # # the equivalent of English adjectives/adverbs
    # "8998": {
    #     "or": [
    #         {"type": "POS", "value": "adjective", "category": "grammatical"},
    #         {"type": "POS", "value": "adverb", "category": "grammatical"}
    #     ]
    # },
    
    # # the equivalent of the English future tense
    # "8999": {"type": "TENSE", "value": "future", "category": "grammatical"},
    # # the equivalent of the English future conditional form
    # "9000": {"type": "TENSE", "value": "future_conditional", "category": "grammatical"},
    # # the equivalent of the English future passive form
    # "9001": {"type": "TENSE", "value": "future_passive", "category": "grammatical"},
    # # the equivalent of the English future passive conditional form
    # "9002": {"type": "TENSE", "value": "future_passive_conditional", "category": "grammatical"},
    # # something is being acted upon; action indicator passive
    # "9003":  {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "passive", "mood": "null"}},
    # # the equivalent of the English past tense
    # "9004": {"type": "TENSE", "value": "past", "category": "grammatical"},
    # # the equivalent of the English past conditional form
    # "9005": {"type": "TENSE", "value": "past_conditional", "category": "grammatical"},
    # # the equivalent of the English past passive conditional form
    # "9006": {"type": "TENSE", "value": "past_passive_conditional", "category": "grammatical"},
    # # the equivalent of the English past passive form
    # "9007": {"type": "TENSE", "value": "past_passive", "category": "grammatical"},
    # # the equivalent of the English present passive conditional form
    # "9008": {"type": "TENSE", "value": "present_passive_conditional", "category": "grammatical"},
    # represent a concrete object
#     "9009": {
#         "and": [
#             {"type": "POS", "value": "noun", "category": "grammatical"},
#             {"type": "TYPE_SHIFT", "value": "concretization", "category": "semantic"},
#         ]
#     },
#     # represent multiple concrete objects
#     "9010": {"type": "NUMBER", "value": "thing_plural", "category": "grammatical"},
#     "9011": {"type": "NUMBER", "value": "plural", "category": "grammatical"},
#     "24667": {"type": "TENSE", "value": "noun", "category": "grammatical", "notes": "for teaching purposes"},
#     "24668": {"type": "GENDER", "value": "feminine", "category": "grammatical"},
#     "24669": {"type": "PERSON", "value": "first_person", "category": "grammatical"},
#     "28043": {"type": "ASPECT", "value": "continuous_verb", "category": "grammatical"},
#     "28044": {
#         "and": [
#             {"type": "DEFINITENESS", "value": "definite", "category": "grammatical"},
#             {"type": "NUMBER", "value": "plural", "category": "grammatical"}
#         ],
#     },
#     "28045":
#     {
#         "and": [
#             {"type": "POS", "value": "noun", "category": "grammatical"},
#             {"type": "NUMBER", "value": "plural", "category": "grammatical"}
#         ]
#     },
#     "28046": {
#         "and": [
#             {"type": "DEFINITENESS", "value": "definite", "category": "grammatical"},
#             {"type": "POS", "value": "noun", "category": "grammatical"},
#             {"type": "NUMBER", "value": "plural", "category": "grammatical"}
#         ]
#     },
#     "24665": {"type": "POS", "value": "adverb", "category": "grammatical"},
#     "24807": {
#         "and": [
#             {"type": "POS", "value": "verb", "category": "grammatical"},
#             {"type": "TENSE", "value": "present", "category": "grammatical"}
#         ]
#     },
#     "25458": {"type": "SIZE", "value": "diminutive", "category": "grammatical", "notes": "for teaching purposes"},
# }

# MODIFIER_SEMANTICS = {
#     # Semantic Modifiers
#     "14647": {"type": "QUANTIFIER", "value": "many", "category": "semantic"},
#     "14947": {"type": "INTENSIFIER", "value": "high", "category": "semantic"},
#     "15474": {"type": "NEGATION", "value": "without", "category": "semantic"},
#     "15927": {"type": "OPERATOR", "value": "opposite", "category": "semantic"},
#     "14430": {"type": "OPERATOR", "value": "generalization", "category": "semantic"},
#     "15972": {"type": "OPERATOR", "value": "part_of", "category": "semantic"},
#     "12352": {"type": "TIME", "value": "ago", "category": "semantic"},
#     "15736": {"type": "TIME", "value": "now", "category": "semantic"},
#     "17705": {"type": "TIME", "value": "future", "category": "semantic"},

#     # Grammatical Modifiers
#     "15654": {"type": "COMPARISON", "value": "more", "category": "grammatical"},
#     "15661": {"type": "COMPARISON", "value": "most", "category": "grammatical"},
#     "12663": {"type": "POSSESSION", "value": "belongs_to", "category": "grammatical"},

#     # Semantic Numerical Modifiers
#     "8510": {"type": "NUMBER", "value": "zero", "category": "semantic"},
#     "8511": {"type": "NUMBER", "value": "one", "category": "semantic"},
#     "8512": {"type": "NUMBER", "value": "two", "category": "semantic"},
#     "8513": {"type": "NUMBER", "value": "three", "category": "semantic"},
#     "8514": {"type": "NUMBER", "value": "four", "category": "semantic"},
#     "8515": {"type": "NUMBER", "value": "five", "category": "semantic"},
#     "8516": {"type": "NUMBER", "value": "six", "category": "semantic"},
#     "8517": {"type": "NUMBER", "value": "seven", "category": "semantic"},
#     "8518": {"type": "NUMBER", "value": "eight", "category": "semantic"},
#     "8519": {"type": "NUMBER", "value": "nine", "category": "semantic"},

#     # Signalling Modifiers
#     "15460": {"type": "USAGE_NOTE", "value": "metaphor", "category": "signalling"},
#     "21624": {"type": "USAGE_NOTE", "value": "blissname", "category": "signalling"},
#     "24961": {"type": "USAGE_NOTE", "value": "slang", "category": "signalling"},
#     "24962": {"type": "USAGE_NOTE", "value": "coarse_slang", "category": "signalling"},
}

# Possible modifiers
# function/gray ones: "21312, 12663, 12858, 25026, 25028, 13382, 15487, 13644, 15722, 15727, 15927, 15967, 15972, 16204, 14454, 14702, 8993, 8994, 24665, 8995, 24667, 8998, 8996, 8997, 8999, 9000, 9001, 9002, 24670, 9003, 9004, 9005, 24674, 24675, 9007, 9006, 9011, 24807, 24677, 9008, 9009, 9010, 16714, 16748, 16984, 16985, 16986, 17214, 17533, 17698, 17963, 18223, 18282, 18294, 18466, 24672, 24671, 24679, 28043, 24676, 24678, 24668, 24673, 24669, 25458, 28044, 28045, 28046"
# https://blissary.com/blissdictionary/?q=B2954|B160|B1309|B4854|B4856|B233|B2025|B1513|B2080|B2084|B486|B2134|B502|B2174|B1739|B1800|B81|B82|B902|B83|B904|B86|B84|B85|B87|B88|B89|B90|B907|B91|B92|B93|B911|B912|B95|B94|B99|B928|B914|B96|B97|B98|B2312|B2341|B2404|B2405|B2406|B2459|B2576|B2578|B2654|B2726|B2768|B2776|B2786|B909|B908|B916|B903|B913|B915|B905|B910|B906|B992|B5996|B5997|B5998

# expression/small word/white ones: "12321, 8551, 8521, 12324, 12333, 12348, 12350, 12351, 12352, 12360, 12361, 12364, 12367, 25653, 12374, 12400, 12401, 12402, 8489, 12580, 12591, 25522, 12602, 8522, 8552, 12610, 12613, 12647, 12656, 25265, 12669, 12849, 12850, 12864, 12865, 12879, 25408, 12910, 12911, 13094, 13100, 8523, 8553, 25852, 8488, 8487, 24879, 8524, 8554, 8490, 13675, 23476, 25869, 13869, 13870, 13871, 13892, 8525, 8555, 13893, 25052, 8504, 8518, 13914, 14117, 8483, 8526, 8556, 8501, 8515, 8533, 8563, 15474, 8534, 8564, 15706, 8505, 8519, 15725, 15729, 15733, 23907, 15736, 15737, 8535, 8565, 15918, 26215, 8497, 8511, 26064, 15929, 15931, 15932, 24011, 15942, 15944, 15943, 25133, 25134, 15948, 8536, 8566, 8486, 16184, 16185, 25595, 16225, 14381, 14382, 14390, 8500, 8514, 14403, 25311, 8527, 8557, 24457, 24458, 24459, 24460, 24461, 24462, 24463, 24464, 14639, 16480, 14641, 14642, 8528, 8558, 24906, 14708, 14906, 14907, 14908, 8529, 8559, 14927, 14932, 14938, 14947, 14952, 25894, 25895, 25896, 14960, 14962, 8530, 8560, 8531, 8561, 15141, 8532, 8562, 16479, 16436, 18014, 8537, 8567, 24932, 8538, 8568, 25364, 16474, 16475, 25931, 8539, 8569, 16713, 8503, 8517, 16762, 8502, 8516, 24944, 8540, 8570, 24309, 17697, 17700, 17702, 17705, 17707, 17708, 17711, 17712, 17720, 17723, 8499, 8513, 26066, 17724, 17739, 25387, 25389, 8498, 8512, 26065, 8541, 8571, 17969, 25628, 17981, 17982, 17983, 20524, 17986, 17987, 8542, 8543, 8572, 8573, 18228, 18229, 18231, 18230, 18234, 18235, 18236, 18237, 18239, 18238, 18242, 16482, 18244, 18245, 18248, 18246, 18247, 18249, 18267, 8544, 8574, 8545, 8575, 18291, 18292, 8546, 8576, 8496, 8510, 25972, 27010, 28052, 28053, 28056, 28055, 28057, 29005, 29051"
# https://blissary.com/blissdictionary/?q=B100|B55|B29|B102|B104|B109|B110|B111|B112|B117|B1186|B1189|B119|B5274|B120|B1213|B1214|B130|B7|B134|B135|B996|B139|B30|B56|B144|B145|B1272|B158|B5053|B162|B1301|B1302|B1314|B1315|B1324|B990|B1345|B1346|B192|B195|B31|B57|B5455|B6|B5|B937|B32|B58|B8|B262|B829|B5472|B271|B272|B273|B277|B33|B59|B1580|B4880|B17|B27|B286|B1619|B1|B34|B60|B14|B24|B41|B67|B449|B42|B68|B2066|B18|B28|B2083|B2086|B2088|B4078|B474|B475|B43|B69|B482|B5725|B10|B20|B1151|B488|B2109|B2110|B4160|B490|B492|B491|B976|B977|B493|B44|B70|B4|B2163|B2164|B1069|B2189|B331|B332|B335|B13|B23|B337|B980|B35|B61|B4518|B4519|B4520|B4521|B4522|B4523|B4524|B4525|B1751|B559|B1753|B1754|B36|B62|B951|B383|B1831|B1832|B1833|B37|B63|B1847|B398|B1856|B401|B402|B1123|B1124|B1125|B405|B1873|B38|B64|B39|B65|B1889|B40|B66|B558|B2226|B2688|B45|B71|B962|B46|B72|B984|B2256|B2257|B5530|B47|B73|B578|B16|B26|B2353|B15|B25|B968|B48|B74|B891|B646|B647|B2581|B648|B649|B650|B2587|B2588|B652|B2597|B12|B22|B1153|B653|B657|B986|B5164|B11|B21|B1152|B49|B75|B676|B1102|B2669|B677|B678|B723|B679|B680|B50|B51|B76|B77|B2730|B699|B2732|B2731|B2733|B2734|B702|B703|B2736|B2735|B2739|B561|B2740|B2741|B2744|B2742|B2743|B2745|B709|B52|B78|B53|B79|B2774|B2775|B54|B80|B9|B19|B1126|B5836|B5999|B6000|B6001|B6002|B6003|B6092|B6138
