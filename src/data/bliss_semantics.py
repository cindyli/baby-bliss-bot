# Blissymbolics Indicators and Modifiers

INDICATOR_SEMANTICS = {
    # UPDATED SECTION
    # action indicators
    # infinitive verb or present tense verb; would tense be null here since we need to differentiate this with ID: 24807?
    "8993": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "null", "voice": "null", "mood": "null"}},
    # active verb; action indicator active
    "8994": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "active", "mood": "null"}},
    # the equivalent of the English present conditional form
    "8995": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "active", "mood": "conditional"}},
    # description indicators
    # the equivalent of the English -ed or -en ending
    "8996": {"type": "POS", "value": ["adjective", "adverb"], "category": "grammatical", "features": {"tense": "past", "voice": "null", "mood": "null"}},
    # equivalent to English words ending in -able
    "8997": {"type": "POS", "value": ["adjective", "adverb"], "category": "grammatical", "features": {"tense": "future", "voice": "null", "mood": "null"}},
    # the equivalent of English adjectives/adverbs
    "8998": {"type": "POS", "value": ["adjective", "adverb"], "category": "grammatical", "features": {"tense": "present", "voice": "null", "mood": "null"}},

    # back to action indicators
    # the equivalent of the English future tense
    "8999": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "null", "mood": "null"}},
    # the equivalent of the English future conditional form
    "9000": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "active", "mood": "conditional"}},
    # the equivalent of the English future passive form
    "9001": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "passive", "mood": "null"}},
    # the equivalent of the English future passive conditional form
    "9002": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "future", "voice": "passive", "mood": "conditional"}},
    # something is being acted upon; action indicator passive
    "9003":  {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "passive", "mood": "null"}},
    # the equivalent of the English past tense
    "9004": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "null", "mood": "null"}},
    # the equivalent of the English past conditional form
    "9005": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "active", "mood": "conditional"}},
    # the equivalent of the English past passive conditional form
    "9006": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "passive", "mood": "conditional"}},
    # the equivalent of the English past passive form
    "9007": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "past", "voice": "passive", "mood": "null"}},
    # the equivalent of the English present passive conditional form
    "9008": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "passive", "mood": "conditional"}},

    ----------------------------
    # distinction between abstract and concrete nouns
     "9009": {
        "and": [
            {"type": "POS", "value": "noun", "category": "grammatical"},
            {"type": "TYPE_SHIFT", "value": "concretization", "category": "semantic"},
        ]
    },
    # represent multiple concrete objects
    "9010": {"type": "NUMBER", "value": "thing_plural", "category": "grammatical"},
    "9011": {"type": "NUMBER", "value": "plural", "category": "grammatical"},
    "24667": {"type": "TENSE", "value": "noun", "category": "grammatical", "notes": "for teaching purposes"},
    "24668": {"type": "GENDER", "value": "feminine", "category": "grammatical"},
    "24669": {"type": "PERSON", "value": "first_person", "category": "grammatical"},
    # indicator (continuous form); has been removed, so I only included the aspect feature in this character only
    "28043": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "passive", "mood": "conditional", "aspect": "continuous"}, "notes": "for teaching purposes"},
    "28044": {
        "and": [
            {"type": "DEFINITENESS", "value": "definite", "category": "grammatical"},
            {"type": "NUMBER", "value": "plural", "category": "grammatical"}
        ],
    },
    "28045":
    {
        "and": [
            {"type": "POS", "value": "noun", "category": "grammatical"},
            {"type": "NUMBER", "value": "plural", "category": "grammatical"}
        ]
    },
    "28046": {
        "and": [
            {"type": "DEFINITENESS", "value": "definite", "category": "grammatical"},
            {"type": "POS", "value": "noun", "category": "grammatical"},
            {"type": "NUMBER", "value": "plural", "category": "grammatical"}
        ]
    },
    --------------------------
    # adverb indicator
    "24665": {"type": "POS", "value": "adverb", "category": "grammatical", "notes": "for teaching purposes"}},
    # verb indicator (present action); similar to ID: 8993
    "24807": {"type": "POS", "value": "verb", "category": "grammatical", "features": {"tense": "present", "voice": "null", "mood": "null"}, "notes": "for teaching purposes"},
    "25458": {"type": "SIZE", "value": "diminutive", "category": "grammatical", "notes": "for teaching purposes"},
}


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

MODIFIER_SEMANTICS = {
    # Semantic Modifiers
    "14647": {"type": "QUANTIFIER", "value": "many", "category": "semantic"},
    "14947": {"type": "INTENSIFIER", "value": "high", "category": "semantic"},
    "15474": {"type": "NEGATION", "value": "without", "category": "semantic"},
    "15927": {"type": "OPERATOR", "value": "opposite", "category": "semantic"},
    "14430": {"type": "OPERATOR", "value": "generalization", "category": "semantic"},
    "15972": {"type": "OPERATOR", "value": "part_of", "category": "semantic"},
    "12352": {"type": "TIME", "value": "ago", "category": "semantic"},
    "15736": {"type": "TIME", "value": "now", "category": "semantic"},
    "17705": {"type": "TIME", "value": "future", "category": "semantic"},

    # Grammatical Modifiers
    "15654": {"type": "COMPARISON", "value": "more", "category": "grammatical"},
    "15661": {"type": "COMPARISON", "value": "most", "category": "grammatical"},
    "12663": {"type": "POSSESSION", "value": "belongs_to", "category": "grammatical"},

    # Semantic Numerical Modifiers
    "8510": {"type": "NUMBER", "value": "zero", "category": "semantic"},
    "8511": {"type": "NUMBER", "value": "one", "category": "semantic"},
    "8512": {"type": "NUMBER", "value": "two", "category": "semantic"},
    "8513": {"type": "NUMBER", "value": "three", "category": "semantic"},
    "8514": {"type": "NUMBER", "value": "four", "category": "semantic"},
    "8515": {"type": "NUMBER", "value": "five", "category": "semantic"},
    "8516": {"type": "NUMBER", "value": "six", "category": "semantic"},
    "8517": {"type": "NUMBER", "value": "seven", "category": "semantic"},
    "8518": {"type": "NUMBER", "value": "eight", "category": "semantic"},
    "8519": {"type": "NUMBER", "value": "nine", "category": "semantic"},

    # Signalling Modifiers
    "15460": {"type": "USAGE_NOTE", "value": "metaphor", "category": "signalling"},
    "21624": {"type": "USAGE_NOTE", "value": "blissname", "category": "signalling"},
    "24961": {"type": "USAGE_NOTE", "value": "slang", "category": "signalling"},
    "24962": {"type": "USAGE_NOTE", "value": "coarse_slang", "category": "signalling"},
}

# Possible modifiers
# function/gray ones: "21312, 12663, 12858, 25026, 25028, 13382, 15487, 13644, 15722, 15727, 15927, 15967, 15972, 16204, 14454, 14702, 8993, 8994, 24665, 8995, 24667, 8998, 8996, 8997, 8999, 9000, 9001, 9002, 24670, 9003, 9004, 9005, 24674, 24675, 9007, 9006, 9011, 24807, 24677, 9008, 9009, 9010, 16714, 16748, 16984, 16985, 16986, 17214, 17533, 17698, 17963, 18223, 18282, 18294, 18466, 24672, 24671, 24679, 28043, 24676, 24678, 24668, 24673, 24669, 25458, 28044, 28045, 28046"
# https://blissary.com/blissdictionary/?q=B2954|B160|B1309|B4854|B4856|B233|B2025|B1513|B2080|B2084|B486|B2134|B502|B2174|B1739|B1800|B81|B82|B902|B83|B904|B86|B84|B85|B87|B88|B89|B90|B907|B91|B92|B93|B911|B912|B95|B94|B99|B928|B914|B96|B97|B98|B2312|B2341|B2404|B2405|B2406|B2459|B2576|B2578|B2654|B2726|B2768|B2776|B2786|B909|B908|B916|B903|B913|B915|B905|B910|B906|B992|B5996|B5997|B5998

# expression/small word/white ones: "12321, 8551, 8521, 12324, 12333, 12348, 12350, 12351, 12352, 12360, 12361, 12364, 12367, 25653, 12374, 12400, 12401, 12402, 8489, 12580, 12591, 25522, 12602, 8522, 8552, 12610, 12613, 12647, 12656, 25265, 12669, 12849, 12850, 12864, 12865, 12879, 25408, 12910, 12911, 13094, 13100, 8523, 8553, 25852, 8488, 8487, 24879, 8524, 8554, 8490, 13675, 23476, 25869, 13869, 13870, 13871, 13892, 8525, 8555, 13893, 25052, 8504, 8518, 13914, 14117, 8483, 8526, 8556, 8501, 8515, 8533, 8563, 15474, 8534, 8564, 15706, 8505, 8519, 15725, 15729, 15733, 23907, 15736, 15737, 8535, 8565, 15918, 26215, 8497, 8511, 26064, 15929, 15931, 15932, 24011, 15942, 15944, 15943, 25133, 25134, 15948, 8536, 8566, 8486, 16184, 16185, 25595, 16225, 14381, 14382, 14390, 8500, 8514, 14403, 25311, 8527, 8557, 24457, 24458, 24459, 24460, 24461, 24462, 24463, 24464, 14639, 16480, 14641, 14642, 8528, 8558, 24906, 14708, 14906, 14907, 14908, 8529, 8559, 14927, 14932, 14938, 14947, 14952, 25894, 25895, 25896, 14960, 14962, 8530, 8560, 8531, 8561, 15141, 8532, 8562, 16479, 16436, 18014, 8537, 8567, 24932, 8538, 8568, 25364, 16474, 16475, 25931, 8539, 8569, 16713, 8503, 8517, 16762, 8502, 8516, 24944, 8540, 8570, 24309, 17697, 17700, 17702, 17705, 17707, 17708, 17711, 17712, 17720, 17723, 8499, 8513, 26066, 17724, 17739, 25387, 25389, 8498, 8512, 26065, 8541, 8571, 17969, 25628, 17981, 17982, 17983, 20524, 17986, 17987, 8542, 8543, 8572, 8573, 18228, 18229, 18231, 18230, 18234, 18235, 18236, 18237, 18239, 18238, 18242, 16482, 18244, 18245, 18248, 18246, 18247, 18249, 18267, 8544, 8574, 8545, 8575, 18291, 18292, 8546, 8576, 8496, 8510, 25972, 27010, 28052, 28053, 28056, 28055, 28057, 29005, 29051"
# https://blissary.com/blissdictionary/?q=B100|B55|B29|B102|B104|B109|B110|B111|B112|B117|B1186|B1189|B119|B5274|B120|B1213|B1214|B130|B7|B134|B135|B996|B139|B30|B56|B144|B145|B1272|B158|B5053|B162|B1301|B1302|B1314|B1315|B1324|B990|B1345|B1346|B192|B195|B31|B57|B5455|B6|B5|B937|B32|B58|B8|B262|B829|B5472|B271|B272|B273|B277|B33|B59|B1580|B4880|B17|B27|B286|B1619|B1|B34|B60|B14|B24|B41|B67|B449|B42|B68|B2066|B18|B28|B2083|B2086|B2088|B4078|B474|B475|B43|B69|B482|B5725|B10|B20|B1151|B488|B2109|B2110|B4160|B490|B492|B491|B976|B977|B493|B44|B70|B4|B2163|B2164|B1069|B2189|B331|B332|B335|B13|B23|B337|B980|B35|B61|B4518|B4519|B4520|B4521|B4522|B4523|B4524|B4525|B1751|B559|B1753|B1754|B36|B62|B951|B383|B1831|B1832|B1833|B37|B63|B1847|B398|B1856|B401|B402|B1123|B1124|B1125|B405|B1873|B38|B64|B39|B65|B1889|B40|B66|B558|B2226|B2688|B45|B71|B962|B46|B72|B984|B2256|B2257|B5530|B47|B73|B578|B16|B26|B2353|B15|B25|B968|B48|B74|B891|B646|B647|B2581|B648|B649|B650|B2587|B2588|B652|B2597|B12|B22|B1153|B653|B657|B986|B5164|B11|B21|B1152|B49|B75|B676|B1102|B2669|B677|B678|B723|B679|B680|B50|B51|B76|B77|B2730|B699|B2732|B2731|B2733|B2734|B702|B703|B2736|B2735|B2739|B561|B2740|B2741|B2744|B2742|B2743|B2745|B709|B52|B78|B53|B79|B2774|B2775|B54|B80|B9|B19|B1126|B5836|B5999|B6000|B6001|B6002|B6003|B6092|B6138
