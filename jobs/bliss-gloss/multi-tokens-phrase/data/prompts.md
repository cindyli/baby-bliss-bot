Prompts used to generate various datasets:

1. Positive context sentences:

Generate a Python list containing 300 unique and grammatically varied sentences. Each sentence must include either the word "lowness" or "shortness", specifically used in the context of height (not emotion or duration). Ensure each sentence is long and contextually rich, naturally leading up to the keyword, which should appear towards the end of the sentence. The set of sentences should include a wide variety of grammatical structures (e.g., conditionals, relative clauses, passive voice, questions, compound/complex sentences etc) and a diverse range of real-world scenarios (e.g., architecture, nature, human height, machinery, etc.). Output the result as a valid Python list of strings.

Go through the given Python list. For each sentence, identify whether it contains the word "lowness" or "shortness". Then, return a new Python list containing only the part of each sentence before the keyword ("lowness" or "shortness"). Ensure the result is still a valid Python list of strings, and remove any trailing punctuation or whitespace after the truncated sentence. Only output the Python list of truncated sentences.

2. Negative context sentences:
# Glosses used in other contexts
Generate a Python list containing 100 sentences. Each sentence must include either the word "lowness" or "shortness", used in various contexts except in the context of height. Ensure each sentence is long and contextually rich, naturally leading up to the keyword, which should appear towards the end of the sentence. The set of sentences should include a wide variety of grammatical structures (e.g., conditionals, relative clauses, passive voice, questions, compound/complex sentences etc) and a diverse range of real-world scenarios (e.g., architecture, nature, human height, machinery, etc.). Output the result as a valid Python list of strings.

# Glosses are unlikely to be used
Generate a Python list containing 100 long, context-rich partial sentences. Each sentence should provide enough information to make the next word very unlikely to be "lowness" or "shortness" when interpreted in the context of physical height. The sentences should cover a diverse range of grammatical structures (e.g., conditionals, passives, relative clauses, interrogatives, compound/complex constructions) and a wide variety of real-world scenarios (e.g., cooking, emotions, finance, astronomy, politics). Avoid including ellipses or any placeholder text. Each sentence should end naturally but remain incomplete. Return the result as a valid Python list of strings.

Improve the dataset above to ensure it covers as many contexts as possible including diverse grammar and sentence structures.

3. Fine tuning sentences:
Now I need to create fine-tuning dataset, give me 150 texts containing either "lowness" or "shortness" or both in the context of height. These sentences should cover as many contexts as possible such as living contexts, tenses and moods, various sentence structures, various grammatical positions, in questions or exclamations, long texts and short, and more. Provide these sentences as a python list.

Improve the dataset above to ensure it covers as many contexts as possible.

4. Text generation prompts for testing:
I am testing a fine-tuned language model's understanding of lowness or shortness in the context of physical height. Please generate 10 diverse partial sentence prompts that contain a "{placeholder}" where a word or phrase representing shortness or lowness in height should appear. These prompts should be partial sentences and diverse as much as possible. Examples are "Because of the {placeholder}," or "The {placeholder} of the fence meant that". Provide these prompts as a python list.
