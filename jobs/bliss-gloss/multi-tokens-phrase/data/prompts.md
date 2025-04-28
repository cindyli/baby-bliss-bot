Prompts used to generate various datasets:

1. Positive context sentences:

Give 100 partial sentences that the next word will contextually lead to "lowness" or "shortness" in the context of height. These partial sentences should be long to provide more context information that leads to the prediction of these words. They should cover varied grammar structures and scenarios. Don't include unnecessary information in these partial sentence such as ellipsis. Provide these sentences as a python list.

Improve the dataset above to ensure it covers as many contexts as possible including diverse grammar and sentence structures.

2. Negative context sentences:
Give 100 partial sentences that the next word will almost impossible to contextually lead to "lowness" or "shortness" in the context of height. These partial sentences should be long to provide more context information that doesn't lead to the prediction of these words. They should cover varied grammar structures and scenarios. Don't include unnecessary information in these partial sentence such as ellipsis. Provide these sentences as a python list.

Improve the dataset above to ensure it covers as many contexts as possible including diverse grammar and sentence structures.

3. Fine tuning sentences:
Now I need to create fine-tuning dataset, give me 150 texts containing either "lowness" or "shortness" or both in the context of height. These sentences should cover as many contexts as possible such as living contexts, tenses and moods, various sentence structures, various grammatical positions, in questions or exclamations, long texts and short, and more. Provide these sentences as a python list.

Improve the dataset above to ensure it covers as many contexts as possible.

4. Text generation prompts for testing:
I am testing a fine-tuned language model's understanding of lowness or shortness in the context of physical height. Please generate 10 diverse partial sentence prompts that contain a "{placeholder}" where a word or phrase representing shortness or lowness in height should appear. These prompts should be partial sentences and diverse as much as possible. Examples are "Because of the {placeholder}," or "The {placeholder} of the fence meant that". Provide these prompts as a python list.
