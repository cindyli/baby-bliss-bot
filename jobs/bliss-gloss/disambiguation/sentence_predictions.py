# Usage: python sentence_predictions.py

"""
This script reports the rank of a certain token in the predictions in a given context.
"""

import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import dataset_29111_wool_shop as dataset
# from data import dataset_animal_bark as dataset
from utils import get_token_prediction


def print_results(title, target_tokens, results):
    # Print results
    print("==============================================================")
    print(f"\n==== {title}")
    for result in results:
        print(f"\nContext: {result['context']}")
        print(f"Rank of {target_tokens}: {result['target_token_ranks']}")
        print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")


training_positive_context_sentences = dataset.training_positive_context_sentences
training_negative_context_sentences = dataset.training_negative_context_sentences
testing_context_sentences = dataset.testing_context_sentences


# Track the total running time of this script
start_time = time.time()

model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
# target_tokens = [" bark"]
target_tokens = [" wool", " yarn", " shop"]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Set the model to evaluation mode
model.eval()

# Get token ids for the target tokens
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Re-verify predictions on training sentences for the new token
results = get_token_prediction(model, tokenizer, training_positive_context_sentences, target_token_ids)
print_results(f"Predictions of token '{target_tokens}' on POSITIVE training sentences:", target_tokens, results)

results = get_token_prediction(model, tokenizer, training_negative_context_sentences, target_token_ids)
print_results(f"Predictions of token '{target_tokens}' on NEGATIVE training sentences:", target_tokens, results)

# Test predictions
results = get_token_prediction(model, tokenizer, testing_context_sentences, target_token_ids)
print_results(f"Predictions of token '{target_tokens}' on TESTING context sentences:", target_tokens, results)

end_time = time.time()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - end_time
print(f"Execution time: {int(elapsed_time // 3600)} hours {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
