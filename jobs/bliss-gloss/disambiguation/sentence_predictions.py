# Usage: python sentence_predictions.py

"""
This script reports the rank of a certain token in the predictions in a given context.
"""

import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import dataset_animal_bark
from utils import get_token_prediction


def print_results(title, results):
    # Print results
    print("==============================================================")
    print(f"\n==== {title}")
    for result in results:
        print(f"\nContext: {result['context']}")
        print(f"Rank of {target_token}: {result['rank_of_target_token_id']}")
        print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")


training_positive_context_sentences = dataset_animal_bark.training_positive_context_sentences
training_negative_context_sentences = dataset_animal_bark.training_negative_context_sentences
testing_context_sentences = dataset_animal_bark.testing_context_sentences


# Track the total running time of this script
start_time = time.time()

# Load model and tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Set the model to evaluation mode
model.eval()

# Get original "bark" token id for training data
target_token = " bark"
tokens = tokenizer.tokenize(target_token)
target_token_id = tokenizer.convert_tokens_to_ids(tokens)[0]
print("target_token_id:", target_token_id)

# Re-verify predictions on training sentences for the new token
results = get_token_prediction(model, tokenizer, training_positive_context_sentences, target_token_id)
print_results(f"Predictions of token '{target_token}' on POSITIVE training sentences:", results)

results = get_token_prediction(model, tokenizer, training_negative_context_sentences, target_token_id)
print_results("Predictions of token '{target_token}' on NEGATIVE training sentences:", results)

# Test predictions
results = get_token_prediction(model, tokenizer, testing_context_sentences, target_token_id)
print_results("Predictions of token '{target_token}' on TESTING context sentences:", results)

end_time = time.time()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - end_time
print(f"Execution time: {int(elapsed_time // 3600)} hours {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
