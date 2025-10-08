# Usage: python input_embedding_PC_and_self-attention.py

"""
This script tests and compares a number of methods to calculate a single input embedding for
a symbol that has multiple synonym glosses:

1. Principal Component Analysis (PCA) on the input embeddings of the synonym glosses
2. Self-attention mechanism on the input embeddings of the synonym glosses

The script tests the prediction and text generation results using the calculated input embedding,
and compares them together with the average input embedding.
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_input_embedding import (
    create_training_data,
    calc_output_embedding,
    get_embeddings_of_words,
    get_average_input_embeddings,
    get_unambiguous_embedding_by_PCA_threshold,
    get_unambiguous_embedding_by_PCA_kneed,
    get_unambiguous_embedding_by_self_attention,
    evaluate_new_token
)

USE_PCA_THRESHOLD_ON_INPUT_EMBEDDING = False  # If True, use principal component analysis on input embedding
USE_PCA_KNEED_ON_INPUT_EMBEDDING = False  # If True, use principal component analysis on input embedding
USE_SELF_ATTENTION_ON_INPUT_EMBEDDING = False  # If True, use self-attention on input embedding
USE_AVERAGE_ON_INPUT_EMBEDDING = True  # If True, use average input embedding

USE_PCA_THRESHOLD_ON_OUTPUT_EMBEDDING = False  # If True, use principal component analysis with explained variance threshold on output embedding
USE_PCA_KNEED_ON_OUTPUT_EMBEDDING = False  # If True, use principal component analysis with kneed library on output embedding
USE_SELF_ATTENTION_ON_OUTPUT_EMBEDDING = False  # If True, use self-attention on output embedding
USE_CALCULATED_OUTPUT_EMBEDDING = True  # If True, use average output embedding

glosses = ["break", "fracture", "injury", "damage"]   # glosses for the symbol 24852
target_tokens = [" break", " fracture", " injury", " damage"]
new_token = "[BLISS_24852]"
# 23085
kneed_sensitivity = 1  # Sensitivity parameter for the KneeLocator to find top N components in PCA
explained_variance_threshold = 0.95  # Explained variance threshold for PCA

print("Parameters:")
print(f"USE_PCA_THRESHOLD_ON_INPUT_EMBEDDING: {USE_PCA_THRESHOLD_ON_INPUT_EMBEDDING}\n"
      f"USE_PCA_KNEED_ON_INPUT_EMBEDDING: {USE_PCA_KNEED_ON_INPUT_EMBEDDING}\n"
      f"USE_SELF_ATTENTION_ON_INPUT_EMBEDDING: {USE_SELF_ATTENTION_ON_INPUT_EMBEDDING}\n"
      f"USE_AVERAGE_ON_INPUT_EMBEDDING: {USE_AVERAGE_ON_INPUT_EMBEDDING}\n")
print(f"USE_PCA_THRESHOLD_ON_OUTPUT_EMBEDDING: {USE_PCA_THRESHOLD_ON_OUTPUT_EMBEDDING}\n"
      f"USE_PCA_KNEED_ON_OUTPUT_EMBEDDING: {USE_PCA_KNEED_ON_OUTPUT_EMBEDDING}\n"
      f"USE_SELF_ATTENTION_ON_OUTPUT_EMBEDDING: {USE_SELF_ATTENTION_ON_OUTPUT_EMBEDDING}\n"
      f"USE_CALCULATED_OUTPUT_EMBEDDING: {USE_CALCULATED_OUTPUT_EMBEDDING}\n\n")

# The dataset contains positive/negative context sentences for calculating the output embedding,
# and data for testing after input embedding and ouptput embedding are added to the model.
initial_dataset_file = os.path.expanduser("~") + "/bliss_gloss/disambiguation/data/dataset_24852_break_fracture_injury_damage.py"

if not os.path.exists(initial_dataset_file):
    print(f"Error: Initial dataset file '{initial_dataset_file}' does not exist.")
    sys.exit(1)

# Initial values
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Read and execute the file to extract the target variables
namespace = {}
target_variable_names_in_initial_dataset = ["training_positive_context_sentences", "training_negative_context_sentences", "testing_positive_context_sentences", "testing_negative_context_sentences", "testing_text_generation_prompts"]
with open(initial_dataset_file, "r") as f:
    initial_dataset = f.read()
exec(initial_dataset, namespace)

training_positive_context_sentences = namespace["training_positive_context_sentences"]
training_negative_context_sentences = namespace["training_negative_context_sentences"]
testing_positive_context_sentences = namespace["testing_positive_context_sentences"]
testing_negative_context_sentences = namespace["testing_negative_context_sentences"]
testing_text_generation_prompts = namespace["testing_text_generation_prompts"]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.to(device)

# Track the total running time of this script
start_time = time.time()

print("Calculating input embedding...")

# Calculate the average input embedding for comparison and for input embedding initialization
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

average_input_embedding = get_average_input_embeddings(model, tokenizer, target_tokens)

if USE_AVERAGE_ON_INPUT_EMBEDDING:
    print("Input embedding: Average input embedding of the synonym glosses.")
    input_embedding = average_input_embedding

if USE_PCA_THRESHOLD_ON_INPUT_EMBEDDING:
    print(f"Input embedding: PCA with explained variance threshold {explained_variance_threshold} on the input embeddings of the synonym glosses.")
    input_embeddings_of_target_tokens = get_embeddings_of_words(model, tokenizer, target_tokens)
    input_embedding = get_unambiguous_embedding_by_PCA_threshold(input_embeddings_of_target_tokens, explained_variance_threshold)

if USE_PCA_KNEED_ON_INPUT_EMBEDDING:
    print(f"Input embedding: PCA with kneed sensitivity {kneed_sensitivity} on the input embeddings of the synonym glosses.")
    input_embeddings_of_target_tokens = get_embeddings_of_words(model, tokenizer, target_tokens)
    input_embedding = get_unambiguous_embedding_by_PCA_kneed(input_embeddings_of_target_tokens, kneed_sensitivity)

if USE_SELF_ATTENTION_ON_INPUT_EMBEDDING:
    print("Input embedding: Self-attention on the input embeddings of the synonym glosses.")
    input_embeddings_of_target_tokens = get_embeddings_of_words(model, tokenizer, target_tokens)
    input_embedding = get_unambiguous_embedding_by_self_attention(input_embeddings_of_target_tokens)

end_time_calc_input_embedding = time.time()
elapsed_time = end_time_calc_input_embedding - start_time
print(f"Execution time for calculating the input embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

print("Calculating output embedding...")
# Calculate output embedding
hidden_states, target_logits = create_training_data(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences, target_token_ids
)
calculated_output_embedding = calc_output_embedding(hidden_states, target_logits)

if USE_PCA_THRESHOLD_ON_OUTPUT_EMBEDDING:
    print(f"Output embedding: PCA with explained variance threshold {explained_variance_threshold} on the output embeddings of the synonym glosses.")
    output_embeddings_of_target_tokens = get_embeddings_of_words(model, tokenizer, target_tokens, type="output_embeddings")
    output_embedding = get_unambiguous_embedding_by_PCA_threshold(output_embeddings_of_target_tokens, explained_variance_threshold)

if USE_PCA_KNEED_ON_OUTPUT_EMBEDDING:
    print(f"Output embedding: PCA with kneed sensitivity {kneed_sensitivity} on the output embeddings of the synonym glosses.")
    output_embeddings_of_target_tokens = get_embeddings_of_words(model, tokenizer, target_tokens, type="output_embeddings")
    output_embedding = get_unambiguous_embedding_by_PCA_kneed(output_embeddings_of_target_tokens, kneed_sensitivity)

if USE_SELF_ATTENTION_ON_OUTPUT_EMBEDDING:
    print("Output embedding: Self-attention on the output embeddings of the synonym glosses.")
    output_embeddings_of_target_tokens = get_embeddings_of_words(model, tokenizer, target_tokens, type="output_embeddings")
    output_embedding = get_unambiguous_embedding_by_self_attention(output_embeddings_of_target_tokens)

if USE_CALCULATED_OUTPUT_EMBEDDING:
    print("Output embedding: Calculated from the positive and negative context sentences.")
    output_embedding = calculated_output_embedding

end_time_calc_output_embedding = time.time()
elapsed_time = end_time_calc_output_embedding - end_time_calc_input_embedding
print(f"Execution time for calculating the output embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Add the new token to the model with the new output embedding. The optimization of the input embedding needs it.
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)

with torch.no_grad():
    model.get_output_embeddings().weight[new_token_id] = output_embedding
    model.get_input_embeddings().weight[new_token_id] = input_embedding

input_embedding_similarity = torch.nn.functional.cosine_similarity(average_input_embedding.to(model.device), input_embedding.to(model.device), dim=0)
print(f"Cosine Similarity between the new token's input embedding and the average: {input_embedding_similarity.item():.4f}")
distance = torch.norm(average_input_embedding.to(model.device) - input_embedding.to(model.device), p=2)
print(f"Euclidean Distance between the new token's input embedding and the average: {distance.item():.4f}")

if not USE_CALCULATED_OUTPUT_EMBEDDING:
    output_embedding_similarity = torch.nn.functional.cosine_similarity(calculated_output_embedding.to(model.device), output_embedding.to(model.device), dim=0)
    print(f"Cosine Similarity between the new token's output embedding and the calculated: {output_embedding_similarity.item():.4f}")
    distance = torch.norm(calculated_output_embedding.to(model.device) - output_embedding.to(model.device), p=2)
    print(f"Euclidean Distance between the new token's output embedding and the calculated: {distance.item():.4f}")

end_time_add_new_token = time.time()
elapsed_time = end_time_add_new_token - end_time_calc_output_embedding
print(f"Execution time for adding a new token: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds\n")

# Evaluation
# Test 1: Embedding initialization
if torch.all(model.get_input_embeddings().weight[new_token_id] == 0):
    print("Error: Input embedding not initialized!")
if torch.all(model.lm_head.weight[new_token_id] == 0):
    print("Error: Output embedding not initialized!")

evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_positive_context_sentences, testing_negative_context_sentences, testing_text_generation_prompts,
    new_token_id, target_token_ids, target_tokens, new_token, target_tokens
)

end_time_validation = time.time()
elapsed_time = end_time_validation - end_time_add_new_token
print(f"Execution time for evaluation: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

total_time = end_time_validation - start_time
print(f"\nTotal execution time: {int(total_time // 60)} minutes and {total_time % 60:.2f} seconds")
