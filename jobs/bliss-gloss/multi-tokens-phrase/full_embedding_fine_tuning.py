# python full_embedding_fine_tuning.py <learning_rate> <epochs> <batch_size>
# python full_embedding_fine_tuning.py 0.0003 10 10

# pip install transformers datasets

import os
import sys
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import dataset_wool_shop as dataset_wool_shop
from utils import evaluate_new_token  # noqa: E402
# from data import dataset_wool_shop as dataset_wool_shop
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
# from utils import evaluate_new_token  # noqa: E402

if len(sys.argv) != 4:
    print("Usage: python qlora_embedding_fine_tuning.py <learning_rate> <epochs> <batch_size>")
    sys.exit(1)

learning_rate = float(sys.argv[1])
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/optimize_input_embedding_lr0.001/epochs1500"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.2-3B-Instruct"
save_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/full_fine_tune_wool_shop_token"
target_tokens = [" wool", " yarn"]
new_token = "[BLISS_29111]"  # Token for "wool shop"

# Load datasets
fine_tuning_sentences = dataset_wool_shop.fine_tuning_sentences
validation_sentences = dataset_wool_shop.validation_sentences

# Track the total running time of this script
start_time = time.time()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load Model without quantization
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# # Save the initial input embedding of the new token
# new_token_id = tokenizer.convert_tokens_to_ids(new_token)
# print(f"Token ID of {new_token}: {new_token_id}")
# new_token_input_embedding_before = model.get_input_embeddings().weight.data[new_token_id].clone()
# new_token_output_embedding_before = model.get_output_embeddings().weight.data[new_token_id].clone()

# Add the new token to the tokenizer and model
new_token = "[BLISS_29111]"
new_token_id = tokenizer.add_tokens(new_token)
print(f"Token ID of the new token {new_token}: {new_token_id}")
model.resize_token_embeddings(len(tokenizer))

# Initialize input/output embeddings of the new token to the same as the initial token
initial_token = " shop"  # Initial token to reset the embeddings for the new token based on the flags
initial_token_id = tokenizer.convert_tokens_to_ids(initial_token)
print(f"Token ID of the intial token {initial_token}: {initial_token_id}")
new_token_input_embedding_before = model.get_input_embeddings().weight.data[initial_token_id].clone()
new_token_output_embedding_before = model.get_output_embeddings().weight.data[initial_token_id].clone()
model.get_input_embeddings().weight.data[new_token_id] = new_token_input_embedding_before
model.get_output_embeddings().weight.data[new_token_id] = new_token_output_embedding_before


# Prepare Dataset (same as before)
def tokenize_func(dataset):
    tokenized = tokenizer(
        dataset["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_dataset = (
    Dataset.from_dict({"text": fine_tuning_sentences})
    .map(tokenize_func, batched=True)
)

validation_dataset = (
    Dataset.from_dict({"text": validation_sentences})
    .map(tokenize_func, batched=True)
)

# Training Arguments (modified for full fine-tuning)
training_args = TrainingArguments(
    output_dir=save_model_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=3,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,    # Use mixed precision training
    logging_steps=1,
    save_strategy="steps",
    save_steps=1,
    evaluation_strategy="steps",
    eval_steps=1,
    load_best_model_at_end=True,
    gradient_checkpointing=True,  # Reduces memory usage
    fp16=False,  # Use bf16 instead
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)

end_time_prepare = time.time()
elapsed_time = end_time_prepare - start_time
print(f"Preparation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Train
print("\nStarting training...")
trainer.train()
print("Training completed!")

end_time_fine_tuning = time.time()
elapsed_time = end_time_fine_tuning - end_time_prepare
print(f"Fine-tuning time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Save Model
model.save_pretrained(save_model_dir, save_embedding_layers=True)
tokenizer.save_pretrained(save_model_dir)

# Compare the input embedding of the new token before and after fine-tuning
new_token_input_embedding_after = model.get_input_embeddings().weight.data[new_token_id].clone()
new_token_output_embedding_after = model.get_output_embeddings().weight.data[new_token_id].clone()
input_embedding_similarity = torch.nn.functional.cosine_similarity(new_token_input_embedding_before, new_token_input_embedding_after, dim=0)
output_embedding_similarity = torch.nn.functional.cosine_similarity(new_token_output_embedding_before, new_token_output_embedding_after, dim=0)
print(f"Similarity of the new token input embedding before and after: {input_embedding_similarity:.4f}")
print(f"Similarity of the new token input embedding before and after: {output_embedding_similarity:.4f}")

print("==============================================================")
print("\n==== Evaluation after fine-tuning ====\n")
# Evaluate the results
training_positive_context_sentences = dataset_wool_shop.training_positive_context_sentences
training_negative_context_sentences = dataset_wool_shop.training_negative_context_sentences
testing_context_sentences = dataset_wool_shop.testing_context_sentences

target_tokens = [" wool", " yarn"]
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

evaluation_results = evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_context_sentences, new_token_id, target_token_ids, target_tokens, new_token, " wool shop"
)

end_time_evaluation_post_fine_tuning = time.time()
elapsed_time = end_time_evaluation_post_fine_tuning - end_time_fine_tuning
print(f"Evaluation time: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
