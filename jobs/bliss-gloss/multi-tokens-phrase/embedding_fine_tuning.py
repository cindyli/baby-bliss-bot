# python embedding_fine_tuning.py 0.001 15 2
# python embedding_fine_tuning.py <learning_rate> <epochs> <batch_size>

import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import dataset_wool_shop as dataset
from utils import evaluate_new_token  # noqa: E402
# from data import dataset_wool_shop as dataset
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
# from utils import evaluate_new_token  # noqa: E402


def prepare_sample(example, tokenizer, max_length=128):
    """Process a single text example for training"""
    # Tokenize with padding and truncation
    tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        example,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Create labels (same as input_ids for causal LM)
    input_ids = encoded.input_ids.squeeze(0)
    attention_mask = encoded.attention_mask.squeeze(0)
    labels = input_ids.clone()

    # Set padding token labels to -100 so they're ignored in loss calculation
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def prepare_dataloader(examples, tokenizer, batch_size=4, max_length=128):
    """Create a dataloader from text examples"""
    # Process all examples
    processed_examples = [prepare_sample(ex, tokenizer, max_length) for ex in examples]

    return DataLoader(
        processed_examples,
        batch_size=batch_size,
        shuffle=True
    )


def fine_tune_new_token(
    model,
    tokenizer,
    save_model_dir,
    new_token_id,
    fine_tuning_sentences,
    learning_rate=1e-4,
    epochs=5,
    batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    gradient_accumulation_steps=5,
    max_grad_norm=1.0
):
    """Fine-tune the new token with the fine-tuning dataset"""

    # Shuffle examples for training
    np.random.shuffle(fine_tuning_sentences)

    print(f"Fine-tuning dataset has {len(fine_tuning_sentences)} examples")

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Prepare for fine-tuning
    # Make sure requires_grad is False for all parameters
    for param in model.parameters():
        param.requires_grad = False

    dataloader = prepare_dataloader(fine_tuning_sentences, tokenizer, batch_size)

    # Create a mask to zero out gradients for tokens we don't want to train
    vocab_size = len(tokenizer)
    input_emb_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    input_emb_mask[new_token_id] = True

    # Unfreeze the input embedding layer to maintain the computation graph
    model.get_input_embeddings().weight.requires_grad = True

    # Set up optimizer for the new token's embeddings and last few layers
    optimizer = torch.optim.AdamW([{"params": model.get_input_embeddings().weight}], lr=learning_rate, weight_decay=weight_decay)

    # Calculate total training steps
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps

    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Empty CUDA cache before training
    torch.cuda.empty_cache()

    # Training loop
    model.train()

    print(f"Beginning training for {epochs} epochs")
    for epoch in range(epochs):
        total_loss = 0
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps

            total_loss += loss.item()
            loss.backward()

            # Zero out gradients for tokens we don't want to train
            with torch.no_grad():
                model.get_input_embeddings().weight.grad[~input_emb_mask] = 0

            # Update weights if needed
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Log progress
        print(f"Epoch {epoch+1}, Loss: {total_loss * gradient_accumulation_steps:.4f}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(save_model_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)

        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    return model, tokenizer


def calculate_mean_embeddings(model, tokenizer, phrases):
    all_token_ids = []

    # Tokenize all phrases and collect token IDs
    for phrase in phrases:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        all_token_ids.extend(tokens)
    print(f"all_token_ids: {all_token_ids}")

    # Get the embeddings for each token ID
    input_embeddings = torch.stack([model.get_input_embeddings().weight[token_id] for token_id in all_token_ids])
    output_embeddings = torch.stack([model.lm_head.weight[token_id] for token_id in all_token_ids])

    # Calculate means
    mean_input_embedding = torch.mean(input_embeddings, dim=0)
    mean_output_embedding = torch.mean(output_embeddings, dim=0)

    return mean_input_embedding, mean_output_embedding


if len(sys.argv) != 4:
    print("Usage: python embedding_fine_tuning.py <learning_rate> <epochs> <batch_size>")
    sys.exit(1)

learning_rate = float(sys.argv[1])
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
# save_model_dir = os.path.expanduser("~") + "/Development/LLMs/fine_tune_wool_shop_token"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/optimize_input_embedding_lr0.001/epochs1500"
save_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/fine_tune_wool_shop_token_new"
phrases = [" wool shop", " yarn shop"]
new_token = "[BLISS_29111]"   # Token for "wool shop"

# Load datasets
fine_tuning_sentences = dataset.fine_tuning_sentences
training_positive_context_sentences = dataset.training_positive_context_sentences
training_negative_context_sentences = dataset.training_negative_context_sentences
testing_context_sentences = dataset.testing_context_sentences

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Track the total running time of this script
start_time = time.time()

# original_token = tokenizer.tokenize(original_phrase)
# original_token_id = tokenizer.convert_tokens_to_ids(original_token)
# print(f"original_token_id: {original_token_id}")

# # Calculate the mean embeddings for the target phrases
# initial_input_emb, initial_output_emb = calculate_mean_embeddings(model, tokenizer, phrases)

# # Add the new token to the tokenizer and model replicating the original token
# tokenizer.add_tokens([new_token])
# model.resize_token_embeddings(len(tokenizer))
# new_token_id = tokenizer.convert_tokens_to_ids(new_token)

# # Assign the original token's embeddings to the new token
# with torch.no_grad():
#     model.get_input_embeddings().weight[new_token_id] = initial_input_emb
#     model.lm_head.weight[new_token_id] = initial_output_emb

new_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_token))
if isinstance(new_token_id, list) and len(new_token_id) == 1:
    new_token_id = new_token_id[0]
else:
    print(f"Error: new_token_id is not a single token ID. It is: {new_token_id}")
    sys.exit(1)

# Run the pipeline
fine_tune_new_token(
    model,
    tokenizer,
    save_model_dir,
    new_token_id,
    fine_tuning_sentences,
    learning_rate,
    epochs,
    batch_size
)

print("\nFine-tuning completed!")

end_time_fine_tuning = time.time()
elapsed_time = end_time_fine_tuning - start_time
print(f"Execution time for the fine-tuning: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

print("==============================================================")
print("\n==== Evaluation after fine-tuning ====\n")
# Evaluate the results
target_tokens = [" wool", " yarn"]
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

evaluation_results = evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_context_sentences, new_token_id, target_token_ids, target_tokens, new_token, " wool shop"
)

end_time_evaluation_post_fine_tuning = time.time()
elapsed_time = end_time_evaluation_post_fine_tuning - end_time_fine_tuning
print(f"Execution time for the evaluation after the fine-tuning: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
