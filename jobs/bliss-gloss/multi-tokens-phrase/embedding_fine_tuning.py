# python embedding_fine_tuning.py 0.003 5 1
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


def prepare_sample(example, tokenizer, max_length=512):
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


def prepare_dataloader(examples, tokenizer, batch_size=4, max_length=512):
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
    batch_size=2,
    warmup_steps=50,
    weight_decay=0.01,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0
):
    """Fine-tune the new token with the fine-tuning dataset"""

    # Shuffle examples for training
    np.random.shuffle(fine_tuning_sentences)

    print(f"Fine-tuning dataset has {len(fine_tuning_sentences)} examples")

    dataloader = prepare_dataloader(fine_tuning_sentences, tokenizer, batch_size)

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Prepare for fine-tuning
    # First, make sure requires_grad is False for all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Now unfreeze the embedding layers completely to maintain the computation graph
    model.get_input_embeddings().weight.requires_grad = True
    model.lm_head.weight.requires_grad = True

    # Create a mask to zero out gradients for tokens we don't want to train
    emb_mask = torch.zeros_like(model.get_input_embeddings().weight, dtype=torch.bool)
    emb_mask[new_token_id] = True

    lm_head_mask = torch.zeros_like(model.lm_head.weight, dtype=torch.bool)
    lm_head_mask[new_token_id] = True

    # Set up optimizer with the full embedding matrices
    optimizer = torch.optim.AdamW([
        {"params": model.get_input_embeddings().weight},
        {"params": model.lm_head.weight}
    ], lr=learning_rate, weight_decay=weight_decay)

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
    global_step = 0
    total_loss = 0

    print(f"Beginning training for {epochs} epochs")
    for epoch in range(epochs):
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Zero out gradients for tokens we don't want to train
            with torch.no_grad():
                model.get_input_embeddings().weight.grad[~emb_mask] = 0
                model.lm_head.weight.grad[~lm_head_mask] = 0

            total_loss += loss.item()

            # Update weights if needed
            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log progress
                if global_step % 10 == 0:
                    print(f"Step {global_step}, Loss: {total_loss * gradient_accumulation_steps:.4f}")
                    total_loss = 0

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(save_model_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)

        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    print(f"Saving fine-tuned model to {save_model_dir}")
    model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)

    return model, tokenizer


if len(sys.argv) != 4:
    print("Usage: python embedding_fine_tuning.py <learning_rate> <epochs> <batch_size>")
    sys.exit(1)

learning_rate = float(sys.argv[1])
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# save_model_dir = os.path.expanduser("~") + "/Development/LLMs/fine_tune_new_token"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
save_model_dir = os.path.expanduser("~") + "/bliss_gloss/multi-tokens-phrase/test_results/models/fine_tune_new_token"
original_phrase = " shop"
new_token = "[BLISS_29111]"

fine_tuning_sentences = dataset.fine_tuning_sentences

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Track the total running time of this script
start_time = time.time()

original_token = tokenizer.tokenize(original_phrase)
original_token_id = tokenizer.convert_tokens_to_ids(original_token)
print(f"original_token_id: {original_token_id}")

# Add the new token to the tokenizer and model replicating the original token
tokenizer.add_tokens([new_token])
model.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print(f"new_token_id: {new_token_id}")

# Assign the original token's embeddings to the new token
with torch.no_grad():
    model.get_input_embeddings().weight[new_token_id] = model.get_input_embeddings().weight[original_token_id]
    model.lm_head.weight[new_token_id] = model.lm_head.weight[original_token_id]

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

end_time_fine_tuning = time.time()
elapsed_time = end_time_fine_tuning - start_time
print(f"Execution time for fine-tuning: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Evaluate the results
training_positive_context_sentences = dataset.training_positive_context_sentences
training_negative_context_sentences = dataset.training_negative_context_sentences
testing_context_sentences = dataset.testing_context_sentences

target_tokens = [" wool", " yarn"]
tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]
target_token_ids = tokenizer.convert_tokens_to_ids(tokens)

evaluation_results = evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_context_sentences, new_token_id, target_token_ids, target_tokens, new_token, "wool shop"
)

print("\nFine-tuning completed!")

end_time_evaluation = time.time()
elapsed_time = end_time_evaluation - start_time
print(f"Execution time for calculating output embedding: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
