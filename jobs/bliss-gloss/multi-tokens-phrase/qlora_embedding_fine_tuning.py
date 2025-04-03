# python qlora_embedding_fine_tuning.py <learning_rate> <epochs> <batch_size>
# python qlora_embedding_fine_tuning.py 0.0001 3 10

# pip install transformers accelerate peft bitsandbytes datasets

# LoRA: https://huggingface.co/docs/peft/main/en/developer_guides/lora
# Quantization with LoRA: https://huggingface.co/docs/peft/en/developer_guides/quantization
# Apply QLoRA to output projections and token embedding: https://github.com/pytorch/torchtune/issues/1000

import os
import sys
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import dataset_wool_shop as dataset
from utils import evaluate_new_token  # noqa: E402
# from data import dataset_wool_shop as dataset
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "disambiguation")))
# from utils import evaluate_new_token  # noqa: E402


if len(sys.argv) != 4:
    print("Usage: python qlora_embedding_fine_tuning.py <learning_rate> <epochs> <batch_size>")
    sys.exit(1)

learning_rate = float(sys.argv[1])
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

# model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# save_model_dir = os.path.expanduser("~") + "/Development/LLMs/fine_tune_wool_shop_token"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/optimize_input_embedding_lr0.001/epochs1500"
save_model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/qlora_fine_tune_wool_shop_token"
target_tokens = [" wool", " yarn"]
new_token = "[BLISS_29111]"   # Token for "wool shop"

# Load datasets
fine_tuning_sentences = dataset.fine_tuning_sentences

# Track the total running time of this script
start_time = time.time()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Configure 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load Model with Quantization
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Save the initial input embedding of the new token
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print(f"Token ID of {new_token}: {new_token_id}")
new_token_input_embedding_before = model.get_input_embeddings().weight.data[new_token_id].clone()

# Preprocess the quantized model for QLoRA Training
model = prepare_model_for_kbit_training(model)

# Print all named modules to find the embedding layer name
print("==============================================================")
print("\n==== Print all named modules ====\n")
for name, module in model.named_modules():
    print(name)

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "embed_tokens"], # Include the input embedding layer
)
model = get_peft_model(model, peft_config)

# Prepare Dataset
def tokenize_func(dataset):
    tokenized = tokenizer(
        dataset["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


dataset = (
    Dataset.from_dict({"text": fine_tuning_sentences})
    .map(tokenize_func, batched=True)
)

# Training Arguments
training_args = TrainingArguments(
    output_dir=save_model_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,  # Reduces memory usage
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    bf16=True,    # Use mixed precision training
    logging_steps=10,
    save_strategy="epoch",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
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
model.save_pretrained(save_model_dir)
tokenizer.save_pretrained(save_model_dir)

# Compare the input embedding of the new token before and after fine-tuning
new_token_input_embedding_after = model.get_input_embeddings().weight.data[new_token_id].clone()
similarity = torch.nn.functional.cosine_similarity(new_token_input_embedding_before, new_token_input_embedding_after)
print(f"Similarity of the new token input embedding before and after: {similarity:.4f}")

print("==============================================================")
print("\n==== Evaluation after fine-tuning ====\n")
# Evaluate the results
training_positive_context_sentences = dataset.training_positive_context_sentences
training_negative_context_sentences = dataset.training_negative_context_sentences
testing_context_sentences = dataset.testing_context_sentences

new_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_token))
if isinstance(new_token_id, list) and len(new_token_id) == 1:
    new_token_id = new_token_id[0]
else:
    print(f"Error: new_token_id is not a single token ID. It is: {new_token_id}")
    sys.exit(1)

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
