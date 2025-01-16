# Usage: python optimize_embeddings.py <epochs> <learning_rate> <similarity_loss_weight>

"""
This script adds a new special token to a llama model to represent specifically
the meaning of an animal bark, disambiguating it from other meanings like tree bark.
This script creates a training dataset from diverse context sentences that may predict
the word "bark" in the sense of an animal bark. It then uses this dataset to optimize
the input and output embeddings for the new token.

The script does:
1. Collecting hidden states output from the last layer and logits from context sentences that may
   predict "bark" in the sense of an animal bark
2. Using these to optimize both input and output embeddings for the new token through:
   - Output embedding optimization: Making the token predict similarly to "bark" in dog contexts
   - Input embedding optimization: Maintaining similarity with output embedding while allowing
     for role-specific variations
   - Joint loss function that balances prediction accuracy and embedding consistency

The script then tests the effectiveness of the new token by:
- Checking the prediction rank of the token " bark" in context sentences for testing
- Checking the prediction rank of the new token in the same context sentences
- Checking top 5 predicted tokens

Note: This approach helps create more semantically precise tokens by learning from contextual
usage patterns rather than directly copying existing token embeddings.
"""

import sys
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


def get_hidden_state_and_next_token_logits(model, tokenizer, text):
    """Get the hidden states before final layer for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get last hidden state before final layer
        hidden_state_from_last_layer = outputs.hidden_states[-1][:, -1, :]  # Token's hidden state from the last layer
        next_token_logits = outputs.logits[:, -1, :]  # Logits for next token
    return hidden_state_from_last_layer, next_token_logits


def create_training_data(model, tokenizer, context_sentences, target_token_id):
    """Create training data from context sentences."""
    hidden_states = []
    target_logits = []

    for context in context_sentences:
        h, logits = get_hidden_state_and_next_token_logits(model, tokenizer, context)

        hidden_states.append(h)
        target_logits.append(logits[0, target_token_id].item())

    return torch.cat(hidden_states, dim=0), torch.tensor(target_logits)


def optimize_embeddings(model, hidden_states, target_logits, embed_dim, epochs, learning_rate, similarity_loss_weight):
    """Joint optimization of input and output embeddings."""
    input_emb = torch.randn(embed_dim, requires_grad=True, device=model.device)
    output_emb = torch.randn(embed_dim, requires_grad=True, device=model.device)

    optimizer = torch.optim.Adam([input_emb, output_emb], lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Output embedding loss
        predicted_logits = torch.matmul(hidden_states, output_emb)
        output_loss = F.mse_loss(predicted_logits, target_logits)

        # Input-output similarity constraint
        similarity_loss = torch.dist(input_emb, output_emb)

        # Total loss
        loss = output_loss + similarity_loss_weight * similarity_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return input_emb, output_emb


def add_token_to_model(model, tokenizer, input_emb, output_emb, new_token):
    """Add new token to model's vocabulary and embedding matrices."""
    # Add token to tokenizer
    num_added_tokens = tokenizer.add_tokens([new_token])
    if num_added_tokens == 0:
        raise ValueError("Token already exists in vocabulary")

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    new_token_id = tokenizer.convert_tokens_to_ids(new_token)

    # Set the embeddings
    with torch.no_grad():
        model.get_input_embeddings().weight[new_token_id] = input_emb
        model.get_output_embeddings().weight[new_token_id] = output_emb

    return new_token_id


def test_token_prediction(model, tokenizer, context_sentences, new_token_id, target_token_id):
    """Test the rank of new token in predictions for each testing context."""
    results = []

    for context in context_sentences:
        with torch.no_grad():
            inputs = tokenizer(context, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

            # Get token rankings
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            rank_of_new_token_id = (sorted_indices == new_token_id).nonzero().item()
            rank_of_target_token_id = (sorted_indices == target_token_id).nonzero().item()

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            results.append({
                'context': context,
                'rank_of_new_token_id': rank_of_new_token_id + 1,  # Convert to 1-based ranking
                'rank_of_target_token_id': rank_of_target_token_id + 1,  # Convert to 1-based ranking
                'top_5_predictions': top_5_tokens
            })

    return results


if len(sys.argv) != 4:
    print("Usage: python optimize_embeddings.py <epochs> <learning_rate> <similarity_loss_weight>")
    sys.exit(1)

epochs = int(sys.argv[1])
learning_rate = float(sys.argv[2])
similarity_loss_weight = float(sys.argv[3])

# Diverse context sentences for training, generated by Claude 3.5 Sonnet and ChatGPT 3.5. They capture different
# aspects that lead to a potential prediction the word "bark" meaning an animal bark. These aspects include:
# basic dog actions, different emotional states, time/situation specific, communication context,
# training and behaviour, and a bark sound without mentioning animals.
context_sentences_for_training = [
    "The excited puppy began to",
    "When strangers approach, guard dogs will",
    "Hearing a noise outside, the dog started to",
    "Seeing the mailman, the German Shepherd would always",
    "To alert its owner of danger, a dog will",
    "The frightened dog let out a loud",
    "Happy to see its owner, the golden retriever would",
    "The aggressive dog growled and then",
    "Feeling threatened, the small dog would",
    "Playfully, the energetic puppy would",
    "Late at night, the neighborhood dogs often",
    "During the full moon, wolves howl and dogs",
    "Whenever the doorbell rings, most dogs",
    "At the sight of a squirrel, the terrier would",
    "While chasing cats, dogs typically",
    "To communicate with other dogs, a puppy will",
    "Instead of wagging its tail, the dog chose to",
    "As a warning signal, the watchdog would",
    "To get attention from its owner, the dog would",
    "Rather than whimper, the dog decided to",
    "Despite training, some dogs still",
    "A well-behaved dog shouldn't randomly",
    "During obedience class, the dog refused to stop",
    "The trainer advised against letting your dog",
    "Even quiet breeds sometimes need to",
    "The sudden noise echoed through the neighborhood - a deep, threatening",
    "At midnight, an unfamiliar sound made me jump: a sharp, piercing",
    "From somewhere in the darkness came a loud, aggressive",
    "Behind the fence, I heard an angry",
    "The strange sound started low, then rose into a harsh"
]

# Diverse context sentences for testing, generated by ChatGPT 3.5.
# The first 5 sentences tend to lead to a prediction of "bark" in the sense of an animal bark.
# The middle 5 sentences tend to lead to a prediction of "bark" in the sense of tree bark.
# The last 5 sentences are random sentences that may not predict "bark" in any sense.
context_sentences_for_testing = [
    "As soon as the stranger stepped into the yard, the dog let out a sharp",
    "The trainer used a clicker to stop the dog from continuing to",
    "I was walking down the quiet street when, out of nowhere, I heard a sudden",
    "The calm evening was interrupted by the echo of a distant",
    "She paused, straining to listen, and then the unmistakable",
    "The forest was dense and quiet, with towering trees whose rough texture was most noticeable on their thick trunks, covered in deep, cracked",
    "The young explorers marveled at the tall trees, each one displaying unique features, such as the coarse and rugged",
    "She paused to examine the large tree, noting how its surface was rough and worn, with the thick layers of",
    "Walking along the trail, he ran his hand over the tree's surface, feeling the texture shift as he brushed against the weathered",
    "As we ventured deeper into the woods, we saw that many of the trees had a distinct pattern on their trunks, a textured surface that resembled hardened",
    "The gentle rustling of leaves filled the quiet forest as the sun set behind the",
    "He couldn’t decide between the red sweater and the blue",
    "After weeks of planning, the team finally launched their new",
    "The aroma of freshly baked bread wafted through the tiny",
    "The sun dipped below the horizon, casting a warm glow over the"
]

# Track the total running time of this script
start_time = time.time()

# Load model and tokenizer
model_dir = os.path.expanduser("~") + "/Development/LLMs/Llama-3.1-8B-Instruct"
# model_dir = os.path.expanduser("~") + "/projects/ctb-whkchun/s2_bliss_LLMs/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Get original "bark" token id for training data
target_token = " bark"
tokens = tokenizer.tokenize(target_token)
bark_token_id = tokenizer.convert_tokens_to_ids(tokens)[0]
print("bark_token_id:", bark_token_id)

# Prepare training data
hidden_states, target_logits = create_training_data(
    model, tokenizer, context_sentences_for_training, bark_token_id
)

# Optimize embeddings
embed_dim = model.config.hidden_size
input_emb, output_emb = optimize_embeddings(
    model, hidden_states, target_logits, embed_dim, epochs, learning_rate, similarity_loss_weight
)

# Add new token to model
new_token = "[BLISS_24020]"
new_token_id = add_token_to_model(model, tokenizer, input_emb, output_emb, new_token)

end_time_training = time.time()
elapsed_time = end_time_training - start_time
print(f"Execution time for training: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

# Test predictions
results = test_token_prediction(model, tokenizer, context_sentences_for_testing, new_token_id, bark_token_id)

# Print results
print("\nTesting predictions for new token:")
for result in results:
    print(f"\nContext: {result['context']}")
    print(f"Rank of {target_token}: {result['rank_of_target_token_id']}")
    print(f"Rank of {new_token}: {result['rank_of_new_token_id']}")
    print(f"Rank difference: {result['rank_of_target_token_id'] - result['rank_of_new_token_id']}")
    print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")

end_time = time.time()

# Calculate elapsed time
end_time_testing = time.time()
elapsed_time = end_time_testing - end_time_training
print(f"Execution time for testing: {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
