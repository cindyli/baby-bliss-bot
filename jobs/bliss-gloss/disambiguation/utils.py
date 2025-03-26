# Utility functions for running the disambiguation scripts that caculates the output embedding
# of the new token that captures the disambiguated meaning of an English word.

import torch
import torch.nn.functional as F


def get_hidden_state_and_next_token_logits(model, tokenizer, text, return_logits=False):
    """Get the hidden states before final layer for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Extract the hidden state of the last token in the sequence from the last layer
        hidden_state_from_last_layer = outputs.hidden_states[-1][:, -1, :]
        # Extract the logit for the next token prediction
        next_token_logits = outputs.logits[:, -1, :] if return_logits else None
    return hidden_state_from_last_layer, next_token_logits


def create_training_data(model, tokenizer, positive_context_sentences, negative_context_sentences, target_token_ids):
    """Create training data from context sentences."""

    # Convert target token ids to list if not already
    target_token_ids = target_token_ids if isinstance(target_token_ids, list) else [target_token_ids]

    hidden_states = []
    target_logits = []

    # Positive context sentences - high target logits
    for context in positive_context_sentences:
        h, logits = get_hidden_state_and_next_token_logits(model, tokenizer, context, True)
        hidden_states.append(h)
        target_logits.append(logits[0, target_token_ids].max().item())

    # Negative context sentences - low target logits
    for context in negative_context_sentences:
        h, logits = get_hidden_state_and_next_token_logits(model, tokenizer, context)
        hidden_states.append(h)
        target_logits.append(-10)  # discourage predicting the target token

    return torch.cat(hidden_states, dim=0).to(model.device), torch.tensor(target_logits, device=model.device)


def calc_embeddings(hidden_states, target_logits):
    output_emb_before_squeeze = torch.linalg.lstsq(hidden_states, target_logits.unsqueeze(1)).solution
    output_emb = output_emb_before_squeeze.squeeze(1)

    return output_emb


def optimize_embeddings(model, hidden_states, target_logits, embed_dim, epochs, learning_rate):
    """Joint optimization of input and output embeddings."""
    output_emb = torch.nn.Parameter(torch.randn(embed_dim, requires_grad=True, device=model.device))

    optimizer = torch.optim.Adam([output_emb], lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Output embedding loss
        predicted_logits = torch.matmul(hidden_states, output_emb)
        loss = F.mse_loss(predicted_logits, target_logits)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return output_emb


def add_token_to_model(model, tokenizer, new_token, input_emb, output_emb):
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


def get_rank_for_tokens(all_ranks, target_token_ids):
    # Retrieve ranks for target token IDs
    target_ranks = []
    for token_id in target_token_ids:
        rank_position = (all_ranks == token_id).nonzero().item()
        target_ranks.append(rank_position + 1)  # Convert to 1-based ranking
    return target_ranks


def test_token_prediction(model, tokenizer, context_sentences, new_token_id, target_token_ids):
    """Test the rank of new token in predictions for each testing context."""

    # Convert target token ids to list if not already
    target_token_ids = target_token_ids if isinstance(target_token_ids, list) else [target_token_ids]

    results = []

    for context in context_sentences:
        with torch.no_grad():
            inputs = tokenizer(context, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

            # Get token rankings
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            rank_of_new_token_id = get_rank_for_tokens(sorted_indices, [new_token_id])
            rank_of_target_token_ids = get_rank_for_tokens(sorted_indices, target_token_ids)

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            results.append({
                'context': context,
                'rank_of_new_token_id': rank_of_new_token_id,  # Convert to 1-based ranking
                'rank_of_target_token_ids': rank_of_target_token_ids,  # Convert to 1-based ranking
                'top_5_predictions': top_5_tokens
            })

    return results


def get_token_prediction(model, tokenizer, context_sentences, target_token_ids):
    """Test the ranks of the target tokens in predictions for each testing context."""
    # Convert target token ids to list if not already
    target_token_ids = target_token_ids if isinstance(target_token_ids, list) else [target_token_ids]

    results = []

    for context in context_sentences:
        with torch.no_grad():
            inputs = tokenizer(context, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

            # Get token rankings
            sorted_indices = torch.argsort(next_token_logits, descending=True)

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            results.append({
                'context': context,
                'target_token_ranks': get_rank_for_tokens(sorted_indices, target_token_ids),  # Now a list of ranks
                'top_5_predictions': top_5_tokens
            })

    return results


def print_results(title, target_tokens, new_token, results):
    # Print results
    print("==============================================================")
    print(f"\n==== {title}")
    for result in results:
        print(f"\nContext: {result['context']}")
        print(f"Rank of {target_tokens}: {result['rank_of_target_token_ids']}")
        print(f"Rank of {new_token}: {result['rank_of_new_token_id'][0]}")
        print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")


def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_new_token(
    model, tokenizer, training_positive_context_sentences, training_negative_context_sentences,
    testing_context_sentences, new_token_id, target_token_ids, target_tokens, new_token, phrase
):
    # Test token prediction
    results = test_token_prediction(model, tokenizer, training_positive_context_sentences, new_token_id, target_token_ids)
    print_results("Re-verify predictions on POSITIVE training sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, training_negative_context_sentences, new_token_id, target_token_ids)
    print_results("Re-verify predictions on NEGATIVE training sentences:", target_tokens, new_token, results)
    results = test_token_prediction(model, tokenizer, testing_context_sentences, new_token_id, target_token_ids)
    print_results("Predictions on TESTING training sentences:", target_tokens, new_token, results)

    # Text Generation
    print("\nValidation - Generation:")

    text_generation_prompts = [
        f"The{phrase} sells a variety of products including",
        f"The{phrase} is a place where people can",
        f"I visited the{phrase} to",
        f"The owner of the{phrase} specializes in",
        f"The{phrase} differs from other stores because it",
        f"A cozy{phrase} is",
        f"I met a friendly alpaca farmer at the{phrase} who",
        f"She spent hours at the{phrase}"
    ]

    for prompt in text_generation_prompts:
        print(f"Prompt: {prompt}")
        print(f"Generated text with{phrase}: {generate_text(model, tokenizer, prompt)}")
        print(f"Generated text with {new_token}: {generate_text(model, tokenizer, prompt.replace(phrase, new_token))}\n")
