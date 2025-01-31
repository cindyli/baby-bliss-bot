# Utility functions for running the disambiguation scripts that caculates the output embedding
# of the new token that captures the disambiguated meaning of an English word.

import torch
import torch.nn.functional as F


def get_hidden_state_and_next_token_logits(model, tokenizer, text, return_logits=False):
    """Get the hidden states before final layer for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get last hidden state before final layer
        hidden_state_from_last_layer = outputs.hidden_states[-1][:, -1, :]  # Token's hidden state from the last layer
        next_token_logits = outputs.logits[:, -1, :] if return_logits else None  # Logits for next token
    return hidden_state_from_last_layer, next_token_logits


def create_training_data(model, tokenizer, positive_context_sentences, negative_context_sentences, target_token_id):
    """Create training data from context sentences."""
    hidden_states = []
    target_logits = []

    # Positive context sentences - high target logits
    for context in positive_context_sentences:
        h, logits = get_hidden_state_and_next_token_logits(model, tokenizer, context, True)
        hidden_states.append(h)
        target_logits.append(logits[0, target_token_id].item())

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


def get_token_prediction(model, tokenizer, context_sentences, target_token_id):
    """Test the rank of the target token in predictions for each testing context."""
    results = []

    for context in context_sentences:
        with torch.no_grad():
            inputs = tokenizer(context, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

            # Get token rankings
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            rank_of_target_token_id = (sorted_indices == target_token_id).nonzero().item()

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            results.append({
                'context': context,
                'rank_of_target_token_id': rank_of_target_token_id + 1,  # Convert to 1-based ranking
                'top_5_predictions': top_5_tokens
            })

    return results
