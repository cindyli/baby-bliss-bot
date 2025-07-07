# Utility functions for running add_symbols.py
import torch


def get_average_embeddings(model, tokenizer, phrase_list, embedding_type="input"):
    """Compute average input embeddings for a list of multi-token words/phrases"""
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer.weight.data

    phrase_vectors = []

    for phrase in phrase_list:
        tokens = tokenizer.tokenize(phrase)
        if not tokens:  # Handle edge case for empty tokens
            continue

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Get embeddings for all sub-tokens
        token_embeddings = embeddings[token_ids]

        # Average across sub-tokens for this phrase
        phrase_embedding = torch.mean(token_embeddings, dim=0)
        phrase_vectors.append(phrase_embedding)

    # Average across all phrases
    if not phrase_vectors:  # Handle empty input case
        return None

    return torch.mean(torch.stack(phrase_vectors), dim=0)


def calc_embeddings_diff(model, tokenizer, pairs_data):
    """Calculate the difference between input and output embeddings for a new token."""
    input_embedding_diffs = []
    output_embedding_diffs = []

    for pair in pairs_data:
        nouns = [noun.strip() for noun in pair["noun"]["noun_only"].split(",")]
        verbs = [verb.strip() for verb in pair["verb-(to)"]["infinitive_form_verbs"].split(",")]

        average_input_embedding_nouns = get_average_embeddings(model, tokenizer, nouns, embedding_type="input")
        average_input_embedding_verbs = get_average_embeddings(model, tokenizer, verbs, embedding_type="input")

        input_embedding_diffs.append(average_input_embedding_verbs - average_input_embedding_nouns)

        average_output_embedding_nouns = get_average_embeddings(model, tokenizer, nouns, embedding_type="output")
        average_output_embedding_verbs = get_average_embeddings(model, tokenizer, verbs, embedding_type="output")

        output_embedding_diffs.append(average_output_embedding_verbs - average_output_embedding_nouns)

    # Average the differences across all pairs
    input_embedding_diff = torch.mean(torch.stack(input_embedding_diffs), dim=0)
    output_embedding_diff = torch.mean(torch.stack(output_embedding_diffs), dim=0)

    return input_embedding_diff, output_embedding_diff


def get_rank_for_tokens(all_ranks, target_token_ids):
    # Retrieve ranks for target token IDs
    target_ranks = []
    for token_id in target_token_ids:
        rank_position = (all_ranks == token_id).nonzero().item()
        target_ranks.append(rank_position + 1)  # Convert to 1-based ranking
    return target_ranks


def test_token_prediction(model, tokenizer, context_sentences, target_token_ids, new_token_id=None):
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
            if new_token_id is not None:
                rank_of_new_token_id = get_rank_for_tokens(sorted_indices, [new_token_id])
            rank_of_target_token_ids = get_rank_for_tokens(sorted_indices, target_token_ids)

            # Get top 5 predicted tokens
            top_5_tokens = tokenizer.convert_ids_to_tokens(sorted_indices[:5])

            result = {
                'context': context,
                'rank_of_target_token_ids': rank_of_target_token_ids,
                'top_5_predictions': top_5_tokens
            }

            if new_token_id is not None:
                result["rank_of_new_token_id"] = rank_of_new_token_id

            results.append(result)

    return results


def print_results(title, target_tokens, new_token, results):
    # Print results
    print("==============================================================")
    print(f"\n==== {title}")
    for result in results:
        print(f"\nContext: {result['context']}")
        print(f"Rank of {target_tokens}: {result['rank_of_target_token_ids']}")
        if new_token:
            print(f"Rank of {new_token}: {result['rank_of_new_token_id'][0]}")
        print(f"Top 5 predictions: {', '.join(result['top_5_predictions'])}")


def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_text_generation(
    model, tokenizer, testing_text_generation_prompts
):
    for prompt in testing_text_generation_prompts:
        print(f"Prompt: {prompt}")
        print(f"Generated: {generate_text(model, tokenizer, prompt)}\n")


def evaluate_new_token(
    model, tokenizer, new_token, testing_predictions_sentences, testing_text_generation_prompts,
    target_tokens, target_token_ids, new_token_id
):
    # Test token prediction
    results = test_token_prediction(model, tokenizer, testing_predictions_sentences, target_token_ids, new_token_id)
    print_results("Predictions test:", target_tokens, new_token, results)

    # Text Generation
    print("\nValidation - Generation:")
    test_text_generation(model, tokenizer, testing_text_generation_prompts)
