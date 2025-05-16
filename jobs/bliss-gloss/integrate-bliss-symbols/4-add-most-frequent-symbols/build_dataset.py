# python build_dataset.py <original_dataset_py_file> <bliss_ids_added_json_file> <glosses> <bci_av_id> <output_dir_in_string>
# python build_dataset.py ./data/original.py ./data/bliss_ids_added.json '["a", "an", "any"]' 12321 "./data/"

import sys
import json
import re
import random

MIN_PARTIAL_SENTENCE_LENGTH = 40
TOKEN_TEMPLATE = "[BLISS_{bciAvId}]"


# Replace special characters in a sentence. This is a must have for sentences in all datasets
# because LLMs tend to use "’" instead of "'".
def replace_special_chars(sentence):
    return sentence.replace("’", "'")


# Replace the whole word in a sentence with a given replacement
# Note: target_words can be a word string or a list of string words
def replace_whole_word(text, target_words, replace_with):
    if isinstance(target_words, str):
        target_words = [target_words]
    for target_word in target_words:
        pattern = r"\s*\b{}\b".format(re.escape(target_word))
        text = re.sub(pattern, replace_with, text, flags=re.IGNORECASE)
    return text


# Truncate the sentence to the last occurrence of any gloss
def get_partial_sentences_by_keywords(sentence, glosses):
    all_indices = []

    for gloss in glosses:
        if not gloss.strip():  # Skip empty or whitespace-only glosses
            continue

        # Match whole word with word boundaries and escape special characters
        pattern = r'\b{}\b'.format(re.escape(gloss))

        # Find all matches and their start positions
        for match in re.finditer(pattern, sentence):
            all_indices.append(match.start())

    if not all_indices:
        print(f"Error: No gloss found in sentence: '{sentence}'")
        return sentence

    last_phrase_start = max(all_indices)
    truncated = sentence[:last_phrase_start]
    return truncated.rstrip()


# Process positive context sentences
# 1. Replace special characters: "’" -> "'"
# 2. Truncate the sentence to the last occurrence of any gloss
# 3. Remove sentences that are too short (less than MIN_PARTIAL_SENTENCE_LENGTH)
# 4. Split the sentences into training and testing sets (80% training, 20% testing)
def process_positive_context_sentences(orig_positive_context_sentences, glosses):
    # Process all sentences first
    processed_sentences = []
    for sentence in orig_positive_context_sentences:
        sentence = replace_special_chars(sentence)
        partial_sentence = get_partial_sentences_by_keywords(sentence, glosses)
        if len(partial_sentence) > MIN_PARTIAL_SENTENCE_LENGTH:
            processed_sentences.append(partial_sentence)

    # Calculate split indices
    total_sentences = len(processed_sentences)
    training_size = int(total_sentences * 0.8)  # 80% for training

    return processed_sentences[:training_size], processed_sentences[training_size:]


# Process negative context sentences
# 1. Replace special characters: "’" -> "'"
# 2. Split the sentences into training and testing sets (80% training, 20% testing)
def process_negative_context_sentences(orig_negative_context_sentences):
    # Replace special characters in the original sentences
    processed_sentences = [replace_special_chars(sentence) for sentence in orig_negative_context_sentences]

    # Calculate split indices
    total_sentences = len(processed_sentences)
    training_size = int(total_sentences * 0.8)  # 80% for training

    return processed_sentences[:training_size], processed_sentences[training_size:]


# Process fine-tuning sentences
# 1. Replace special characters: "’" -> "'"
# 2. Replace glosses of the in-processing bci_av_id with the corresponding token. These sentences compose
#    the dataset for fine-tuning.
# 3. Go through the list of BCI-AV-IDs already in the model and replace their glosses with the corresponding tokens.
#    These sentences are appended to the fine-tuning dataset to ensure the model learns how different Bliss tokens
#    interact with each other.
# 4. Randomize the order of the sentences to ensure a good mix of training and validation data.
# 5. Split the sentences into training and validation sets (80% training, 20% validation)
def process_fine_tuning_sentences(orig_fine_tuning_sentences, bci_av_id, existing_id_gloss_list):
    # Replace special characters in the original sentences
    normalized_sentences = [replace_special_chars(sentence) for sentence in orig_fine_tuning_sentences]

    # Replace glosses of the in-processing bci_av_id with the corresponding token
    glosses_for_current_id = existing_id_gloss_list[bci_av_id]
    processed_single_sentences = [replace_whole_word(sentence, glosses_for_current_id, TOKEN_TEMPLATE.format(bciAvId=bci_av_id)) for sentence in normalized_sentences]

    # Go through the list of BCI-AV-IDs already in the model and replace their glosses with the corresponding tokens
    # This is to ensure the fine-tuning learns how different Bliss tokens interact with each other
    processed_sentences = []
    for sentence in processed_single_sentences:
        modified = sentence
        for gloss_id, target_glosses in existing_id_gloss_list.items():
            modified = replace_whole_word(modified, target_glosses, TOKEN_TEMPLATE.format(bciAvId=gloss_id))
        if modified != sentence:
            processed_sentences.append(modified)

    # Merge the processed sentences with the ones that only glosses for the in-processing bci_av_id were replaced
    processed_sentences.extend(processed_single_sentences)

    # Randomnize the order of the sentences to ensure a good mix of training and validation data
    random.shuffle(processed_sentences)

    # Calculate split indices
    total_sentences = len(processed_sentences)
    fine_tuning_size = int(total_sentences * 0.8)  # 80% for training

    return processed_sentences[:fine_tuning_size], processed_sentences[fine_tuning_size:]


# Remove the empty space before the placeholder in the testing text generation prompts
def process_testing_text_generation_prompts(orig_testing_text_generation_prompts):
    # Replace special characters in the original sentences
    return [prompt.replace(" {placeholder}", "{placeholder}") for prompt in orig_testing_text_generation_prompts]


# The format of the output file is: <output_dir>/dataset_<bci_av_id>_<glosses>.py
# The glosses are processed to remove all non-alphanumeric characters and joined with underscores.
def get_output_file_location(output_dir, bci_av_id, glosses):
    # Remove all non-alphanumeric characters from each gloss
    processed_glosses = [re.sub(r'[^a-zA-Z0-9]', '', gloss) for gloss in glosses]

    # Join processed glosses with underscores
    gloss_part = '_'.join(processed_glosses)

    # Construct the full file path
    return f"{output_dir}dataset_{bci_av_id}_{gloss_part}.py"


if len(sys.argv) != 6:
    print("Usage: python build_dataset.py <original_dataset_py_file> <bliss_ids_added_json_file> <glosses> <bci_av_id> <output_dir_in_string>")
    print("Example: python build_dataset.py ./data/original.py ./data/bliss_ids_added.json '[\"a\", \"an\", \"any\"]' 12321 \"./data/\"")
    exit()

original_dataset_py_file = sys.argv[1]
existing_id_gloss_list = json.load(open(sys.argv[2], "r"))
glosses = json.loads(sys.argv[3])
bci_av_id = sys.argv[4]
output_dir = sys.argv[5]

# Prepare the existing ID to gloss list. The glosses in the list will be replaced by their corresponding tokens
# when processing the fine-tuning sentences.
existing_id_gloss_list[bci_av_id] = glosses

# Read and execute the file to extract the target variables
namespace = {}
target_variable_names_in_original_dataset = ["positive_context_sentences", "negative_context_sentences", "fine_tuning_sentences", "testing_text_generation_prompts"]
with open(original_dataset_py_file, "r") as f:
    original_dataset = f.read()
exec(original_dataset, namespace)

# Get datasets and process them
orig_positive_context_sentences = namespace["positive_context_sentences"]
orig_negative_context_sentences = namespace["negative_context_sentences"]
orig_fine_tuning_sentences = namespace["fine_tuning_sentences"]
orig_testing_text_generation_prompts = namespace["testing_text_generation_prompts"]

training_positive_context_sentences, testing_positive_context_sentences = process_positive_context_sentences(orig_positive_context_sentences, glosses)
training_negative_context_sentences, testing_negative_context_sentences = process_negative_context_sentences(orig_negative_context_sentences)
fine_tuning_sentences, validation_sentences = process_fine_tuning_sentences(orig_fine_tuning_sentences, bci_av_id, existing_id_gloss_list)
testing_text_generation_prompts = process_testing_text_generation_prompts(orig_testing_text_generation_prompts)

# Save the processed dataset into a new output file
# The location of the output file is in the format: <output_dir>/dataset_<bci_av_id>_<glosses>.py
output_file = get_output_file_location(output_dir, bci_av_id, glosses)
with open(output_file, "w") as f:
    f.write(f"\ntraining_positive_context_sentences = {json.dumps(training_positive_context_sentences, indent=4, ensure_ascii=False)}\n")
    print(f"len(training_positive_context_sentences) = {len(training_positive_context_sentences)}")

    f.write(f"\ntesting_positive_context_sentences = {json.dumps(testing_positive_context_sentences, indent=4, ensure_ascii=False)}\n")
    print(f"len(testing_positive_context_sentences) = {len(testing_positive_context_sentences)}")

    f.write(f"\ntraining_negative_context_sentences = {json.dumps(training_negative_context_sentences, indent=4, ensure_ascii=False)}\n")
    print(f"len(training_negative_context_sentences) = {len(training_negative_context_sentences)}")

    f.write(f"\ntesting_negative_context_sentences = {json.dumps(testing_negative_context_sentences, indent=4, ensure_ascii=False)}\n")
    print(f"len(testing_negative_context_sentences) = {len(testing_negative_context_sentences)}")

    f.write(f"\nfine_tuning_sentences = {json.dumps(fine_tuning_sentences, indent=4, ensure_ascii=False)}\n")
    print(f"len(fine_tuning_sentences) = {len(fine_tuning_sentences)}")

    f.write(f"\nvalidation_sentences = {json.dumps(validation_sentences, indent=4, ensure_ascii=False)}\n")
    print(f"len(validation_sentences) = {len(validation_sentences)}")

    f.write(f"\ntesting_text_generation_prompts = {json.dumps(testing_text_generation_prompts, indent=4, ensure_ascii=False)}\n")
    print(f"len(testing_text_generation_prompts) = {len(testing_text_generation_prompts)}")
