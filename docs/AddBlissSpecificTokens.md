# Adding a New Token for a Bliss Symbol in LLaMA

This documentation outlines the steps to add a new token for a Bliss symbol into the LLaMA model. As an example, we will use the Bliss symbol with ID 24819, which represents "lowness" or "shortness" in the context of height.

## Step 1: Create and Prepare the Dataset for a New Bliss Token

### Step 1.1 Generate Synthetic Datasets using LLMs

You can use a large language model like ChatGPT to generate synthetic data. Refer to prompts.md for sample prompts used to create various datasets.

Each dataset should be diverse and span a wide range of contexts and grammatical structures. All datasets are Python lists.

The datasets required are:
1. **Positive Partial Context Sentences****

Sentences that provide strong contextual cues leading to the prediction of "lowness" or "shortness" in the context of height.

2. **Negative Partial Context Sentences**

Sentences that should not lead to the prediction of "lowness" or "shortness" in the height context. These may include:
* Irrelevant contexts where those words might appear
* Contexts leading to different word predictions

3. **Fine-Tuning Sentences**

Sentences that clearly use "lowness", "shortness", or both in the height context. These will be used to fine-tune the model on the new token.

4. **Text Generation Prompts for Testing**

Partial sentences that contain the placeholder {placeholder} where the new token or gloss word should appear. These are used to evaluate how well the fine-tuned model understands the connection between the new token and its context.

#### Handling Rare Glosses

For glosses that are rare or contextually hard to place in partial sentences, you can ask the LLM to generate full sentences containing the gloss. Then use the helper script below to truncate those sentences from the position of the gloss onward.

* Script: `generate_partial_context_sentences.py`

This script truncates full sentences starting from the target gloss, retaining the sentence as-is if the gloss is not found.

* Example usage:
```
python generate_partial_context_sentences.py "./data/dataset_24918_lowness_shortness.py" '[" lowness", " shortness"]'
```

### Step 1.2: Screen Datasets

After generating the datasets, verify their quality by checking whether the glosses rank appropriately:

* Positive partial sentences should rank highly for the target glosses.

* Negative partial sentences should rank low or incorrectly.

Use the following script to assess sentence quality:

* Script: `sentence_predictions.py`
It processes both `training_positive_context_sentences` and `training_negative_context_sentences`.

Adjust these threshold variables in the script as needed:

* `RANK_POSITIVE_THRESHOLD`: Maximum rank allowed for at least one of the target gloss tokens to be accepted as positive.
* `RANK_NEGATIVE_THRESHOLD`: Minimum rank below which all target gloss tokens must fall to be accepted as negative.

### Step 3: Finalize the Dataset

Once the datasets are validated, split the first three datasets into training and testing subsets. Save them in a file named:

```
dataset_{BlissID}_{gloss1}_{gloss2}_..._{glossN}.py
```

* Note: Remove spaces from gloss names in the filename.
* Place the file in the directory: `jobs/bliss-gloss/data/`

The dataset file should define the following Python list variables:

```
training_positive_context_sentences     # 80% of positive partial context sentences
testing_positive_context_sentences      # 20% of positive partial context sentences
training_negative_context_sentences     # 80% of negative partial context sentences
testing_negative_context_sentences      # 20% of negative partial context sentences
fine_tuning_sentences                   # 80% of fine-tuning sentences
validation_sentences                    # 20% of fine-tuning sentences
testing_text_generation_prompts         # 100% of generation prompts
```
