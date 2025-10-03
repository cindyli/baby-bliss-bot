# Word Disambiguation in Blissymbolics

## Overview

Our current approach to associating Blissymbols with language models like Llama relies on using the English
gloss of each symbol as a bridge. A gloss consists of one or more words that convey the meaning of a Bliss
symbol. However, this introduces ambiguity because English words often have multiple meanings, while
Blissymbols tend to be more precise.

For example, the English word *bark* can refer to either the sound made by a dog or the outer layer of a tree.
Blissymbolics, on the other hand, has distinct symbols for each meaning, eliminating ambiguity.

As part of our research into word disambiguation, we introduce a new special token into the Llama model to
explicitly represent *bark* in the sense of an animal sound, distinguishing it from *bark* as tree bark.

## Challenges

This approach introduces two main challenges, depending on the structure of the gloss:

1. **Single-token glosses (e.g., "bark")**
   The new token's input embedding can be initialized using the input embedding of the existing token "bark."
   Contextual processing in the hidden layers will disambiguate the meaning. However, we must compute a
   distinct output embedding for the new token.

2. **Multi-token glosses (e.g., "wool shop")**
   Both the input and output embeddings must be computed for the new token to capture the combined meaning of
   multiple tokens.

This document explores solutions for these two challenges.

## Computing the Output Embedding

We use a data-driven approach to compute the output embedding for the new token by leveraging context sentences.
Two sets of sentences are created:

**1. Positive Context Sentences**
These sentences strongly predict "bark" in the sense of an animal sound. For each sentence, we extract:
- The **contextual embedding** (hidden states from the last hidden layer) of the last token.
- The **logits** for the token "bark"

**2. Negative Context Sentences**
These sentences either predict "bark" in the sense of tree bark or do not predict "bark" at all. For these
sentences, we extract:
- The **contextual embedding** of the last token.
- The **logits** for "bark," assigning a negative value to ensure the new token does not receive a high
probability in these contexts.

**Mathematical Formulation**
For each context sentence, the calculation of the logits follows this formula:
```
logit_c = dot_product(contextual_embedding_c, output_embedding_of_new_token)
```

Given:
- **N**: Total number of context sentences
- **D**: Model embedding dimension
- **OE**: Output embedding of the new token

We aggregate extracted data into:
1. **T_CE**: A tensor of shape *(N, D)* holding all contextual embeddings.
2. **T_L**: A tensor of shape *(N, 1)* holding all logits.

The matrix equation to calculate logits for all context sentences in one shot is:
```
T_CE * OE = T_L
```

Two approaches are used to solve for OE:

1. **Optimization Approach**
2. **Inversion of Matrix Multiplication Approach**

### Approach 1: Optimization

This approach starts with a random output embedding and optimizes it to minimize the difference between
expected and actual logits.

Three versions of this approach were tested. See the comments in
[optimize_output_embedding.py](../jobs/bliss_gloss/disambiguation/optimize_output_embedding.py) for details on each version.

#### Results
The optimization approach produced inconsistent results. The ranking of the new token was good in some contexts
but poor in others. Test results are available in:

- [OE-optimization-V1-result_1000_0.01_0.2.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V1-result_1000_0.01_0.2.log)
- [OE-optimization-V1-result_500_0.01_0.3.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V1-result_500_0.01_0.3.log)
- [OE-optimization-V1-result_800_0.01_0.1.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V1-result_800_0.01_0.1.log)
- [OE-optimization-V2-result_700_0.01_0.2.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V2-result_700_0.01_0.2.log)
- [OE-optimization-V2-result_800_0.001_0.3.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V2-result_800_0.001_0.3.log)
- [OE-optimization-V3-result_800_0.01.log](../jobs/bliss_gloss/disambiguation/test_results/OE-optimization-V3-result_800_0.01.log)

### Approach 2: Inversion of Matrix Multiplication

This approach inverts the matrix equation directly using
[torch.linalg.lstsq()](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html).

- **Script**: [calc_output_embedding.py](../jobs/bliss_gloss/disambiguation/calc_output_embedding.py)

#### Results
This method produced significantly better results across all test sentences. The output embedding calculated
with this method led to more accurate predictions. The test result is available in:

- [OE-invert-matrix-multiplication.log](../jobs/bliss_gloss/disambiguation/test_results/OE-invert-matrix-multiplication.log)

### Supporting Data and Scripts

#### Dataset
- **Script**: [dataset_animal_bark.py](../jobs/bliss_gloss/disambiguation/data/dataset_animal_bark.py)

This script defines three arrays that contain context sentences for training or testing:
1. Positive context sentences: The model assigns a high probability to "bark" in the sense of animal bark as
a top prediction
2. Negative context sentences: The model assigns a low probability to "bark" in the sense of animal bark in
these contexts
3. Testing context sentences: Mixed-context examples for testing

#### Sentence Prediction Evaluation
Checks the model’s prediction of "bark" across all training and testing sentences.
- **Script**: [sentence_predictions.py](../jobs/bliss_gloss/disambiguation/sentence_predictions.py)
- **Results**: [dataset_animal_bark_predictions.log](../jobs/bliss_gloss/disambiguation/test_results/dataset_animal_bark_predictions.log)

#### Output Embedding Comparison
Compares cosine similarity between embeddings generated by the two approaches. The optimization approach uses
version 3 with:
- **Epochs**: 200
- **Learning Rate**: 0.01

- **Script**: [output_embedding_compare.py](../jobs/bliss_gloss/disambiguation/output_embedding_compare.py)
- **Results**: [output_embedding_compare.log](../jobs/bliss_gloss/disambiguation/test_results/output_embedding_compare.log)

### Conclusion
The **inversion of matrix multiplication** approach provides the best results for computing output embeddings
of disambiguated Blissymbol tokens. The optimization approach, while feasible, produces inconsistent outcomes.Future work will explore refining input embeddings for multi-token glosses.
