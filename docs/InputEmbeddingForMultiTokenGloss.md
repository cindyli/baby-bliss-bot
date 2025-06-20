# Calculate Input Embedding for Multi-Token Glosses

Bliss symbols often have glosses composed of multiple tokens. When introducing a new token to represent such a symbol, it's common to use its multi-token gloss as a semantic bridge to integrate it into a language model. One critical step in this process is initializing the **input embedding** for the new token effectively, as it influences the success of subsequent fine-tuning.

This document outlines and evaluates different approaches for initializing the input embedding and their impact on fine-tuning and final model performance.

---

## Overview

- **Goal:** Introduce a new token with a multi-token gloss (e.g., `"wool shop"`) and determine the most effective way to initialize its input embedding.
- **Process:** 
  1. Initialize both input and output embeddings.
  2. Fine-tune the model using a dataset that contains the new token.
  3. Evaluate the model's understanding and generation capabilities with the new token.

---

## Step 1: Initialize Input and Output Embeddings

We use a consistent method to compute the **output embedding** based on previous work on [output embedding via matrix inversion](./docs/OutputEmbeddingForWordDisambiguation.md). This document focuses on different strategies for initializing the **input embedding**.

### Input Embedding Initialization Strategies:

1. **Optimized input embedding** (via contextual optimization)
2. **Averaged input embedding** of all composing tokens
3. **Input embedding of the last token** in the gloss
4. **Random input embedding**
5. **Random input and output embeddings**

---

Since Strategies 2–5 involve straightforward initialization techniques, we focus here on **Strategy 1**, which uses contextual optimization to compute the input embedding:

### Strategy 1: Optimized Input Embedding via Contextual Optimization

- **Example Gloss:** `"wool shop"`

- **Overview:**  
  This method generates an optimized input embedding for the new token by aligning it with the contextual behavior of its gloss in natural sentences.

- **Steps:**
  1. **Prepare Context Sentences**
     - Create two sets of partial sentences:
       - **Positive Contexts**: Sentences where `"wool shop"` is a highly likely next phrase.
       - **Negative Contexts**: Sentences where `"wool shop"` is an unlikely continuation.
     - *Examples can be found in the variables* `training_positive_context_sentences` *and* `training_negative_context_sentences` *in* [`dataset_29111_wool_shop.py`](../jobs/bliss-gloss/multi-tokens-phrase/data/dataset_29111_wool_shop.py).

  2. **Run Optimization**
     - Use the script [`input_embedding_optimization.py`](../jobs/bliss-gloss/multi-tokens-phrase/input_embedding_optimization.py) to:
       - Compute the **target contextual embeddings** of `"wool shop"` across the context sentences.
       - **Iteratively optimize** a randomly initialized input embedding so that it matches the averaged contextual behavior of the gloss.

- **Result:**  
  This approach ensures the new token embedding reflects the semantic context in which the gloss commonly appears, improving its initialization for downstream fine-tuning.

---

## Step 2: Fine-Tuning Procedure

Once the input and output embeddings are initialized, fine-tuning adjusts:
- The new token's input and output embeddings
- Attention layers in the transformer

This step uses the script [`qlora_embedding_exploration.py`](../jobs/bliss-gloss/multi-tokens-phrase/qlora_embedding_exploration.py) to perform a QLoRA fine-tuning.

*The Example of the fine-tuning dataset can be found in the variables* `fine_tuning_sentences` *in* [`dataset_29111_wool_shop.py`](../jobs/bliss-gloss/multi-tokens-phrase/data/dataset_29111_wool_shop.py).

---

## Experimental Results

| Input Embedding Initialization | Best Epoch | Eval Loss | Train Loss | Cosine Similarity (Before/After Fine-Tuning) | Test Quality | Time/Epoch |
|-------------------------------|------------|-----------|------------|----------------------------------------------|--------------|------------|
| **Optimized input; calculated output** | 6 | 0.3367 | 0.1824 | Input: 1.000 | Good | ~0.87 min |
| **Averaged input; calculated output** | 6 | **0.3336** | **0.1816** | Input: 0.0344 (vs random), 1 (vs optimized) | **Best** | ~0.84 min |
| **Last token input ("shop"); calculated output** | 6.33 | 0.3371 | 0.2011 | Input: 0.0533 (vs random), 1 (vs optimized) | Good | ~0.84 min |
| **Random input; calculated output** | 5.33 | 0.3789 | 0.2846 | Input: -0.0389, Output: 1.000 | Mostly Good, one anomaly | ~0.87 min |
| **Random input & output** | 16.33 | 0.8693 | 0.2347 | Input: 0.0370, Output: 0.0081 (vs random) | Inconsistent; worse generation | ~0.87 min |

---

## ✅ Conclusion

The **most effective approach** is:

> **Initialize the input embedding using the averaged embeddings of all composing tokens** in the gloss, combined with the calculated output embedding.

### Why this works best:
- Lowest **evaluation and training loss**
- High **semantic alignment** with optimized embedding
- Best performance in **generation** and **prediction tasks**
- Faster convergence with fewer fine-tuning epochs

---

## 📎 References

- [Output Embedding for Word Disambiguation](./docs/OutputEmbeddingForWordDisambiguation.md)
- [Optimization Script](../jobs/bliss-gloss/multi-tokens-phrase/input_embedding_optimization.py)
- [QLoRA fine-tuning Script](../jobs/bliss-gloss/multi-tokens-phrase/qlora_embedding_exploration.py)
- [Context Dataset: `dataset_29111_wool_shop.py`](../jobs/bliss-gloss/multi-tokens-phrase/data/)
