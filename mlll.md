# Diffusion-Based Generative Masked Language Models for Indic Languages

## Overview

Master’s thesis proposal on adapting pre-existing Indic language encoder models (such as IndicBERT) into diffusion-based generative masked language models, and designing an infilling-focused benchmark built from IndicXTREME, IndicGLUE, and IndicNLG datasets. The work targets both a novel architecture for Indic generative masked language modeling and a reusable evaluation suite for text infilling and generation in Indic languages.

## Thesis Topic and Aim

### Topic

The core topic can be formulated as:

> **Diffusion-Based Generative Masked Language Models for Indic Languages: Adapting IndicBERT for Flexible-Length Text Infilling and Evaluation on IndicXTREME, IndicGLUE, and IndicNLG.**

This has three key elements:

- A diffusion-based generative masked language model architecture for Indic languages, built on top of a pre-trained Indic encoder such as IndicBERT or IndicBERT-v3.
- A focus on flexible-length or fixed-length text infilling as a primary generative task, inspired by discrete diffusion models for text infilling(DDOT & DreamOn).
- A comprehensive evaluation across established NLU and NLG benchmarks for Indic languages (IndicXTREME, IndicGLUE, IndicNLG), plus an additional infilling-specific benchmark derived from them.

### Aim

The overarching aim is to:

1. **Design and implement a diffusion-based generative masked language model for Indic languages** by adapting a pre-trained Indic encoder (e.g., IndicBERT) with a discrete diffusion process over token representations.
2. **Create an infilling-focused evaluation benchmark for Indic text** by transforming existing NLU/NLG datasets (IndicXTREME, IndicGLUE, IndicNLG) into span infilling tasks at multiple granularities (word, phrase, sentence).
3. **Systematically compare diffusion-based and conventional masked/autoregressive models** on Indic infilling and generation tasks, measuring quality, diversity, robustness, and cross-lingual behavior.

The aim is not merely to “beat a benchmark”, but to provide a clear modeling contribution and a structured evaluation framework for diffusion language models in Indic settings.

## Background and Literature Context

### Diffusion Language Models

Diffusion language models (DLMs) generate discrete text by iteratively denoising noisy token sequences or latent representations, rather than predicting tokens left-to-right as in autoregressive language models. Recent work has introduced several families of DLMs, including GENIE, Diffusion-LM, DiffusionBERT, and energy-based diffusion models that incorporate sequence-level energy corrections.

Surveys on diffusion for non-autoregressive text generation highlight that diffusion models can improve quality compared to earlier non-autoregressive approaches, while offering parallel generation and a natural framework for editing and infilling. In discrete token spaces, diffusion models must handle non-differentiable sampling and discrete noise processes, which has led to various approximations and design choices that affect speed and quality.

### DiffusionBERT and Generative Masked Language Models

DiffusionBERT is an important prior that demonstrates how to build a generative masked language model by combining a BERT backbone with a discrete diffusion process. It introduces a non-Markovian diffusion model that conditions on both the noisy state and the original context, and uses a time-step-free decoding strategy for efficient generation. Empirically, DiffusionBERT improves perplexity and BLEU on several English text generation tasks compared to previous diffusion-based text models and conventional generative masked LMs.

The proposed thesis extends this idea to Indic languages by using IndicBERT as the backbone and tailoring the diffusion process to the multilingual, code-mixed, and often low-resource nature of Indic text.

### Indic Language Benchmarks and Models

AI4Bharat and others have developed resources for Indic NLP:

- **IndicBERT / IndicBERT-v2/v3**: Multilingual BERT/ALBERT-style encoders trained on large Indic corpora (IndicCorp v2), covering more than a dozen Indic languages and Indian English.
- **IndicXTREME**: A human-supervised benchmark with nine diverse NLU tasks across around 20 Indic languages, designed to test zero-shot capabilities of pretrained language models.
- **IndicGLUE**: A general NLU benchmark comprising tasks such as article genre classification, news classification, headline prediction, Wikipedia section title prediction, cloze-style QA, and masked entity prediction across multiple Indian languages.
- **IndicNLG Benchmark / IndicNLG Suite**: A collection of NLG datasets for Indic languages spanning biography generation from infoboxes, news headline generation, sentence summarization, paraphrase generation, and question generation, amounting to roughly 8M examples across five tasks.

These benchmarks focus on NLU and NLG for Indic languages but have not yet been systematically explored with diffusion-based generative masked language models.

## Proposed Research Questions

The thesis can be structured around the following research questions:

1. **Architectural Question**: How should a diffusion-based generative masked language model be designed on top of IndicBERT to support flexible-length span infilling in Indic languages, while preserving or improving downstream NLU performance?
2. **Benchmarking Question**: How can existing Indic NLU/NLG benchmarks (IndicXTREME, IndicGLUE, IndicNLG) be transformed into a principled infilling-focused evaluation suite for diffusion language models?
3. **Comparative Question**: How do diffusion-based generative masked LMs compare to conventional masked and autoregressive models on Indic text infilling and generation tasks in terms of quality, diversity, robustness, and cross-lingual transfer?

## Methodology

### 1. Benchmark Construction from IndicXTREME, IndicGLUE, and IndicNLG

#### 1.1 Task Selection

From **IndicXTREME**, tasks that involve sentence-level understanding and classification can be used as contexts for infilling, by masking spans from the input or output sentences.

From **IndicGLUE**, the following tasks are especially suitable:

- Cloze-style QA: mask a key entity or phrase in the passage and ask the model to infill the missing text rather than selecting from predefined options.
- Masked entity prediction: transform the multiple-choice setup into a generative infilling task, evaluating how closely the generated span matches the correct entity.

From **IndicNLG**, tasks like headline generation, summarization, paraphrasing, and question generation can be adapted into infilling problems by masking parts of the source or target text:

- Headline generation: mask important content words in the headline and ask the model to reconstruct the full headline.
- Summarization: mask sentences or clauses from the summary and require infilling to restore a coherent summary.
- Paraphrase generation: create partial paraphrase pairs with masked spans that need infilling.

#### 1.2 Span Masking Strategies

A key design choice is how spans are masked. The benchmark should include multiple masking strategies:

- **Random contiguous spans**: mask uniformly random contiguous sequences of tokens of varying lengths (short phrases, full sentences).
- **Semantic spans**: mask spans corresponding to named entities, noun phrases, or clauses identified via syntactic or semantic heuristics.
- **Long-span masking**: mask entire paragraphs or sections to test long-range infilling capabilities.

Each example is thus turned into a triple: (context, masked span position, original span), enabling evaluation of infilling quality.

#### 1.3 Metrics for Infilling

Evaluation should use both automatic and human-compatible metrics:

- **Token-level metrics**: exact match, F1, and accuracy for short spans like entities or short phrases.
- **Sequence-level metrics**: BLEU, ROUGE, and BERTScore for longer spans to capture fluency and semantic similarity.
- **Diffusion-specific metrics**: trajectory-based metrics that examine how infilling quality improves across diffusion steps, or canvas consistency metrics for partial completions.

Where feasible, small-scale **human evaluation** can be conducted to assess fluency, faithfulness to the context, and cross-lingual appropriateness.

### 2. Model Architecture: Adapting IndicBERT to a Diffusion Generative Masked LM

#### 2.1 Backbone Selection

The backbone will be a **pre-trained Indic encoder**:

- IndicBERT (ALBERT-style, multilingual encoder for 12+ Indic languages).
- Alternatively, newer **IndicBERT-v3** models (270M, 1B, 4B parameter variants) that offer improved performance on Indic benchmarks.

These encoders provide contextual token representations for multilingual Indic text, which form the basis of the diffusion process.

#### 2.2 Diffusion Process Design

Inspired by DiffusionBERT and other discrete diffusion models, the thesis will define a discrete diffusion process over token representations:

- **Forward (noising) process**: progressively corrupt the token sequence (or its embeddings) by replacing tokens with noise tokens, masked tokens, or sampled alternatives, according to a chosen noise schedule.
- **Reverse (denoising) process**: train a neural network (based on the IndicBERT encoder and additional layers) to predict the clean token distribution at each step from the noisy sequence.

Possible design choices include:

- Non-Markovian diffusion that conditions on the original context and the noisy sequence at each step, as in DiffusionBERT.
- Time-step-free decoding where the model learns to map noisy inputs to clean outputs without explicit conditional dependence on time steps, simplifying inference.
- Self-conditioning, where the model’s own previous predictions are fed back as auxiliary inputs to improve stability.

#### 2.3 Output Parameterization and Decoding

The model must output a distribution over the vocabulary for each masked or noisy position at each diffusion step. Decoding strategies may include:

- **Iterative infilling**: run the diffusion process for a fixed number of steps and take the final token predictions as the infilled spans.
- **Adaptive stopping**: use confidence thresholds or entropy measures to stop early when predictions stabilize, reducing inference cost.

Comparisons can be made between full diffusion decoding and simplified approximations (e.g., fewer steps, partially collapsed schedules) to study speed–quality trade-offs.

### 3. Training Procedure

#### 3.1 Pretraining and Fine-Tuning Setup

Training can proceed in stages:

1. **Pretraining the diffusion generative masked LM** on large Indic corpora (e.g., IndicCorp v2) using span corruption objectives similar to generative masked language modeling.
2. **Fine-tuning on benchmark-derived infilling tasks** constructed from IndicXTREME, IndicGLUE, and IndicNLG.

The loss function typically combines:

- Cross-entropy over the true tokens at corrupted positions.
- Additional regularization encouraging consistency across diffusion steps.

The training pipeline will leverage existing AI4Bharat scripts for IndicBERT and adapt them to incorporate the diffusion process, reusing tokenizers, vocabularies, and data preprocessing utilities.

#### 3.2 Baseline Models

To contextualize results, several baselines should be trained or evaluated:

- **Standard IndicBERT / IndicBERT-v2/v3** without diffusion, used as masked LMs and fine-tuned on the same tasks.
- **Autoregressive multilingual models** (e.g., mBERT, XLM-R, generic AR LLMs) adapted for infilling via special tokens or span generation techniques.
- **Existing discrete diffusion models for text** (where feasible) adapted to Indic data to provide a diffusion baseline not tuned specifically for Indic languages.

These baselines allow comparison of diffusion-based architectures with the current state of Indic NLP and generic diffusion text models.

### 4. Experimental Design

#### 4.1 Tasks and Datasets

For each chosen benchmark:

- IndicXTREME: select tasks representing classification, structure prediction, and QA; derive infilling variants by masking spans in inputs/outputs.
- IndicGLUE: focus on cloze QA and masked entity prediction; generate span infilling tasks from these datasets.
- IndicNLG: use headline generation, summarization, paraphrasing, and question generation; create infilling tasks by masking parts of the outputs.

The datasets should be split into training, validation, and test sets in a manner consistent with the original benchmark splits.

#### 4.2 Evaluation Protocol

Experiments should be designed to answer the main research questions:

- Compare diffusion-based and baseline models on:
  - Infilling metrics (exact match, F1, BLEU, ROUGE, BERTScore).
  - NLG metrics on original tasks (headline generation, summarization).
  - NLU accuracy for original tasks (classification, QA).
- Evaluate behavior across:
  - Languages (high-resource vs low-resource Indic languages).
  - Masking strategies (short vs long spans).
  - Code-mixed vs monolingual text.

Where possible, conduct ablations to understand the impact of:

- Noise schedules (linear, cosine, learned).
- Number of diffusion steps.
- Self-conditioning and non-Markovian aspects.

#### 4.3 Compute and Implementation

Experiments will be conducted on a high-memory GPU (such as an RTX 6000 with 96GB VRAM), enabling training of moderately large models (hundreds of millions of parameters) and efficient evaluation across languages and tasks. Existing open-source repositories from AI4Bharat and discrete diffusion model collections can be used as starting points for implementation.

## Expected Results and Contributions

### Expected Results

Although exact performance is unknown, reasonable expectations include:

- **Comparable or improved infilling quality** compared to standard masked LMs on many Indic languages, particularly for medium-length spans where iterative refinement is beneficial.
- **Improved diversity and robustness** in generated spans, as diffusion models tend to explore a richer space of outputs than deterministic masked LM decoding.
- **Competitive NLU performance** on IndicXTREME and IndicGLUE tasks, demonstrating that the diffusion adaptation does not significantly harm downstream understanding tasks and may improve cross-lingual generalization.

Negative or neutral results are also informative, for example if diffusion adds computational overhead without strong gains on certain tasks or languages.

### Contributions

The thesis is expected to make contributions in three areas:

1. **Modeling Contribution**: A diffusion-based generative masked language model for Indic languages, built on top of IndicBERT, with implementation details and open-source code.
2. **Benchmark Contribution**: An infilling-focused evaluation suite constructed from IndicXTREME, IndicGLUE, and IndicNLG, with clearly defined tasks, masking strategies, and metrics tailored to diffusion language models.
3. **Empirical Analysis Contribution**: A systematic comparison of diffusion-based and conventional masked/autoregressive models on Indic infilling and generation tasks, including analysis of language-specific behaviors, robustness, and trade-offs between speed and quality.

Collectively, these contributions address a clear gap in the literature: the lack of diffusion-based generative masked language models and infilling benchmarks for Indic languages.

## Risks and Mitigations

### Potential Risks

- **Implementation complexity**: Discrete diffusion models and non-Markovian architectures can be challenging to implement and debug.
- **Compute demands**: Training large models across many languages and tasks may be compute-intensive.
- **Benchmark adaptation quality**: Poorly designed masking strategies could lead to unnatural infilling tasks that do not reflect real-world usage.

### Mitigation Strategies

- Start from existing implementations of DiffusionBERT and discrete diffusion models to reduce engineering overhead.
- Prioritize a subset of languages and tasks (e.g., focusing first on a few high-resource Indic languages) and extend coverage as time permits.
- Iteratively refine the masking strategies and evaluation metrics with small pilot studies and human inspections to ensure task naturalness.

## Conclusion

This thesis proposal leverages established Indic language resources (IndicBERT, IndicXTREME, IndicGLUE, IndicNLG) and recent advances in diffusion language modeling (DiffusionBERT, discrete diffusion infilling) to design a realistic yet impactful master’s project. By focusing on architectural adaptation, benchmark construction, and empirical comparison, the work can produce contributions that are both academically publishable and practically useful for future research on diffusion language models for low-resource and multilingual settings.

---