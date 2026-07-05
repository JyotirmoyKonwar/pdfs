# Diffusion-Based Generative Masked Language Models for Indic Languages

## Overview

Masters thesis proposal on adapting a pre-trained Indic encoder into a diffusion-based generative masked language model for text infilling, and building an evaluation benchmark from IndicXTREME, IndicGLUE, and IndicNLG. The work is intentionally scoped to 2-3 Indic languages so that it remains feasible within six months while addressing an underexplored problem in Indic NLP.

The main contribution is an Indic-language adaptation of diffusion-based masked generation together with a structured benchmark for evaluating text infilling in Indic settings.

## Thesis Topic and Aim

### Topic

The thesis topic can be stated as:

> **Diffusion-Based Generative Masked Language Models for Indic Languages: Adapting IndicBERT for Text Infilling and Benchmarking on IndicXTREME, IndicGLUE, and IndicNLG.**

The topic has three main components:

- Adapt a pretrained Indic encoder such as IndicBERT into a diffusion-based generative masked language model.
- Build an infilling-oriented benchmark by transforming selected tasks from IndicXTREME, IndicGLUE, and IndicNLG into span reconstruction problems.
- Evaluate the model on 2-3 Indic languages, such as Hindi, Tamil, and Bengali.

### Aim

The thesis has three core aims:

1. **Modeling aim:** design and implement a diffusion-based generative masked language model for Indic languages by adapting a pretrained encoder such as IndicBERT.
2. **Benchmark aim:** create an evaluation suite for Indic text infilling by converting selected tasks from IndicXTREME, IndicGLUE, and IndicNLG into benchmarkable span reconstruction settings.
3. **Empirical aim:** compare the proposed diffusion-based model with a small set of baseline models to study quality, robustness, and cross-lingual behavior in Indic text infilling.

## Background and Motivation

Diffusion language models have become an important research direction in text generation because they support iterative refinement and naturally suit editing-style tasks such as text completion and infilling. Text infilling is especially significant because it captures realistic language modeling behavior: restoring missing spans, filling incomplete drafts, and reconstructing contextually appropriate content within existing text.

Indic NLP now has strong pretrained encoders and benchmark resources, including IndicBERT-family models, IndicXTREME, IndicGLUE, and IndicNLG. However, these resources have not yet been systematically explored using diffusion-based generative masked language models for Indic text infilling.

This creates a clear research opportunity: to study whether diffusion-based masked generation can provide effective infilling for linguistically diverse Indic languages and to establish a benchmark that future work can reuse.

## Research Questions

The thesis can be organized around the following research questions:

1. How can a pretrained Indic encoder such as IndicBERT be adapted into a diffusion-based generative masked language model for Indic text infilling?
2. How can selected tasks from IndicXTREME, IndicGLUE, and IndicNLG be transformed into a principled evaluation suite for Indic text infilling?
3. How does the proposed diffusion-based model compare with standard masked and autoregressive baselines on quality, robustness, and cross-lingual transfer?
4. How does model performance vary across 2-3 Indic languages and across different masking strategies?

## Methodology

### 1. Scope Definition

The study will focus on 2-3 Indic languages, preferably chosen to provide linguistic diversity while remaining tractable for a six-month thesis. A practical set would be Hindi, Bengali and another.

The thesis will focus on monolingual text infilling. This keeps the data preparation, training, and evaluation pipeline manageable and helps maintain a clear experimental story.

### 2. Benchmark Construction

The benchmark will be created by deriving infilling tasks from selected portions of IndicXTREME, IndicGLUE, and IndicNLG. The goal is to convert existing NLU and NLG resources into a unified evaluation setting for span reconstruction.

Suitable task types include:

- Cloze-style and masked prediction tasks from IndicGLUE.
- Headline generation, summarization, paraphrasing, and question generation from IndicNLG, adapted into span reconstruction settings.
- Selected context-sensitive tasks from IndicXTREME where masking yields meaningful completion problems.

### 3. Masking Strategy

The benchmark should include multiple masking strategies so that the evaluation is not restricted to one easy pattern. These may include random contiguous spans, linguistically meaningful spans such as named entities or phrases, and medium-length spans that test contextual reasoning.

Each example will be converted into a context-plus-masked-span format, enabling direct evaluation of infilling quality.

### 4. Model Architecture

The backbone will be a pretrained Indic encoder such as IndicBERT or a recent IndicBERT-family variant. The model will be adapted into a diffusion-based generative masked language model using a discrete noising and denoising process over token sequences.

The overall design will follow the general logic of diffusion-style masked generation:

- A forward process that gradually corrupts tokens according to a noise schedule.
- A reverse model that predicts the clean token distribution from noisy inputs.
- Iterative denoising for span reconstruction during inference.

### 5. Training Procedure

Training will proceed in two stages:

1. **Adaptation stage:** task-adaptive or continued training of the pretrained Indic encoder under a diffusion-style corruption objective.
2. **Fine-tuning stage:** fine-tuning on the benchmark-derived Indic infilling tasks.

This approach keeps the work practical by adapting an existing model rather than training a new language model from scratch.

### 6. Baselines

The empirical comparison should use a small, realistic set of baselines:

- A standard IndicBERT-style masked language model baseline.
- One autoregressive or encoder-decoder multilingual baseline adapted for infilling.

This is sufficient to establish whether the diffusion-based approach offers meaningful advantages in Indic text infilling.

### 7. Evaluation

Evaluation will be carried out across languages, masking strategies, and span lengths.[4] Metrics should include Exact Match and token-level F1 for short spans, and BLEU, ROUGE, and BERTScore for longer spans.

Where feasible, a small human evaluation can be added to assess fluency, contextual fit, and adequacy of the infilled text.

## Expected Results and Contributions

The expected outcome is a diffusion-based Indic text infilling model that performs competitively with standard masked baselines and provides useful insight into how diffusion-style generation behaves across Indic languages. Even where gains are modest, the thesis would still contribute a benchmark and a structured empirical analysis for an underexplored research area.

The thesis is expected to contribute in three main ways:

1. **Model contribution:** an Indic adaptation of a diffusion-based generative masked language model.
2. **Benchmark contribution:** a reusable infilling benchmark derived from IndicXTREME, IndicGLUE, and IndicNLG.
3. **Empirical contribution:** a systematic evaluation of diffusion-based versus standard baselines on selected Indic languages.

## Feasibility

The project is feasible within six months if the scope is kept narrow and the work is staged properly. Restricting the thesis to 2-3 languages, selected benchmark tasks, and a small number of baselines makes the implementation and evaluation manageable.

A practical timeline would include literature review and benchmark design in the first two months, model adaptation and initial experiments in the middle phase, and evaluation, analysis, and writing in the final months.

## Conclusion

This thesis is a focused and feasible study of diffusion-based generative masked language models for Indic languages, centered on text infilling and benchmark construction using existing Indic NLP resources.[3][4] It offers a strong master’s-level contribution by combining a current modeling paradigm with a meaningful multilingual evaluation gap in Indic NLP.