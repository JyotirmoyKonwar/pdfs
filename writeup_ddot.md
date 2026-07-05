# Indic Diffusion Infilling with DDOT-Style and DreamOn-Style Architectures

## Overview

This report presents an updated, in-depth thesis outline for a master’s project on diffusion language models for Indic languages, focusing on **flexible-length text infilling** using ideas from DDOT (Discrete Diffusion with Optimal Transport Position Coupling) and DreamOn (diffusion LMs for code infilling) adapted on top of Indic encoders such as IndicBERT. The work combines architectural adaptation, benchmark construction from existing Indic datasets, and systematic evaluation of diffusion-based infilling against masked and autoregressive baselines.[1][2][3][4]

## Thesis Topic and Aim

### Topic

> **Flexible-Length Indic Text Infilling with Diffusion Language Models: Adapting DDOT-Style and DreamOn-Style Architectures on IndicBERT and Evaluating on Benchmarks Derived from IndicXTREME, IndicGLUE, and IndicNLG.**

This topic emphasizes:

- **Flexible-length, flexible-position text infilling** using discrete diffusion, inspired by DDOT’s joint token–position diffusion and OT-based position coupling.[3][5]
- **Variable-length span control mechanisms** adapted from DreamOn’s expand/delete and remasking strategies, originally designed for code infilling beyond fixed canvases.[4][6]
- **Application to Indic languages** via backbone models like IndicBERT and its variants, which provide strong multilingual representations for Indian languages.[7][2][1]
- **Evaluation on constructed infilling benchmarks** that extend existing NLU/NLG suites (IndicXTREME, IndicGLUE, IndicNLG) to span-infilling tasks.

### Aim

The main aims of the thesis are:

1. **Architectural Adaptation**: Design and implement a diffusion-based infilling architecture for Indic languages that combines DDOT-style joint token–position diffusion with DreamOn-style variable-length span control on top of IndicBERT (or similar Indic encoders).[1][3][4]
2. **Benchmark Construction**: Build a principled, multi-task infilling benchmark for Indic languages by transforming existing datasets from IndicXTREME, IndicGLUE, and IndicNLG into span-corruption and infilling tasks at multiple granularities.[8][9][10]
3. **Empirical Comparison and Analysis**: Compare the proposed diffusion-infilling models against conventional masked LMs and autoregressive baselines on infilling quality, diversity, robustness, and cross-lingual performance across Indic languages, including code-mixed and low-resource scenarios.[11][12][13]

## Background and Related Work

### Diffusion Language Models for Discrete Text

Diffusion language models (DLMs) generalize diffusion processes to discrete token spaces, generating text by iteratively denoising noisy token sequences or latent representations instead of predicting tokens sequentially. They offer advantages such as parallel generation, bi-directional context usage, and natural support for editing and infilling, but must contend with discrete noise processes and sampling approximations.[12][14][11]

Notable DLM frameworks include GENIE (a diffusion pre-training framework for text generation), Diffusion-LM, and energy-based diffusion language models that incorporate sequence-level energy corrections to improve approximations and reduce performance degradation when reducing sampling steps. Surveys on diffusion text generation highlight both the promise of DLMs and open challenges in efficiency, variable-length generation, and evaluation.[14][15][16][11][12]

### DDOT: Flexible-Length Text Infilling via Joint Token–Position Diffusion

DDOT (Discrete Diffusion with Optimal Transport Position Coupling) introduces a discrete diffusion model that jointly denoises token values and token positions to support flexible-length and flexible-position text infilling. Unlike previous diffusion text models that assume fixed token positions and fixed canvas sizes, DDOT models token positions as continuous variables and uses sample-level Optimal Transport (OT) coupling to preserve relative token ordering while allowing dynamic adjustment of infilled segment length and position.[5][3]

The DDOT framework:

- Treats positions as continuous vectors \(Z_t \in [-1, 1]^L\) evolving under an ODE, with a learnable velocity field that moves noisy positions towards ground truth positions.[3]
- Uses OT coupling between initial and final positions to ensure non-crossing paths within prompt and response subsets, preserving order and enabling flexible span placement.[5][3]
- Employs a masked discrete diffusion process for token values (e.g., SEDD-style score entropy loss) and combines token and position losses in a single training objective.[3]

Experiments on English infilling benchmarks (One-Billion-Word, Yelp, CodeParrot) show that DDOT significantly outperforms naive diffusion baselines and achieves competitive performance with state-of-the-art non-autoregressive models, especially on block masking and long sequences.[17][5][3]

### DreamOn: Variable-Length Infilling Beyond Fixed Canvas

DreamOn is a diffusion language model designed for code infilling, capable of expanding and contracting masked regions during inference to support variable-length infilling beyond a fixed-size canvas. It uses entropy-based remasking, special expand/delete tokens, and dynamic recalculation of attention masks to allow flexible generation of code sequences.[6][18][4]

While DreamOn targets code, its mechanisms for variable-length infilling and any-order editing (infilling at arbitrary positions) are conceptually relevant to natural-language infilling, especially for tasks where the desired span length is not known a priori. The thesis proposes to adapt these ideas for Indic text.[19][4]

### Indic Language Benchmarks and Encoders

AI4Bharat and collaborators provide foundational resources for Indic NLP:

- **IndicBERT and IndicBERT-v2/v3**: Multilingual ALBERT/BERT-style encoders trained on large Indic corpora (IndicCorp v2) covering major Indian languages and Indian English; models span 270M to multi-billion parameter scales and are evaluated on benchmarks like IndicXTREME.[2][20][7][1]
- **IndicXTREME**: A multi-task benchmark with nine diverse NLU tasks across ~18–20 languages, designed to test zero-shot performance of multilingual models.[13][21][8]
- **IndicGLUE**: A general NLU benchmark comprising classification, headline prediction, cloze-style QA, and masked entity prediction tasks in multiple Indian languages.[10][22][23]
- **IndicNLG Suite**: A multilingual NLG benchmark with datasets for biography generation, summarization, headline generation, paraphrasing, and question generation across 11 Indic languages, totaling around 8M examples.[9][24][25]

These resources are ideal for evaluating new generative architectures for Indic languages and for building derived infilling benchmarks.

## Research Questions

The thesis can be structured around the following research questions:

1. **Architecture RQ**: How can DDOT-style joint token–position diffusion and DreamOn-style variable-length infilling mechanisms be adapted on top of an Indic encoder (e.g., IndicBERT) to support flexible-length text infilling in Indic languages?
2. **Benchmark RQ**: How can existing Indic benchmarks (IndicXTREME, IndicGLUE, IndicNLG) be systematically transformed into a family of span-infilling tasks that capture real-world infilling scenarios for Indic text (short spans, long spans, code-mixed spans)?[8][9][10]
3. **Comparative RQ**: How do diffusion-based infilling models compare to masked and autoregressive baselines on Indic infilling tasks in terms of quality, diversity, robustness to noise, and performance across languages and resource levels?[11][12][13]

## Methodology

### 1. Benchmark Construction from IndicXTREME, IndicGLUE, and IndicNLG

#### 1.1 Task Selection and Transformation

The thesis will construct an **IndicInfilling** benchmark by transforming existing tasks:

- **IndicXTREME**: Select tasks involving sentence-level NLU and QA. For classification tasks, mask informative spans (e.g., key phrases) in input sequences and require infilling of those spans given the remaining context. For QA tasks, mask answer spans in passages and ask the model to infill them.[21][13][8]
- **IndicGLUE**: Focus on cloze-style QA and masked entity prediction; convert multiple-choice setups into generative infilling tasks by removing options and evaluating whether the model can infill the correct entity or phrase.[22][23][10]
- **IndicNLG**: Use headline generation, summarization, paraphrasing, and question generation tasks as sources for infilling by masking parts of the target text (e.g., key content words, clauses, or entire sentences) and requiring reconstruction.[26][25][9]

For each dataset, the transformation yields triples \((\text{context}, \text{mask pattern}, \text{ground-truth span})\), which define infilling instances.

#### 1.2 Masking Regimes

The benchmark will define several masking regimes inspired by DDOT and NLG practices:[9][3]

- **Random short-span masking**: Mask random contiguous sequences of 1–5 tokens to test local infilling.
- **Block masking**: Mask longer contiguous spans (phrases, clauses, sentences) to test coherent reconstruction over larger spans, similar to DDOT’s block experiments on long sequences.[3]
- **Semantic masking**: Mask spans corresponding to named entities, noun phrases, or syntactic constituents identified via simple heuristics or taggers.
- **Code-mixed masking**: For code-mixed or multilingual data (e.g., Hindi–English, Tamil–English), mask mixed-language segments to test cross-lingual and script-robust infilling.

Each original benchmark dataset will be augmented with multiple masking patterns, creating a diverse suite of infilling tasks.

#### 1.3 Evaluation Metrics

The evaluation will combine standard NLG metrics with DDOT-style infilling metrics:[10][9][3]

- **Token-level metrics**: exact match, accuracy, and F1 for short spans (e.g., masked entities, short answers).
- **Sequence-level metrics**: BLEU, ROUGE, METEOR, and BERTScore for longer spans, capturing fluency and semantic similarity.[25][9]
- **Success Rate (SR)**: proportion of infilling instances where the model correctly reconstructs spans within an acceptable tolerance, as used in DDOT.[3]
- **Diffusion-specific trajectory metrics**: optional measures of how infilling quality changes across diffusion steps, such as improvement in BLEU/ROUGE over time or changes in entropy.

Where feasible, small-scale human evaluation can assess fluency, faithfulness to the context, and appropriateness in different languages and scripts.

### 2. Model Architecture: Adapting DDOT and DreamOn on IndicBERT

#### 2.1 Backbone: Indic Encoders

The backbone will be a pre-trained Indic encoder:

- **IndicBERT / IndicBERT-v2/v3**: selected based on parameter scale and available resources; these encoders provide contextual representations for multiple Indic languages and are already tuned on Indic benchmarks.[20][7][2][1]

Using a pre-trained encoder reduces the need to train from scratch and ensures strong contextual understanding across languages.

#### 2.2 DDOT-Style Joint Token–Position Diffusion

The thesis will implement a DDOT-style discrete diffusion process on top of IndicBERT:[5][3]

- **Token diffusion**: A masked discrete diffusion process, similar to SEDD, where token values are progressively corrupted and a score function is learned to denoise them via a score-entropy loss.[3]
- **Position diffusion**: A continuous position diffusion process, modelling positions as continuous vectors and learning a velocity field that transforms noisy positions towards structured ground-truth positions.[5][3]
- **OT coupling**: A sample-level Optimal Transport coupling between initial and final positions, computed separately for prompt and infilled response subsets, preserving relative ordering while allowing flexibility in infilled span placement.[5][3]

Architecturally, this may involve adding:

- A **token score head** on top of IndicBERT, predicting token distributions at each diffusion step.
- A **position velocity head**, predicting position velocities for continuous position variables.
- **Type embeddings** to distinguish prompt tokens from response/infilled tokens.

#### 2.3 DreamOn-Style Variable-Length Infilling Mechanisms

In addition to DDOT’s position diffusion, the thesis will adapt selected ideas from DreamOn for variable-length infilling:[4][6]

- **Expand/delete tokens**: Introduce special control tokens that allow the model to expand or shrink masked segments during diffusion, controlling infilled span length.
- **Entropy-based remasking**: Use entropy or uncertainty measures to decide which positions to re-mask and refine across steps, improving difficult spans.[18]
- **Dynamic attention masks**: Recalculate attention masks at each step to reflect updated prompt and infilled regions, echoing DreamOn’s handling of code canvases.[27][4]

These mechanisms can be combined or compared with DDOT’s OT-based position control to explore different approaches to variable-length infilling in Indic text.

#### 2.4 Decoding Strategies

Two families of decoding strategies will be explored:

- **DDOT-style decoding**: Jointly simulate token and position diffusion over a fixed number of steps, then read off infilled spans from the final token distributions and positions.[3]
- **DreamOn-style decoders**: Use expand/delete operations and remasking to iteratively refine the masked region, possibly with fewer explicit position variables.

Comparisons between these strategies will be part of the experimental analysis.

### 3. Training Procedure

#### 3.1 Pretraining on Large Indic Corpora

The diffusion-infilling models can be first pre-trained on large Indic monolingual corpora (e.g., IndicCorp v2) using span-corruption objectives similar to generative masked LM training.[28][13]

- Random and block span corruption across multiple languages to encourage strong infilling capabilities.
- Possibly language-specific or script-specific sampling to balance high- and low-resource languages.

#### 3.2 Fine-Tuning on IndicInfilling Benchmark

After pretraining, models will be fine-tuned on the infilling tasks derived from IndicXTREME, IndicGLUE, and IndicNLG.[8][9][10]

Fine-tuning objectives include:

- Cross-entropy loss over true tokens in masked spans.
- Position velocity loss for DDOT-style position diffusion.
- Optional auxiliary losses (e.g., classification or QA losses) to jointly optimize NLU performance.

#### 3.3 Baseline Models

Baselines for comparison:

- **Standard masked LMs**: IndicBERT/IndicBERT-v2/v3 used with conventional masked LM decoding, fine-tuned on the same tasks.[2][20][1]
- **Autoregressive models**: Multilingual AR models (e.g., XLM-R or mT5) adapted for infilling via templates or span generation.[13][10]
- **Existing discrete diffusion baselines**: Where feasible, reimplement SEDD-style diffusion without DDOT’s position coupling for Indic data to gauge the added value of OT coupling.[29][30][3]

### 4. Experimental Design

#### 4.1 Task Coverage

Experiments will cover:

- Multiple Indic languages (e.g., Hindi, Tamil, Telugu, Bengali), balancing resource levels based on dataset sizes.[9][8]
- Different masking regimes (random short spans, block spans, semantic spans, code-mixed spans).
- NLU and NLG tasks transformed into infilling tasks, plus original tasks for downstream performance checks.

#### 4.2 Evaluation Protocol

For each model and task:

- Compute infilling metrics (token-level and sequence-level) on test sets.
- Evaluate NLU accuracy on original tasks (IndicXTREME, IndicGLUE) to ensure that diffusion adaptation does not degrade understanding performance.[10][8]
- Evaluate NLG metrics on original IndicNLG tasks (summarization, headline generation) for generative quality.[25][9]

Ablation studies will examine:

- Impact of OT coupling vs simple position noise on infilling quality.[3]
- Impact of DreamOn-style expand/delete and remasking vs pure DDOT-style position diffusion.[18][4]
- Effect of diffusion step count on quality and inference speed.[15][3]

#### 4.3 Compute Considerations

Experiments will be run on a high-memory GPU (e.g., RTX 6000 96GB VRAM), allowing training of moderately large models (hundreds of millions of parameters) and large-batch evaluation across languages. Computational budgets will be managed by prioritizing key languages and tasks and sharing the backbone across variants.[2][13]

## Expected Results and Contributions

### Expected Results

While actual results are uncertain, plausible expectations include:

- **Improved infilling quality for medium and long spans** compared to standard masked LMs, thanks to joint token–position diffusion and variable-length control mechanisms.[4][3]
- **Competitive or improved diversity and robustness** in generated spans relative to autoregressive baselines, especially on noisy or code-mixed inputs.[12][11]
- **Maintained or modestly improved NLU performance** on benchmarks like IndicXTREME and IndicGLUE, demonstrating that diffusion adaptation does not significantly harm representation quality for downstream tasks.[8][10]

Negative or mixed results (e.g., higher compute cost without clear gains on some tasks) will also be analyzed, contributing insights into where diffusion infilling is most beneficial.

### Contributions

The thesis aims to contribute in three primary ways:

1. **Architectural Contribution**: A diffusion-based infilling architecture for Indic languages that combines DDOT-style joint token–position diffusion and DreamOn-style variable-length control on top of IndicBERT, with code and design details suitable for reuse.[1][4][3]
2. **Benchmark Contribution**: An IndicInfilling benchmark suite derived from IndicXTREME, IndicGLUE, and IndicNLG, with clearly specified masking regimes, tasks, and evaluation metrics tailored to flexible infilling.[9][10][8]
3. **Empirical and Analytical Contribution**: A comparative study of diffusion-based, masked, and autoregressive models on Indic infilling tasks, including language-wise performance, robustness analysis, speed–quality trade-offs, and ablations of DDOT/DreamOn-style components.[11][12][4][3]

Collectively, these contributions deepen understanding of diffusion language models in multilingual, low-resource, and code-mixed contexts, and provide tools and baselines for future work on Indic generative modeling.

## Risks and Mitigations

### Risks

- **Implementation complexity**: DDOT and DreamOn architectures involve non-trivial position diffusion, OT coupling, and variable-length control mechanisms.
- **Compute intensity**: Training and evaluating large diffusion models across multiple languages and tasks may be compute-intensive.
- **Benchmark construction quality**: Poorly designed masking strategies could produce infilling tasks that are either trivial or unnatural.

### Mitigations

- Start from existing DDOT and DreamOn descriptions and any available code, reusing their training recipes and architectural modules where possible.[31][4][3]
- Begin with a subset of languages and tasks (e.g., 3–4 high-resource languages and a handful of tasks) and extend coverage as time allows.[8][9]
- Iteratively refine masking strategies based on pilot studies and manual inspection, ensuring that infilling tasks reflect realistic text editing and completion scenarios.

## Conclusion

The updated thesis outline integrates DDOT and DreamOn—state-of-the-art diffusion infilling frameworks—with Indic encoders and benchmarks, defining a realistic but ambitious master’s project. By focusing on architectural adaptation, benchmark construction, and systematic evaluation, the work can produce contributions of both scientific and practical relevance to diffusion language models and Indic NLP.[1][4][8][3]