# Thesis

## 1. An empirical study of diffusion language models for low-resource language generation and adaptation

### Good research questions

- Do diffusion LMs degrade more slowly than AR LMs as training data shrinks?

- Are they more robust on morphologically rich low-resource languages?

- Does iterative denoising help or hurt when token statistics are sparse?

- Can adaptation methods like adapters or LoRA transfer effectively to diffusion LMs in low-resource settings?

### Objectives

- At least one high-resource language baseline and several low-resource languages.

- Multiple data scales, for example 100%, 10%, 1%, and 0.1% of training data.

- Metrics beyond perplexity, such as exact match, chrF, BLEU, morphological accuracy, or human evaluation if feasible.

- A comparison against an autoregressive baseline and, ideally, a multilingual baseline.

## 2. Adaptability of Diffusion LMs to Low‑Resource Indic Languages

### Same as #1.

A first high‑impact direction is to study how well diffusion LMs adapt to low‑resource Indic languages compared to autoregressive baselines under matched data and parameter budgets. The core question is whether DLMs’ bidirectional context and iterative denoising confer advantages in low‑resource or cross‑lingual transfer settings. This builds on evidence that low‑resource languages remain challenging for LMs and that architectures with stronger conditional signal can sometimes compensate for data scarcity.

### Possible Research Questions
- For a fixed total training token budget, do diffusion LMs achieve better perplexity or downstream task performance (e.g., classification, QA, summarization) than autoregressive LMs on low‑resource Indic languages?

- How does cross‑lingual transfer behave when pretraining on high‑resource languages (e.g., Hindi, English) and adapting to lower‑resource ones (e.g., Assamese, Manipuri), using shared vocabularies and script unification strategies similar to IndicTrans2?

- Do discrete diffusion formulations (over tokens or subwords) behave differently from continuous diffusion over embeddings in the low‑resource regime?

### Methodology Sketch
- Data: Use subsets of AI4Bharat corpora (e.g., Samanantar and BPCC) focusing on 2–4 target languages that span high‑ and low‑resource regimes.

- Models: Implement a small diffusion LM (e.g., Diffusion‑LM‑style continuous model or a discrete diffusion model over tokens), plus matched‑size autoregressive baselines such as GPT‑like or T5‑style decoders.

- Training: Pretrain multilingual models on a mixture of English and Indic languages, then fine‑tune on individual low‑resource languages. Control total training tokens and parameter counts to isolate architectural effects.

- Evaluation: Compare perplexity, translation quality (if using parallel data), and downstream tasks such as sentiment classification or NLI in the target languages. Use public benchmarks where available (e.g., IndicXTREME for some tasks).

### Novelty and Feasibility
The diffusion‑LM literature has focused primarily on English or high‑resource multilingual corpora, with little published evidence about low‑resource adaptation or Indic languages. AI4Bharat’s resources offer a unique testbed to analyze cross‑lingual transfer in DLMs, especially when combined with script unification and multilingual vocabularies as in IndicTrans2. The project is feasible at research scale because it can restrict to modest model sizes and token budgets while still yielding insightful comparative results.

**Aim**: Compare a small diffusion LM against a matched autoregressive LM on a few Indic languages under the same token and parameter budget.

**Core idea***:

- Use AI4Bharat corpora like Samanantar and BPCC (parallel + monolingual data for 22 Indic languages).

- Implement a modest diffusion LM (e.g., Diffusion‑LM style over embeddings or a discrete diffusion model) and an autoregressive baseline with similar size and training tokens.

- Pretrain multilingual models on English + a few higher‑resource Indic languages, then fine‑tune on low‑resource ones (e.g., Assamese, Manipuri).

- Evaluate on perplexity and a couple of downstream tasks (e.g., sentiment, NLI, simple QA) using benchmarks like IndicXTREME where available.

**Why this is good**:

- Clear, publishable research question: “Do DLMs transfer better to low‑resource Indic languages than AR LMs under matched budgets?”

- Compute‑feasible: you can keep models in the 50–300M parameter range and token budgets in tens of millions.

### What's been done here till now?

**Diffutron: A Masked Diffusion Language Model for Turkish (2026)**
- Diffutron is a masked diffusion language model tailored to Turkish, explicitly motivated by the question of whether masked diffusion LMs can handle a morphologically rich, agglutinative language efficiently.

- It starts from a multilingual BERT encoder and uses LoRA‑based continual pre‑training on large Turkish corpora, followed by multi‑stage instruction tuning.

- On Turkish benchmarks (including CETVEL subsets), Diffutron (≈307M params) achieves competitive performance with multi‑billion‑parameter autoregressive baselines, i.e., LLMs that are about seven times larger.

- The paper explicitly frames masked diffusion LMs (MDLMs) as a non‑autoregressive alternative to standard LLMs and shows that, at least for one morphologically rich language, you can get similar quality with much lower parameter counts and a more resource‑efficient training pipeline.

> This is probably the single best “proof‑of‑concept” paper if your thesis angle is “can diffusion LMs adapt well to non‑English / morphologically complex languages compared to standard LLMs?”.

**Massively multilingual diffusion‑style language models (speech / TTS)**

**OmniVoice: Towards Omnilingual Zero‑Shot Text‑to‑Speech with a Diffusion‑LM‑style Architecture (2026)**

OmniVoice is a zero‑shot TTS system for 600+ languages, using a discrete non‑autoregressive architecture explicitly described as “diffusion language model‑style”.

- It directly maps text to multi‑codebook acoustic tokens (no separate semantic‑token stage), trained on a 581k‑hour multilingual dataset built entirely from open‑source resources, with hundreds of low‑resource languages.

- Key design choices: a full‑codebook random masking strategy and initialization from a pre‑trained LLM for better intelligibility.

- Empirically it achieves state‑of‑the‑art or near‑SOTA performance across Chinese, English and diverse multilingual benchmarks while providing the widest language coverage reported to date in TTS.

> While OmniVoice is not a text‑only LM, it is strong evidence that diffusion‑LM‑style discrete models can scale to hundreds of low‑resource languages in a real system, and the paper discusses architectural reasons why this style scales well under data imbalance.

### Diffusion LMs used specifically in low‑resource NLP scenarios

**DiffusionCLS: Diffusion LM for Data Augmentation in Low‑Resource Sentiment Classification (EMNLP 2024)**

- Uses a diffusion language model to generate pseudo‑samples by reconstructing strong label‑related tokens for sentiment classification, focusing explicitly on low‑resource scenarios (domain‑specific, imbalanced, and few‑shot).

- Experiments show that diffusion‑LM‑based augmentation improves performance versus conventional data‑augmentation methods (often based on autoregressive LMs) in multiple low‑resource settings.

**Tiny Recursive Language Diffusion Models (TR‑LDM, 2026, OpenReview)**

- Introduces a very small (≤20M parameters) diffusion language model with a recursive “reasoning/proposal” scheme, explicitly optimized for compute‑constrained and low‑resource settings.

- They emphasize feasibility on a single H100 and show competitive results on algorithmic and reasoning benchmarks relative to larger baselines, arguing that the diffusion‑LM training/inference recipe is well‑suited to low‑compute / low‑resource contexts.


### Surveys and overviews that discuss data efficiency vs PLM/LLM baselines

Several surveys explicitly compare diffusion LMs to autoregressive pre‑trained language models (PLMs) / LLMs, and some touch on low‑resource data efficiency:

**Diffusion models in text generation: a survey (PeerJ Comput. Sci., 2024)**

- Provides a comprehensive overview of diffusion models for text (conditional, unconstrained, multi‑modal) and explicitly compares diffusion models with PLMs/LLMs across multiple dimensions (quality, diversity, controllability, efficiency).

- Notes that diffusion models often have better controllability and global coherence, but suffer from more complex inference; it does not have a deep, dedicated section on low‑resource languages, but it frames where diffusion LMs might be more data‑efficient or easier to fine‑tune.

**A Survey on Diffusion Language Models (2025)**

- Focuses specifically on “Diffusion‑LMs”, covering both continuous and discrete variants and their applications.

- Discusses how diffusion LMs can be non‑autoregressive, parallel, and bidirectional, and summarizes comparative results against autoregressive LLMs on standard benchmarks, though again not specifically cross‑lingual.

**Emergent Mind topic: Diffusion‑LMs**

- This synthesis article summarizes results from multiple papers, explicitly highlighting “data efficiency in low‑resource settings: diffusion‑LMs consistently …” (the rest of the sentence is truncated in the snippet, but the page positions low‑resource data efficiency as a key point).

- Also links to work on accelerated diffusion‑LM inference and masked diffusion LMs.

>These sources give you high‑level comparative arguments (e.g., diffusion LMs are more parallel, offer better global conditioning, may adapt well with less data) that you can connect to low‑resource/multilingual scenarios, even if they do not run many non‑English experiments.


---


## 3. Diffusion‑Based Text Infilling and Editing for Indic Languages

Discrete diffusion models have recently been extended to flexible‑length text infilling, where both token identities and positions are denoised jointly using optimal transport couplings. This enables powerful editing operations such as inserting or rewriting spans of variable length, which is particularly useful for tasks like machine‑assisted translation post‑editing, grammatical error correction, and style transfer.

### Possible Research Questions

- Can discrete diffusion models provide higher‑quality or more controllable infilling and editing for Indic languages than autoregressive infilling baselines at similar scales?

- How well do positional diffusion methods handle scripts with complex morphology or word order variation, such as Dravidian or Indo‑Aryan languages?

- Can a diffusion infilling model be integrated into a translation pipeline (e.g., IndicTrans2) as a post‑editing module for hallucination reduction or style control?

### Methodology Sketch

- Start from an existing discrete diffusion or text infilling implementation (e.g., DDOT‑style models) and adapt it to a multilingual vocabulary including selected Indic languages.

- Train on parallel corpora and monolingual corpora to perform masked‑span reconstruction and style or formality control in target languages.

- Evaluate on tasks such as: grammar correction, sentence simplification, style transfer, and translation post‑editing, using human and automatic metrics.

- Compare against autoregressive span‑infilling baselines (e.g., T5‑style models) and analyze differences in controllability and error profiles.

### Novelty and Feasibility
Flexible‑length discrete diffusion for text is very recent, and there is little to no work applying it to Indic languages or low‑resource settings. A focused thesis could provide the first systematic study of diffusion‑based infilling for Indic scripts, highlighting both strengths and limitations with respect to grammatical complexity and script diversity. Compute costs are moderate: infilling models can be trained at relatively small scales with short sequence lengths and still show interesting behavior.

**Aim**: Build a discrete diffusion infilling model for Indic scripts and compare it to span‑infilling autoregressive models.

**Core idea**:

- Start from recent discrete diffusion infilling work (e.g., optimal‑transport‑based position + token denoising).

- Train a multilingual infilling model on Indic monolingual and parallel data: masked‑span reconstruction, style/formality control, or MT post‑editing.

- Evaluate on:

    - Grammar correction or sentence simplification in 1–2 Indic languages.

    - MT post‑editing on top of IndicTrans2 outputs.

**Why this is good**:

- Very recent area; essentially no work in Indic/low‑resource languages.

- Infilling/editing models can be relatively small with short sequence lengths, so your dual‑GPU setup is sufficient.

### SOTA Text In-filling models

The strongest text-infilling models right now are mostly code-focused families rather than general chat LLMs, with StarCoder2 and newer FIM-tuned decoder-only models leading the open side. For general-purpose infilling, BART/UL2-style denoising models remain important baselines, while FiLM is a newer research direction that improves middle-span generation.

**What is most relevant now**

- StarCoder2-3B/7B/15B: one of the clearest current open-weight families trained with the Fill-in-the-Middle objective, and the 15B model is described as best-in-class for its size.

- Qwen2.5-Coder: a strong practical choice for infilling in code workflows, though the public material I found emphasizes the broader Qwen2.5 family more than infilling specifically.

- FIM-trained causal LLMs: OpenAI’s training recipe showed that decoder-only models can learn infilling without architecture changes by applying FIM-style data transformation.

**Research-grade models**

- FiLM is a 2024 method aimed directly at any-position text generation and reportedly outperforms earlier infilling methods in evaluations.

- UL2 and BART are still foundational because their denoising objectives include text infilling or closely related span corruption, and they are often used as references when discussing infilling capability.

> For the popular open infilling-capable models, the dominant setup is still an autoregressive Transformer trained with a Fill-in-the-Middle (FIM) objective or related denoising objective.

**Architecture used**

- StarCoder2 is a decoder-only Transformer with grouped-query attention and sliding-window attention; it was trained with the FIM objective, so it can generate missing middle spans even though the backbone is still autoregressive.

- FIM-tuned LLMs usually keep the same left-to-right decoder architecture and just change the training data format so the model learns to predict a middle span from prefix + suffix context.

- BART is different: it is a seq2seq Transformer with a bidirectional encoder and autoregressive decoder, and its pretraining explicitly uses text infilling as one of the denoising corruptions.

- UL2 is also Transformer-based, but it is a unified denoising framework with Mixture-of-Denoisers rather than a diffusion architecture.

- FiLM is a newer any-order language model that extends masked language modeling; it is not diffusion either, and it can even be fine-tuned from a left-to-right LM.

> Diffusion-style language models do exist as a separate research direction, but the current widely used infilling models are mainly autoregressive or masked/denoising seq2seq models.

## 4. Evaluation and Robustness of Diffusion LMs for Low‑Resource Text Classification

Diffusion models have begun to be explored for text classification, mainly motivated by robustness and uncertainty estimation rather than pure accuracy gains. Work such as ROIC‑DM proposes diffusion‑based classifiers that add noise to label vectors and denoise conditioned on the input text, achieving stronger robustness against adversarial attacks compared to standard classifiers.

### Possible Research Questions
- Do diffusion‑based classifiers offer robustness advantages for Indic‑language text classification tasks (e.g., sentiment, hate‑speech detection, topic classification) compared to standard fine‑tuned transformers?

- How does performance degrade as labeled data becomes scarce, and does the diffusion classifier retain an advantage in the low‑label regime?

- Can pre‑trained Indic transformers (e.g., IndicBERT, multilingual BERT) serve as feature extractors or “advisors” inside a diffusion classifier to improve performance on low‑resource tasks?

### Methodology Sketch
- Collect or reuse existing labeled datasets for 2–3 classification tasks in a few Indic languages (e.g., Hindi, Bengali, Tamil), possibly leveraging platforms like IndicXTREME.

- Implement a diffusion‑based classifier along the lines of ROIC‑DM, conditioning on features from pre‑trained Indic transformers.

- Compare robustness under adversarial or noisy perturbations and performance under varying label budgets to strong baselines such as fine‑tuned IndicBERT or XLM‑R.

### Novelty and Feasibility
Very few papers apply diffusion models to text classification at all, and those that do focus on English and robustness benchmarks. Extending this line of work to Indic languages and low‑label regimes would be novel, especially if it combines diffusion classifiers with pre‑trained Indic encoders. The compute footprint is modest because classification models are smaller and the number of diffusion steps can be limited.

**Aim**: Use diffusion as a classifier (not generator) for Indic sentiment/toxic‑speech/etc., focusing on robustness and low‑label regimes.

**Core idea**:

- Implement a diffusion‑based classifier similar to ROIC‑DM, where labels are denoised conditioned on textual features.

- Use IndicBERT or multilingual transformers as frozen feature extractors / “advisors” for the diffusion classifier.

- Compare robustness to adversarial/noisy perturbations and performance when labeled data is scarce, versus fine‑tuned IndicBERT/XLM‑R baselines.

**Why this is good**:

- Diffusion for text classification is still niche; adding Indic + low‑label analysis is new.

- Models are small and training is cheap, very friendly to your hardware.

## 5. Multilingual Diffusion Pre‑Training from Indic Resources

A more ambitious but still scoped direction is to pre‑train a moderate‑scale multilingual diffusion LM on a curated subset of Indic corpora and compare its behavior to masked LMs like IndicBERT on generative and understanding tasks. This builds directly on resources such as IndicCorp, Samanantar, and BPCC, which provide large‑scale monolingual and parallel text for Indic languages.

### Possible Research Questions

- For a fixed compute budget, does a multilingual diffusion LM offer better bidirectional generative capabilities (e.g., infilling, paraphrasing) than a similarly sized masked LM trained on the same data?

- How well does the diffusion LM’s pre‑training transfer to downstream tasks like NER, QA, and MT fine‑tuning compared to IndicBERT or XLM‑R baselines?

- Does script unification (as used in IndicTrans2) improve multilingual diffusion pre‑training by encouraging lexical sharing across related languages?

### Methodology Sketch

- Curate a multilingual training corpus from IndicCorp, Samanantar, and BPCC, selecting a manageable subset of languages and documents.

- Implement a masked diffusion LM similar to LLaDA but at a significantly reduced scale (e.g., 100–300M parameters, shorter sequences), using random masking ratios and multilingual vocabularies.

- Train for a fixed token budget and compare to a matched masked LM baseline trained under the same conditions.

- Evaluate on generative tasks (infilling, paraphrasing, translation) and understanding tasks (classification, NER) using standard benchmarks where available.

### Novelty and Feasibility

While LLaDA demonstrates that diffusion LMs can be scaled to 8B parameters and rival LLaMA‑class autoregressive models, it is trained solely on massive general‑purpose corpora. A focused multilingual diffusion LM for Indic languages has not been reported publicly and would provide a valuable reference point for the community, even at moderate scales. The main risk is training time; careful scoping of model size and token budgets is essential to keep the project within a master’s timeframe.

**Aim**: Train a 100–300M parameter masked diffusion LM on a curated subset of Indic corpora and compare it to a masked LM baseline like IndicBERT.

**Core idea**:

- Use IndicCorp/Samanantar/BPCC to build a controlled multilingual corpus over, say, 4–6 Indic languages plus English.

- Implement a scaled‑down LLaDA‑style masked diffusion LM: random masking ratios, no causal mask, Transformer backbone.

- Train under a fixed token budget and compare to a masked LM trained under identical conditions.

- Evaluate on:

    - Generative tasks: infilling, paraphrasing, simple translation.

    - Understanding tasks: classification, NER, QA.

***Why this is good**:

- First “Indic‑centric” multilingual DLM would be novel even at moderate scale.

- More ambitious computationally; you’d likely rely on your server plus some cloud time, but still doable if you keep scope tight.


## Evolutionary Strategies vs GRPO(RL)

### 1. Multi‑task and continual ES post‑training

**What’s known:**

- Abdi et al. show ES can match GRPO on single tasks but causes significantly more forgetting and broader KL drift during sequential training over four tasks (Countdown, Math, Chemistry QA, BoolQ).

- Multi‑task GRPO is being actively studied because naive GRPO training across tasks leads to imbalance and some tasks dominating others.

**Gaps / open questions:**

- No one has designed an ES variant explicitly aimed at multi‑task robustness (e.g., task‑aware noise, per‑task population splits, or parameter‑grouped perturbations) and compared it to multi‑task GRPO.

- We do not know whether combining ES in parameter space with action‑space methods (e.g., SPPO, sequence‑level RL) yields better multi‑task trade‑offs.

**Impactful directions:**

- Design and test multi‑task ES for LLMs: shared base, task‑specific heads/adapters, and explicit mechanisms to limit off‑task drift (e.g., per‑task noise scales, task‑conditioned perturbations, or projection of updates into shared vs task‑specific subspaces).

- Measure not only average accuracy but task imbalance, catastrophic forgetting, KL drift, and cross‑task interference, directly extending Abdi et al. but with new algorithm ideas.

### 2. ES for alignment and safety‑centric objectives

**What’s known:**

- Qiu et al. and Cognizant blogs emphasize that ES is less prone to reward hacking than GRPO on conciseness and other long‑horizon objectives.

- EGGROLL demonstrates ES can optimize non‑differentiable metrics like pass@k and can handle unusual architectures (e.g., integer RNNs) where gradients are awkward.

**Gaps / open questions:**

- Very little systematic work on using ES to enforce safety/alignment constraints, e.g., toxicity, honesty, or refusal behavior, especially under sparse or delayed feedback.

- No clear analysis of whether ES’s “solution distribution” property actually leads to more robust alignment than GRPO in realistic safety benchmarks (e.g., red‑team test suites).

**Impactful directions:**

- Use ES or low‑rank ES to fine‑tune a base model on a safety‑oriented dataset with outcome‑level rewards (e.g., reward = 1 if a safety checker and a human proxy both approve, 0 otherwise).

- Compare to GRPO on:

    - Reward hacking tendencies,

    - Robustness to distribution shift in harmful prompts,

    - Stability across random seeds.

- This would directly test the “ES for less reward hacking” claim in an alignment context, which is not fully explored in current work.

### 3. ES under extreme resource constraints and tiny adapters

**What’s known:**

- EGGROLL shows that low‑rank ES with rank‑1 adapters can train integer‑only RNNs and quantized LLMs, and that large populations can be used on a single GPU by clever batching.

- LoRA work suggests many adaptation problems are intrinsically low‑rank, and some recent work suggests reasoning improvements can be achieved in a very small number of parameters.

**Gaps / open questions:**

- Almost no work on “tiny parameter budget + ES”: e.g., can ES reliably optimize a 10k–100k parameter controller on top of a frozen LLM to improve reasoning under strict compute / memory constraints?

- No clear study of the trade‑off between adapter rank, population size, and sample efficiency for reasoning; EGGROLL shows rank‑1 works, but mostly for specific architectures and setups.

**Impactful directions:**

-Study “micro‑ES adapters”: extremely low‑rank (or small MLP) controllers trained via ES on top of a frozen LLM, targeting reasoning benchmarks.

-Map out which combinations of adapter size, population size, and dataset size still yield meaningful gains over the base model—and where GRPO cannot even run due to optimizer memory.

- This would be directly useful for low‑resource labs and is not yet systematically explored.


### ES for structured and tool‑augmented reasoning

**What’s known:**

- Qiu et al. extend ES to Sudoku and ARC‑AGI puzzles, which are highly structured and closer to algorithmic reasoning, and ES shows strong improvements there.

- There is growing interest in tool‑augmented and program‑like reasoning (e.g., code interpreters, external calculators, search tools), but almost all post‑training is RL‑based.

**Gaps / open questions:**

- Little to no work on ES fine‑tuning where the policy is a tool‑using agent that interacts with external APIs, planners, or search—especially with sparse success signals.

- No systematic comparison of ES vs RL for agentic reasoning (multi‑step tool calls, planning sequences) instead of single‑response reasoning.

**Impactful directions:**

- Treat the entire tool‑using agent as a black‑box policy and optimize its parameters (or a small controller) with ES, using episode‑level success metrics (e.g., success at multi‑step puzzles, API sequences).

- Compare ES vs GRPO in environments where credit assignment is especially hard (e.g., only final answer matters after a long tool‑use trajectory).


### 5. ES for process‑level supervision and metacognition

**What’s known:**

- Most ES work so far uses outcome rewards (correct/incorrect). Process‑level supervision (rewarding intermediate steps, chain‑of‑thought quality, self‑critique) has been mainly explored via RL or supervised learning.

- There is increasing interest in reasoning under token budget and semantic exploration strategies (e.g., SD‑E²) but those are largely gradient‑based.

**Gaps / open questions:**

- No strong exploration of ES on process‑level rewards such as:

- Agreement between multiple sampled chains,

- Self‑consistency,

- Internal uncertainty measures (semantic entropy, density) as in some of Qiu’s related work.

- No comparison of ES vs GRPO when rewards depend on the entire reasoning trace, not just the final answer.

**Impactful directions:**

- Design an ES objective where reward is a combination of final correctness and process metrics (chain diversity, self‑consistency, self‑critique quality).

- Analyze whether ES’s parameter‑space smoothing helps avoid degenerate “shortcut” processes that can sometimes arise in RLHF‑style setups.


### 6. Theory + practice of ES–GRPO hybrids

**What’s known:**

- Qiu et al. connect ES to smoothed versions of the reward landscape; Abdi et al. show ES behaves like a random walk in flat subspaces and GRPO like a sharp, low‑dimensional gradient path.

- There is parallel work on GRPO variants and sequence‑level PPO that try to improve exploration and stability.

**Gaps / open questions:**

- Almost no hybrid methods that mix ES and GRPO in a principled way for LLMs, e.g., ES for exploring coarse directions and GRPO for local refinement, or ES on some parameter groups and GRPO on others.

- No theoretical or empirical characterization of when such hybrids outperform either pure method.

**Impactful directions:**

- Propose and test a two‑phase or alternating ES–GRPO algorithm: ES for a small number of outer steps to find robust high‑reward regions, GRPO for local, low‑norm refinement.

- Or partition parameters: ES on LoRA components, GRPO on base or head layers, to balance exploration and forgetting.

- Tie this to the geometry story: show that the hybrid sits between ES and GRPO in norm, KL, and forgetting.

---

## Starting papers for end to end training-

### 1. Diffusion-LM Improves Controllable Text Generation

One of the foundational papers for diffusion over language and explicitly discusses end-to-end learning of embeddings as part of the training setup.

[Diffusion-LM GitHub](https://github.com/xiangli1999/diffusion-lm)

### 2. Large Language Diffusion Models (LLaDA)

Trains a diffusion model from scratch under a pre-training + supervised fine-tuning paradigm.

### 3. A Survey on Diffusion Language Models

Useful for getting the full picture of training strategies, from pre-training to post-training methods.

