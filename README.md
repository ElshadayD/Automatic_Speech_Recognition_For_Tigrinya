# Amharic-to-Tigrinya Cross-Lingual Transfer Learning for Speech Recognition

## Overview

This repository contains the implementation of seven experiments investigating cross-lingual transfer learning for **low-resource Tigrinya Automatic Speech Recognition (ASR)**.

The study explores whether ASR models adapted to Amharic, a linguistically related Ethio-Semitic language, can be transferred effectively to Tigrinya using limited amounts of labeled target-language speech data.

The experiments investigate the effects of:
- Multilingual pretraining versus language-specific adaptation.
- Default tokenizers versus a custom Tigrinya Byte Pair Encoding (BPE) tokenizer.
- Full model fine-tuning versus Low-Rank Adaptation (LoRA).
- Different amounts of Tigrinya training data (2, 4, 8, 16, and 22 hours).
- Performance comparison against the multilingual Ethio-ASR benchmark.

The main objective is to determine whether transferring knowledge from a related language can provide a more data-efficient alternative to training multilingual ASR models.

## Research Questions

This study addresses four main research questions:

1. **Tokenizer Selection:** How does using a custom Tigrinya tokenizer instead of the default tokenizer affect zero-shot ASR performance?
2. **Fine-Tuning Strategy:** How do full fine-tuning and LoRA compare when adapting Amharic ASR models to Tigrinya?
3. **Benchmark Comparison:** Can Amharic-to-Tigrinya transfer learning outperform the multilingual Ethio-ASR model?
4. **Data Requirements:** How much Tigrinya speech data is required for single-source cross-lingual transfer to achieve competitive performance?

## Experimental Setup

Seven experimental configurations were designed to investigate these questions.

| Experiment | Models | Adaptation | Tokenizer | Tigrinya Data |
|---|---|---|---|---|
| Exp 1 | XLSR-300M, MMS-1B, Whisper Large V3 | Zero-shot | Default | 0 hours |
| Exp 2 | XLSR-300M, MMS-1B, Whisper Large V3 | Zero-shot | Custom Tigrinya BPE | 0 hours |
| Exp 3 | Wav2Vec2-Amharic, Whisper-Small-Amharic | Zero-shot | Default | 0 hours |
| Exp 4 | Wav2Vec2-Amharic, Whisper-Small-Amharic | Zero-shot | Custom Tigrinya BPE | 0 hours |
| Exp 5 | Ethio-ASR | Zero-shot | Native | 0 hours |
| Exp 6 | Wav2Vec2-Amharic, Whisper-Small-Amharic | Full Fine-Tuning | Custom Tigrinya BPE | 2–22 hours |
| Exp 7 | Wav2Vec2-Amharic, Whisper-Small-Amharic | LoRA | Custom Tigrinya BPE | 2–22 hours |

## Models Evaluated

### 1. Multilingual Pretrained Models

These models were evaluated without additional Tigrinya adaptation.

- **Wav2Vec2 XLSR-300M:** `facebook/wav2vec2-xls-r-300m`
- **MMS-1B-All:** Massively Multilingual Speech model.
- **Whisper Large V3:** Large-scale multilingual speech recognition model.

### 2. Amharic-Adapted Models

These models were selected to investigate knowledge transfer from a linguistically related language.

- **Wav2Vec2-Amharic:** `jonatasgrosman/wav2vec2-large-xlsr-53-amharic`
- **Whisper-Small-Amharic:** `amankhazi/whisper-small-am` 

### 3. Benchmark Model

- **Ethio-ASR:** A multilingual Ethiopian ASR model based on Wav2Vec-BERT 2.0, trained on five Ethiopian languages, including Tigrinya.

## Dataset

The experiments use the Tigrinya portion of the Large Vocabulary Read Speech Corpus developed by Abate et al. (2020).

The dataset contains:

- 12,323 transcribed utterances.
- Approximately 24 hours of speech.
- 98 native Tigrinya speakers.
- Audio sampled at 16 kHz.

### Dataset Distribution

The dataset was divided at the speaker level to prevent speaker overlap between training, development, and test sets.

| Split | Speakers | Utterances | Duration |
|---|---:|---:|---:|
| Training | 90 | 11,305 | ~22 hours |
| Development | 4 | 511 | ~1 hour |
| Testing | 4 | 507 | ~1 hour |
| **Total** | **98** | **12,323** | **~24 hours** |

For the data-scaling experiments, five nested training subsets were created:

- 2 hours
- 4 hours
- 8 hours
- 16 hours
- 22 hours

The same held-out test set was used across all experiments to ensure comparable evaluation.

*Note: The dataset is not included in this repository. Access to the original corpus may be required to reproduce the experiments.*

## Methodology

### Custom Tigrinya Tokenizer

A custom Byte Pair Encoding (BPE) tokenizer was trained using the 11,305 Tigrinya training transcripts.

The motivation was to address the limited or inefficient Tigrinya character coverage of the default tokenizers.

The custom tokenizer was used in Experiments 2, 4, 6, and 7.

For models using the custom tokenizer, the output layer was resized to match the new vocabulary.

### Full Fine-Tuning

In Experiment 6, all model parameters were updated during training.

Main training configurations:

- Optimizer: AdamW
- Learning rate: 1e-5
- Batch size: 16
- Maximum epochs: 50
- Early stopping based on development-set performance.

### Low-Rank Adaptation (LoRA)

In Experiment 7, LoRA was used to adapt the pretrained models while keeping their original weights frozen.

LoRA configurations:

- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: Query and Value projections.

For Wav2Vec2, LoRA was applied to the encoder. For Whisper, it was applied to both the encoder and decoder.

This approach updated less than 1% of the model parameters, significantly reducing the number of trainable parameters.

## Evaluation Metrics

All models were evaluated using two standard ASR metrics:

**Word Error Rate (WER)**

Measures the percentage of word-level errors, including substitutions, deletions, and insertions.

**Character Error Rate (CER)**

Measures character-level transcription errors using Levenshtein distance.

Both metrics were calculated on the same 507-utterance held-out test set.

Lower WER and CER indicate better recognition performance.

## Experimental Results

### 1. Zero-Shot Evaluation

The following results compare the default tokenizers with the custom Tigrinya tokenizer.

| Model | Default WER | Default CER | Tigrinya BPE WER | Tigrinya BPE CER |
|---|---:|---:|---:|---:|
| Wav2Vec2 XLSR-300M | 137.3% | 89.2% | 129.0% | 86.2% |
| Whisper Large V3 | 113.9% | 86.4% | 101.1% | 84.3% |
| MMS-1B-All | 97.3% | 75.1% | 91.7% | 72.4% |
| Wav2Vec2-Amharic | 113.5% | 46.2% | 86.4% | 35.6% |
| Whisper-Small-Amharic | 91.0% | 42.3% | **71.4%** | **20.4%** |

**Key observation:** Amharic-adapted models performed substantially better than general multilingual models. The custom Tigrinya tokenizer improved the performance of all evaluated models, with particularly significant improvements for the Amharic-adapted models.

### 2. Fine-Tuning Results

The following table presents the results of full fine-tuning and LoRA using different amounts of Tigrinya training data.

| Data | Wav2Vec2 Full FT | Wav2Vec2 LoRA | Whisper Full FT | Whisper LoRA |
|---|---:|---:|---:|---:|
| 2 hours | 92.3% | 87.4% | 87.1% | 81.0% |
| 4 hours | 84.1% | 75.9% | 82.5% | 76.4% |
| 8 hours | 67.1% | 68.9% | 54.9% | 56.8% |
| 16 hours | 43.1% | 45.5% | 36.9% | 37.7% |
| 22 hours | 28.3% | 30.0% | **26.2%** | 28.2% |

*Values represent WER (%).*

### 3. Comparison with Ethio-ASR

| Model | Tigrinya Training Data | WER | CER |
|---|---:|---:|---:|
| Ethio-ASR | ~182 hours | 31.86% | 16.32% |
| Wav2Vec2-Amharic (Full FT) | 22 hours | 28.3% | 15.0% |
| Whisper-Small-Amharic (Full FT) | 22 hours | **26.2%** | **11.0%** |

**Key finding:** With only 22 hours of Tigrinya speech data, both Amharic-adapted models outperformed the Ethio-ASR benchmark on the held-out read-speech test set.

### 4. Main Findings

The experiments reveal several important findings:

1. **Related-language transfer is effective:** Amharic-adapted models performed substantially better than general multilingual models in zero-shot Tigrinya recognition.
2. **Tokenizer alignment matters:** A custom Tigrinya BPE tokenizer improved recognition performance across all evaluated models.
3. **Very small datasets can hurt performance:** Fine-tuning with 2–4 hours of Tigrinya speech degraded performance compared to zero-shot recognition.
4. **LoRA is more stable with limited data:** LoRA outperformed full fine-tuning at 2–4 hours while updating less than 1% of the model parameters.
5. **More data improves performance:** Both adaptation approaches improved considerably from 8 hours onward.
6. **Single-source transfer can outperform multilingual training:** With 22 hours of Tigrinya speech, both Amharic-adapted models surpassed the Ethio-ASR benchmark.

## Technologies and Tools

The experiments were implemented using:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- PEFT (Parameter-Efficient Fine-Tuning)
- Librosa
- SentencePiece / BPE tokenization
- CUDA and mixed-precision training

### Hardware

Experiments were conducted using a single:

- NVIDIA RTX 4060 Laptop GPU
- 8 GB VRAM

## Limitations and Future Work

This study focuses on a single clean, read-speech Tigrinya corpus. The results may not generalize directly to spontaneous conversational speech, noisy recordings, or different dialects.

Potential future research directions include:

- Evaluating models on noisy and conversational Tigrinya speech.
- Scaling training data beyond 22 hours.
- Fine-tuning MMS-1B-All on Amharic before transferring to Tigrinya.
- Exploring transfer between other Semitic and Ethio-Semitic languages.
- Investigating bidirectional and multidirectional cross-lingual transfer among Ethiopian languages.

## Citation

If you use this work in your research, please cite:

**Asfaw, E. D., Abate, S. T., & Tachbelie, M. Y.** *Amharic to Tigrinya Cross-Lingual Transfer Learning for Speech Recognition.*

## Acknowledgments

We acknowledge the researchers who developed the Tigrinya read-speech corpus and the publicly available multilingual and Amharic-adapted pretrained speech recognition models used in this study.

---

**Research focus:** Low-Resource Speech Recognition | Cross-Lingual Transfer Learning | Ethiopian Languages | Parameter-Efficient Fine-Tuning | Automatic Speech Recognition
