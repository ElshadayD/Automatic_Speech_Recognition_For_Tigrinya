# Amharic-to-Tigrinya Cross-Lingual Transfer Learning for Speech Recognition

## Overview

This project investigates cross-lingual transfer learning for low-resource Tigrinya Automatic Speech Recognition (ASR). It explores whether ASR models adapted to Amharic can be transferred effectively to Tigrinya using limited amounts of labeled speech data.

Seven experiments were conducted to investigate:
- Multilingual versus Amharic-adapted ASR models.
- Default versus custom Tigrinya BPE tokenizers.
- Full fine-tuning versus Low-Rank Adaptation (LoRA).
- The effect of different training data sizes (2, 4, 8, 16, and 22 hours).
- Comparison with the Ethio-ASR benchmark.

## Models

- Wav2Vec2 XLSR-300M
- MMS-1B-All
- Whisper Large V3
- Wav2Vec2-Amharic
- Whisper-Small-Amharic
- Ethio-ASR

## Dataset

The experiments use the Tigrinya portion of the Large Vocabulary Read Speech Corpus.

| Split | Utterances | Duration |
|---|---:|---:|
| Train | 11,305 | ~22 hours |
| Dev | 511 | ~1 hour |
| Test | 507 | ~1 hour |
| **Total** | **12,323** | **~24 hours** |

Audio was sampled at 16 kHz, with speaker-independent train, development, and test splits.

## Experiments

| Experiment | Description |
|---|---|
| 1 | Multilingual models with default tokenizers |
| 2 | Multilingual models with custom Tigrinya BPE |
| 3 | Amharic-adapted models with default tokenizers |
| 4 | Amharic-adapted models with custom Tigrinya BPE |
| 5 | Ethio-ASR baseline evaluation |
| 6 | Full fine-tuning with 2–22 hours of Tigrinya data |
| 7 | LoRA fine-tuning with 2–22 hours of Tigrinya data |

## Key Results

| Model | Training Data | WER | CER |
|---|---:|---:|---:|
| Ethio-ASR | ~182 hours | 31.86% | 16.32% |
| Wav2Vec2-Amharic (Full FT) | 22 hours | 28.3% | 15.0% |
| Whisper-Small-Amharic (Full FT) | 22 hours | **26.2%** | **11.0%** |

**Main findings:**
- Amharic-adapted models performed better than general multilingual models.
- Custom Tigrinya tokenization improved recognition performance.
- LoRA was more stable with very limited training data (2–4 hours).
- With 22 hours of Tigrinya data, both Amharic-adapted models outperformed the Ethio-ASR benchmark.

## Technologies

Python · PyTorch · Hugging Face Transformers · Datasets · Evaluate · PEFT · Librosa · CUDA

## Hardware

NVIDIA RTX 4060 Laptop GPU (8 GB VRAM).

## Citation

Asfaw, E. D., Abate, S. T., & Tachbelie, M. Y. *Amharic to Tigrinya Cross-Lingual Transfer Learning for Speech Recognition.*

