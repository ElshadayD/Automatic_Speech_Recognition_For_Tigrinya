# Generated from: Experiment 1 (pretrained models not adapted for either amharic or tigrigna+ tig tokenizer).ipynb
# Converted at: 2026-11-12T10:28:42.012Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## Experimenting on multilingual models trained on neither amharic nor tigrigna for tigrigna ASR and using the tigrigna tokenizer


# ## facebook/wav2vec2-xls-r-300m


import torch
import librosa
from datasets import Dataset
import pandas as pd
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import evaluate
from tqdm import tqdm
import gc
import os
import time
import psutil

# ---------------- CONFIG ----------------
DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_ID = "facebook/wav2vec2-xls-r-300m"
TOKENIZER_PATH = "./tigrinya_tokenizer"   # path to your custom tokenizer
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- TIMER START ----------------
start_time = time.time()

# ---------------- NORMALIZE ----------------
def normalize(text):
    text = text.replace("፡", " ")
    text = " ".join(text.split())
    return text.strip()

# ---------------- LOAD DATA ----------------
def load_split(split):
    text_path = f"{DATA_ROOT}/text/{split}/text"
    wavscp_path = f"{DATA_ROOT}/text/{split}/wav.scp"

    text_dict = {}

    with open(text_path, encoding="utf-8") as f:
        for line in f:
            utt, sentence = line.strip().split(" ", 1)
            text_dict[utt] = normalize(sentence)

    rows = []

    with open(wavscp_path, encoding="utf-8") as f:
        for line in f:
            utt, wav = line.strip().split(" ", 1)

            wav = wav.replace(
                "/share/documents/solomon/TigrignaASR",
                DATA_ROOT
            )

            if utt in text_dict and os.path.exists(wav):
                rows.append({
                    "audio_path": wav,
                    "sentence": text_dict[utt]
                })

    return Dataset.from_pandas(pd.DataFrame(rows))

# ---------------- LOAD DATASETS ----------------
print("Loading dataset...")
train_ds = load_split("train")
test_ds = load_split("eval")

print("Train samples:", len(train_ds))
print("eval samples:", len(test_ds))

# ---------------- LOAD PROCESSOR + CUSTOM TOKENIZER ----------------
print("Loading model...")

# Load base processor (feature extractor)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Replace tokenizer with your custom Tigrinya tokenizer
custom_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_PATH)
processor.tokenizer = custom_tokenizer

# Load model
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Resize output layer to match tokenizer vocab size
model.resize_token_embeddings(len(processor.tokenizer))

model.to(DEVICE)
model.eval()

print("Model loaded.")

# ---------------- METRICS ----------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ---------------- STORAGE ----------------
predictions = []
references = []
sample_wers = []
sample_cers = []

print("Starting low-memory evaluation...\n")

# ---------------- EVALUATION LOOP ----------------
for sample in tqdm(test_ds, desc="Transcribing audio"):

    audio_path = sample["audio_path"]

    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Normalize audio
        audio = audio / (abs(audio).max() + 1e-9)

        # Processor
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(DEVICE)

        # Inference
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(input_values).logits

        # Greedy decode
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        transcription = normalize(transcription)
        reference = sample["sentence"]

        predictions.append(transcription)
        references.append(reference)

        # Per-sample metrics
        sample_wer = wer_metric.compute(
            predictions=[transcription],
            references=[reference]
        )

        sample_cer = cer_metric.compute(
            predictions=[transcription],
            references=[reference]
        )

        sample_wers.append(sample_wer)
        sample_cers.append(sample_cer)

        # Cleanup
        del input_values, logits, predicted_ids
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        continue

# ---------------- SHOW SAMPLE PREDICTIONS ----------------
num_examples = min(10, len(predictions))

print(f"\n--- Showing {num_examples} Sample Predictions vs References ---\n")

for i in range(num_examples):
    print(f"Sample {i+1} | WER: {sample_wers[i]:.2f} | CER: {sample_cers[i]:.2f}")
    print(f"Reference : {references[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 50)

# ---------------- COMPUTE FINAL METRICS ----------------
print("\nComputing metrics...\n")

final_wer = wer_metric.compute(
    predictions=predictions,
    references=references
)

final_cer = cer_metric.compute(
    predictions=predictions,
    references=references
)

# ---------------- SAVE PREDICTIONS ----------------
df = pd.DataFrame({
    "reference": references,
    "prediction": predictions
})
df.to_csv("eval_predictions.csv", index=False)

# ---------------- TIMER END ----------------
end_time = time.time()
total_time = end_time - start_time

# ---------------- MEMORY ----------------
peak_gpu_mem = (
    torch.cuda.max_memory_allocated() / (1024 ** 3)
    if torch.cuda.is_available()
    else 0
)

process = psutil.Process(os.getpid())
cpu_mem = process.memory_info().rss / (1024 ** 3)

# ---------------- FINAL RESULTS ----------------
print("==============================")
print("FINAL TEST RESULTS")
print("==============================")
print("Dataset : eval")
print(f"WER     : {final_wer:.4f}")
print(f"CER     : {final_cer:.4f}")

print("\nSaved predictions → eval_predictions.csv")

print("\n⏱ Total Time (seconds):", round(total_time, 2))
print("⏱ Total Time (minutes):", round(total_time / 60, 2))

print("\n💾 Peak GPU Memory (GB):", round(peak_gpu_mem, 2))
print("💾 CPU RAM Usage (GB):", round(cpu_mem, 2))

print("\nDONE ✅")

# ## mms-1b-all


import torch
import librosa
from datasets import Dataset
import pandas as pd
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import evaluate
from tqdm import tqdm
import gc
import os
import time
import psutil

# ---------------- CONFIG ----------------
DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_ID = "facebook/mms-1b-all"
TOKENIZER_PATH = "./tigrinya_tokenizer"   # Path to your custom tokenizer
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- TIMER START ----------------
start_time = time.time()

# ---------------- NORMALIZE ----------------
def normalize(text):
    text = text.replace("፡", " ")
    text = " ".join(text.split())
    return text.strip()

# ---------------- LOAD DATA ----------------
def load_split(split):
    text_path = f"{DATA_ROOT}/text/{split}/text"
    wavscp_path = f"{DATA_ROOT}/text/{split}/wav.scp"

    text_dict = {}

    with open(text_path, encoding="utf-8") as f:
        for line in f:
            utt, sentence = line.strip().split(" ", 1)
            text_dict[utt] = normalize(sentence)

    rows = []

    with open(wavscp_path, encoding="utf-8") as f:
        for line in f:
            utt, wav = line.strip().split(" ", 1)

            wav = wav.replace(
                "/share/documents/solomon/TigrignaASR",
                DATA_ROOT
            )

            if utt in text_dict and os.path.exists(wav):
                rows.append({
                    "audio_path": wav,
                    "sentence": text_dict[utt]
                })

    return Dataset.from_pandas(pd.DataFrame(rows))

# ---------------- LOAD DATASETS ----------------
print("Loading dataset...")
train_ds = load_split("train")
test_ds = load_split("eval")

print("Train samples:", len(train_ds))
print("eval samples:", len(test_ds))

# ---------------- LOAD PROCESSOR + CUSTOM TOKENIZER ----------------
print("Loading model...")

# Load base processor (feature extractor)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Replace tokenizer with custom Tigrinya tokenizer
custom_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_PATH)
processor.tokenizer = custom_tokenizer

# Load model
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Resize output layer to match tokenizer vocab size
model.resize_token_embeddings(len(processor.tokenizer))

model.to(DEVICE)
model.eval()

print("Model loaded.")

# ---------------- METRICS ----------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ---------------- STORAGE ----------------
predictions = []
references = []
sample_wers = []
sample_cers = []

print("Starting low-memory evaluation...\n")

# ---------------- EVALUATION LOOP ----------------
for sample in tqdm(test_ds, desc="Transcribing audio"):

    audio_path = sample["audio_path"]

    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Normalize audio
        audio = audio / (abs(audio).max() + 1e-9)

        # Prepare inputs
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(DEVICE)

        # Inference
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(input_values).logits

        # Greedy decode
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        transcription = normalize(transcription)
        reference = sample["sentence"]

        predictions.append(transcription)
        references.append(reference)

        # Per-sample metrics
        sample_wer = wer_metric.compute(
            predictions=[transcription],
            references=[reference]
        )

        sample_cer = cer_metric.compute(
            predictions=[transcription],
            references=[reference]
        )

        sample_wers.append(sample_wer)
        sample_cers.append(sample_cer)

        # Cleanup
        del input_values, logits, predicted_ids
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        continue

# ---------------- SHOW SAMPLE PREDICTIONS ----------------
num_examples = min(10, len(predictions))

print(f"\n--- Showing {num_examples} Sample Predictions vs References ---\n")

for i in range(num_examples):
    print(f"Sample {i+1} | WER: {sample_wers[i]:.2f} | CER: {sample_cers[i]:.2f}")
    print(f"Reference : {references[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 50)

# ---------------- COMPUTE FINAL METRICS ----------------
print("\nComputing metrics...\n")

final_wer = wer_metric.compute(
    predictions=predictions,
    references=references
)

final_cer = cer_metric.compute(
    predictions=predictions,
    references=references
)

# ---------------- SAVE PREDICTIONS ----------------
df = pd.DataFrame({
    "reference": references,
    "prediction": predictions
})
df.to_csv("eval_predictions.csv", index=False)

# ---------------- TIMER END ----------------
end_time = time.time()
total_time = end_time - start_time

# ---------------- MEMORY ----------------
peak_gpu_mem = (
    torch.cuda.max_memory_allocated() / (1024 ** 3)
    if torch.cuda.is_available()
    else 0
)

process = psutil.Process(os.getpid())
cpu_mem = process.memory_info().rss / (1024 ** 3)

# ---------------- FINAL RESULTS ----------------
print("==============================")
print("FINAL TEST RESULTS")
print("==============================")
print("Dataset : eval")
print(f"WER     : {final_wer:.4f}")
print(f"CER     : {final_cer:.4f}")

print("\nSaved predictions → eval_predictions.csv")

print("\n⏱ Total Time (seconds):", round(total_time, 2))
print("⏱ Total Time (minutes):", round(total_time / 60, 2))

print("\n💾 Peak GPU Memory (GB):", round(peak_gpu_mem, 2))
print("💾 CPU RAM Usage (GB):", round(cpu_mem, 2))

print("\nDONE ✅")

# ## Whisper-large-v3


import torch
import librosa
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer
)
import evaluate
from tqdm import tqdm
import gc
import os
import time
import psutil

# ---------------- CONFIG ----------------
DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_ID = "openai/whisper-large-v3"
TOKENIZER_PATH = "./tigrinya_tokenizer"   # Your custom tokenizer path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- TIMER ----------------
start_time = time.time()

# ---------------- NORMALIZATION ----------------
def normalize(text):
    text = text.replace("፡", " ")
    text = text.replace("።", "")
    text = " ".join(text.split())
    return text.strip()

# ---------------- LOAD DATA ----------------
def load_split(split):
    text_path = f"{DATA_ROOT}/text/{split}/text"
    wavscp_path = f"{DATA_ROOT}/text/{split}/wav.scp"

    text_dict = {}

    with open(text_path, encoding="utf-8") as f:
        for line in f:
            utt, sentence = line.strip().split(" ", 1)
            text_dict[utt] = normalize(sentence)

    rows = []

    with open(wavscp_path, encoding="utf-8") as f:
        for line in f:
            utt, wav = line.strip().split(" ", 1)

            wav = wav.replace(
                "/share/documents/solomon/TigrignaASR",
                DATA_ROOT
            )

            if utt in text_dict and os.path.exists(wav):
                rows.append({
                    "audio_path": wav,
                    "sentence": text_dict[utt]
                })

    return Dataset.from_pandas(pd.DataFrame(rows))

# ---------------- LOAD DATASETS ----------------
print("Loading dataset...")
train_ds = load_split("train")
test_ds = load_split("eval")

print("Train samples:", len(train_ds))
print("eval samples:", len(test_ds))

# ---------------- LOAD PROCESSOR + CUSTOM TOKENIZER ----------------
print("Loading model...")

# Load base processor
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Replace tokenizer with custom Tigrinya tokenizer
custom_tokenizer = WhisperTokenizer.from_pretrained(TOKENIZER_PATH)
processor.tokenizer = custom_tokenizer

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    use_safetensors=True
)

# Resize decoder embeddings for custom tokenizer
model.resize_token_embeddings(len(processor.tokenizer))

model.to(DEVICE)
model.eval()

print("Model loaded.")

# ---------------- GENERATION SETTINGS ----------------
# No forced language tokens since custom tokenizer is used
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

# ---------------- METRICS ----------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

predictions = []
references = []

sample_wers = []
sample_cers = []

example_outputs = []
num_examples_to_show = 5

print("Starting low-memory evaluation...\n")

# ---------------- EVALUATION ----------------
for idx, sample in enumerate(tqdm(test_ds, desc="Transcribing")):

    audio_path = sample["audio_path"]

    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Feature extraction
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(
            DEVICE,
            dtype=DTYPE
        )

        # Generate transcription
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                generated_ids = model.generate(
                    input_features,
                    max_new_tokens=128,
                    num_beams=5,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

        # Decode
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        transcription = normalize(transcription)
        reference = sample["sentence"]

        predictions.append(transcription)
        references.append(reference)

        # Sample metrics
        sample_wer = wer_metric.compute(
            predictions=[transcription],
            references=[reference]
        )

        sample_cer = cer_metric.compute(
            predictions=[transcription],
            references=[reference]
        )

        sample_wers.append(sample_wer)
        sample_cers.append(sample_cer)

        # Save examples
        if idx < num_examples_to_show:
            example_outputs.append({
                "Reference": reference,
                "Prediction": transcription
            })

        # Cleanup
        del input_features, generated_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# ---------------- FINAL METRICS ----------------
print("\nComputing metrics...\n")

wer = wer_metric.compute(
    predictions=predictions,
    references=references
)

cer = cer_metric.compute(
    predictions=predictions,
    references=references
)

# ---------------- SAVE PREDICTIONS ----------------
df = pd.DataFrame({
    "reference": references,
    "prediction": predictions
})
df.to_csv("eval_predictions.csv", index=False)

# ---------------- TIMER ----------------
end_time = time.time()
total_time = end_time - start_time

# ---------------- MEMORY ----------------
if torch.cuda.is_available():
    peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
else:
    peak_gpu_mem = 0

process = psutil.Process(os.getpid())
cpu_mem = process.memory_info().rss / (1024 ** 3)

# ---------------- RESULTS ----------------
print("==============================")
print("FINAL TEST RESULTS")
print("==============================")
print("Dataset : eval")
print(f"WER     : {wer:.4f}")
print(f"CER     : {cer:.4f}")

print("\nSaved predictions → eval_predictions.csv")

print("\n⏱ Total Time (minutes):", round(total_time / 60, 2))

print("\n💾 Peak GPU Memory (GB):", round(peak_gpu_mem, 2))
print("💾 CPU RAM Usage (GB):", round(cpu_mem, 2))

# ---------------- SAMPLE OUTPUTS ----------------
print(f"\n--- Showing {len(example_outputs)} Sample Predictions ---\n")



print("\nDONE ✅")

import os
import librosa

DATA_ROOT = "/share/documents/solomon/TigrignaASR"

# Candidate names for each dataset split
SPLIT_CANDIDATES = {
    "train": ["train"],
    "validation": ["valid", "dev", "validation"],
    "test": ["eval", "test"],
}


def find_existing_split(candidates):
    """
    Returns the first split name whose wav.scp file exists.
    """
    for split in candidates:
        wavscp_path = os.path.join(DATA_ROOT, "text", split, "wav.scp")
        if os.path.exists(wavscp_path):
            return split
    return None


def get_total_hours(split):
    """
    Computes total duration (in hours) for a given split.
    """
    wavscp_path = os.path.join(DATA_ROOT, "text", split, "wav.scp")

    if not os.path.exists(wavscp_path):
        raise FileNotFoundError(f"wav.scp not found: {wavscp_path}")

    total_seconds = 0.0
    num_files = 0

    with open(wavscp_path, encoding="utf-8") as f:
        for line in f:
            utt, wav_path = line.strip().split(" ", 1)

            # Replace original hardcoded root if needed
            wav_path = wav_path.replace(
                "/share/documents/solomon/TigrignaASR",
                DATA_ROOT
            )

            if not os.path.exists(wav_path):
                print(f"Missing file: {wav_path}")
                continue

            try:
                # Get duration without loading the full waveform
                duration = librosa.get_duration(path=wav_path)
                total_seconds += duration
                num_files += 1
            except Exception as e:
                print(f"Error reading {wav_path}: {e}")

    total_hours = total_seconds / 3600.0

    print(f"\nSplit: {split}")
    print(f"Files processed : {num_files}")
    print(f"Total seconds   : {total_seconds:.2f}")
    print(f"Total hours     : {total_hours:.2f}")

    return num_files, total_hours


# Store results
results = {}

# Process train, validation, and test splits
for split_label, candidates in SPLIT_CANDIDATES.items():
    actual_split = find_existing_split(candidates)

    if actual_split is None:
        print(f"No {split_label} split found. Checked: {candidates}")
        continue

    num_files, hours = get_total_hours(actual_split)

    results[split_label] = {
        "actual_name": actual_split,
        "num_files": num_files,
        "hours": hours,
    }

# Print summary
print("\n==============================")
print("DATASET DURATION SUMMARY")
print("==============================")

for split_label, info in results.items():
    print(
        f"{split_label.capitalize():<12}"
        f"({info['actual_name']:<10}) : "
        f"{info['hours']:.2f} hours "
        f"({info['num_files']} files)"
    )

if results:
    total_hours = sum(info["hours"] for info in results.values())
    total_files = sum(info["num_files"] for info in results.values())
    print("------------------------------")
    print(f"{'Total':<24}: {total_hours:.2f} hours ({total_files} files)")

print("==============================")