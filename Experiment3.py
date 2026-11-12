import torch
import librosa
from datasets import Dataset
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import evaluate
from tqdm import tqdm
import gc
import os
import time
import psutil

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_ID = "openai/whisper-large-v3"

# Whisper has no Tigrinya language token. Amharic ("amharic") uses the same
# Ge'ez script, so it is the closest option for a zero-shot test.
# Set to None to let Whisper auto-detect the language.
LANGUAGE = "amharic"
TASK = "transcribe"

NUM_BEAMS = 5            # use 1 for much faster (greedy) decoding
MAX_SAMPLES = None       # e.g. 20 for a quick test, None = full eval set
LOG_EVERY = 25           # print running WER/CER every N files

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print("Device:", DEVICE, "| dtype:", TORCH_DTYPE)
if DEVICE == "cpu":
    print("WARNING: whisper-large-v3 (1.5B params) on CPU is very slow. Try MAX_SAMPLES=20 and NUM_BEAMS=1 first.")

# =========================================================
# TIMER START
# =========================================================
start_time = time.time()

# =========================================================
# TEXT NORMALIZATION
# =========================================================
def normalize_text(text):
    text = text.replace("፡", " ")
    text = " ".join(text.split())
    return text.strip()

def remove_spaces(text):
    return text.replace(" ", "")

# =========================================================
# LOAD DATA
# =========================================================
def load_split(split):
    text_path = f"{DATA_ROOT}/text/{split}/text"
    wavscp_path = f"{DATA_ROOT}/text/{split}/wav.scp"

    text_dict = {}
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            utt, sentence = line.strip().split(" ", 1)
            text_dict[utt] = normalize_text(sentence)

    rows, missing = [], 0
    with open(wavscp_path, encoding="utf-8") as f:
        for line in f:
            utt, wav = line.strip().split(" ", 1)
            wav = wav.replace("/share/documents/solomon/TigrignaASR", DATA_ROOT)
            if utt in text_dict and os.path.exists(wav):
                rows.append({"audio_path": wav, "sentence": text_dict[utt]})
            else:
                missing += 1

    print(f"[{split}] loaded {len(rows)} utterances ({missing} skipped: missing text or audio)")
    return Dataset.from_pandas(pd.DataFrame(rows))

# =========================================================
# LOAD DATASET
# =========================================================
test_ds = load_split("eval")
if MAX_SAMPLES:
    test_ds = test_ds.select(range(min(MAX_SAMPLES, len(test_ds))))
print("Test samples:", len(test_ds))

# =========================================================
# LOAD MODEL + PROCESSOR (original Whisper tokenizer)
# =========================================================
# NOTE: the custom ./tigrinya_tokenizer and embedding resize from the
# fine-tuned experiment are intentionally NOT used here. The pretrained
# whisper-large-v3 weights only make sense with its own tokenizer; swapping
# the tokenizer would scramble every token id and give garbage output.
print(f"\nLoading {MODEL_ID} ...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
)
model.to(DEVICE)
model.eval()
print(f"Model loaded in {time.time() - t0:.1f}s | "
      f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

generate_kwargs = {"max_new_tokens": 440, "num_beams": NUM_BEAMS, "task": TASK}
if LANGUAGE:
    generate_kwargs["language"] = LANGUAGE
print("Generate kwargs:", generate_kwargs)

# =========================================================
# METRICS
# =========================================================
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# =========================================================
# STORAGE
# =========================================================
predictions, references = [], []
predictions_nospace, references_nospace = [], []
sample_wers, sample_cers = [], []
audio_paths = []

example_outputs = []
num_examples_to_show = 5
failed = 0

# =========================================================
# EVALUATION LOOP
# =========================================================
print("\nTranscribing test set...\n")
loop_start = time.time()

for idx, sample in enumerate(tqdm(test_ds, desc="Transcribing", unit="file")):
    audio_path = sample["audio_path"]

    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device=DEVICE, dtype=TORCH_DTYPE)

        with torch.no_grad():
            generated_ids = model.generate(input_features, **generate_kwargs)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcription = normalize_text(transcription)
        reference = normalize_text(sample["sentence"])

        predictions.append(transcription)
        references.append(reference)
        predictions_nospace.append(remove_spaces(transcription))
        references_nospace.append(remove_spaces(reference))
        audio_paths.append(audio_path)

        sample_wer = wer_metric.compute(predictions=[transcription], references=[reference])
        sample_cer = cer_metric.compute(predictions=[transcription], references=[reference])
        sample_wers.append(sample_wer)
        sample_cers.append(sample_cer)

        if idx < num_examples_to_show:
            example_outputs.append({
                "Reference": reference,
                "Prediction": transcription,
                "WER": sample_wer,
                "CER": sample_cer,
            })

        # periodic progress log
        if (idx + 1) % LOG_EVERY == 0:
            run_wer = wer_metric.compute(predictions=predictions, references=references)
            run_cer = cer_metric.compute(predictions=predictions, references=references)
            rate = (idx + 1) / (time.time() - loop_start)
            print(f"\n[{idx + 1}/{len(test_ds)}] running WER {run_wer:.4f} | CER {run_cer:.4f} | "
                  f"{rate:.2f} files/s | failed {failed}")
            print(f"  REF: {reference}")
            print(f"  HYP: {transcription}")

        del input_features, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        failed += 1
        print(f"Error processing {audio_path}: {e}")
        continue

# =========================================================
# FINAL METRICS
# =========================================================
wer = wer_metric.compute(predictions=predictions, references=references)
cer = cer_metric.compute(predictions=predictions, references=references)
wer_nospace = wer_metric.compute(predictions=predictions_nospace, references=references_nospace)

# =========================================================
# TIMER & MEMORY
# =========================================================
total_time = time.time() - start_time

peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

# =========================================================
# SAMPLE OUTPUTS
# =========================================================
print(f"\n--- Showing {min(num_examples_to_show, len(example_outputs))} Sample Predictions ---\n")
for i, ex in enumerate(example_outputs):
    print(f"Sample {i + 1}")
    print(f"WER: {ex['WER']:.3f} | CER: {ex['CER']:.3f}")
    print("Reference :", ex["Reference"])
    print("Prediction:", ex["Prediction"])
    print("-" * 60)

# =========================================================
# FINAL RESULTS
# =========================================================
print("\n==============================")
print(f"FINAL TEST RESULTS  ({MODEL_ID}, zero-shot, language={LANGUAGE})")
print("==============================")
print(f"Evaluated    : {len(predictions)} files ({failed} failed)")
print(f"Standard WER : {wer:.4f} ({wer * 100:.2f}%)")
print(f"CER          : {cer:.4f} ({cer * 100:.2f}%)")
print(f"No-space WER : {wer_nospace:.4f} ({wer_nospace * 100:.2f}%)")

print("\n⏱ Total Time")
print(f"Seconds: {total_time:.2f}")
print(f"Minutes: {total_time / 60:.2f}")

print("\n💾 Memory Usage")
print(f"Peak GPU Memory: {peak_gpu_mem:.2f} GB")
print(f"CPU RAM Usage : {cpu_mem:.2f} GB")

# =========================================================
# SAVE PREDICTIONS
# =========================================================
out_csv = "eval_predictions_whisper_large_v3.csv"
pd.DataFrame({
    "audio_path": audio_paths,
    "reference": references,
    "prediction": predictions,
    "reference_nospace": references_nospace,
    "prediction_nospace": predictions_nospace,
    "sample_wer": sample_wers,
    "sample_cer": sample_cers,
}).to_csv(out_csv, index=False)

print(f"\nSaved predictions → {out_csv}")
print("DONE ✅")
