#!/usr/bin/env python
"""
Evaluate a wav2vec2 CTC model on the test split of a Tigrigna ASR dataset.

Install:  pip install torch transformers librosa soundfile jiwer pandas tqdm
Usage:    python eval_asr.py --model facebook/wav2vec2-xls-r-300m
          python eval_asr.py --model /path/to/finetuned-checkpoint --batch_size 16
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import jiwer
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_NAME = "facebook/mms-1b-all"
SAMPLE_RATE = 16000
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
AUDIO_COLS = ["path", "audio", "audio_path", "audio_filepath", "file", "filename", "wav", "wav_path"]
TEXT_COLS = ["text", "sentence", "transcript", "transcription", "label", "normalized_text"]

log = logging.getLogger("eval_asr")


def setup_logging(log_file):
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.DEBUG)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def show_tree(root, max_entries=30):
    log.info(f"Contents of {root}:")
    for i, p in enumerate(sorted(Path(root).iterdir())):
        if i >= max_entries:
            log.info("  ...")
            break
        kind = "dir " if p.is_dir() else "file"
        log.info(f"  [{kind}] {p.name}")


def resolve_audio(p, base_dirs):
    p = str(p)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for b in base_dirs:
        cand = os.path.join(b, p)
        if os.path.exists(cand):
            return cand
    return p  # let the loader fail loudly (and be logged)


def pick_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {what} column in {list(df.columns)}")


def from_dataframe(df, manifest_path, root, split):
    a, t = pick_col(df, AUDIO_COLS, "audio"), pick_col(df, TEXT_COLS, "text")
    log.info(f"Using columns: audio='{a}', text='{t}'")
    bases = [str(Path(manifest_path).parent), root, os.path.join(root, split)]
    return [(resolve_audio(p, bases), str(s)) for p, s in zip(df[a], df[t])]


def load_test_set(root, split="test"):
    root = Path(root)
    show_tree(root)

    # 1) manifest files: test.tsv / test.csv / test.json / test.jsonl (also in subfolders)
    for pattern in (f"{split}.tsv", f"{split}.csv", f"{split}.jsonl", f"{split}.json",
                    f"*{split}*.tsv", f"*{split}*.csv", f"*{split}*.jsonl", f"*{split}*.json"):
        hits = sorted(root.glob(pattern)) + sorted(root.glob(f"*/{pattern}"))
        if hits:
            m = hits[0]
            log.info(f"Found manifest: {m}")
            if m.suffix == ".tsv":
                df = pd.read_csv(m, sep="\t", quoting=3)
            elif m.suffix == ".csv":
                df = pd.read_csv(m)
            else:
                try:
                    df = pd.read_json(m, lines=m.suffix == ".jsonl")
                except ValueError:
                    df = pd.read_json(m, lines=True)
            return from_dataframe(df, m, str(root), split)

    # 2) Kaldi style: <split>/wav.scp + <split>/text
    d = next((c for c in (root / "text" / split, root / split) if (c / "wav.scp").exists()), root / split)
    if (d / "wav.scp").exists() and (d / "text").exists():
        log.info(f"Found Kaldi-style dir: {d}")
        wavs = dict(l.strip().split(maxsplit=1) for l in open(d / "wav.scp", encoding="utf-8") if l.strip())
        out = []
        for l in open(d / "text", encoding="utf-8"):
            if not l.strip():
                continue
            uid, *txt = l.strip().split(maxsplit=1)
            if uid in wavs:
                out.append((resolve_audio(wavs[uid], [str(root), str(root / "wav"), str(root / "wav" / split), str(d)]),
                            txt[0] if txt else ""))
        missing = sum(not os.path.exists(p) for p, _ in out)
        log.info(f"Kaldi set: {len(out)} utterances ({len(wavs)} in wav.scp), {missing} audio paths not found on disk")
        if missing:
            log.warning(f"Example unresolved path: {next(p for p, _ in out if not os.path.exists(p))}")
        return out

    # 3) folder of audio files with matching .txt transcripts
    if d.is_dir():
        log.info(f"Scanning {d} for audio + .txt pairs")
        out = []
        for a in sorted(d.rglob("*")):
            if a.suffix.lower() in AUDIO_EXTS and a.with_suffix(".txt").exists():
                out.append((str(a), a.with_suffix(".txt").read_text(encoding="utf-8").strip()))
        if out:
            return out

    raise FileNotFoundError(
        f"Could not find a '{split}' set under {root}. See the directory listing above "
        f"and adapt load_test_set() or pass --manifest."
    )


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def load_audio(path):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--split", default="eval", help="train / dev / eval")
    ap.add_argument("--manifest", default=None, help="explicit manifest (tsv/csv/json/jsonl)")
    ap.add_argument("--lang", default="tir", help="MMS adapter language code, e.g. tir")
    ap.add_argument("--allow_untrained", action="store_true", help="run raw pretrained model with random CTC head")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None, help="debug: limit # of samples")
    ap.add_argument("--out_dir", default="eval_output")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "eval.log"))
    log.info(f"Args: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ---- data
    t0 = time.time()
    if args.manifest:
        m = Path(args.manifest)
        df = pd.read_csv(m, sep="\t" if m.suffix == ".tsv" else ",") if m.suffix in (".tsv", ".csv") \
            else pd.read_json(m, lines=m.suffix == ".jsonl")
        samples = from_dataframe(df, m, args.data_root, args.split)
    else:
        samples = load_test_set(args.data_root, args.split)
    log.info(f"Loaded {len(samples)} test samples in {time.time() - t0:.1f}s")
    if args.max_samples:
        samples = samples[: args.max_samples]
        log.info(f"Limiting to {len(samples)} samples")
    for p, t in samples[:3]:
        log.info(f"  example: {p} -> {t!r}")

    # ---- model
    log.info(f"Loading model: {args.model}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(args.model)
        model = Wav2Vec2ForCTC.from_pretrained(args.model)
        if args.lang:  # MMS-style per-language adapter
            log.info(f"Loading language adapter: {args.lang}")
            processor.tokenizer.set_target_lang(args.lang)
            model.load_adapter(args.lang)
    except Exception as e:
        log.error(f"Could not load a trained CTC model/processor from '{args.model}': {e}")
        if not args.allow_untrained:
            log.error(
                "'facebook/wav2vec2-xls-r-300m' is pretrained-only: NO CTC head, NO vocabulary. "
                "Options: (a) --allow_untrained to run it anyway (random head, WER ~100%, meaningless baseline), "
                "(b) a model trained for ASR, e.g. --model facebook/mms-1b-all --lang tir (true zero-shot), "
                "(c) fine-tune first with finetune_asr.py."
            )
            sys.exit(1)
        import tempfile
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
        log.warning("=" * 60)
        log.warning("UNTRAINED MODE: building a dummy char vocab from eval transcripts and attaching a")
        log.warning("RANDOMLY INITIALIZED CTC head. Results below are NOT meaningful ASR performance.")
        log.warning("=" * 60)
        chars = sorted({c for _, t in samples for c in t.replace(" ", "|")})
        vocab = {c: i for i, c in enumerate(chars)}
        vocab["[UNK]"], vocab["[PAD]"] = len(vocab), len(vocab) + 1
        vp = os.path.join(tempfile.mkdtemp(), "vocab.json")
        json.dump(vocab, open(vp, "w", encoding="utf-8"), ensure_ascii=False)
        tok = Wav2Vec2CTCTokenizer(vp, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0,
                                      do_normalize=True, return_attention_mask=True)
        processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
        model = Wav2Vec2ForCTC.from_pretrained(args.model, vocab_size=len(tok), pad_token_id=tok.pad_token_id)
    model = model.to(device).eval()
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | "
             f"vocab size: {model.config.vocab_size}")

    # ---- inference
    preds, refs, bad = [], [], 0
    nb = (len(samples) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for bi in tqdm(range(nb), desc="Evaluating"):
        batch = samples[bi * args.batch_size:(bi + 1) * args.batch_size]
        audios, texts = [], []
        for path, text in batch:
            try:
                audios.append(load_audio(path))
                texts.append(text)
            except Exception as e:
                bad += 1
                log.warning(f"Failed to load {path}: {e}")
        if not audios:
            continue
        inputs = processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(
                inputs.input_values.to(device),
                attention_mask=inputs.get("attention_mask", None).to(device)
                if inputs.get("attention_mask", None) is not None else None,
            ).logits
        batch_preds = processor.batch_decode(torch.argmax(logits, dim=-1))
        preds += batch_preds
        refs += texts

        if bi % 10 == 0:
            run_wer = jiwer.wer(refs, preds)
            log.info(f"batch {bi + 1}/{nb} | running WER {run_wer:.4f} | "
                     f"{len(refs) / (time.time() - t0):.2f} utt/s")
            log.debug(f"  REF: {texts[0]}")
            log.debug(f"  HYP: {batch_preds[0]}")

    # ---- metrics
    log.info("=" * 60)
    log.info(f"Evaluated {len(refs)} utterances ({bad} failed to load) in {time.time() - t0:.1f}s")
    wer, cer = jiwer.wer(refs, preds), jiwer.cer(refs, preds)
    log.info(f"WER: {wer:.4f} ({wer * 100:.2f}%)")
    log.info(f"CER: {cer:.4f} ({cer * 100:.2f}%)")

    out = os.path.join(args.out_dir, "predictions.tsv")
    pd.DataFrame({"reference": refs, "prediction": preds}).to_csv(out, sep="\t", index=False)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"wer": wer, "cer": cer, "n": len(refs), "model": args.model}, f, indent=2)
    log.info(f"Saved predictions to {out}")


if __name__ == "__main__":
    main()
