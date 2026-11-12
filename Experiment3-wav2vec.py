#!/usr/bin/env python
"""
Zero-shot evaluation of agkphysics/wav2vec2-large-xlsr-53-amharic on the Tigrigna eval split,
using the model's BUILT-IN Amharic tokenizer / vocabulary (no custom tokenizer, no head resizing).

Install:  pip install torch transformers librosa soundfile jiwer pandas tqdm psutil
          pip install pyctcdecode        # only for --beam_search
Usage:    python eval_amharic_w2v2.py --max_samples 20
          python eval_amharic_w2v2.py
          python eval_amharic_w2v2.py --beam_search --beam_width 50
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import jiwer
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    import psutil
except ImportError:
    psutil = None

DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_ID = "agkphysics/wav2vec2-large-xlsr-53-amharic"
SR = 16000

log = logging.getLogger("eval_amharic")


def setup_logging(log_file):
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.DEBUG)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def normalize_text(text):
    text = text.replace("፡", " ")
    return " ".join(text.split()).strip()


def load_kaldi(root, split):
    root = Path(root)
    d = root / "text" / split
    wavs = dict(l.strip().split(maxsplit=1) for l in open(d / "wav.scp", encoding="utf-8") if l.strip())
    out, missing = [], 0
    for l in open(d / "text", encoding="utf-8"):
        parts = l.strip().split(maxsplit=1)
        if len(parts) < 2 or parts[0] not in wavs:
            continue
        p = wavs[parts[0]]
        if not os.path.isabs(p):
            p = str(root / p) if (root / p).exists() else str(root / "wav" / p)
        if not os.path.exists(p):
            missing += 1
            continue
        out.append((p, normalize_text(parts[1])))
    log.info(f"[{split}] {len(out)} utterances loaded ({missing} skipped: audio not found)")
    return out


def build_beam_decoder(tokenizer):
    from pyctcdecode import build_ctcdecoder
    vocab = tokenizer.get_vocab()
    labels = [None] * len(vocab)
    for tok, i in vocab.items():
        labels[i] = tok
    mapped = []
    for t in labels:
        if t == tokenizer.pad_token:
            mapped.append("")                       # CTC blank
        elif t == tokenizer.word_delimiter_token:
            mapped.append(" ")                      # word boundary
        elif t in tokenizer.all_special_tokens:
            mapped.append("⁇")                      # unk / bos / eos
        else:
            mapped.append(t)
    return build_ctcdecoder(mapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="1 matches per-file inference; this xlsr-53 model uses group-norm, so padding "
                         "in larger batches can slightly change results")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--beam_search", action="store_true", help="pyctcdecode beam search (no LM) instead of greedy")
    ap.add_argument("--beam_width", type=int, default=100)
    ap.add_argument("--out_dir", default="eval_output_amharic")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "eval.log"))
    log.info(f"Args: {vars(args)}")
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ---- data
    samples = load_kaldi(args.data_root, args.split)
    if args.max_samples:
        samples = samples[: args.max_samples]
        log.info(f"Limiting to {len(samples)} samples")
    for p, t in samples[:3]:
        log.info(f"  example: {p} -> {t!r}")

    # ---- model + BUILT-IN processor
    log.info(f"Loading model and built-in tokenizer: {args.model}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()
    tok = processor.tokenizer
    vocab = tok.get_vocab()
    log.info(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | "
             f"vocab size {len(vocab)} | pad/blank={tok.pad_token!r} | delimiter={tok.word_delimiter_token!r}")
    log.info(f"Feature extractor returns attention_mask: {processor.feature_extractor.return_attention_mask}")

    # ---- how well can this vocabulary even express the Tigrigna references?
    ref_chars = Counter("".join(t for _, t in samples).replace(" ", ""))
    total = sum(ref_chars.values())
    missing = {c: n for c, n in ref_chars.items() if c not in vocab}
    miss_n = sum(missing.values())
    log.info(f"Vocabulary coverage: {len(missing)} of {len(ref_chars)} distinct reference characters are NOT in the "
             f"Amharic vocab, covering {miss_n}/{total} character occurrences ({100 * miss_n / max(total, 1):.2f}%). "
             f"These can never be predicted correctly (CER floor).")
    log.info(f"Most frequent missing characters: {Counter(missing).most_common(15)}")

    decoder = None
    if args.beam_search:
        decoder = build_beam_decoder(tok)
        log.info(f"Beam search enabled (width {args.beam_width}, no language model)")
    else:
        log.info("Greedy CTC decoding")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- inference
    preds, refs, paths, failed = [], [], [], 0
    nb = (len(samples) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for bi in tqdm(range(nb), desc="Transcribing"):
        batch = samples[bi * args.batch_size:(bi + 1) * args.batch_size]
        audios, keep = [], []
        for p, t in batch:
            try:
                audios.append(librosa.load(p, sr=SR, mono=True)[0])
                keep.append((p, t))
            except Exception as e:
                failed += 1
                log.warning(f"Failed to load {p}: {e}")
        if not audios:
            continue
        inp = processor(audios, sampling_rate=SR, return_tensors="pt", padding=True)
        am = inp.get("attention_mask")
        with torch.no_grad():
            logits = model(inp.input_values.to(device),
                           attention_mask=am.to(device) if am is not None else None).logits
        if decoder is None:
            texts = processor.batch_decode(torch.argmax(logits, dim=-1))
        else:
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            texts = [decoder.decode(x, beam_width=args.beam_width) for x in lp]
        for txt, (p, t) in zip(texts, keep):
            preds.append(normalize_text(txt))
            refs.append(t)
            paths.append(p)

        if bi % 20 == 0:
            log.info(f"batch {bi + 1}/{nb} | running WER {jiwer.wer(refs, preds):.4f} | "
                     f"CER {jiwer.cer(refs, preds):.4f} | {len(refs) / (time.time() - t0):.2f} utt/s")
            log.debug(f"  REF: {refs[-1]}")
            log.debug(f"  HYP: {preds[-1]}")
        del logits
        gc.collect()

    # ---- metrics
    wer, cer = jiwer.wer(refs, preds), jiwer.cer(refs, preds)
    wer_ns = jiwer.wer([r.replace(" ", "") for r in refs], [p.replace(" ", "") for p in preds])
    total_time = time.time() - start
    peak_gpu = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3 if psutil else float("nan")

    log.info("=" * 60)
    log.info(f"FINAL RESULTS ({args.model}, zero-shot, built-in tokenizer)")
    log.info(f"Evaluated    : {len(refs)} utterances ({failed} failed)")
    log.info(f"Standard WER : {wer:.4f} ({wer * 100:.2f}%)")
    log.info(f"CER          : {cer:.4f} ({cer * 100:.2f}%)")
    log.info(f"No-space WER : {wer_ns:.4f} ({wer_ns * 100:.2f}%)")
    log.info(f"Total time   : {total_time / 60:.2f} min | peak GPU {peak_gpu:.2f} GB | CPU RAM {cpu_mem:.2f} GB")
    for i in range(min(5, len(refs))):
        log.info(f"Sample {i + 1}\n  REF: {refs[i]}\n  HYP: {preds[i]}")

    pd.DataFrame({
        "audio_path": paths, "reference": refs, "prediction": preds,
        "reference_nospace": [r.replace(" ", "") for r in refs],
        "prediction_nospace": [p.replace(" ", "") for p in preds],
        "sample_wer": [jiwer.wer(r, p) for r, p in zip(refs, preds)],
        "sample_cer": [jiwer.cer(r, p) for r, p in zip(refs, preds)],
    }).to_csv(os.path.join(args.out_dir, "eval_predictions.csv"), index=False)
    json.dump({"wer": wer, "cer": cer, "wer_nospace": wer_ns, "n": len(refs), "model": args.model,
               "beam_search": args.beam_search, "oov_char_rate": miss_n / max(total, 1)},
              open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)
    log.info(f"Saved eval_predictions.csv and metrics.json to {args.out_dir}")


if __name__ == "__main__":
    main()
