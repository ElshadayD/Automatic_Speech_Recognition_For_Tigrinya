#!/usr/bin/env python
"""
agkphysics/wav2vec2-large-xlsr-53-amharic + a custom BPE tokenizer (default 100 tokens) for Tigrigna CTC ASR.

The Amharic model already has a CTC head for Amharic characters. Here that head is replaced by a new one
sized for the BPE vocabulary (the encoder, already adapted to Amharic/Ge'ez speech, is kept as the starting
point), then the model is fine-tuned on Tigrigna and evaluated.

Pipeline:  train BPE on text/train/text  ->  fine-tune XLS-R with CTC on those tokens
           ->  evaluate on the eval split (WER / CER / no-space WER), with logs throughout.

Install:  pip install torch transformers tokenizers librosa soundfile jiwer pandas tqdm accelerate
Usage:    python amharic_w2v2_bpe_ctc.py --max_samples 50 --epochs 1       # smoke test
          python amharic_w2v2_bpe_ctc.py                                    # full run (train + eval)
          python amharic_w2v2_bpe_ctc.py --eval_only                        # re-evaluate ./amharic-bpe100/final
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import librosa
import pandas as pd
import soundfile as sf
import torch
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)

DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_NAME = "agkphysics/wav2vec2-large-xlsr-53-amharic"
SR = 16000

log = logging.getLogger("amharic_bpe")


def setup_logging(log_file):
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.DEBUG)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")]
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    transformers.logging.set_verbosity_info()
    transformers.logging.disable_default_handler()
    for h in handlers:
        transformers.logging.add_handler(h)


# ---- codec begin -----------------------------------------------------------
SP = "▁"            # word-boundary marker (acts as the "space" token in CTC)
MARK0 = 0xE000      # private-use chars used as vowel-order markers (8 orders)
ETH_LO, ETH_HI = 0x1200, 0x1357  # Ethiopic syllables block (rows of 8)


def normalize_text(text):
    text = text.replace("፡", " ")
    return " ".join(text.split()).strip()


def decompose(text):
    """Each Ethiopic syllable -> (row base char, vowel-order marker). Exactly reversible."""
    out = []
    for ch in text:
        cp = ord(ch)
        if ETH_LO <= cp <= ETH_HI:
            off = cp - ETH_LO
            out.append(chr(ETH_LO + (off // 8) * 8))
            out.append(chr(MARK0 + off % 8))
        else:
            out.append(ch)
    return "".join(out)


def compose(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if MARK0 <= cp < MARK0 + 8:
            if out and ETH_LO <= ord(out[-1]) <= ETH_HI and (ord(out[-1]) - ETH_LO) % 8 == 0:
                out[-1] = chr(ord(out[-1]) + cp - MARK0)
            continue
        out.append(ch)
    return "".join(out)


class BPECodec:
    """Text <-> BPE token ids for CTC. id of [PAD] is the CTC blank."""

    SPECIALS = ("[PAD]", "[UNK]")

    def __init__(self, tok=None, decomp=False):
        self.tok, self.decomp = tok, decomp

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    @property
    def pad_id(self):
        return self.tok.token_to_id("[PAD]")

    def prep(self, text):
        text = normalize_text(text)
        if self.decomp:
            text = decompose(text)
        return " ".join(SP + w for w in text.split())

    def encode(self, text):
        return self.tok.encode(self.prep(text)).ids

    def decode(self, ids, collapse=False):
        ids = [int(i) for i in ids]
        if collapse:  # CTC: merge repeats first, then drop blanks
            ids = [i for k, i in enumerate(ids) if k == 0 or i != ids[k - 1]]
        toks = [self.tok.id_to_token(i) for i in ids if i >= 0 and i != self.pad_id]
        text = "".join(t for t in toks if t and t not in self.SPECIALS).replace(SP, " ")
        if self.decomp:
            text = compose(text)
        return normalize_text(text)

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        self.tok.save(os.path.join(d, "bpe_tokenizer.json"))
        json.dump({"decompose": self.decomp}, open(os.path.join(d, "bpe_meta.json"), "w"))

    @classmethod
    def load(cls, d):
        tok = Tokenizer.from_file(os.path.join(d, "bpe_tokenizer.json"))
        meta = json.load(open(os.path.join(d, "bpe_meta.json")))
        return cls(tok, meta["decompose"])


def train_codec(texts, vocab_size, mode):
    chars = set("".join(normalize_text(t) for t in texts)) - {" "}
    log.info(f"Unique characters in training text: {len(chars)} (requested BPE vocab: {vocab_size})")
    decomp = mode == "yes" or (mode == "auto" and len(chars) + 3 > vocab_size)
    if decomp:
        log.info("Alphabet is larger than the BPE vocab -> using syllable decomposition "
                 "(consonant row + vowel order) so a small BPE vocab is possible.")
    codec = BPECodec(None, decomp)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(BPECodec.SPECIALS), show_progress=False)
    tok.train_from_iterator([codec.prep(t) for t in texts], trainer)
    codec.tok = tok
    return codec
# ---- codec end -------------------------------------------------------------


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
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


def filter_samples(samples, codec, max_sec, split):
    kept, total, d_long, d_bad, d_ctc = [], 0.0, 0, 0, 0
    for p, t in tqdm(samples, desc=f"Checking {split}"):
        try:
            dur = sf.info(p).duration
        except Exception as e:
            log.warning(f"Cannot read {p}: {e}")
            d_bad += 1
            continue
        if not t:
            d_bad += 1
        elif dur > max_sec:
            d_long += 1
        elif len(codec.encode(t)) >= dur * 50 * 0.8:  # CTC needs enough frames (~50/s)
            d_ctc += 1
        else:
            kept.append((p, t))
            total += dur
    log.info(f"[{split}] kept {len(kept)} ({total / 3600:.2f} h) | dropped: {d_long} too long, "
             f"{d_ctc} too many tokens for CTC, {d_bad} unreadable/empty")
    return kept


class ASRDataset(Dataset):
    def __init__(self, samples, codec, fe):
        self.samples, self.codec, self.fe = samples, codec, fe

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        wav, _ = librosa.load(path, sr=SR, mono=True)
        return {"input_values": self.fe(wav, sampling_rate=SR).input_values[0],
                "labels": self.codec.encode(text)}


@dataclass
class Collator:
    fe: Wav2Vec2FeatureExtractor

    def __call__(self, feats):
        batch = self.fe.pad([{"input_values": f["input_values"]} for f in feats], padding=True, return_tensors="pt")
        L = max(len(f["labels"]) for f in feats)
        batch["labels"] = torch.tensor([f["labels"] + [-100] * (L - len(f["labels"])) for f in feats])
        return batch


class LogCallback(TrainerCallback):
    def __init__(self):
        self.t0 = time.time()

    def on_train_begin(self, args, state, control, **kw):
        log.info(f"Training started: {state.max_steps} optimizer steps")

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            log.info(f"step {state.global_step}/{state.max_steps} | epoch {state.epoch:.2f} | "
                     f"{(time.time() - self.t0) / 60:.1f} min | {logs}")

    def on_save(self, args, state, control, **kw):
        log.info(f"Checkpoint saved at step {state.global_step}")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_eval(model, codec, fe, samples, device, batch_size, out_dir):
    model.to(device).eval()
    preds, refs, paths, failed = [], [], [], 0
    nb = (len(samples) + batch_size - 1) // batch_size
    t0 = time.time()
    for bi in tqdm(range(nb), desc="Evaluating"):
        batch = samples[bi * batch_size:(bi + 1) * batch_size]
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
        inp = fe(audios, sampling_rate=SR, return_tensors="pt", padding=True)
        am = inp.get("attention_mask")
        with torch.no_grad():
            logits = model(inp.input_values.to(device),
                           attention_mask=am.to(device) if am is not None else None).logits
        for row, (p, t) in zip(logits.argmax(-1).cpu().tolist(), keep):
            preds.append(codec.decode(row, collapse=True))
            refs.append(t)
            paths.append(p)
        if bi % 10 == 0:
            log.info(f"batch {bi + 1}/{nb} | running WER {jiwer.wer(refs, preds):.4f} | "
                     f"{len(refs) / (time.time() - t0):.2f} utt/s")
            log.debug(f"  REF: {refs[-1]}")
            log.debug(f"  HYP: {preds[-1]}")

    wer, cer = jiwer.wer(refs, preds), jiwer.cer(refs, preds)
    wer_ns = jiwer.wer([r.replace(" ", "") for r in refs], [p.replace(" ", "") for p in preds])
    log.info("=" * 60)
    log.info(f"Evaluated {len(refs)} utterances ({failed} failed) in {(time.time() - t0) / 60:.1f} min")
    log.info(f"WER          : {wer:.4f} ({wer * 100:.2f}%)")
    log.info(f"CER          : {cer:.4f} ({cer * 100:.2f}%)")
    log.info(f"No-space WER : {wer_ns:.4f} ({wer_ns * 100:.2f}%)")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"audio_path": paths, "reference": refs, "prediction": preds}).to_csv(
        os.path.join(out_dir, "eval_predictions.csv"), index=False)
    json.dump({"wer": wer, "cer": cer, "wer_nospace": wer_ns, "n": len(refs)},
              open(os.path.join(out_dir, "eval_metrics.json"), "w"), indent=2)
    log.info(f"Saved eval_predictions.csv and eval_metrics.json to {out_dir}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default="./amharic-bpe100")
    ap.add_argument("--vocab_size", type=int, default=100, help="BPE vocabulary size")
    ap.add_argument("--decompose", choices=["auto", "yes", "no"], default="auto",
                    help="split Ethiopic syllables into consonant+vowel before BPE")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_sec", type=float, default=15.0)
    ap.add_argument("--eval_batch_size", type=int, default=1,
                    help="1 = per-file inference; this xlsr-53 model uses group-norm, so padding in larger "
                         "batches can slightly change results")
    ap.add_argument("--max_samples", type=int, default=None, help="debug: limit samples per split")
    ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate <out_dir>/final")
    ap.add_argument("--no_eval", action="store_true", help="skip the eval run after training")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "run.log"))
    log.info(f"Args: {vars(args)}")
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    log.info(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))

    try:
        fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    except Exception as e:
        log.warning(f"Could not load feature extractor from {args.model} ({e}); using defaults")
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0,
                                      do_normalize=True, return_attention_mask=False)
    log.info(f"Feature extractor: do_normalize={fe.do_normalize}, return_attention_mask={fe.return_attention_mask}")
    final = os.path.join(args.out_dir, "final")
    lim = (lambda x, n=None: x[:n or args.max_samples] if args.max_samples else x)

    # ------------------------------------------------------------------ eval only
    if args.eval_only:
        codec = BPECodec.load(final)
        fe = Wav2Vec2FeatureExtractor.from_pretrained(final)
        model = Wav2Vec2ForCTC.from_pretrained(final)
        log.info(f"Loaded model + BPE codec from {final} (vocab {codec.vocab_size}, decompose={codec.decomp})")
        ev = lim(load_kaldi(args.data_root, "eval"))
        run_eval(model, codec, fe, ev, device, args.eval_batch_size, args.out_dir)
        return

    if not cuda:
        log.warning("No GPU detected: fine-tuning a 300M model on CPU is impractically slow.")

    # ------------------------------------------------------------------ data + BPE
    train_raw = lim(load_kaldi(args.data_root, "train"))
    dev_raw = lim(load_kaldi(args.data_root, "dev"), max((args.max_samples or 0) // 5, 5))
    eval_raw = lim(load_kaldi(args.data_root, "eval"))

    codec = train_codec([t for _, t in train_raw], args.vocab_size, args.decompose)
    codec.save(args.out_dir)
    vocab = sorted(codec.tok.get_vocab().items(), key=lambda x: x[1])
    log.info(f"BPE vocab size: {codec.vocab_size} (requested {args.vocab_size}) | CTC blank id = {codec.pad_id}")
    if codec.vocab_size > args.vocab_size:
        log.warning(f"Vocab is {codec.vocab_size} > {args.vocab_size}: the base alphabet alone is larger than "
                    f"requested; try --decompose yes.")
    log.info(f"Tokens: {[t for t, _ in vocab]}")

    # round-trip sanity check: text -> ids -> text
    ok = sum(codec.decode(codec.encode(t)) == normalize_text(t) for _, t in train_raw[:200])
    log.info(f"Round-trip check (encode->decode) exact on {ok}/{min(200, len(train_raw))} train sentences")
    for _, t in train_raw[:2]:
        ids = codec.encode(t)
        log.info(f"  text: {t}\n  ids ({len(ids)}): {ids[:30]}\n  tokens: {[codec.tok.id_to_token(i) for i in ids[:15]]}")

    train = filter_samples(train_raw, codec, args.max_sec, "train")
    dev = filter_samples(dev_raw, codec, args.max_sec, "dev")
    train_ds, dev_ds = ASRDataset(train, codec, fe), ASRDataset(dev, codec, fe)

    # ------------------------------------------------------------------ model
    log.info(f"Loading {args.model}; its Amharic CTC head is replaced by a new head of {codec.vocab_size} outputs")
    log.info("A 'some weights were not initialized ... size mismatch' warning for lm_head is expected.")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
        mask_time_prob=0.05,
        ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=codec.pad_id, vocab_size=codec.vocab_size,
        ignore_mismatched_sizes=True,   # drop the Amharic lm_head, use our BPE head
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    log.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    def preprocess_logits(logits, labels):
        return (logits[0] if isinstance(logits, tuple) else logits).argmax(-1)

    def compute_metrics(pred):
        hyp = [codec.decode(r, collapse=True) for r in pred.predictions]
        ref = [codec.decode(r) for r in pred.label_ids]
        log.info(f"  sample REF: {ref[0]}")
        log.info(f"  sample HYP: {hyp[0]}")
        return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}

    steps_per_epoch = max(len(train_ds) // (args.batch_size * args.grad_accum), 1)
    every = max(steps_per_epoch, 50)
    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=min(500, steps_per_epoch * 2),
        fp16=cuda,
        eval_strategy="steps",      # transformers < 4.41: use evaluation_strategy
        eval_steps=every,
        save_steps=every,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, data_collator=Collator(fe),
        train_dataset=train_ds, eval_dataset=dev_ds,
        compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits,
        callbacks=[LogCallback()],
    )
    trainer.train()

    trainer.save_model(final)
    fe.save_pretrained(final)
    codec.save(final)
    log.info(f"Final model + BPE codec saved to {final}")

    # ------------------------------------------------------------------ eval
    if not args.no_eval:
        log.info("Running final evaluation on the eval split")
        run_eval(trainer.model, codec, fe, eval_raw, device, args.eval_batch_size, args.out_dir)


if __name__ == "__main__":
    main()
