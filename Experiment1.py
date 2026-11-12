#!/usr/bin/env python
"""
openai/whisper-large-v3 + a custom BPE tokenizer (default 100 tokens) for Tigrigna ASR.

Whisper's built-in ~51.8k-token vocabulary is replaced by a small BPE trained on text/train/text.
The decoder's token embedding and output projection are re-created for the new vocabulary
(new ids: BPE tokens, then <sot>, then <eos>), then the model is fine-tuned and evaluated.

Pipeline: train BPE -> swap vocab -> fine-tune -> greedy-decode eval split (WER / CER / no-space WER).

Training modes:
  --train_mode lora    (default) encoder + decoder frozen, LoRA on the decoder + new embedding/output head.
                       Needs `pip install peft`. Base weights are loaded in bf16 to fit smaller GPUs.
  --train_mode decoder  freeze the encoder, fully train the decoder (needs a large GPU, ~24 GB+).

Install:  pip install torch transformers tokenizers peft librosa soundfile jiwer pandas tqdm accelerate
Usage:    python whisper_bpe.py --max_samples 50 --epochs 1       # smoke test
          python whisper_bpe.py                                    # full run (train + eval)
          python whisper_bpe.py --eval_only                        # re-evaluate ./whisper-bpe100/final
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
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

DATA_ROOT = "/share/documents/solomon/TigrignaASR"
MODEL_NAME = "openai/whisper-large-v3"
SR = 16000
MAX_TARGET = 440  # Whisper decoder has 448 positions

log = logging.getLogger("whisper_bpe")


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




def special_ids(codec):
    """New-vocab layout: [0..V-1] BPE tokens (incl. [PAD],[UNK]), V = <sot>, V+1 = <eos>."""
    V = codec.vocab_size
    return V, V + 1


def filter_samples(samples, codec, max_sec, split):
    kept, total, d_long, d_bad, d_tok = [], 0.0, 0, 0, 0
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
        elif len(codec.encode(t)) + 1 > MAX_TARGET:
            d_tok += 1
        else:
            kept.append((p, t))
            total += dur
    log.info(f"[{split}] kept {len(kept)} ({total / 3600:.2f} h) | dropped: {d_long} too long, "
             f"{d_tok} too many tokens, {d_bad} unreadable/empty")
    return kept


def amp_dtype():
    return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16


class ASRDataset(Dataset):
    def __init__(self, samples, codec, fe):
        self.samples, self.codec, self.fe = samples, codec, fe
        self.eos = special_ids(codec)[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        wav, _ = librosa.load(path, sr=SR, mono=True)
        feats = self.fe(wav, sampling_rate=SR, return_tensors="np").input_features[0]
        # labels = text tokens + <eos>; the model right-shifts them and prepends <sot> itself
        return {"input_features": feats, "labels": self.codec.encode(text) + [self.eos]}


class Collator:
    def __call__(self, feats):
        x = torch.tensor(np.stack([f["input_features"] for f in feats]))
        L = max(len(f["labels"]) for f in feats)
        y = torch.tensor([f["labels"] + [-100] * (L - len(f["labels"])) for f in feats])
        return {"input_features": x, "labels": y}


# ----------------------------------------------------------------------------
# Model surgery: new vocabulary
# ----------------------------------------------------------------------------
def swap_vocabulary(model, codec):
    sot, eos = special_ids(codec)
    n, d = eos + 1, model.config.d_model
    emb = nn.Embedding(n, d)
    head = nn.Linear(d, n, bias=False)
    nn.init.normal_(emb.weight, std=0.02)
    nn.init.normal_(head.weight, std=0.02)
    model.model.decoder.embed_tokens = emb
    model.proj_out = head                      # untied from the embedding
    c = model.config
    c.vocab_size, c.tie_word_embeddings = n, False
    c.decoder_start_token_id, c.bos_token_id, c.eos_token_id, c.pad_token_id = sot, sot, eos, codec.pad_id
    c.forced_decoder_ids, c.suppress_tokens, c.begin_suppress_tokens = None, [], None
    model.generation_config = GenerationConfig(decoder_start_token_id=sot, bos_token_id=sot,
                                               eos_token_id=eos, pad_token_id=codec.pad_id)
    log.info(f"New vocabulary: {n} ids ({codec.vocab_size} BPE + <sot>={sot} + <eos>={eos})")
    return model


def get_inner(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


@torch.no_grad()
def greedy_decode(inner, feats, codec, max_new):
    sot, eos = special_ids(codec)
    enc = inner.model.encoder(feats).last_hidden_state
    B, dev = feats.size(0), feats.device
    ban = [codec.pad_id, codec.tok.token_to_id("[UNK]"), sot]
    cur = torch.full((B, 1), sot, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    past, out = None, []
    for _ in range(max_new):
        o = inner(encoder_outputs=BaseModelOutput(last_hidden_state=enc), decoder_input_ids=cur,
                  past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1].float()
        logits[:, ban] = -float("inf")
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        out.append(nxt)
        done |= nxt == eos
        if bool(done.all()):
            break
        cur = nxt[:, None]
    return torch.stack(out, 1).tolist()


def transcribe(model, codec, fe, samples, device, batch_size, max_new, desc="Decoding", log_every=10):
    inner = get_inner(model)
    inner.to(device).eval()
    wdtype = inner.model.encoder.conv1.weight.dtype
    eos = special_ids(codec)[1]
    preds, refs, paths, failed = [], [], [], 0
    nb = (len(samples) + batch_size - 1) // batch_size
    t0 = time.time()
    for bi in tqdm(range(nb), desc=desc):
        batch = samples[bi * batch_size:(bi + 1) * batch_size]
        feats, keep = [], []
        for p, t in batch:
            try:
                wav = librosa.load(p, sr=SR, mono=True)[0]
                feats.append(fe(wav, sampling_rate=SR, return_tensors="np").input_features[0])
                keep.append((p, t))
            except Exception as e:
                failed += 1
                log.warning(f"Failed to load {p}: {e}")
        if not feats:
            continue
        x = torch.tensor(np.stack(feats)).to(device=device, dtype=wdtype)
        with torch.autocast(device_type="cuda", dtype=amp_dtype(), enabled=(device == "cuda")):
            ids = greedy_decode(inner, x, codec, max_new)
        for row, (p, t) in zip(ids, keep):
            row = row[:row.index(eos)] if eos in row else row
            preds.append(codec.decode(row))
            refs.append(t)
            paths.append(p)
        if bi % log_every == 0:
            log.info(f"batch {bi + 1}/{nb} | running WER {jiwer.wer(refs, preds):.4f} | "
                     f"{len(refs) / (time.time() - t0):.2f} utt/s")
            log.debug(f"  REF: {refs[-1]}")
            log.debug(f"  HYP: {preds[-1]}")
    return preds, refs, paths, failed


def run_eval(model, codec, fe, samples, device, batch_size, max_new, out_dir):
    t0 = time.time()
    preds, refs, paths, failed = transcribe(model, codec, fe, samples, device, batch_size, max_new, "Evaluating")
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


class LogCallback(TrainerCallback):
    def __init__(self, codec, fe, dev_samples, device, bs, max_new):
        self.t0 = time.time()
        self.codec, self.fe, self.dev, self.device, self.bs, self.max_new = codec, fe, dev_samples, device, bs, max_new

    def on_train_begin(self, args, state, control, **kw):
        log.info(f"Training started: {state.max_steps} optimizer steps")

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            log.info(f"step {state.global_step}/{state.max_steps} | epoch {state.epoch:.2f} | "
                     f"{(time.time() - self.t0) / 60:.1f} min | {logs}")

    def on_save(self, args, state, control, **kw):
        log.info(f"Checkpoint saved at step {state.global_step}")

    def on_evaluate(self, args, state, control, model=None, **kw):
        if model is None or not self.dev:
            return
        preds, refs, _, _ = transcribe(model, self.codec, self.fe, self.dev, self.device, self.bs,
                                       self.max_new, "Dev WER", log_every=10 ** 9)
        log.info(f"[dev subset of {len(refs)}] WER {jiwer.wer(refs, preds):.4f} | CER {jiwer.cer(refs, preds):.4f}")
        log.info(f"  sample REF: {refs[0]}\n  sample HYP: {preds[0]}")
        get_inner(model).train()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default="./whisper-bpe100")
    ap.add_argument("--vocab_size", type=int, default=100, help="BPE vocabulary size")
    ap.add_argument("--decompose", choices=["auto", "yes", "no"], default="auto",
                    help="split Ethiopic syllables into consonant+vowel before BPE")
    ap.add_argument("--train_mode", choices=["lora", "decoder"], default="lora")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 (lora) / 5e-5 (decoder)")
    ap.add_argument("--max_sec", type=float, default=15.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--dev_wer_samples", type=int, default=50, help="dev utterances decoded at each eval step")
    ap.add_argument("--max_samples", type=int, default=None, help="debug: limit samples per split")
    ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate <out_dir>/final")
    ap.add_argument("--no_eval", action="store_true", help="skip the eval run after training")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "run.log"))
    if args.lr is None:
        args.lr = 1e-3 if args.train_mode == "lora" else 5e-5
    log.info(f"Args: {vars(args)}")
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    log.info(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)}, "
                                    f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB)" if cuda else ""))
    lim = (lambda x, n=None: x[:n or args.max_samples] if args.max_samples else x)
    final = os.path.join(args.out_dir, "final")
    load_dtype = amp_dtype() if cuda and args.train_mode == "lora" else torch.float32

    # ------------------------------------------------------------------ eval only
    if args.eval_only:
        codec = BPECodec.load(final)
        fe = WhisperFeatureExtractor.from_pretrained(final)
        model = WhisperForConditionalGeneration.from_pretrained(
            final, torch_dtype=amp_dtype() if cuda else torch.float32)
        log.info(f"Loaded model + BPE codec from {final} (vocab {codec.vocab_size}, decompose={codec.decomp})")
        run_eval(model, codec, fe, lim(load_kaldi(args.data_root, "eval")), device,
                 args.eval_batch_size, args.max_new_tokens, args.out_dir)
        return

    if not cuda:
        log.warning("No GPU detected: fine-tuning whisper-large-v3 on CPU is impractically slow.")

    # ------------------------------------------------------------------ data + BPE
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    log.info(f"Feature extractor: {fe.feature_size} mel bins, 30 s windows")
    train_raw = lim(load_kaldi(args.data_root, "train"))
    dev_raw = lim(load_kaldi(args.data_root, "dev"), max((args.max_samples or 0) // 5, 5))
    eval_raw = lim(load_kaldi(args.data_root, "eval"))

    codec = train_codec([t for _, t in train_raw], args.vocab_size, args.decompose)
    codec.save(args.out_dir)
    vocab = sorted(codec.tok.get_vocab().items(), key=lambda x: x[1])
    log.info(f"BPE vocab size: {codec.vocab_size} (requested {args.vocab_size})")
    if codec.vocab_size > args.vocab_size:
        log.warning(f"Vocab is {codec.vocab_size} > {args.vocab_size}: the base alphabet alone is larger than "
                    f"requested; try --decompose yes.")
    log.info(f"Tokens: {[t for t, _ in vocab]}")
    ok = sum(codec.decode(codec.encode(t)) == normalize_text(t) for _, t in train_raw[:200])
    log.info(f"Round-trip check (encode->decode) exact on {ok}/{min(200, len(train_raw))} train sentences")
    for _, t in train_raw[:2]:
        ids = codec.encode(t)
        log.info(f"  text: {t}\n  ids ({len(ids)}): {ids[:30]}\n  tokens: {[codec.tok.id_to_token(i) for i in ids[:15]]}")

    train = filter_samples(train_raw, codec, args.max_sec, "train")
    dev = filter_samples(dev_raw, codec, args.max_sec, "dev")
    train_ds, dev_ds = ASRDataset(train, codec, fe), ASRDataset(dev, codec, fe)

    # ------------------------------------------------------------------ model
    log.info(f"Loading {args.model} in {load_dtype} (train_mode={args.train_mode})")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=load_dtype, low_cpu_mem_usage=True)
    log.info(f"Total parameters before vocab swap: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    model = swap_vocabulary(model, codec)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:  # older transformers
        model.gradient_checkpointing_enable()

    if args.train_mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            log.error("peft is required for --train_mode lora:  pip install peft")
            sys.exit(1)
        for p in model.parameters():
            p.requires_grad = False
        cfg = LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
            target_modules=r".*decoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|"
                           r"encoder_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)",
            modules_to_save=["embed_tokens", "proj_out"],   # the new vocabulary layers are fully trained + saved
        )
        model = get_peft_model(model, cfg)
    else:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable parameters: {n_tr / 1e6:.1f}M of {n_all / 1e9:.2f}B")

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
        bf16=cuda and amp_dtype() == torch.bfloat16,
        fp16=cuda and amp_dtype() == torch.float16,
        eval_strategy="steps",      # transformers < 4.41: use evaluation_strategy
        eval_steps=every,
        save_steps=every,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        label_names=["labels"],
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )
    dev_sub = dev[:args.dev_wer_samples]
    trainer = Trainer(
        model=model, args=targs, data_collator=Collator(),
        train_dataset=train_ds, eval_dataset=dev_ds,
        callbacks=[LogCallback(codec, fe, dev_sub, device, args.eval_batch_size, args.max_new_tokens)],
    )
    trainer.train()
    if cuda:
        log.info(f"Peak GPU memory during training: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    # ------------------------------------------------------------------ save
    model = trainer.model
    if args.train_mode == "lora":
        try:
            model = model.merge_and_unload()      # fold LoRA into the base weights -> plain Whisper model
        except Exception as e:
            log.error(f"merge_and_unload failed ({e}); saving the LoRA adapter only")
            model.save_pretrained(os.path.join(args.out_dir, "final_adapter"))
    get_inner(model).config.use_cache = True
    get_inner(model).save_pretrained(final)
    fe.save_pretrained(final)
    codec.save(final)
    log.info(f"Final model + BPE codec saved to {final}")

    # ------------------------------------------------------------------ eval
    if not args.no_eval:
        log.info("Running final evaluation on the eval split")
        run_eval(model, codec, fe, eval_raw, device, args.eval_batch_size, args.max_new_tokens, args.out_dir)


if __name__ == "__main__":
    main()
