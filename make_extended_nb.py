# Builds extended_eval.ipynb: 3 seeds, full-test official metrics (M-ETA, COMET-22, overall), per-entity-type breakdown.
# Run: py -3 make_extended_nb.py
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Extended evaluation: NLLB-600M + LoRA on SemEval-2025 EA-MT (English→Spanish)

Same setup as `nllb_lora_eamt_es.ipynb`, but evaluated the way the task scores systems:
- **3 training runs** (seeds 1, 2, 3), reported as mean ± std, so the gain isn't one lucky run
- **all 5,338 test sentences** for every metric
- the task's official metrics: **M-ETA** (entity accuracy), **COMET-22**, and their harmonic mean (the leaderboard score), plus sacreBLEU
- **M-ETA by entity type** (movie, place, person, ...)

Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. Takes about an hour.
At the end it downloads `results.zip` (the numbers + every prediction). That file is what the paper is written from.""")

code("""# 1. installs (torchao conflicts with peft on Colab, so it goes first)
!pip -q uninstall -y torchao
!pip -q install datasets sacrebleu peft sentencepiece unbabel-comet""")

code("""# 2. settings: identical to the original notebook, plus seeds
SEEDS = [1, 2, 3]
MODEL = "facebook/nllb-200-distilled-600M"
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
TRAIN = dict(per_device_train_batch_size=8, learning_rate=2e-4, num_train_epochs=3, fp16=True)
MAX_NEW_TOKENS, GEN_BS = 128, 32

from datasets import load_dataset
ds = load_dataset("sapienzanlp/ea-mt-benchmark", "en-es")
train_split, test = ds["validation"], ds["test"]
print(len(train_split), "train (validation split) |", len(test), "test")
assert not set(train_split["id"]) & set(test["id"]), "train and test overlap"
""")

code("""# 3. metrics, exactly as the task defines them
# M-ETA (task paper Eq. 2; official ea-mt-eval notebook): correct if any gold mention, casefolded,
# is a substring of the casefolded prediction.
def meta_hits(preds, data):
    return [any(t["mention"] and t["mention"].casefold() in (p or "").casefold() for t in ex["targets"])
            for p, ex in zip(preds, data)]

def meta(preds, data):
    h = meta_hits(preds, data)
    return 100 * sum(h) / len(h)

def meta_by_type(preds, data):
    per = {}
    for hit, ex in zip(meta_hits(preds, data), data):
        for t in set(ex["entity_types"]) or {"(none)"}:
            n, k = per.get(t, (0, 0))
            per[t] = (n + 1, k + hit)
    return {t: {"n": n, "meta": 100 * k / n} for t, (n, k) in sorted(per.items(), key=lambda x: -x[1][0])}

def refs_of(data):
    # sacreBLEU wants one list per reference slot; short rows are padded with their first reference
    m = max(len(ex["targets"]) for ex in data)
    return [[(ex["targets"][j] if j < len(ex["targets"]) else ex["targets"][0])["translation"] for ex in data] for j in range(m)]

def overall(comet, m):   # task paper Eq. 3, both on a 0-100 scale
    return 2 * comet * m / (comet + m) if comet + m else 0.0

def best_of_refs(scores, owners, n):
    # COMET is scored against every reference; each sentence keeps its best (official scoring)
    best = [float("-inf")] * n
    for s, i in zip(scores, owners):
        best[i] = max(best[i], s)
    return best

# quick self-check on toy data
_toy = [{"targets": [{"mention": "El guardavía", "translation": "x"}], "entity_types": ["Artwork"]},
        {"targets": [{"mention": "Roma", "translation": "y"}, {"mention": "Rome", "translation": "z"}], "entity_types": ["Place"]}]
assert meta(["es EL GUARDAVÍA hoy", "nothing"], _toy) == 50.0
assert meta_by_type(["x", "rome!"], _toy)["Place"]["meta"] == 100.0
assert best_of_refs([0.2, 0.9, 0.5], [0, 0, 1], 2) == [0.9, 0.5]
assert round(overall(90.0, 30.0), 2) == 45.0
print("metric self-checks pass")
""")

code("""# 4. translation helper (greedy decoding, the model's defaults, same as the original notebook)
import torch, gc, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

tok = AutoTokenizer.from_pretrained(MODEL)
def load_base():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL).to("cuda")

@torch.no_grad()
def translate_all(model, texts):
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length-sorted batches run faster; order restored below
    out = [None] * len(texts)
    tok.src_lang = "eng_Latn"
    for s in range(0, len(order), GEN_BS):
        idx = order[s:s + GEN_BS]
        b = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True).to("cuda")
        g = model.generate(**b, forced_bos_token_id=tok.convert_tokens_to_ids("spa_Latn"), max_new_tokens=MAX_NEW_TOKENS)
        for i, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)):
            out[i] = t
    return out

sources = test["source"]
""")

code("""# 5. base model on the full test set
t0 = time.time()
model = load_base()
preds = {"base": translate_all(model, sources)}
del model; gc.collect(); torch.cuda.empty_cache()
print(f"base done in {time.time() - t0:.0f}s | M-ETA {meta(preds['base'], test):.2f}")
""")

code("""# 6. fine-tune once per seed, each time from a fresh base model, then translate the full test set
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

def preprocess(batch):
    tok.src_lang, tok.tgt_lang = "eng_Latn", "spa_Latn"
    enc = tok(batch["source"], max_length=128, truncation=True)
    enc["labels"] = tok(text_target=[t[0]["translation"] for t in batch["targets"]], max_length=128, truncation=True)["input_ids"]
    return enc
train_ds = train_split.map(preprocess, batched=True, remove_columns=train_split.column_names)

train_loss = {}
for seed in SEEDS:
    t0 = time.time()
    set_seed(seed)
    model = get_peft_model(load_base(), LoraConfig(task_type="SEQ_2_SEQ_LM", **LORA))
    args = Seq2SeqTrainingArguments(output_dir=f"lora-s{seed}", seed=seed, save_strategy="no", logging_steps=20, report_to="none", **TRAIN)
    tr = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds,
                        data_collator=DataCollatorForSeq2Seq(tok, model=model), processing_class=tok)
    train_loss[seed] = tr.train().training_loss
    preds[f"seed{seed}"] = translate_all(model, sources)
    if seed == SEEDS[0]:
        model.save_pretrained("adapter-seed1")   # kept for publishing on Hugging Face
    del model, tr; gc.collect(); torch.cuda.empty_cache()
    print(f"seed {seed}: loss {train_loss[seed]:.4f} | M-ETA {meta(preds[f'seed{seed}'], test):.2f} | {time.time() - t0:.0f}s")
""")

code("""# 7. sacreBLEU + COMET-22 for every system
import sacrebleu
from comet import download_model, load_from_checkpoint

refs = refs_of(test)
bleu = {k: sacrebleu.corpus_bleu(v, refs) for k, v in preds.items()}
bleu_sig = str(sacrebleu.BLEU().get_signature())

comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
def comet_score(hyps):
    rows, owners = [], []
    for i, (ex, h) in enumerate(zip(test, hyps)):
        for t in ex["targets"]:
            rows.append({"src": ex["source"], "mt": h, "ref": t["translation"]}); owners.append(i)
    out = comet_model.predict(rows, batch_size=64, gpus=1, progress_bar=False)
    best = best_of_refs(out.scores, owners, len(hyps))
    return 100 * sum(best) / len(best)

comet = {}
for k, v in preds.items():
    t0 = time.time(); comet[k] = comet_score(v); print(f"COMET {k}: {comet[k]:.2f} ({time.time() - t0:.0f}s)")
""")

code("""# 8. results table, per-type breakdown, and the download
import json, csv, statistics as st, zipfile, sys, transformers, peft, comet as comet_pkg
from google.colab import files

rows = {k: {"bleu": bleu[k].score, "meta": meta(v, test), "comet": comet[k]} for k, v in preds.items()}
for r in rows.values(): r["overall"] = overall(r["comet"], r["meta"])
ms = lambda key: (st.mean(rows[f"seed{s}"][key] for s in SEEDS), st.stdev(rows[f"seed{s}"][key] for s in SEEDS))

print(f"{'system':<14}{'BLEU':>8}{'M-ETA':>8}{'COMET':>8}{'Overall':>9}")
for k, r in rows.items():
    print(f"{k:<14}{r['bleu']:>8.2f}{r['meta']:>8.2f}{r['comet']:>8.2f}{r['overall']:>9.2f}")
print("fine-tuned mean ± std over seeds:", {m: f"{ms(m)[0]:.2f} ± {ms(m)[1]:.2f}" for m in ["bleu", "meta", "comet", "overall"]})

by_type = {k: meta_by_type(v, test) for k, v in preds.items()}
print("\\nM-ETA by entity type (base → seed1):")
for t, d in by_type["base"].items():
    print(f"  {t:<28} n={d['n']:<5} {d['meta']:6.1f} → {by_type['seed1'][t]['meta']:6.1f}")

results = {
    "task": "SemEval-2025 Task 2 EA-MT, en-es", "n_test": len(test), "n_train": len(train_split), "seeds": SEEDS,
    "lora": LORA, "train": TRAIN, "decoding": {"strategy": "model default (greedy)", "max_new_tokens": MAX_NEW_TOKENS},
    "systems": rows, "train_loss": train_loss, "by_type": by_type, "bleu_signature": bleu_sig,
    "comet_model": "Unbabel/wmt22-comet-da (best of references per sentence)",
    "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                 "peft": peft.__version__, "sacrebleu": sacrebleu.__version__, "comet": getattr(comet_pkg, "__version__", "?"),
                 "gpu": torch.cuda.get_device_name(0)},
}
json.dump(results, open("results.json", "w"), indent=2, ensure_ascii=False)
with open("predictions.tsv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\\t")
    w.writerow(["id", "source", "gold_mentions", "entity_types", *preds])
    for i, ex in enumerate(test):
        w.writerow([ex["id"], ex["source"], " | ".join(t["mention"] for t in ex["targets"]), " | ".join(ex["entity_types"]), *(preds[k][i] for k in preds)])
with zipfile.ZipFile("results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in ["results.json", "predictions.tsv"]: z.write(p)
    for root in ["adapter-seed1"]:
        import os
        for dp, _, fs in os.walk(root):
            for fn in fs: z.write(os.path.join(dp, fn))
files.download("results.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("extended_eval.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote extended_eval.ipynb with", len(cells), "cells")
