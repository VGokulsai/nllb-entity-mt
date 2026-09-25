# nllb-entity-mt

A reproduction of **"Zero at SemEval-2025 Task 2: Entity-Aware Machine Translation"** (Gundam, Marri, Malladi, Mamidi — LTRC, IIIT Hyderabad). The paper fine-tunes a small translation model to translate **named entities** (movie titles, place names, people) correctly, which general models get wrong even when the rest of the sentence is fine.

I reproduced the core method for **one language pair, English → Spanish**: fine-tune `facebook/nllb-200-distilled-600M` with **LoRA** on the SemEval-2025 EA-MT data, and measure whether it improves translation over the base model.

Everything runs in one Colab notebook on a free T4 GPU: [`nllb_lora_eamt_es.ipynb`](nllb_lora_eamt_es.ipynb).

## What the method is

Named entities are hard for machine translation because the correct translation is often *not* a word-for-word one. For example:

- English: *"What kind of artwork is The Signal-Man?"*
- Base NLLB: *"¿Qué tipo de obra de arte es **El Hombre de la Señal**?"* (literal — wrong)
- Correct: *"¿Qué tipo de obra artística es **El guardavía**?"* (the real Spanish title)

The idea: instead of a huge general model, fine-tune a small specialist. LoRA trains under 1% of the model's parameters (here **4.7M of 620M, 0.76%**), so it fits on a free GPU.

## What I reproduced, and what I changed

- **Same:** model (NLLB-600M), method (LoRA fine-tune), task (en→es EA-MT), evaluation on the official test set.
- **Changed, on purpose:** the paper builds a "silver" training set by re-translating everything through the paid Google Translate API. I skipped that and trained directly on the benchmark's provided gold translations (the `validation` split, 739 examples). This removes a paid dependency and tests the core fine-tuning claim more directly.
- Trained on `validation` (739), evaluated on `test` (5,338). Different examples — no data leakage.
- Data: [`sapienzanlp/ea-mt-benchmark`](https://huggingface.co/datasets/sapienzanlp/ea-mt-benchmark), config `en-es`.

## Results

Fine-tuning the base model with LoRA, measured on the full 5,338-example test set:

| Metric | Base NLLB-600M | + LoRA fine-tune | Change |
|---|---|---|---|
| BLEU (full test, 5,338) | 60.22 | **65.25** | **+5.0** (~8% relative) |
| Entity accuracy (300-example estimate) | 19.7% | **23.7%** | **+4.0 pts** (~20% relative) |

The main finding: **fine-tuning helped named-entity accuracy proportionally more than it helped overall BLEU** (~20% vs ~8% relative), which matches the paper's argument that a specialist fine-tune targets exactly the entity problem.

## What this is not (honest limits)

- **Not a claim that I beat the paper.** The paper reports Spanish BLEU 59.14, and my numbers run higher — but they were computed with a different BLEU implementation (sacreBLEU) than the paper's, and BLEU scores from different tools are not directly comparable. The trustworthy result here is the **within-experiment** comparison (base vs fine-tuned, identical metric and data): **+5 BLEU**. The cross-paper number is not a fair head-to-head.
- **The entity metric is a crude substring match**, not the paper's official M-ETA, so the absolute ~20% is rough. Both models are scored the same crude way, so the *direction* (fine-tuning up) is what to trust, not the exact figure. It's a 300-example estimate.
- **One language only** (Spanish). The paper covers ten; Spanish is high-resource, so the base model already scores ~60 and fine-tuning's absolute gains are smaller than they would be on a harder language.

## Run it

Open [`nllb_lora_eamt_es.ipynb`](nllb_lora_eamt_es.ipynb) in Google Colab, set the runtime to a T4 GPU, and run the cells top to bottom. It installs its dependencies, downloads the data and model, fine-tunes, and prints the numbers above. The full-test evaluation cell takes ~15–20 minutes; everything else is quick.

## Credits

- Method reproduced from: Gundam, Marri, Malladi, Mamidi. *Zero at SemEval-2025 Task 2: Entity-Aware Machine Translation.* SemEval-2025. https://aclanthology.org/2025.semeval-1.157/
- Task and data: Conia, Li, Navigli, Potdar. *SemEval-2025 Task 2: Entity-Aware Machine Translation.* Dataset: `sapienzanlp/ea-mt-benchmark`.
- Model: `facebook/nllb-200-distilled-600M` (NLLB, Costa-jussà et al. 2022).
