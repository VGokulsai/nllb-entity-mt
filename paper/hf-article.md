---
title: "Teaching a 600M translation model to get names right: a small SemEval-2025 reproduction"
---

# Teaching a 600M translation model to get names right

Translation models are good at sentences and bad at names. Ask NLLB-600M to translate "What kind of artwork is The Signal-Man?" into Spanish and you get "¿Qué tipo de obra de arte es El Hombre de la Señal?". The grammar is fine. The title is wrong: in Spanish it's "El guardavía". Ask about the film About Joan and the model leaves it in English, where the accepted title is "À propos de Joan".

SemEval-2025 Task 2 (Entity-Aware Machine Translation) is built around exactly this problem. One of the teams, Zero from IIIT Hyderabad, tried something cheap: take the small distilled NLLB model and fine-tune it with LoRA for each language. I wanted to know if that actually helps, so I reran it for English→Spanish on a free Colab GPU.

## What I did

- Model: `facebook/nllb-200-distilled-600M`, with LoRA on the attention projections (rank 16, alpha 32). That trains 4.7M of 620M parameters, 0.76%.
- Data: [`sapienzanlp/ea-mt-benchmark`](https://huggingface.co/datasets/sapienzanlp/ea-mt-benchmark), `en-es`. I trained on the 739-sentence validation split and tested on all 5,338 test sentences. No overlap.
- Training: 3 epochs, batch size 8, learning rate 2e-4. It took 62 seconds on a T4.

The paper doesn't list its LoRA settings, so these are mine. It also built its training data by running the sentences through the Google Translate API. I skipped that and trained on the benchmark's own human translations, which is free and tests the fine-tuning claim more directly, but it's a real difference from their setup.

## Results

| | Base NLLB-600M | + LoRA | Change |
|---|---|---|---|
| BLEU, all 5,338 test sentences | 60.22 | 65.25 | +5.0 |
| Entity accuracy, first 300 sentences | 19.7% | 23.7% | +4.0 points |

BLEU went up about 8% in relative terms. Entity accuracy went up about 20%. So fine-tuning helped the names more than it helped the sentences overall, which is the whole argument of the original paper.

The absolute entity number is still low, though. Even after fine-tuning, about three out of four sentences in the sample miss the correct name. That makes sense: the model never looks anything up, and 739 examples can't teach it every film title in Spanish.

## What this doesn't show

I almost wrote "I beat the paper" because they report Spanish BLEU of 59.14 and I got 65. That comparison doesn't hold. BLEU scores from different tools and settings aren't comparable, and the paper doesn't say which split it scored. The number I trust is the one inside my own experiment: same base model, same data, same metric, +5 BLEU.

Other limits I know about:

- It's one training run. I don't know yet how much the gain moves with a different random seed.
- My entity check is a case-sensitive substring match on 300 sentences. The official metric (M-ETA) ignores case and I haven't run it on the full test set.
- I didn't compute COMET, the task's other official metric.
- It's one language pair out of the paper's ten, and Spanish is one of the easier ones for NLLB.

The repo has a second notebook that fixes the first three: three seeds, official M-ETA and COMET-22 on all 5,338 sentences, and a breakdown by entity type (films, places, people and so on). When it's run I'll post the numbers and the fine-tuned adapter here on the Hub.

## Links

- Full report (Zenodo): [doi.org/10.5281/zenodo.23008141](https://doi.org/10.5281/zenodo.23008141)
- Code and notebook: [github.com/VGokulsai/nllb-entity-mt](https://github.com/VGokulsai/nllb-entity-mt)
- Original paper: [Zero at SemEval-2025 Task 2](https://aclanthology.org/2025.semeval-1.157/) (Gundam, Marri, Malladi, Mamidi)
- Task paper: [SemEval-2025 Task 2: Entity-Aware Machine Translation](https://aclanthology.org/2025.semeval-1.326/) (Conia, Li, Navigli, Potdar)

I'm a high-school student in Hyderabad and this is my first reproduction, written up with help from Claude. If you work on MT and spot something wrong, I'd like to hear it.
