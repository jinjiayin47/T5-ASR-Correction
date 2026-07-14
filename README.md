# T5-Lev: Dynamic Error-Correction Enhanced Speech Translation

> Undergraduate thesis project — Shanghai Normal University, School of Business
> Author: Jinjia Yin (印晋佳) · Advisor: Liu Yuan · April 2025

A joint optimization pipeline that corrects Automatic Speech Recognition (ASR)
errors before they propagate into machine translation. The core contribution
is **T5-Lev**, an error-correction module that combines a fine-tuned **T5**
text-generation model with **Levenshtein-distance-based similarity scoring**
to distinguish real ASR mistakes (misspellings, homophone confusions) from
harmless variation, then repairs them — producing cleaner, more readable
input for downstream translation.

## Overview

Speech recognition systems frequently produce errors due to accents,
background noise, and speaking rate. These errors don't just hurt ASR
readability — they compound and degrade the quality of any translation
built on top of them. This project addresses that problem with a four-stage
pipeline:

```
Audio → ASR (Wav2Vec2) → T5-Lev Error Correction → Translation (MarianMT) → Evaluation
```

1. **ASR** — `facebook/wav2vec2-large-960h` transcribes English speech to text.
2. **T5-Lev correction** — a fine-tuned T5 model rewrites the ASR output,
   guided by a Levenshtein-similarity filter that flags likely real errors
   (0.75 < similarity < 1, excluding simple suffix variants like `-ing`/`-ed`/`-s`)
   so the model learns to fix genuine mistakes without over-correcting.
3. **Translation** — `Helsinki-NLP/opus-mt-en-zh` (MarianMT) translates the
   corrected English text into Chinese, with punctuation normalization.
4. **Evaluation** — WER and CER measure ASR/correction quality; METEOR
   measures translation quality (chosen over BLEU because it better handles
   Chinese synonymy and flexible word order).

## Results

Evaluated on 2,910 test samples from the CoVoST (Common Voice-based Speech
Translation) dataset:

| Metric | Original ASR | After T5-Lev Correction | Improvement |
|---|---|---|---|
| WER  | 0.2955 | 0.1716 | **−41.2%** |
| CER  | 0.0935 | 0.0746 | −20.2% |
| METEOR (translation) | 0.4129 | 0.4461 | +8.0% |

Example correction:

| | Text |
|---|---|
| Original ASR | `there is only one ware to learn the alchemist answered` |
| T5-Lev corrected | `"there is only one way to learn," the alchemist answered.` |
| Reference | `"there is only one way to learn," the alchemist answered.` |
| Translation METEOR | 0.0625 → **0.6310** |

## Repository Structure

```
.
├── README.md
├── notebook/
│   └── asr_correction_translation_pipeline.ipynb   # full pipeline, runnable end-to-end
├── thesis/
│   └── thesis.pdf                                  # full write-up (methodology, related work, references)
└── data/                                            # not included — see Dataset section
```

## Dataset

Experiments use a 1/20 subset of **[CoVoST](https://github.com/facebookresearch/covost)**
(English → Chinese), split 80/20 into train/test:

| Split | Duration (h) | Utterances | English words | Chinese chars |
|---|---|---|---|---|
| Train | 13.98 | 11,637 | 101,537 | 13,165 |
| Test  | 3.53  | 2,910  | 25,662  | 3,315  |

The full dataset is not redistributed in this repo. Download it from the
[CoVoST release page](https://github.com/facebookresearch/covost) and arrange
it as:

```
data/
├── train/{audio,transcripts,translations}/
└── test/{audio,transcripts,translations}/
```

Each split expects `<name>.mp3` audio files paired with `<name>.txt`
transcript and translation files sharing the same base filename.

## Installation

```bash
git clone https://github.com/<your-username>/t5-lev-asr-correction.git
cd t5-lev-asr-correction
pip install -r requirements.txt
```

`requirements.txt`:
```
torch
torchaudio
transformers
python-Levenshtein
nltk
jieba
```

## Usage

Open `notebook/asr_correction_translation_pipeline.ipynb` in Jupyter and run
the cells in order. Update the dataset paths in the "Load Training and Test
Data" section to point at your local `data/train` and `data/test` folders,
then run through training, correction, translation, and evaluation.

```bash
jupyter notebook notebook/asr_correction_translation_pipeline.ipynb
```

## Method Summary

**Similarity filtering.** To avoid "false-positive" error pairs (e.g. tagging
`there`→`the` as an error when both are valid words in other contexts), word
pairs are only treated as correctable ASR errors when:
- the ASR output and reference have the same word count for that utterance, and
- their Levenshtein similarity falls strictly between 0.75 and 1.0, and
- they aren't simple suffix variants (`-ing`, `-er`, `-ed`, `-s`).

$$\text{similarity} = 1 - \frac{\text{Levenshtein distance}}{\max(\text{len}_1, \text{len}_2)}$$

**Correction model.** `T5ForConditionalGeneration` (t5-base) is fine-tuned
with the task prefix `"fix ASR errors: "`, using AdamW (lr = 3e-5), batch
size 8, 10 epochs, gradient clipping at 1.0.

**Evaluation metrics.** WER and CER (edit-distance-based) for ASR/correction
quality; METEOR (with a fragmentation penalty, γ=0.5, β=3.0) for translation
quality, using `jieba` for Chinese tokenization.

## Limitations & Future Work

- Trained on a small (1/20) subset of CoVoST due to compute constraints (CPU-only, 8GB RAM); results may not generalize to full-scale, noisier, or accented data.
- Doesn't yet account for regional-accent variation present in the underlying dataset.
- Future directions: scaling to the full dataset and additional language pairs, incorporating multi-task/reinforcement learning for longer and more complex sentences, and exploring dictionary-constrained correction to reduce dependence on reference transcripts.

## Citation

If you use this work, please cite the thesis:

```
Yin, J. (2025). Research on Joint Optimization Method for Speech Translation
Based on Dynamic Error Correction Enhancement. Undergraduate thesis,
Shanghai Normal University.
```

## Acknowledgments

Thesis advisor: Liu Yuan, Shanghai Normal University, School of Business.
Built on [Wav2Vec2](https://arxiv.org/abs/2006.11477),
[T5](https://arxiv.org/abs/1910.10683), and
[MarianMT](https://aclanthology.org/P18-4020/).
