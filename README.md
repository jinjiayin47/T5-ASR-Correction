# T5-Lev: A Hybrid Framework for ASR Error Correction and Speech Translation

> Improving Automatic Speech Recognition (ASR) outputs using Transformer-based text generation and Levenshtein Distance.

## Overview

Automatic Speech Recognition (ASR) systems often generate transcription errors due to accents, background noise, pronunciation variations, and speaking speed. These errors not only reduce transcription quality but also propagate into downstream machine translation systems.

This project proposes **T5-Lev**, a hybrid post-processing framework that combines a **Transformer-based T5 model** with **Levenshtein Distance** to detect and correct ASR errors before machine translation.

The framework improves transcription quality and provides cleaner input for downstream translation models.

---

## Pipeline

```
Speech Audio
      │
      ▼
Wav2Vec2 ASR
      │
      ▼
Error Detection
(Levenshtein Distance)
      │
      ▼
T5 Error Correction
      │
      ▼
MarianMT Translation
      │
      ▼
Evaluation
(CER, WER, METEOR)
```

---

## Features

- Automatic Speech Recognition using **Wav2Vec2**
- Transformer-based text correction using **T5**
- Similarity-based error detection with **Levenshtein Distance**
- English → Chinese translation using **MarianMT**
- End-to-end evaluation with multiple NLP metrics

---

## Dataset

The project uses the **CoVoST 2** multilingual speech translation dataset.

Each sample contains:

- Speech audio
- English transcription
- Chinese translation

For computational efficiency, approximately **1/20 of the original dataset** was used during experimentation.

---

## Models

| Module | Model |
|---------|------|
| ASR | facebook/wav2vec2-large-960h |
| Error Correction | T5-base |
| Machine Translation | Helsinki-NLP/opus-mt-en-zh |

---

## Methodology

### 1. Speech Recognition

Speech signals are converted into English transcripts using the pre-trained Wav2Vec2 model.

### 2. Error Detection

Potential spelling errors are identified by comparing ASR outputs with reference transcripts using Levenshtein Distance.

Only word pairs satisfying

```
0.75 < similarity < 1
```

are selected as correction candidates.

### 3. Error Correction

The detected ASR outputs are reformulated into a text-to-text generation task:

```
fix ASR errors:
```

The T5 model generates corrected sentences with improved spelling, punctuation, and readability.

### 4. Machine Translation

Corrected transcripts are translated into Chinese using MarianMT.

### 5. Evaluation

The framework is evaluated using:

- Word Error Rate (WER)
- Character Error Rate (CER)
- METEOR

---

## Experimental Results

| Metric | Original ASR | Corrected ASR |
|---------|-------------:|--------------:|
| CER | 0.0935 | 0.0746 |
| WER | 0.2955 | 0.1716 |
| METEOR | 0.4129 | 0.4461 |

The proposed framework achieved:

- **41.2% reduction in Word Error Rate (WER)**
- Lower Character Error Rate (CER)
- Improved machine translation quality measured by METEOR

---

## Repository Structure

```
.
├── notebook.ipynb
├── code.py
├── README.md
├── figures/
│   ├── pipeline.png
│   ├── architecture.png
│   └── results.png
├── requirements.txt
└── data/
```

---

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Torchaudio
- NLTK
- Jieba
- Levenshtein
- Pandas
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

1. Download the CoVoST dataset.
2. Configure the dataset paths.
3. Run the notebook or Python script.

```bash
python code.py
```

or

```bash
jupyter notebook
```

---

## Future Improvements

- Fine-tune larger language models (Flan-T5, UL2, Llama)
- Support multilingual correction beyond English
- Replace rule-based similarity filtering with neural error detection
- Evaluate on larger speech translation benchmarks
- Integrate Retrieval-Augmented Generation (RAG) for context-aware correction

---

## Citation

If you find this project useful, please cite:

```
Yin, J.
T5-Lev: A Hybrid Framework for ASR Error Correction and Speech Translation.
2025.
```

---

## Author

**Jinjia Yin**

M.Sc. Student  
Institut Polytechnique de Paris

Research Interests:

- Natural Language Processing
- Speech Translation
- Large Language Models
- Machine Learning
