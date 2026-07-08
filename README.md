# NER Synthetic Data Generation

Synthetic data generation pipeline for creating **historical newspaper-style named entity recognition (NER)** examples compatible with **GLiNER** fine-tuning.

The repository generates short historical passages with entity annotations inspired by HIPE/Impresso-style historical NER guidelines, then converts them into token-span format usable by GLiNER.

Generated dataset:

[emanuelaboros/synthetic-historical-ner-data](https://huggingface.co/datasets/emanuelaboros/synthetic-historical-ner-data)

---

## Overview

This project uses an instruction-tuned large language model to generate fully synthetic training examples for historical NER. Each generated example contains:

1. a realistic historical newspaper-style text passage;
2. a list of entity mentions;
3. entity labels from a controlled historical NER schema;
4. a converted GLiNER-compatible representation.

The main goal is to produce low-cost synthetic data for experimentation, pretraining, data augmentation, and rapid prototyping before evaluating on manually annotated historical NER benchmarks.

---

## Entity schema

The generated annotations use a simplified historical NER schema:

| Label | Meaning | Examples |
|---|---|---|
| `pers` | Person names | politicians, authors, witnesses, officers |
| `loc` | Locations | cities, countries, regions, streets, geopolitical places |
| `org` | Organizations | ministries, councils, parties, companies, universities |
| `prod` | Human or media productions | newspapers, books, laws, decrees, reports, named works |
| `time` | Temporal expressions | dates, years, months, periods |

The labels are intentionally kept coarse to avoid noisy synthetic taxonomies and to remain close to historical NER use cases.

---

## Output format

The generated data is saved in GLiNER-compatible format.

Example:

```json
{
  "tokenized_text": [
    "Le",
    "Conseil",
    "fédéral",
    "se",
    "réunit",
    "à",
    "Berne",
    "."
  ],
  "ner": [
    [1, 2, "org"],
    [6, 6, "loc"]
  ]
}
```

Each `ner` span follows the format:

```text
[start_token_index, end_token_index, entity_label]
```

Token indices are inclusive.

---

## Repository structure

```text
ner-synthetic-data-generation/
├── generate.py
├── requirements.txt
├── README.md
└── LICENSE
```

Expected output files:

```text
outputs/
├── synthetic_historical_ner_gliner.jsonl
├── synthetic_historical_ner_gliner.json
└── synthetic_historical_ner_raw.json
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/EmanuelaBoros/ner-synthetic-data-generation.git
cd ner-synthetic-data-generation
```

Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

Minimal requirements:

```txt
vllm
huggingface_hub
```

---

## Generate a small test dataset

Run a small local test first:

```bash
python generate.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-samples 5 \
  --tensor-parallel-size 1 \
  --output-jsonl outputs/synthetic_historical_ner_gliner.jsonl \
  --output-json outputs/synthetic_historical_ner_gliner.json \
  --raw-output-json outputs/synthetic_historical_ner_raw.json
```

---

## Generate and push to Hugging Face Hub

First authenticate with Hugging Face:

```bash
hf auth login
```

Then run:

```bash
python generate.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-samples 1000 \
  --tensor-parallel-size 1 \
  --output-jsonl outputs/synthetic_historical_ner_gliner.jsonl \
  --output-json outputs/synthetic_historical_ner_gliner.json \
  --raw-output-json outputs/synthetic_historical_ner_raw.json \
  --push-to-hub \
  --hub-dataset-id emanuelaboros/synthetic-historical-ner-data
```

This uploads the generated files to:

```text
https://huggingface.co/datasets/emanuelaboros/synthetic-historical-ner-data
```

---

## Suggested workflow

1. Generate a small sample, for example 5 to 20 examples.
2. Manually inspect the raw generations and converted GLiNER spans.
3. Check that all entity strings appear in the generated text.
4. Verify that labels remain within the controlled schema.
5. Generate a larger dataset.
6. Use the synthetic data for GLiNER fine-tuning or data augmentation.
7. Evaluate on real manually annotated historical NER data.

---

## Quality control

Synthetic NER data can be useful, but it should be inspected carefully. Common issues include:

- missing entity mentions;
- entity strings that do not exactly match the text;
- wrong entity types;
- overly modern language;
- unrealistic historical contexts;
- inconsistent OCR-like noise;
- duplicated examples.

The script includes post-processing to keep only valid examples where entity spans can be matched in the tokenized text.

---

## Limitations

This dataset is fully synthetic. It is not a gold-standard benchmark and should not be used as the only evaluation source for historical NER systems.

The generated data is most useful for:

- warm-starting NER models;
- testing GLiNER fine-tuning pipelines;
- comparing prompts;
- augmenting low-resource historical NER settings;
- prototyping before training on curated data.

For serious evaluation, models should be tested on manually annotated corpora.

---

## Dataset

The generated dataset is available on Hugging Face:

[emanuelaboros/synthetic-historical-ner-data](https://huggingface.co/datasets/emanuelaboros/synthetic-historical-ner-data)

---

## License

This repository is released under the MIT License.

---

## Citation

If you use this repository, please cite it as:

```bibtex
@misc{boros2026synthetichistoricalner,
  author = {Boros, Emanuela},
  title = {NER Synthetic Data Generation},
  year = {2026},
  url = {https://github.com/EmanuelaBoros/ner-synthetic-data-generation}
}
```
