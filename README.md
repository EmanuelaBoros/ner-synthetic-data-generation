# NER Synthetic Data Generation

Synthetic data generation pipeline for creating **historical newspaper-style named entity recognition (NER)** examples compatible with **GLiNER** fine-tuning.

The repository generates short historical passages with entity annotations inspired by HIPE/Impresso-style historical NER guidelines, then converts them into token-span format usable by GLiNER.

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
