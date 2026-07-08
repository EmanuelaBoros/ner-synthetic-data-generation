from __future__ import annotations
from huggingface_hub import HfApi
from pathlib import Path
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


def upload_to_hub(repo_id, files, private=False):
    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    for file_path in files:
        file_path = Path(file_path)
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                repo_type="dataset",
            )

    print(f"Uploaded files to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------
# Tokenization and span extraction
# ---------------------------------------------------------------------


def tokenize_text(text: str) -> list[str]:
    """
    Simple tokenizer compatible with GLiNER-style token-span data.
    Keeps punctuation as separate tokens.
    """
    return re.findall(r"\w+(?:[-_’']\w+)*|\S", text, flags=re.UNICODE)


def normalize_entity_type(label: str) -> str:
    """
    Normalize labels to HIPE/Impresso-style coarse entity types.
    """
    label = label.lower().strip()
    label = label.replace("_", " ").replace("-", " ")

    mapping = {
        "person": "pers",
        "pers": "pers",
        "per": "pers",
        "location": "loc",
        "place": "loc",
        "loc": "loc",
        "organization": "org",
        "organisation": "org",
        "org": "org",
        "product": "prod",
        "human production": "prod",
        "media product": "prod",
        "newspaper": "prod",
        "journal": "prod",
        "prod": "prod",
        "date": "time",
        "time": "time",
        "period": "time",
        "temporal expression": "time",
    }

    return mapping.get(label, label)


def find_all_token_spans(tokens: list[str], entity_text: str) -> list[tuple[int, int]]:
    """
    Find all exact token-level occurrences of an entity string in the tokenized text.
    Case-insensitive.
    """
    entity_tokens = tokenize_text(entity_text)

    if not entity_tokens:
        return []

    spans = []
    lowered_tokens = [t.lower() for t in tokens]
    lowered_entity = [t.lower() for t in entity_tokens]

    n = len(lowered_entity)

    for i in range(len(tokens) - n + 1):
        if lowered_tokens[i : i + n] == lowered_entity:
            spans.append((i, i + n - 1))

    return spans


def extract_gliner_example(record: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert generated JSON into GLiNER format:
    {
      "tokenized_text": [...],
      "ner": [[start, end, label], ...]
    }
    """
    try:
        text = record["text"]
        entities = record["entities"]
    except Exception:
        return None

    if not isinstance(text, str) or not isinstance(entities, list):
        return None

    tokens = tokenize_text(text)
    ner: list[list[Any]] = []

    seen = set()

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        entity_text = ent.get("entity")
        types = ent.get("types", [])

        if not entity_text or not isinstance(types, list):
            continue

        spans = find_all_token_spans(tokens, str(entity_text))

        for start, end in spans:
            for label in types:
                label = normalize_entity_type(str(label))

                if label not in {"pers", "loc", "org", "prod", "time"}:
                    continue

                key = (start, end, label)
                if key not in seen:
                    ner.append([start, end, label])
                    seen.add(key)

    if not ner:
        return None

    ner = sorted(ner, key=lambda x: (x[0], x[1], x[2]))

    return {
        "tokenized_text": tokens,
        "ner": ner,
        "metadata": {
            "source": "synthetic",
            "guidelines": "Impresso/HIPE-style coarse NER",
        },
    }


def extract_json_from_generation(text: str) -> dict[str, Any] | None:
    """
    Robustly extract the JSON object from an LLM generation.
    Handles outputs with surrounding text or <start>/<end> markers.
    """
    text = text.strip()

    text = text.replace("<end>", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start : end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------


def create_hipe_synthetic_prompt(
    language: str = "french",
    year: int | str = "1905",
    country: str = "Switzerland",
    newspaper_section: str = "politics",
    topic: str = "municipal elections",
    ocr_noise: str = "light",
) -> str:
    """
    Create a prompt for synthetic historical newspaper NER examples.

    The generated example follows a simplified HIPE/Impresso-style coarse schema:
    - pers: person names and person mentions with titles when part of the name mention
    - loc: places, cities, countries, regions, streets, geopolitical entities
    - org: organizations, institutions, parties, companies, associations
    - prod: human/media productions, especially newspapers, journals, books, laws, reports
    - time: dates, years, historical periods, temporal expressions
    """

    return f"""
You generate synthetic named-entity-recognition data for historical newspapers.

The output must be one valid JSON object and nothing else.

Task:
Write one realistic short historical newspaper passage and annotate named entities.

Context attributes:
- language: {language}
- approximate year: {year}
- country or region: {country}
- newspaper section: {newspaper_section}
- topic: {topic}
- OCR noise level: {ocr_noise}

Entity types:
Use only these lowercase labels:

1. "pers"
   Persons, including historical figures, politicians, authors, ministers, mayors,
   military officers, artists, witnesses, or named private individuals.

2. "loc"
   Locations, including cities, countries, regions, streets, buildings when used
   primarily as places, rivers, mountains, geopolitical territories.

3. "org"
   Organizations, including governments, ministries, councils, political parties,
   companies, associations, universities, churches, committees, armies, newspapers
   when they act as institutions.

4. "prod"
   Human productions, especially newspapers, journals, books, reports, laws,
   decrees, artistic works, published documents, named articles, or named media products.

5. "time"
   Dates, years, months, days, historical periods, temporal expressions.

Annotation rules:
- Annotate only named or clearly referential entities.
- Do not annotate generic common nouns alone, such as "le ministre", "la ville",
  "le journal", unless they are part of a named expression.
- Prefer full entity mentions when possible.
- Preserve historical style.
- The text may include mild OCR-like noise only if requested.
- Make sure every entity string in the entities list appears exactly in the text.
- Avoid hallucinated labels outside the allowed set.
- Nested mentions are allowed in the entities list, but keep the output usable.

Output schema:

{{
  "text": "historical newspaper passage",
  "entities": [
    {{"entity": "surface form exactly as it appears in text", "types": ["pers"]}},
    {{"entity": "surface form exactly as it appears in text", "types": ["loc"]}}
  ]
}}

Example:

{{
  "text": "Le Conseil fédéral a reçu hier à Berne M. Louis Ruchonnet, dont le discours sera publié dans la Gazette de Lausanne du 12 mars 1885.",
  "entities": [
    {{"entity": "Conseil fédéral", "types": ["org"]}},
    {{"entity": "Berne", "types": ["loc"]}},
    {{"entity": "Louis Ruchonnet", "types": ["pers"]}},
    {{"entity": "Gazette de Lausanne", "types": ["prod"]}},
    {{"entity": "Lausanne", "types": ["loc"]}},
    {{"entity": "12 mars 1885", "types": ["time"]}}
  ]
}}

Now generate one new example.
<end>
""".strip()


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------


def generate_from_prompts(
    prompts: list[str],
    llm: LLM,
    sampling_params: SamplingParams,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate raw JSON records and processed GLiNER examples.
    """
    outputs = llm.generate(prompts, sampling_params)

    raw_records = []
    gliner_records = []

    for output in outputs:
        generated_text = output.outputs[0].text
        record = extract_json_from_generation(generated_text)

        if record is None:
            continue

        gliner_record = extract_gliner_example(record)

        if gliner_record is None:
            continue

        raw_records.append(record)
        gliner_records.append(gliner_record)

    return raw_records, gliner_records


def save_jsonl(data: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Sampling attributes
# ---------------------------------------------------------------------

LANGUAGES = ["french", "german", "english"]

COUNTRIES = [
    "Switzerland",
    "France",
    "Germany",
    "Luxembourg",
    "Belgium",
    "Italy",
    "Austria",
    "United Kingdom",
]

SECTIONS = [
    "politics",
    "local news",
    "international news",
    "culture",
    "economy",
    "legal affairs",
    "military news",
    "society",
    "science",
    "advertisements",
]

TOPICS = [
    "municipal elections",
    "railway construction",
    "public health measures",
    "diplomatic visit",
    "new theatre performance",
    "university ceremony",
    "court trial",
    "workers association meeting",
    "banking scandal",
    "agricultural exhibition",
    "colonial debate",
    "newspaper controversy",
    "military appointment",
    "church reform",
    "school inspection",
]

YEARS = list(range(1850, 1951))

OCR_NOISE_LEVELS = ["none", "light"]


def build_prompts(num_samples: int, seed: int = 13) -> list[str]:
    random.seed(seed)

    prompts = []

    for _ in range(num_samples):
        prompt = create_hipe_synthetic_prompt(
            language=random.choice(LANGUAGES),
            year=random.choice(YEARS),
            country=random.choice(COUNTRIES),
            newspaper_section=random.choice(SECTIONS),
            topic=random.choice(TOPICS),
            ocr_noise=random.choice(OCR_NOISE_LEVELS),
        )
        prompts.append(prompt)

    return prompts


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------


def print_stats(data: list[dict[str, Any]]) -> None:
    if not data:
        print("No valid examples generated.")
        return

    lengths = [len(row["tokenized_text"]) for row in data]
    num_entities = [len(row["ner"]) for row in data]

    labels = []
    for row in data:
        for _, _, label in row["ner"]:
            labels.append(label)

    print(f"Valid examples: {len(data)}")
    print(f"Average tokens: {sum(lengths) / len(lengths):.2f}")
    print(f"Average entities: {sum(num_entities) / len(num_entities):.2f}")
    print("Entity distribution:")
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic HIPE/Impresso-style NER data for GLiNER."
    )

    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="LLM name or local path.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of synthetic examples to generate.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism.",
    )

    parser.add_argument(
        "--dtype",
        default="half",
        choices=["half", "float16", "bfloat16", "float32", "auto"],
        help="Model dtype.",
    )

    parser.add_argument(
        "--output-jsonl",
        default="outputs/synthetic_hipe_gliner.jsonl",
        help="Output path for GLiNER JSONL data.",
    )

    parser.add_argument(
        "--output-json",
        default="outputs/synthetic_hipe_gliner.json",
        help="Output path for GLiNER JSON data.",
    )

    parser.add_argument(
        "--raw-output-json",
        default="outputs/synthetic_hipe_raw.json",
        help="Output path for raw LLM generations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Maximum generated tokens per example.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling.",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload generated dataset to Hugging Face Hub.",
    )

    parser.add_argument(
        "--hub-dataset-id",
        default="emanuelaboros/synthetic-historical-ner-data",
        help="Hugging Face dataset repository ID.",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload as a private dataset.",
    )

    parser.add_argument(
        "--output-jsonl",
        default="outputs/synthetic_historical_ner_gliner.jsonl",
    )

    parser.add_argument(
        "--output-json",
        default="outputs/synthetic_historical_ner_gliner.json",
    )

    parser.add_argument(
        "--raw-output-json",
        default="outputs/synthetic_historical_ner_raw.json",
    )
    args = parser.parse_args()

    print("Loading model...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["<end>"],
    )

    prompts = build_prompts(args.num_samples, seed=args.seed)

    print(f"Generating {len(prompts)} examples...")
    raw_records, gliner_records = generate_from_prompts(
        prompts=prompts,
        llm=llm,
        sampling_params=sampling_params,
    )

    save_jsonl(gliner_records, args.output_jsonl)
    save_json(gliner_records, args.output_json)
    save_json(raw_records, args.raw_output_json)

    print_stats(gliner_records)

    print(f"\nSaved GLiNER JSONL to: {args.output_jsonl}")
    print(f"Saved GLiNER JSON to: {args.output_json}")
    print(f"Saved raw generations to: {args.raw_output_json}")

    if args.push_to_hub:
        upload_to_hub(
            repo_id=args.hub_dataset_id,
            files=[
                args.output_jsonl,
                args.output_json,
                args.raw_output_json,
            ],
            private=args.private,
        )


if __name__ == "__main__":
    main()
