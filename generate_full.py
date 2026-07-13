from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from vllm import LLM, SamplingParams

COLUMNS = [
    "TOKEN",
    "NE-COARSE-LIT",
    "NE-COARSE-METO",
    "NE-FINE-LIT",
    "NE-FINE-METO",
    "NE-FINE-COMP",
    "NE-NESTED",
    "NEL-LIT",
    "NEL-METO",
    "MISC",
]

COARSE_TYPES = {"pers", "loc", "org", "prod", "time"}

FINE_TYPES = {
    "pers.ind",
    "pers.coll",
    "pers.ind.articleauthor",
    "org.adm",
    "org.ent",
    "org.ent.pressagency",
    "loc.adm.town",
    "loc.adm.reg",
    "loc.adm.nat",
    "loc.adm.sup",
    "loc.phys.geo",
    "loc.phys.hydro",
    "loc.phys.astro",
    "loc.oro",
    "loc.fac",
    "loc.add.phys",
    "loc.add.elec",
    "loc.unk",
    "prod.media",
    "prod.doctr",
    "time.date.abs",
}

COMP_TYPES = {
    "comp.name",
    "comp.title",
    "comp.function",
    "comp.func",
    "comp.qualifier",
    "comp.demonym",
}

BIO_PATTERN = re.compile(r"^[BI]-(.+)$")

LANGUAGES = ["fr", "de", "en"]
DOCUMENT_TYPES = ["newspaper"]
DATASETS = ["synthetic"]
PUBLICATION_TITLES = [
    "EXP",
    "GDL",
    "JDG",
    "IMP",
    "LUX",
    "NZZ",
    "LCD",
    "LTF",
]
SECTIONS = [
    "politics",
    "local news",
    "international affairs",
    "court notice",
    "commercial announcement",
    "cultural chronicle",
    "military report",
    "scientific note",
    "shipping news",
    "public administration",
]
TOPICS = [
    "municipal elections",
    "administrative decree",
    "railway construction",
    "court summons",
    "public health notice",
    "ministerial visit",
    "newspaper controversy",
    "theatre review",
    "trade exhibition",
    "military appointment",
    "missing person notice",
    "school ceremony",
    "workers association meeting",
    "diplomatic declaration",
]
YEARS = list(range(1798, 1951))


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------


def create_hipe_tsv_prompt(
    *,
    language: str,
    publication_title: str,
    year: int,
    document_type: str,
    section: str,
    topic: str,
    ocr_noise: str,
) -> str:
    """
    Prompt the LLM to return one complete HIPE-style TSV document as JSON rows.
    The script later validates and serializes these rows to real TSV.
    """

    return f"""
You generate synthetic HIPE-style named entity annotation data for historical newspapers.

Return exactly one valid JSON object and nothing else.

The output must contain:
1. "metadata": an object with document-level metadata.
2. "rows": a list of token-level rows.

Context:
- language: {language}
- publication title: {publication_title}
- approximate year: {year}
- document type: {document_type}
- newspaper section: {section}
- topic: {topic}
- OCR noise: {ocr_noise}

You must follow this TSV column schema exactly:
TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC

Column meanings:
- TOKEN: one token.
- NE-COARSE-LIT: literal coarse entity BIO tag: pers, loc, org, prod, time, or O.
- NE-COARSE-METO: metonymic coarse entity BIO tag, or O.
- NE-FINE-LIT: literal fine entity BIO tag, or O.
- NE-FINE-METO: metonymic fine entity BIO tag, or O.
- NE-FINE-COMP: entity component BIO tag, or O.
- NE-NESTED: nested entity BIO tag, or O.
- NEL-LIT: Wikidata QID, NIL, or _.
- NEL-METO: Wikidata QID, NIL, or _.
- MISC: _, NoSpaceAfter, EndOfLine, EndOfSentence, or a pipe-separated combination.

Allowed coarse types:
- pers
- loc
- org
- prod
- time

Allowed fine types:
- pers.ind
- pers.coll
- pers.ind.articleauthor
- org.adm
- org.ent
- org.ent.pressagency
- loc.adm.town
- loc.adm.reg
- loc.adm.nat
- loc.adm.sup
- loc.phys.geo
- loc.phys.hydro
- loc.phys.astro
- loc.oro
- loc.fac
- loc.add.phys
- loc.add.elec
- loc.unk
- prod.media
- prod.doctr
- time.date.abs

Allowed component types:
- comp.name
- comp.title
- comp.function
- comp.qualifier
- comp.demonym

Important annotation rules:
- Use BIO format: B-type for the first token of a mention, I-type for following tokens.
- Keep coarse and fine literal columns consistent.
  Example: B-pers must correspond to B-pers.ind, B-pers.coll, or B-pers.ind.articleauthor.
- Use the most specific fine label, not just the coarse type.
- Use O when no annotation applies.
- For literal entities that should be linked but no Wikidata QID is known, use NIL in NEL-LIT.
- For non-entities, use _ in NEL-LIT and NEL-METO.
- Add component labels for person titles, names, functions, qualifiers, and demonyms.
- Add comp.name for the name-bearing part of organisations, locations, and productions when useful.
- Use NE-NESTED for one-level nested entities, especially locations inside organisations or productions.
- Include at least one example with a component annotation.
- Include at least one location, one person, and one organisation or production when natural.
- Metonymy is optional, but if used, literal and metonymic columns must both be filled consistently.
  Example: "l'Élysée" literally loc.fac but metonymically org.adm.
- Keep the passage realistic for a historical newspaper.
- You may include light OCR noise only if requested, but keep the token table valid.
- Do not invent columns.
- Each row must contain all 10 columns as JSON keys.

Metadata keys:
- hipe2022:document_id
- hipe2022:date
- hipe2022:language
- hipe2022:document_type
- hipe2022:dataset
- hipe2022:original_source
- hipe2022:applicable_columns
- synthetic:publication_title
- synthetic:section
- synthetic:topic

Return JSON in this form:
{{
  "metadata": {{
    "hipe2022:document_id": "SYN-000001",
    "hipe2022:date": "{year}-01-01",
    "hipe2022:language": "{language}",
    "hipe2022:document_type": "{document_type}",
    "hipe2022:dataset": "synthetic",
    "hipe2022:original_source": "generated",
    "hipe2022:applicable_columns": "TOKEN NE-COARSE-LIT NE-COARSE-METO NE-FINE-LIT NE-FINE-METO NE-FINE-COMP NE-NESTED NEL-LIT NEL-METO MISC",
    "synthetic:publication_title": "{publication_title}",
    "synthetic:section": "{section}",
    "synthetic:topic": "{topic}"
  }},
  "rows": [
    {{
      "TOKEN": "Le",
      "NE-COARSE-LIT": "O",
      "NE-COARSE-METO": "O",
      "NE-FINE-LIT": "O",
      "NE-FINE-METO": "O",
      "NE-FINE-COMP": "O",
      "NE-NESTED": "O",
      "NEL-LIT": "_",
      "NEL-METO": "_",
      "MISC": "_"
    }},
    {{
      "TOKEN": "Dr",
      "NE-COARSE-LIT": "B-pers",
      "NE-COARSE-METO": "O",
      "NE-FINE-LIT": "B-pers.ind",
      "NE-FINE-METO": "O",
      "NE-FINE-COMP": "B-comp.title",
      "NE-NESTED": "O",
      "NEL-LIT": "NIL",
      "NEL-METO": "_",
      "MISC": "NoSpaceAfter"
    }},
    {{
      "TOKEN": ".",
      "NE-COARSE-LIT": "I-pers",
      "NE-COARSE-METO": "O",
      "NE-FINE-LIT": "I-pers.ind",
      "NE-FINE-METO": "O",
      "NE-FINE-COMP": "I-comp.title",
      "NE-NESTED": "O",
      "NEL-LIT": "NIL",
      "NEL-METO": "_",
      "MISC": "_"
    }},
    {{
      "TOKEN": "Martin",
      "NE-COARSE-LIT": "I-pers",
      "NE-COARSE-METO": "O",
      "NE-FINE-LIT": "I-pers.ind",
      "NE-FINE-METO": "O",
      "NE-FINE-COMP": "B-comp.name",
      "NE-NESTED": "O",
      "NEL-LIT": "NIL",
      "NEL-METO": "_",
      "MISC": "EndOfSentence"
    }}
  ]
}}
<end>
""".strip()


# ---------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip().replace("<end>", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def clean_cell(value: Any) -> str:
    if value is None:
        return "_"
    value = str(value).strip()
    value = value.replace("\t", " ").replace("\n", " ")
    return value if value else "_"


def coarse_from_fine(fine_type: str) -> str | None:
    if fine_type.startswith("pers."):
        return "pers"
    if fine_type.startswith("org."):
        return "org"
    if fine_type.startswith("loc."):
        return "loc"
    if fine_type.startswith("prod."):
        return "prod"
    if fine_type.startswith("time."):
        return "time"
    return None


def validate_bio_label(label: str, allowed_types: set[str]) -> str:
    label = clean_cell(label)
    if label in {"O", "_"}:
        return "O"

    m = BIO_PATTERN.match(label)
    if not m:
        return "O"

    prefix, entity_type = label[:1], m.group(1)

    if entity_type == "comp.func":
        entity_type = "comp.function"

    if entity_type not in allowed_types:
        return "O"

    return f"{prefix}-{entity_type}"


def validate_nel(value: str, is_entity: bool) -> str:
    value = clean_cell(value)

    if not is_entity:
        return "_"

    if value in {"_", "NIL"}:
        return "NIL"

    if re.match(r"^Q\d+$", value):
        return value

    # Keep URLs out of the TSV to avoid messy synthetic linking.
    return "NIL"


def validate_misc(value: str) -> str:
    value = clean_cell(value)
    if value == "_":
        return "_"

    allowed = {"NoSpaceAfter", "EndOfLine", "EndOfSentence"}
    parts = re.split(r"[|,;]", value)
    cleaned = [p.strip() for p in parts if p.strip() in allowed]
    return "|".join(cleaned) if cleaned else "_"


def validate_row(row: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(row, dict):
        return None

    out = {col: clean_cell(row.get(col, "_")) for col in COLUMNS}

    if out["TOKEN"] in {"", "_"}:
        return None

    out["NE-COARSE-LIT"] = validate_bio_label(out["NE-COARSE-LIT"], COARSE_TYPES)
    out["NE-COARSE-METO"] = validate_bio_label(out["NE-COARSE-METO"], COARSE_TYPES)
    out["NE-FINE-LIT"] = validate_bio_label(out["NE-FINE-LIT"], FINE_TYPES)
    out["NE-FINE-METO"] = validate_bio_label(out["NE-FINE-METO"], FINE_TYPES)
    out["NE-FINE-COMP"] = validate_bio_label(out["NE-FINE-COMP"], COMP_TYPES)
    out["NE-NESTED"] = validate_bio_label(out["NE-NESTED"], FINE_TYPES)

    # Repair coarse/fine inconsistencies conservatively.
    for suffix in ["LIT", "METO"]:
        fine = out[f"NE-FINE-{suffix}"]
        coarse = out[f"NE-COARSE-{suffix}"]

        if fine != "O":
            prefix, fine_type = fine.split("-", 1)
            expected_coarse = coarse_from_fine(fine_type)
            if expected_coarse:
                out[f"NE-COARSE-{suffix}"] = f"{prefix}-{expected_coarse}"
        elif coarse != "O":
            # If coarse exists without fine, drop it because HIPE expects specific fine labels.
            out[f"NE-COARSE-{suffix}"] = "O"

    is_lit_entity = out["NE-FINE-LIT"] != "O"
    is_meto_entity = out["NE-FINE-METO"] != "O"

    out["NEL-LIT"] = validate_nel(out["NEL-LIT"], is_lit_entity)
    out["NEL-METO"] = validate_nel(out["NEL-METO"], is_meto_entity)
    out["MISC"] = validate_misc(out["MISC"])

    return out


def validate_record(
    record: dict[str, Any], fallback_metadata: dict[str, str]
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    merged_metadata = dict(fallback_metadata)
    for key, value in metadata.items():
        merged_metadata[str(key)] = clean_cell(value)

    rows = record.get("rows", [])
    if not isinstance(rows, list):
        return None

    validated_rows = []
    for row in rows:
        validated = validate_row(row)
        if validated is not None:
            validated_rows.append(validated)

    if len(validated_rows) < 5:
        return None

    return {
        "metadata": merged_metadata,
        "rows": validated_rows,
    }


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def record_to_tsv(record: dict[str, Any]) -> str:
    lines = []

    for key, value in record["metadata"].items():
        lines.append(f"# {key} = {value}")

    lines.append("\t".join(COLUMNS))

    for row in record["rows"]:
        lines.append("\t".join(row[col] for col in COLUMNS))

    return "\n".join(lines)


def save_tsv(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            if i:
                f.write("\n\n")
            f.write(record_to_tsv(record))
            f.write("\n")


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def upload_to_hub(repo_id: str, files: list[str | Path], private: bool = False) -> None:
    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
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
# Generation
# ---------------------------------------------------------------------


def build_jobs(num_samples: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    jobs = []

    for i in range(num_samples):
        year = random.choice(YEARS)
        language = random.choice(LANGUAGES)
        publication_title = random.choice(PUBLICATION_TITLES)
        section = random.choice(SECTIONS)
        topic = random.choice(TOPICS)
        document_type = random.choice(DOCUMENT_TYPES)
        month = random.randint(1, 12)
        day = random.randint(1, 28)

        doc_id = f"SYN-{year}-{i + 1:06d}"
        doc_date = date(year, month, day).isoformat()

        metadata = {
            "hipe2022:document_id": doc_id,
            "hipe2022:date": doc_date,
            "hipe2022:language": language,
            "hipe2022:document_type": document_type,
            "hipe2022:dataset": "synthetic",
            "hipe2022:original_source": "generated",
            "hipe2022:applicable_columns": "TOKEN NE-COARSE-LIT NE-COARSE-METO NE-FINE-LIT NE-FINE-METO NE-FINE-COMP NE-NESTED NEL-LIT NEL-METO MISC",
            "synthetic:publication_title": publication_title,
            "synthetic:section": section,
            "synthetic:topic": topic,
        }

        prompt = create_hipe_tsv_prompt(
            language=language,
            publication_title=publication_title,
            year=year,
            document_type=document_type,
            section=section,
            topic=topic,
            ocr_noise=random.choice(["none", "light"]),
        )

        jobs.append({"prompt": prompt, "metadata": metadata})

    return jobs


def generate_records(
    llm: LLM,
    sampling_params: SamplingParams,
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    prompts = [job["prompt"] for job in jobs]
    outputs = llm.generate(prompts, sampling_params)

    records = []
    failed = []

    for output, job in zip(outputs, jobs):
        generated_text = output.outputs[0].text
        raw = extract_json_object(generated_text)
        if raw is None:
            failed.append(generated_text)
            continue

        record = validate_record(raw, job["metadata"])
        if record is None:
            failed.append(generated_text)
            continue

        records.append(record)

    return records, failed


def print_stats(records: list[dict[str, Any]], failed_count: int) -> None:
    print(f"Valid records: {len(records)}")
    print(f"Failed records: {failed_count}")

    if not records:
        return

    token_counts = [len(r["rows"]) for r in records]
    print(f"Average tokens/document: {sum(token_counts) / len(token_counts):.2f}")

    fine_lit = Counter()
    fine_meto = Counter()
    comps = Counter()
    nested = Counter()

    for record in records:
        for row in record["rows"]:
            for column, counter in [
                ("NE-FINE-LIT", fine_lit),
                ("NE-FINE-METO", fine_meto),
                ("NE-FINE-COMP", comps),
                ("NE-NESTED", nested),
            ]:
                label = row[column]
                if label != "O":
                    counter[label.split("-", 1)[1]] += 1

    print("Literal fine labels:", dict(fine_lit.most_common()))
    print("Metonymic fine labels:", dict(fine_meto.most_common()))
    print("Component labels:", dict(comps.most_common()))
    print("Nested labels:", dict(nested.most_common()))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic HIPE-style full-column TSV NER/NEL data."
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-tsv", default="outputs/synthetic_historical_ner_hipe_full.tsv"
    )
    parser.add_argument(
        "--output-jsonl", default="outputs/synthetic_historical_ner_hipe_full.jsonl"
    )
    parser.add_argument("--failed-output", default="outputs/failed_generations.jsonl")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--hub-dataset-id", default="emanuelaboros/synthetic-historical-ner-data"
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.num_samples, seed=args.seed)

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

    print(f"Generating {len(jobs)} synthetic HIPE-style documents...")
    records, failed = generate_records(llm, sampling_params, jobs)

    save_tsv(records, args.output_tsv)
    save_jsonl(records, args.output_jsonl)

    if failed:
        failed_path = Path(args.failed_output)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_path.open("w", encoding="utf-8") as f:
            for item in failed:
                f.write(json.dumps({"generation": item}, ensure_ascii=False) + "\n")

    print_stats(records, failed_count=len(failed))
    print(f"Saved TSV: {args.output_tsv}")
    print(f"Saved JSONL: {args.output_jsonl}")

    files_to_upload = [args.output_tsv, args.output_jsonl]
    if failed:
        files_to_upload.append(args.failed_output)

    if args.push_to_hub:
        upload_to_hub(
            repo_id=args.hub_dataset_id,
            files=files_to_upload,
            private=args.private,
        )


if __name__ == "__main__":
    main()
