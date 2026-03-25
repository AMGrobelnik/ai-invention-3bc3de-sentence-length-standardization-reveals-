#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru"]
# ///
"""Convert raw typological metadata into exp_sel_data_out.json schema format.

Creates 6 dataset views from the compiled WALS + UD typological data,
each with language-level examples suitable for typological analysis.
"""

import json
import sys
from pathlib import Path

from loguru import logger

WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "data.log"), rotation="30 MB", level="DEBUG")


def make_input_str(rec: dict, *, include_word_order: bool = True, include_case: bool = True) -> str:
    """Build a JSON-string input from a language record's features."""
    features = {
        "language_iso": rec["language_iso"],
        "language_name": rec["language_name"],
        "language_family": rec["language_family"],
    }
    if include_word_order and rec.get("word_order_81A"):
        features["word_order_81A"] = rec["word_order_81A"]
    if include_case:
        features["case_count_49A"] = rec.get("case_count_49A")
        features["ud_case_richness"] = rec["ud_case_richness"]
        features["ud_case_values"] = rec["ud_case_values"]
    features["latitude"] = rec.get("latitude")
    features["longitude"] = rec.get("longitude")
    features["glottocode"] = rec.get("glottocode")
    return json.dumps(features, ensure_ascii=False)


@logger.catch
def main() -> None:
    # Load raw data
    raw_path = WORKSPACE / "temp" / "datasets" / "full_typological_metadata.json"
    if not raw_path.exists():
        raw_path = WORKSPACE / "data_out.json"
    logger.info(f"Loading raw data from {raw_path}")
    records = json.loads(raw_path.read_text())
    logger.info(f"Loaded {len(records)} language records")

    datasets = []

    # --- Dataset 1: Full typological profiles (all languages) ---
    ds1_examples = []
    for i, rec in enumerate(records):
        inp = make_input_str(rec, include_word_order=True, include_case=True)
        # Output: full typological profile summary
        profile_parts = []
        if rec.get("word_order_81A"):
            profile_parts.append(f"word_order={rec['word_order_81A']}")
        if rec.get("case_count_49A"):
            profile_parts.append(f"wals_cases={rec['case_count_49A']}")
        profile_parts.append(f"ud_case_richness={rec['ud_case_richness']}")
        if rec.get("language_family"):
            profile_parts.append(f"family={rec['language_family']}")
        output = "; ".join(profile_parts) if profile_parts else "no_typological_data"

        ds1_examples.append({
            "input": inp,
            "output": output,
            "metadata_row_index": i,
            "metadata_task_type": "typological_profile",
            "metadata_language_code_ud": rec["language_code_ud"],
            "metadata_ud_treebank_ids": rec["ud_treebank_ids"],
            "metadata_has_wals": rec.get("wals_code") is not None,
        })
    datasets.append({"dataset": "full_typological_profiles", "examples": ds1_examples})
    logger.info(f"Dataset 1 (full_typological_profiles): {len(ds1_examples)} examples")

    # --- Dataset 2: Word order prediction (languages with 81A) ---
    ds2_examples = []
    for i, rec in enumerate(records):
        if not rec.get("word_order_81A"):
            continue
        # Input: everything EXCEPT word order
        inp = make_input_str(rec, include_word_order=False, include_case=True)
        ds2_examples.append({
            "input": inp,
            "output": rec["word_order_81A"],
            "metadata_row_index": i,
            "metadata_task_type": "classification",
            "metadata_n_classes": 7,
            "metadata_wals_code": rec["wals_code"],
            "metadata_language_code_ud": rec["language_code_ud"],
        })
    datasets.append({"dataset": "word_order_prediction", "examples": ds2_examples})
    logger.info(f"Dataset 2 (word_order_prediction): {len(ds2_examples)} examples")

    # --- Dataset 3: UD case richness (languages with case > 0) ---
    ds3_examples = []
    for i, rec in enumerate(records):
        if rec["ud_case_richness"] <= 0:
            continue
        features = {
            "language_iso": rec["language_iso"],
            "language_name": rec["language_name"],
            "language_family": rec["language_family"],
            "word_order_81A": rec.get("word_order_81A"),
            "case_count_49A": rec.get("case_count_49A"),
            "latitude": rec.get("latitude"),
            "longitude": rec.get("longitude"),
        }
        ds3_examples.append({
            "input": json.dumps(features, ensure_ascii=False),
            "output": json.dumps({"ud_case_richness": rec["ud_case_richness"], "case_values": rec["ud_case_values"]}),
            "metadata_row_index": i,
            "metadata_task_type": "regression",
            "metadata_language_code_ud": rec["language_code_ud"],
            "metadata_ud_treebank_ids": rec["ud_treebank_ids"],
        })
    datasets.append({"dataset": "ud_case_richness", "examples": ds3_examples})
    logger.info(f"Dataset 4 (ud_case_richness): {len(ds3_examples)} examples")

    # Assemble output
    output = {
        "metadata": {
            "description": "Typological Metadata for UD Languages (WALS + UD Case Richness)",
            "sources": ["WALS CLDF (81A word order, 49A case count)", "Universal Dependencies (case morphological features)"],
            "total_languages": len(records),
            "total_datasets": len(datasets),
            "selected_datasets": ["full_typological_profiles", "word_order_prediction", "ud_case_richness"],
        },
        "datasets": datasets,
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Saved {len(datasets)} datasets to {out_path}")
    for ds in datasets:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")


if __name__ == "__main__":
    main()
