#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "loguru", "huggingface-hub", "psutil"]
# ///
"""Extract per-dependency records from Universal Dependencies treebanks via HuggingFace."""

import json
import math
import os
import resource
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev

import psutil
from datasets import load_dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
LOG_FMT = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"

logger.remove()
logger.add(sys.stdout, level="INFO", format=LOG_FMT)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware limits
# ---------------------------------------------------------------------------
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.6, 14) * 1e9)  # 60% of available, max 14GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f}GB, total: {TOTAL_RAM_GB:.1f}GB")

# ---------------------------------------------------------------------------
# UPOS label mapping (from HF ClassLabel)
# ---------------------------------------------------------------------------
UPOS_NAMES = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART",
    "DET", "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX",
]

# ---------------------------------------------------------------------------
# Mini-20 diverse treebank subset
# ---------------------------------------------------------------------------
MINI_TREEBANKS = [
    "en_ewt", "de_gsd", "ja_gsd", "zh_gsd", "ar_padt", "hi_hdtb",
    "ru_syntagrus", "cs_cac", "fi_tdt", "tr_imst", "ko_gsd", "es_ancora",
    "fr_gsd", "pt_bosque", "it_isdt", "pl_pdb", "hu_szeged", "eu_bdt",
    "ta_ttb", "he_htb",
]

DATASET_ID = "commul/universal_dependencies"


def parse_feats(feats_str: str | None) -> dict[str, str]:
    """Parse 'Case=Nom|Number=Sing' into {'Case': 'Nom', 'Number': 'Sing'}."""
    if not feats_str:
        return {}
    result = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def process_treebank(config_name: str) -> tuple[list[dict], list[dict], dict]:
    """Process one treebank config. Returns (dep_records, sent_records, treebank_meta)."""
    lang_code = config_name.split("_")[0]
    dep_records: list[dict] = []
    sent_records: list[dict] = []
    all_case_values: set[str] = set()
    all_sent_lengths: list[int] = []

    for split_name in ["train", "dev", "test"]:
        try:
            ds = load_dataset(DATASET_ID, config_name, split=split_name)
        except (ValueError, KeyError):
            logger.debug(f"{config_name}/{split_name} not available, skipping")
            continue

        for sent_idx, row in enumerate(ds):
            tokens = row["tokens"]
            heads = row["head"]
            deprels = row["deprel"]
            upos_ids = row["upos"]
            feats_list = row["feats"]
            n_tokens = len(tokens)

            if n_tokens == 0:
                continue

            all_sent_lengths.append(n_tokens)
            sentence_id = f"{config_name}_{split_name}_{sent_idx}"

            # Collect per-dependency records for this sentence
            distances: list[int] = []
            for tok_idx_0, (head_str, deprel, upos_id, feat_str) in enumerate(
                zip(heads, deprels, upos_ids, feats_list)
            ):
                tok_pos = tok_idx_0 + 1  # 1-based
                head_pos = int(head_str)
                if head_pos == 0:  # root
                    continue

                dist = abs(tok_pos - head_pos)
                distances.append(dist)

                dep_feats = parse_feats(feat_str)
                dep_case = dep_feats.get("Case")
                if dep_case:
                    all_case_values.add(dep_case)

                # head feats
                head_idx_0 = head_pos - 1
                head_feat_str = feats_list[head_idx_0] if 0 <= head_idx_0 < n_tokens else None
                head_feats = parse_feats(head_feat_str)
                head_case = head_feats.get("Case")
                if head_case:
                    all_case_values.add(head_case)

                head_upos_id = upos_ids[head_idx_0] if 0 <= head_idx_0 < n_tokens else None
                head_upos = UPOS_NAMES[head_upos_id] if head_upos_id is not None and 0 <= head_upos_id < len(UPOS_NAMES) else None
                dep_upos = UPOS_NAMES[upos_id] if 0 <= upos_id < len(UPOS_NAMES) else None

                # Compact: input encodes features, output is distance
                # Format: treebank|rel|depPOS|headPOS|sentLen|depCase|headCase
                dc = dep_case or "_"
                hc = head_case or "_"
                dp = dep_upos or "_"
                hp = head_upos or "_"
                dep_records.append({
                    "input": f"{config_name}|{deprel}|{dp}|{hp}|{n_tokens}|{dc}|{hc}",
                    "output": str(dist),
                    "metadata_fold": split_name,
                    "metadata_sentence_id": sentence_id,
                    "metadata_tok_pos": tok_pos,
                })

            # Sentence-level record
            mdd_obs = mean(distances) if distances else 0.0
            mdd_rand = (n_tokens + 1) / 3
            sent_records.append({
                "treebank_id": config_name,
                "language_code": lang_code,
                "sentence_id": sentence_id,
                "sentence_length": n_tokens,
                "num_dependencies": len(distances),
                "mdd_observed": round(mdd_obs, 4),
                "mdd_random_analytic": round(mdd_rand, 4),
                "split": split_name,
            })

        logger.info(f"{config_name}/{split_name}: {len(ds)} sentences")

    # Treebank metadata
    treebank_meta = {
        "treebank_id": config_name,
        "language_code": lang_code,
        "num_sentences": len(all_sent_lengths),
        "mean_sentence_length": round(mean(all_sent_lengths), 2) if all_sent_lengths else 0,
        "std_sentence_length": round(stdev(all_sent_lengths), 2) if len(all_sent_lengths) > 1 else 0,
        "num_case_values": len(all_case_values),
        "case_values": sorted(all_case_values),
    }

    logger.info(
        f"✓ {config_name}: {len(dep_records)} deps, {len(sent_records)} sents, "
        f"{len(all_case_values)} case values"
    )
    return dep_records, sent_records, treebank_meta


@logger.catch
def main() -> None:
    out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dep_records: list[dict] = []
    all_sent_records: list[dict] = []
    all_treebank_meta: list[dict] = []

    # Support TEST_LIMIT env var for gradual scaling
    test_limit = int(os.environ.get("TEST_LIMIT", "0"))
    treebanks = MINI_TREEBANKS[:test_limit] if test_limit > 0 else MINI_TREEBANKS
    logger.info(f"Processing {len(treebanks)} treebanks with ThreadPoolExecutor")

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_treebank, tb): tb for tb in treebanks}
        for fut in as_completed(futures):
            tb_name = futures[fut]
            try:
                deps, sents, meta = fut.result()
                all_dep_records.extend(deps)
                all_sent_records.extend(sents)
                all_treebank_meta.append(meta)
            except Exception:
                logger.exception(f"Failed processing {tb_name}")

    logger.info(f"Total: {len(all_dep_records)} dependency records from {len(all_treebank_meta)} treebanks")

    # Group records by treebank for per-treebank dataset entries
    from collections import defaultdict
    by_treebank: dict[str, list[dict]] = defaultdict(list)
    for rec in all_dep_records:
        tb = rec["input"].split("|")[0]
        by_treebank[tb].append(rec)

    meta_block = {
        "description": "Universal Dependencies per-dependency records for DDM analysis",
        "source": "commul/universal_dependencies (HuggingFace)",
        "num_treebanks": len(all_treebank_meta),
        "num_records": len(all_dep_records),
        "num_sentences": len(all_sent_records),
        "schema_version": "1.0",
        "treebank_metadata": all_treebank_meta,
        "sentence_metadata_sample": all_sent_records[:20],
    }

    # Split into parts of ~50MB each
    # Each part is a valid exp_sel_data_out with subset of treebank datasets
    split_dir = out_dir / "data_out"
    split_dir.mkdir(exist_ok=True)

    treebank_names = sorted(by_treebank.keys())
    part_datasets: list[dict] = []
    part_num = 1
    current_size_est = 0

    def write_part(datasets: list[dict], num: int) -> None:
        output = {"metadata": meta_block, "datasets": datasets}
        path = split_dir / f"full_data_out_{num}.json"
        path.write_text(json.dumps(output, ensure_ascii=False))
        sz = path.stat().st_size / 1e6
        logger.info(f"Wrote {path}: {sz:.1f} MB ({len(datasets)} treebanks)")

    MAX_PART_MB = 80  # keep under 100MB limit
    MAX_RECORDS_PER_PART = 400_000  # ~80MB at ~200 bytes/record

    for tb_name in treebank_names:
        examples = by_treebank[tb_name]
        # Large treebanks: split across multiple parts
        offset = 0
        while offset < len(examples):
            chunk = examples[offset:offset + MAX_RECORDS_PER_PART]
            est_mb = len(chunk) * 200 / 1e6
            if current_size_est + est_mb > MAX_PART_MB and part_datasets:
                write_part(part_datasets, part_num)
                part_num += 1
                part_datasets = []
                current_size_est = 0
            ds_label = tb_name if offset == 0 and offset + MAX_RECORDS_PER_PART >= len(examples) else f"{tb_name}_p{offset // MAX_RECORDS_PER_PART}"
            part_datasets.append({"dataset": ds_label, "examples": chunk})
            current_size_est += est_mb
            offset += MAX_RECORDS_PER_PART

    if part_datasets:
        write_part(part_datasets, part_num)

    # Also write mini (3 examples per treebank) and preview (1000 total)
    mini_datasets = []
    for tb_name in treebank_names[:5]:
        mini_datasets.append({"dataset": tb_name, "examples": by_treebank[tb_name][:3]})
    mini_out = {"metadata": meta_block, "datasets": mini_datasets}
    mini_path = out_dir / "mini_data_out.json"
    mini_path.write_text(json.dumps(mini_out, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {mini_path}: {mini_path.stat().st_size/1e6:.1f} MB")

    preview_datasets = []
    remaining = 1000
    for tb_name in treebank_names:
        if remaining <= 0:
            break
        take = min(remaining, len(by_treebank[tb_name]))
        preview_datasets.append({"dataset": tb_name, "examples": by_treebank[tb_name][:take]})
        remaining -= take
    preview_out = {"metadata": meta_block, "datasets": preview_datasets}
    preview_path = out_dir / "preview_data_out.json"
    preview_path.write_text(json.dumps(preview_out, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {preview_path}: {preview_path.stat().st_size/1e6:.1f} MB")

    logger.info(f"Wrote {part_num} split files to {split_dir}")


if __name__ == "__main__":
    main()
