# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "numpy", "pandas", "loguru", "psutil"]
# ///
"""
Download UD treebanks from HuggingFace (commul/universal_dependencies),
fetch WALS 81A word order data, compute per-treebank DDM statistics,
and assemble into exp_sel_data_out.json schema.

Datasets produced:
  1. ud_treebank_summaries — per-treebank DDM stats + metadata
  2. ud_core_argument_deps — nsubj/obj/iobj records for Cox modeling
  3. wals_word_order — WALS 81A word order classifications
  4. ud_sentence_length_ddm — DDM(n) by sentence length per treebank
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
    level="DEBUG",
)

# ── Hardware detection ───────────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

NUM_CPUS = _detect_cpus()
logger.info(f"Detected {NUM_CPUS} CPUs")

# RAM limit: 14GB on 30GB machine
import psutil
_avail = psutil.virtual_memory().available
RAM_BUDGET = min(14 * 1024**3, int(_avail * 0.85))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────
WS = Path(__file__).parent
TEMP = WS / "temp" / "datasets"
TEMP.mkdir(parents=True, exist_ok=True)
OUT_FILE = WS / "full_data_out.json"

# ── Constants ────────────────────────────────────────────────────────
MIN_SENTENCES = 50          # skip treebanks with fewer
MIN_SENT_LEN = 3            # minimum sentence length for DDM
MAX_SENT_LEN = 80           # cap sentence length
MIN_SENTS_PER_LEN = 10      # need this many sentences per length bin
N_PERMUTATIONS = 30         # random baseline permutations
CORE_ARG_DEPRELS = {"nsubj", "obj", "iobj"}
MAX_CORE_ARG_ROWS = 500_000  # sample limit

# ── Load WALS 81A ───────────────────────────────────────────────────
# ISO 639-1 (2-letter) → ISO 639-3 (3-letter) mapping for UD languages
ISO1_TO_ISO3 = {
    "ab": "abk", "af": "afr", "akk": "akk", "am": "amh", "an": "arg",
    "ar": "arb", "az": "aze", "ba": "bak", "be": "bel", "bg": "bul",
    "bho": "bho", "bm": "bam", "bn": "ben", "br": "bre", "ca": "cat",
    "ceb": "ceb", "ckb": "ckb", "cop": "cop", "cs": "ces", "cu": "chu",
    "cy": "cym", "da": "dan", "de": "deu", "el": "ell", "en": "eng",
    "eo": "epo", "es": "spa", "et": "est", "eu": "eus", "fa": "pes",
    "fi": "fin", "fo": "fao", "fr": "fra", "ga": "gle", "gd": "gla",
    "gl": "glg", "got": "got", "grc": "grc", "gu": "guj", "gv": "glv",
    "ha": "hau", "hbo": "hbo", "he": "heb", "hi": "hin", "hr": "hrv",
    "hsb": "hsb", "ht": "hat", "hu": "hun", "hy": "hye", "hyw": "hyw",
    "id": "ind", "is": "isl", "it": "ita", "ja": "jpn", "jv": "jav",
    "ka": "kat", "kk": "kaz", "kmr": "kmr", "ko": "kor", "la": "lat",
    "lb": "ltz", "lt": "lit", "lv": "lav", "lzh": "lzh", "mk": "mkd",
    "ml": "mal", "mr": "mar", "mt": "mlt", "my": "mya", "myv": "myv",
    "nds": "nds", "nl": "nld", "no": "nor", "oc": "oci", "or": "ori",
    "pa": "pan", "pl": "pol", "ps": "pus", "pt": "por", "ro": "ron",
    "ru": "rus", "sa": "san", "sd": "snd", "si": "sin", "sk": "slk",
    "sl": "slv", "sme": "sme", "sq": "sqi", "sr": "srp", "sv": "swe",
    "swl": "swl", "ta": "tam", "te": "tel", "th": "tha", "tl": "tgl",
    "tn": "tsn", "tr": "tur", "tt": "tat", "ug": "uig", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "vi": "vie", "wo": "wol", "xnr": "xnr",
    "yi": "yid", "yo": "yor", "yue": "yue", "zh": "cmn",
}


@logger.catch
def load_wals_81a() -> dict[str, dict[str, str]]:
    """Load WALS 81A word order data. Returns {iso_code: {word_order, family, ...}}.

    Keys include both ISO 639-3 codes AND ISO 639-1 codes (via mapping).
    """
    # Load codes for 81A
    codes = {}
    with open(TEMP / "wals_codes.csv") as f:
        for row in csv.DictReader(f):
            if row["Parameter_ID"] == "81A":
                codes[row["ID"]] = row["Name"]  # e.g. 81A-1 -> SOV

    # Load languages
    langs = {}  # wals_id -> {iso, family, name, ...}
    with open(TEMP / "wals_languages.csv") as f:
        for row in csv.DictReader(f):
            langs[row["ID"]] = {
                "name": row["Name"],
                "iso": row["ISO639P3code"],
                "family": row["Family"],
                "genus": row.get("Genus", ""),
            }

    # Load values for 81A
    result = {}  # iso_code -> {word_order, family, ...}
    with open(TEMP / "wals_values.csv") as f:
        for row in csv.DictReader(f):
            if row["Parameter_ID"] != "81A":
                continue
            lang_id = row["Language_ID"]
            code_id = row["Code_ID"]
            if lang_id in langs and code_id in codes:
                lang = langs[lang_id]
                iso3 = lang["iso"]
                if iso3:
                    entry = {
                        "word_order": codes[code_id],
                        "family": lang["family"],
                        "name": lang["name"],
                        "genus": lang["genus"],
                    }
                    result[iso3] = entry

    # Build reverse mapping: ISO 639-1 → entry (via ISO1_TO_ISO3)
    for iso1, iso3 in ISO1_TO_ISO3.items():
        if iso3 in result and iso1 not in result:
            result[iso1] = result[iso3]

    logger.info(f"WALS 81A: {len(result)} entries (incl. ISO 639-1 aliases)")
    return result


# ── Process a single treebank ────────────────────────────────────────
def compute_random_mdd_for_tree(heads: list[int], n_perms: int = N_PERMUTATIONS) -> float:
    """Compute expected MDD under random linearization of a dependency tree.

    heads: list of head indices (1-based, 0=root) for each token position 1..n.
    Returns mean MDD across n_perms random permutations.
    """
    n = len(heads)
    if n < 2:
        return 0.0

    # Collect arcs (dependent_idx, head_idx) — 0-based
    arcs = []
    for dep_idx in range(n):
        h = heads[dep_idx]
        if h > 0:  # skip root
            arcs.append((dep_idx, h - 1))  # convert to 0-based

    if not arcs:
        return 0.0

    rng = np.random.default_rng()
    total_mdd = 0.0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        # position of each original token in the permuted order
        pos = np.empty(n, dtype=np.int64)
        pos[perm] = np.arange(n)
        dist_sum = sum(abs(int(pos[d]) - int(pos[h])) for d, h in arcs)
        total_mdd += dist_sum / len(arcs)

    return total_mdd / n_perms


def process_single_treebank(config: str) -> dict[str, Any] | None:
    """Process one UD treebank config. Returns summary dict or None if too small."""
    try:
        from datasets import load_dataset

        # Load all splits
        ds = load_dataset("commul/universal_dependencies", config, trust_remote_code=False)

        all_rows = []
        for split_name in ds:
            for row in ds[split_name]:
                all_rows.append(row)

        n_sentences = len(all_rows)
        if n_sentences < MIN_SENTENCES:
            return None

        # Extract per-sentence data
        sent_data_by_length: dict[int, list[float]] = defaultdict(list)  # n -> [mdd_obs values]
        sent_random_by_length: dict[int, list[float]] = defaultdict(list)  # n -> [mdd_random values]
        case_values_all: set[str] = set()
        core_arg_records: list[dict] = []

        for row in all_rows:
            tokens = row.get("tokens") or []
            heads = row.get("head") or row.get("heads") or []
            deprels = row.get("deprel") or row.get("deprels") or []
            feats_list = row.get("feats") or []

            n = len(tokens)
            if n < MIN_SENT_LEN or n > MAX_SENT_LEN:
                continue
            if len(heads) != n or len(deprels) != n:
                continue

            # Parse heads to int
            try:
                heads_int = [int(h) for h in heads]
            except (ValueError, TypeError):
                continue

            # Compute observed MDD
            arcs = []
            for i in range(n):
                h = heads_int[i]
                if h > 0:  # skip root
                    dist = abs((i + 1) - h)  # 1-based positions
                    arcs.append(dist)

                    # Core argument records
                    dep_deprel = deprels[i] if i < len(deprels) else ""
                    if dep_deprel in CORE_ARG_DEPRELS:
                        # Extract case from feats
                        feat_str = feats_list[i] if i < len(feats_list) else ""
                        case_val = None
                        if feat_str and feat_str != "_":
                            for feat_pair in str(feat_str).split("|"):
                                if feat_pair.startswith("Case="):
                                    case_val = feat_pair.split("=", 1)[1]
                                    case_values_all.add(case_val)
                                    break

                        core_arg_records.append({
                            "deprel": dep_deprel,
                            "distance": dist,
                            "sentence_length": n,
                            "case_value": case_val,
                        })

            if not arcs:
                continue

            mdd_obs = sum(arcs) / len(arcs)
            sent_data_by_length[n].append(mdd_obs)

            # Compute random baseline
            mdd_random = compute_random_mdd_for_tree(heads_int, n_perms=N_PERMUTATIONS)
            sent_random_by_length[n].append(mdd_random)

        # Aggregate DDM by length
        ddm_by_length = {}
        mdd_obs_by_length = {}
        mdd_random_by_length = {}
        n_sents_by_length = {}
        total_sents_used = 0

        for length in sorted(sent_data_by_length.keys()):
            obs_vals = sent_data_by_length[length]
            rand_vals = sent_random_by_length[length]
            if len(obs_vals) < MIN_SENTS_PER_LEN:
                continue

            mean_obs = np.mean(obs_vals)
            mean_rand = np.mean(rand_vals)

            mdd_obs_by_length[str(length)] = round(float(mean_obs), 4)
            mdd_random_by_length[str(length)] = round(float(mean_rand), 4)
            n_sents_by_length[str(length)] = len(obs_vals)
            total_sents_used += len(obs_vals)

            if mean_rand > 0:
                ddm_by_length[str(length)] = round(1.0 - mean_obs / mean_rand, 4)
            else:
                ddm_by_length[str(length)] = 0.0

        if not ddm_by_length:
            return None

        # Sentence length distribution
        sent_len_dist = {}
        for length, count in n_sents_by_length.items():
            sent_len_dist[length] = round(int(count) / total_sents_used, 6)

        # Case richness
        case_richness = len(case_values_all)

        # Extract language info from config name
        lang_code = config.split("_")[0]

        # Mean sentence length
        all_lengths = []
        for length_str, count in n_sents_by_length.items():
            all_lengths.extend([int(length_str)] * count)
        mean_sent_len = round(float(np.mean(all_lengths)), 2) if all_lengths else 0.0

        return {
            "config": config,
            "lang_code": lang_code,
            "n_sentences": n_sentences,
            "n_sentences_used": total_sents_used,
            "mean_sentence_length": mean_sent_len,
            "case_richness": case_richness,
            "ddm_by_length": ddm_by_length,
            "mdd_obs_by_length": mdd_obs_by_length,
            "mdd_random_by_length": mdd_random_by_length,
            "n_sentences_by_length": n_sents_by_length,
            "sentence_length_distribution": sent_len_dist,
            "core_arg_records": core_arg_records,
            "case_values": sorted(case_values_all),
        }
    except Exception as e:
        logger.warning(f"Failed to process {config}: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────
@logger.catch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-configs", type=int, default=0, help="Max configs to process (0=all)")
    args = parser.parse_args()

    t0 = time.time()

    # Load WALS
    wals = load_wals_81a()

    # Get all configs
    from datasets import get_dataset_config_names
    all_configs = get_dataset_config_names("commul/universal_dependencies")
    logger.info(f"Total UD configs: {len(all_configs)}")
    if args.max_configs > 0:
        all_configs = all_configs[:args.max_configs]
        logger.info(f"Limited to {len(all_configs)} configs")

    # Process treebanks in parallel
    results: list[dict] = []
    n_workers = max(1, NUM_CPUS - 1)  # leave 1 core free
    logger.info(f"Processing with {n_workers} workers")

    # Process in batches to manage memory
    BATCH_SIZE = 30
    all_core_arg_records: list[dict] = []

    for batch_start in range(0, len(all_configs), BATCH_SIZE):
        batch_configs = all_configs[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = math.ceil(len(all_configs) / BATCH_SIZE)
        logger.info(f"Batch {batch_num}/{total_batches}: configs {batch_start}-{batch_start + len(batch_configs) - 1}")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_treebank, cfg): cfg for cfg in batch_configs}
            for future in as_completed(futures):
                cfg = futures[future]
                try:
                    result = future.result(timeout=300)
                    if result is not None:
                        # Extract core arg records before storing summary
                        core_recs = result.pop("core_arg_records", [])
                        for rec in core_recs:
                            rec["treebank_id"] = result["config"]
                            rec["lang_code"] = result["lang_code"]
                        all_core_arg_records.extend(core_recs)
                        results.append(result)
                        logger.debug(f"  ✓ {cfg}: {result['n_sentences_used']} sents, DDM lengths: {len(result['ddm_by_length'])}")
                except Exception as e:
                    logger.warning(f"  ✗ {cfg}: {e}")

        gc.collect()
        elapsed = time.time() - t0
        logger.info(f"  Elapsed: {elapsed:.0f}s, {len(results)} treebanks done, {len(all_core_arg_records)} core arg records")

    logger.info(f"Processed {len(results)} treebanks successfully")

    # ── Compute pooled reference distribution ────────────────────────
    total_sents_global = 0
    global_sents_by_length: dict[str, int] = defaultdict(int)
    for r in results:
        for length_str, count in r["n_sentences_by_length"].items():
            global_sents_by_length[length_str] += count
            total_sents_global += count

    ref_dist = {}
    for length_str in sorted(global_sents_by_length.keys(), key=int):
        ref_dist[length_str] = round(global_sents_by_length[length_str] / total_sents_global, 6)

    # ── Compute naive and standardized DDM per treebank ──────────────
    for r in results:
        # Naive DDM = Σ DDM(n) × P_tb(n)
        naive_ddm = 0.0
        for length_str, ddm_val in r["ddm_by_length"].items():
            p_tb = r["sentence_length_distribution"].get(length_str, 0.0)
            naive_ddm += ddm_val * p_tb
        r["naive_ddm"] = round(naive_ddm, 4)

        # Standardized DDM = Σ DDM(n) × P_ref(n)
        std_ddm = 0.0
        for length_str, ddm_val in r["ddm_by_length"].items():
            p_ref = ref_dist.get(length_str, 0.0)
            std_ddm += ddm_val * p_ref
        r["standardized_ddm"] = round(std_ddm, 4)

    # ── Add WALS metadata to treebank results ────────────────────────
    for r in results:
        iso = r["lang_code"]
        wals_info = wals.get(iso, {})
        r["word_order"] = wals_info.get("word_order", None)
        r["language_family"] = wals_info.get("family", None)
        r["language_name"] = wals_info.get("name", None)

    # ── Sample core argument records if too many ─────────────────────
    if len(all_core_arg_records) > MAX_CORE_ARG_ROWS:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_core_arg_records), size=MAX_CORE_ARG_ROWS, replace=False)
        all_core_arg_records = [all_core_arg_records[i] for i in sorted(indices)]
        logger.info(f"Sampled core arg records to {MAX_CORE_ARG_ROWS}")

    # Add WALS info to core arg records
    for rec in all_core_arg_records:
        iso = rec["lang_code"]
        wals_info = wals.get(iso, {})
        rec["word_order"] = wals_info.get("word_order", None)
        rec["language_family"] = wals_info.get("family", None)

    logger.info(f"Final: {len(results)} treebanks, {len(all_core_arg_records)} core arg records")

    # ── Build output in exp_sel_data_out.json schema ─────────────────
    # Dataset 1: ud_treebank_summaries
    treebank_examples = []
    for r in sorted(results, key=lambda x: x["config"]):
        input_data = {
            "treebank_id": r["config"],
            "language_name": r.get("language_name"),
            "iso_code": r["lang_code"],
            "language_family": r.get("language_family"),
            "word_order": r.get("word_order"),
            "case_richness": r["case_richness"],
            "n_sentences": r["n_sentences"],
            "n_sentences_used": r["n_sentences_used"],
            "mean_sentence_length": r["mean_sentence_length"],
        }
        output_data = {
            "naive_ddm": r["naive_ddm"],
            "standardized_ddm": r["standardized_ddm"],
            "ddm_by_length": r["ddm_by_length"],
            "mdd_obs_by_length": r["mdd_obs_by_length"],
            "mdd_random_by_length": r["mdd_random_by_length"],
            "n_sentences_by_length": r["n_sentences_by_length"],
            "sentence_length_distribution": r["sentence_length_distribution"],
        }
        treebank_examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_treebank_id": r["config"],
            "metadata_lang_code": r["lang_code"],
            "metadata_word_order": r.get("word_order"),
            "metadata_language_family": r.get("language_family"),
            "metadata_n_sentences": r["n_sentences"],
            "metadata_naive_ddm": r["naive_ddm"],
            "metadata_standardized_ddm": r["standardized_ddm"],
            "metadata_case_richness": r["case_richness"],
        })

    # Dataset 2: ud_core_argument_deps
    core_arg_examples = []
    for rec in all_core_arg_records:
        input_data = {
            "treebank_id": rec["treebank_id"],
            "deprel": rec["deprel"],
            "sentence_length": rec["sentence_length"],
            "case_value": rec["case_value"],
            "word_order": rec.get("word_order"),
            "language_family": rec.get("language_family"),
        }
        output_data = {
            "distance": rec["distance"],
        }
        core_arg_examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_treebank_id": rec["treebank_id"],
            "metadata_deprel": rec["deprel"],
            "metadata_distance": rec["distance"],
            "metadata_sentence_length": rec["sentence_length"],
            "metadata_word_order": rec.get("word_order"),
            "metadata_language_family": rec.get("language_family"),
        })

    # Dataset 3: wals_word_order
    wals_examples = []
    for iso, info in sorted(wals.items()):
        input_data = {
            "iso_code": iso,
            "language_name": info["name"],
            "language_family": info["family"],
            "genus": info["genus"],
        }
        output_data = {
            "word_order": info["word_order"],
        }
        wals_examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_iso_code": iso,
            "metadata_word_order": info["word_order"],
            "metadata_language_family": info["family"],
        })

    # Dataset 4: ud_sentence_length_ddm (one row per treebank×length)
    len_ddm_examples = []
    for r in sorted(results, key=lambda x: x["config"]):
        for length_str in sorted(r["ddm_by_length"].keys(), key=int):
            input_data = {
                "treebank_id": r["config"],
                "sentence_length": int(length_str),
                "n_sentences": r["n_sentences_by_length"][length_str],
                "word_order": r.get("word_order"),
                "language_family": r.get("language_family"),
            }
            output_data = {
                "ddm": r["ddm_by_length"][length_str],
                "mdd_obs": r["mdd_obs_by_length"][length_str],
                "mdd_random": r["mdd_random_by_length"][length_str],
            }
            len_ddm_examples.append({
                "input": json.dumps(input_data),
                "output": json.dumps(output_data),
                "metadata_treebank_id": r["config"],
                "metadata_sentence_length": int(length_str),
                "metadata_ddm": r["ddm_by_length"][length_str],
                "metadata_word_order": r.get("word_order"),
            })

    # Assemble final output
    output = {
        "metadata": {
            "description": "UD Treebank DDM Standardization & Core-Argument Dataset",
            "n_treebanks": len(results),
            "n_languages": len(set(r["lang_code"] for r in results)),
            "n_core_arg_records": len(core_arg_examples),
            "n_wals_languages": len(wals_examples),
            "n_length_ddm_records": len(len_ddm_examples),
            "reference_distribution": ref_dist,
            "min_sentence_length_for_ddm": MIN_SENT_LEN,
            "min_sentences_per_length": MIN_SENTS_PER_LEN,
            "max_sentence_length": MAX_SENT_LEN,
            "random_baseline_method": f"permutation_sampling_{N_PERMUTATIONS}",
            "source_ud": "commul/universal_dependencies (HuggingFace, UD v2.17)",
            "source_wals": "cldf-datasets/wals (GitHub, WALS Online)",
        },
        "datasets": [
            {"dataset": "ud_treebank_summaries", "examples": treebank_examples},
            {"dataset": "ud_core_argument_deps", "examples": core_arg_examples},
            {"dataset": "wals_word_order", "examples": wals_examples},
            {"dataset": "ud_sentence_length_ddm", "examples": len_ddm_examples},
        ],
    }

    # Save
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_mb = OUT_FILE.stat().st_size / 1e6
    elapsed = time.time() - t0
    logger.info(f"Saved {OUT_FILE} ({size_mb:.1f} MB) in {elapsed:.0f}s")
    ds_summary = [f"{d['dataset']} ({len(d['examples'])} examples)" for d in output["datasets"]]
    logger.info(f"Datasets: {ds_summary}")


if __name__ == "__main__":
    main()
