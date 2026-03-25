#!/usr/bin/env python3
"""Feasibility Check: Inter-treebank DDM(n) Variance at Fixed Sentence Lengths.

Loads per-dependency records from 19 data files, reconstructs sentence-level MDD,
computes DDM(n) per treebank at fixed n values, measures inter-treebank variance,
and compares naive vs length-standardized DDM rankings.
"""

import json
import math
import os
import resource
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
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

def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 30.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.6 * 1e9)  # 60% of RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_yqz"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus/data_out"
)
WORKSPACE = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_yqz"
    "/3_invention_loop/iter_2/gen_art/exp_id4_it2__opus"
)
TARGET_LENGTHS = [5, 10, 15, 20, 25, 30]
BIN_HALF_WIDTH = 2  # use n±2 bins if exact n has <20 sentences
MIN_SENTENCES = 10
NUM_FILES = 19
MAX_EXAMPLES: int | None = None  # Set to limit for testing


def process_file(file_path: str) -> dict[str, dict[str, list[float]]]:
    """Process one data file, return {sentence_id: {treebank_id, length, distances[]}}."""
    with open(file_path) as f:
        data = json.load(f)

    sentences: dict[str, dict[str, Any]] = {}

    for ds in data["datasets"]:
        examples = ds["examples"]
        limit = MAX_EXAMPLES if MAX_EXAMPLES else len(examples)

        for ex in examples[:limit]:
            sid = ex["metadata_sentence_id"]
            dist = int(ex["output"])
            if sid not in sentences:
                parts = ex["input"].split("|")
                # Strip _p0, _p1 etc. suffixes from split treebank names
                tb_raw = parts[0]
                import re
                tb_id = re.sub(r"_p\d+$", "", tb_raw)
                sentences[sid] = {
                    "treebank_id": tb_id,
                    "sentence_length": int(parts[4]),
                    "distances": [],
                }
            sentences[sid]["distances"].append(dist)

    return sentences


def aggregate_sentences(all_sentences: dict) -> tuple[
    dict[tuple[str, int], dict[str, list[float]]],
    dict[str, list[float]],
    dict[str, dict[str, float]],
]:
    """Group sentence data by (treebank, length) and compute per-sentence MDD.

    Returns:
        grouped: {(tb, length): {"obs": [mdd_obs...], "rand": [mdd_rand...]}}
        tb_all: {tb: [mdd_obs...]} for naive DDM
        tb_rand_all: {tb: {"obs_sum", "rand_sum"}} for naive DDM
    """
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"obs": [], "rand": []}
    )
    tb_naive: dict[str, dict[str, float]] = defaultdict(
        lambda: {"obs_sum": 0.0, "rand_sum": 0.0, "count": 0}
    )
    tb_lengths: dict[str, list[int]] = defaultdict(list)

    for sid, sdata in all_sentences.items():
        tb = sdata["treebank_id"]
        n = sdata["sentence_length"]
        dists = sdata["distances"]
        mdd_obs = np.mean(dists)
        mdd_rand = (n + 1) / 3.0

        grouped[(tb, n)]["obs"].append(mdd_obs)
        grouped[(tb, n)]["rand"].append(mdd_rand)
        tb_naive[tb]["obs_sum"] += mdd_obs
        tb_naive[tb]["rand_sum"] += mdd_rand
        tb_naive[tb]["count"] += 1
        tb_lengths[tb].append(n)

    return grouped, tb_lengths, tb_naive


@logger.catch
def main() -> None:
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")

    # ── Step 1: Load all files in parallel ───────────────────────────────
    logger.info(f"Loading {NUM_FILES} data files from {DATA_DIR}")
    all_sentences: dict[str, Any] = {}

    workers = min(NUM_CPUS, NUM_FILES)
    file_paths = [str(DATA_DIR / f"full_data_out_{i}.json") for i in range(1, NUM_FILES + 1)]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, fp): fp for fp in file_paths}
        for fut in as_completed(futures):
            try:
                sentences = fut.result()
                all_sentences.update(sentences)
                logger.info(f"Loaded {len(sentences)} sentences from {Path(futures[fut]).name}")
            except Exception:
                logger.error(f"Failed to process {futures[fut]}")
                raise

    logger.info(f"Total sentences loaded: {len(all_sentences)}")

    # ── Step 2: Aggregate ────────────────────────────────────────────────
    logger.info("Aggregating sentence-level MDD values")
    grouped, tb_lengths, tb_naive = aggregate_sentences(all_sentences)

    treebank_ids = sorted(tb_naive.keys())
    logger.info(f"Found {len(treebank_ids)} treebanks: {treebank_ids}")

    # ── Step 3: Compute DDM(n) per treebank at each target length ────────
    logger.info("Computing DDM(n) per treebank at target lengths")
    ddm_by_n: dict[int, dict[str, float]] = {}
    coverage_by_n: dict[int, dict[str, int]] = {}

    for n in TARGET_LENGTHS:
        ddm_by_n[n] = {}
        coverage_by_n[n] = {}
        for tb in treebank_ids:
            # Try exact length first
            key = (tb, n)
            if key in grouped and len(grouped[key]["obs"]) >= MIN_SENTENCES:
                mean_obs = np.mean(grouped[key]["obs"])
                mean_rand = np.mean(grouped[key]["rand"])
                ddm_by_n[n][tb] = 1.0 - mean_obs / mean_rand
                coverage_by_n[n][tb] = len(grouped[key]["obs"])
            else:
                # Fall back to bin n ± BIN_HALF_WIDTH
                obs_all, rand_all = [], []
                for delta in range(-BIN_HALF_WIDTH, BIN_HALF_WIDTH + 1):
                    bkey = (tb, n + delta)
                    if bkey in grouped:
                        obs_all.extend(grouped[bkey]["obs"])
                        rand_all.extend(grouped[bkey]["rand"])
                if len(obs_all) >= MIN_SENTENCES:
                    mean_obs = np.mean(obs_all)
                    mean_rand = np.mean(rand_all)
                    ddm_by_n[n][tb] = 1.0 - mean_obs / mean_rand
                    coverage_by_n[n][tb] = len(obs_all)

        logger.info(f"  n={n}: {len(ddm_by_n[n])}/{len(treebank_ids)} treebanks covered")

    # ── Step 4: Inter-treebank variance of DDM(n) at each n ─────────────
    logger.info("Computing inter-treebank variance stats")
    variance_stats: dict[str, dict[str, Any]] = {}
    for n in TARGET_LENGTHS:
        vals = list(ddm_by_n[n].values())
        if len(vals) >= 2:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            variance_stats[str(n)] = {
                "mean": mean_v,
                "std": std_v,
                "cv": std_v / mean_v if mean_v != 0 else float("inf"),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "num_treebanks": len(vals),
            }
            logger.info(
                f"  n={n}: mean={mean_v:.4f}, std={std_v:.4f}, "
                f"CV={variance_stats[str(n)]['cv']:.4f}, range={variance_stats[str(n)]['range']:.4f}"
            )

    # ── Step 5: Sentence-length distributions P(n) per treebank ──────────
    logger.info("Computing P(n) distributions per treebank")
    pn_distributions: dict[str, dict[str, float]] = {}
    for tb in treebank_ids:
        lengths = tb_lengths[tb]
        hist = Counter(lengths)
        total = sum(hist.values())
        pn_distributions[tb] = {str(n): hist.get(n, 0) / total for n in range(1, max(lengths) + 1)}

    # KL divergence between P(n) distributions
    all_lengths_set = set()
    for tb in treebank_ids:
        all_lengths_set.update(int(k) for k in pn_distributions[tb].keys())
    max_len = max(all_lengths_set)

    # Build P(n) arrays with smoothing
    pn_arrays: dict[str, np.ndarray] = {}
    for tb in treebank_ids:
        arr = np.array([float(pn_distributions[tb].get(str(n), 0)) for n in range(1, max_len + 1)])
        arr = arr + 1e-10  # smoothing
        arr = arr / arr.sum()
        pn_arrays[tb] = arr

    # Mean pairwise KL divergence
    kl_values = []
    for i, tb1 in enumerate(treebank_ids):
        for tb2 in treebank_ids[i + 1:]:
            kl = float(np.sum(pn_arrays[tb1] * np.log(pn_arrays[tb1] / pn_arrays[tb2])))
            kl_values.append(kl)
    mean_kl = float(np.mean(kl_values))
    logger.info(f"Mean pairwise KL divergence of P(n): {mean_kl:.4f}")

    # ── Step 6: Naive vs Standardized DDM rankings ──────────────────────
    logger.info("Computing naive vs standardized DDM rankings")

    # Naive DDM per treebank
    naive_ddm: dict[str, float] = {}
    for tb in treebank_ids:
        d = tb_naive[tb]
        naive_ddm[tb] = 1.0 - d["obs_sum"] / d["rand_sum"]

    # Pooled reference distribution P_ref(n) = average across all treebanks
    p_ref = np.zeros(max_len)
    for tb in treebank_ids:
        p_ref += pn_arrays[tb]
    p_ref /= len(treebank_ids)
    p_ref /= p_ref.sum()

    # Standardized DDM: sum_n DDM(n, tb) * P_ref(n)
    # Need DDM(n) for all n, not just target lengths. Use grouped data.
    standardized_ddm: dict[str, float] = {}
    for tb in treebank_ids:
        ddm_weighted_sum = 0.0
        weight_sum = 0.0
        for n_idx in range(max_len):
            n = n_idx + 1
            key = (tb, n)
            if key in grouped and len(grouped[key]["obs"]) >= 3:
                mean_obs = np.mean(grouped[key]["obs"])
                mean_rand = np.mean(grouped[key]["rand"])
                ddm_n = 1.0 - mean_obs / mean_rand
                w = p_ref[n_idx]
                ddm_weighted_sum += ddm_n * w
                weight_sum += w
        if weight_sum > 0:
            standardized_ddm[tb] = ddm_weighted_sum / weight_sum

    # Spearman correlation between naive and standardized rankings
    common_tbs = sorted(set(naive_ddm.keys()) & set(standardized_ddm.keys()))
    if len(common_tbs) < 3:
        logger.warning(f"Only {len(common_tbs)} treebanks have both naive and standardized DDM — need ≥3")
        spearman_rho, spearman_p = float("nan"), float("nan")
        rank_changes = {}
        max_rank_change = 0
    else:
        naive_vals = [naive_ddm[tb] for tb in common_tbs]
        std_vals = [standardized_ddm[tb] for tb in common_tbs]
        spearman_rho, spearman_p = stats.spearmanr(naive_vals, std_vals)
        logger.info(f"Spearman rho (naive vs standardized): {spearman_rho:.4f} (p={spearman_p:.4e})")

        naive_ranks = stats.rankdata(naive_vals)
        std_ranks = stats.rankdata(std_vals)
        rank_changes = {
            tb: {"naive_rank": int(nr), "std_rank": int(sr), "change": int(sr - nr)}
            for tb, nr, sr in zip(common_tbs, naive_ranks, std_ranks)
        }
        max_rank_change = max(abs(v["change"]) for v in rank_changes.values())
    logger.info(f"Max rank change: {max_rank_change}")

    # ── Step 7: Feasibility verdict ──────────────────────────────────────
    cv_values = [variance_stats[str(n)]["cv"] for n in TARGET_LENGTHS if str(n) in variance_stats]
    all_cv_low = all(cv < 0.05 for cv in cv_values)
    pn_similar = mean_kl < 0.1

    if all_cv_low:
        verdict = "FAIL"
        reason = "DDM(n) shows near-zero inter-treebank variance (all CV < 0.05)"
    elif pn_similar and spearman_rho > 0.99:
        verdict = "MARGINAL"
        reason = f"P(n) distributions similar (mean KL={mean_kl:.4f}) and rankings nearly identical (rho={spearman_rho:.4f})"
    else:
        verdict = "PASS"
        reason = (
            f"Meaningful DDM(n) variation across treebanks (mean CV={np.mean(cv_values):.4f}), "
            f"diverse P(n) distributions (mean KL={mean_kl:.4f}), "
            f"standardization changes rankings (rho={spearman_rho:.4f}, max rank change={max_rank_change})"
        )

    logger.info(f"Feasibility verdict: {verdict} — {reason}")

    # ── Build output ─────────────────────────────────────────────────────
    # Format as exp_gen_sol_out schema: {datasets: [{dataset, examples}]}
    examples = []
    for tb in common_tbs:
        # Input: treebank info
        input_str = f"treebank={tb}"

        # Build detailed output
        ddm_at_targets = {
            str(n): round(ddm_by_n[n][tb], 6)
            for n in TARGET_LENGTHS
            if tb in ddm_by_n.get(n, {})
        }
        output_data = {
            "naive_ddm": round(naive_ddm[tb], 6),
            "standardized_ddm": round(standardized_ddm[tb], 6),
            "ddm_at_target_lengths": ddm_at_targets,
            "naive_rank": rank_changes[tb]["naive_rank"],
            "standardized_rank": rank_changes[tb]["std_rank"],
            "rank_change": rank_changes[tb]["change"],
        }
        output_str = json.dumps(output_data)

        examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_treebank_id": tb,
            "predict_naive_ddm": str(round(naive_ddm[tb], 6)),
            "predict_standardized_ddm": str(round(standardized_ddm[tb], 6)),
        })

    method_out = {
        "metadata": {
            "method_name": "DDM_length_standardization_feasibility",
            "description": "Feasibility check for inter-treebank DDM(n) variance at fixed sentence lengths",
            "target_lengths": TARGET_LENGTHS,
            "bin_half_width": BIN_HALF_WIDTH,
            "min_sentences": MIN_SENTENCES,
            "num_treebanks": len(treebank_ids),
            "num_sentences_total": len(all_sentences),
            "spearman_rho_naive_vs_std": round(spearman_rho, 6),
            "spearman_p_value": float(spearman_p),
            "max_rank_change": max_rank_change,
            "mean_pairwise_kl": round(mean_kl, 6),
            "feasibility_verdict": verdict,
            "feasibility_reason": reason,
            "variance_stats_by_n": variance_stats,
        },
        "datasets": [
            {
                "dataset": "ud_ddm_feasibility",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Output saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
