#!/usr/bin/env python3
"""Standardized DDM Rankings: Spearman ρ, Cohen's d, and Sensitivity Analysis.

Loads pre-aggregated UD DDM dataset (314 treebanks), computes naive vs standardized
DDM rankings, measures rank agreement, identifies most-shifted language families,
and runs sensitivity with alternative reference distributions.
"""

import glob
import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware / memory limits
# ---------------------------------------------------------------------------
_avail = psutil.virtual_memory().available
RAM_BUDGET = int(4 * 1024**3)  # 4 GB — plenty for this analysis
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

NUM_CPUS = len(os.sched_getaffinity(0))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/"
    "comp-ling-dobrovoljc_yqz/3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)
WORK_DIR = Path(__file__).resolve().parent
MAX_EXAMPLES: int | None = None  # Set to int for scaling tests


def load_dataset() -> dict:
    """Load all full_data_out shards and combine."""
    shards = sorted(glob.glob(str(DATA_DIR / "full_data_out" / "full_data_out_*.json")))
    if not shards:
        logger.warning("No full shards found, falling back to mini_data_out.json")
        return json.loads((DATA_DIR / "mini_data_out.json").read_text())

    logger.info(f"Loading {len(shards)} data shards")
    combined: dict | None = None
    for path in shards:
        logger.info(f"  Loading {Path(path).name}")
        shard = json.loads(Path(path).read_text())
        if combined is None:
            combined = shard
        else:
            # Merge datasets by name
            for ds_shard in shard["datasets"]:
                matched = False
                for ds_combined in combined["datasets"]:
                    if ds_combined["dataset"] == ds_shard["dataset"]:
                        ds_combined["examples"].extend(ds_shard["examples"])
                        matched = True
                        break
                if not matched:
                    combined["datasets"].append(ds_shard)
    return combined


def parse_treebanks(dataset: dict) -> list[dict]:
    """Extract treebank summaries into flat records."""
    ds = next(d for d in dataset["datasets"] if d["dataset"] == "ud_treebank_summaries")
    examples = ds["examples"]
    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]

    records = []
    for ex in examples:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        rec = {
            "treebank_id": inp["treebank_id"],
            "language_family": inp.get("language_family"),
            "word_order": inp.get("word_order"),
            "case_richness": inp.get("case_richness", 0) or 0,
            "n_sentences": inp.get("n_sentences", 0),
            "mean_sentence_length": inp.get("mean_sentence_length", 0),
            "naive_ddm": out["naive_ddm"],
            "standardized_ddm": out["standardized_ddm"],
            "ddm_by_length": {int(k): v for k, v in out.get("ddm_by_length", {}).items()},
            "sentence_length_distribution": {
                int(k): v for k, v in out.get("sentence_length_distribution", {}).items()
            },
        }
        records.append(rec)
    logger.info(f"Parsed {len(records)} treebank records")
    return records


def rank_descending(values: np.ndarray) -> np.ndarray:
    """Rank values descending (highest = rank 1). Uses average for ties."""
    return stats.rankdata(-values, method="average")


def compute_std_ddm(
    ddm_by_length: dict[int, float],
    ref_dist: dict[int, float],
) -> float:
    """Recompute standardized DDM using given reference distribution."""
    total_weight = 0.0
    weighted_sum = 0.0
    for n, ddm_n in ddm_by_length.items():
        w = ref_dist.get(n, 0.0)
        if w > 0:
            weighted_sum += ddm_n * w
            total_weight += w
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def build_pooled_ref(records: list[dict], filter_fn=None) -> dict[int, float]:
    """Build pooled reference distribution from treebank sentence length distributions."""
    pool: dict[int, float] = {}
    for rec in records:
        if filter_fn and not filter_fn(rec):
            continue
        for n, p in rec["sentence_length_distribution"].items():
            pool[n] = pool.get(n, 0.0) + p
    # Normalize
    total = sum(pool.values())
    if total > 0:
        pool = {n: v / total for n, v in pool.items()}
    return pool


@logger.catch
def main():
    # ------------------------------------------------------------------
    # Step 1-2: Load and parse
    # ------------------------------------------------------------------
    dataset = load_dataset()
    ref_dist = {int(k): v for k, v in dataset["metadata"]["reference_distribution"].items()}
    records = parse_treebanks(dataset)
    n = len(records)
    logger.info(f"Total treebanks: {n}")

    # ------------------------------------------------------------------
    # Step 3-4: Rankings
    # ------------------------------------------------------------------
    naive_vals = np.array([r["naive_ddm"] for r in records])
    std_vals = np.array([r["standardized_ddm"] for r in records])

    naive_ranks = rank_descending(naive_vals)
    std_ranks = rank_descending(std_vals)

    # Verify recomputed std DDM for first 5
    logger.info("Verifying standardized DDM for first 5 treebanks...")
    for i in range(min(5, n)):
        recomputed = compute_std_ddm(records[i]["ddm_by_length"], ref_dist)
        diff = abs(recomputed - records[i]["standardized_ddm"])
        logger.info(
            f"  {records[i]['treebank_id']}: dataset={records[i]['standardized_ddm']:.4f}, "
            f"recomputed={recomputed:.4f}, diff={diff:.4f}"
        )
        if diff > 0.02:
            logger.warning(f"  Large discrepancy for {records[i]['treebank_id']}")

    # ------------------------------------------------------------------
    # Step 5: Primary metrics
    # ------------------------------------------------------------------
    rho, rho_p = stats.spearmanr(naive_ranks, std_ranks)
    rank_shifts = std_ranks - naive_ranks
    abs_shifts = np.abs(rank_shifts)
    mean_abs_shift = float(np.mean(abs_shifts))
    max_shift = float(np.max(abs_shifts))
    # Cohen's d: mean rank shift / pooled SD
    cohens_d = float(np.mean(abs_shifts) / np.std(abs_shifts)) if np.std(abs_shifts) > 0 else 0.0

    logger.info(f"Spearman ρ = {rho:.4f} (p = {rho_p:.2e})")
    logger.info(f"Mean |rank shift| = {mean_abs_shift:.1f}, Max = {max_shift:.1f}")
    logger.info(f"Cohen's d (|shifts|) = {cohens_d:.4f}")

    # Sanity checks (only meaningful with enough data)
    if n > 20:
        assert 0 < rho < 1.0, f"Spearman ρ = {rho}, expected (0,1)"

    # ------------------------------------------------------------------
    # Step 6: Language family analysis
    # ------------------------------------------------------------------
    from collections import defaultdict

    family_data: dict[str, list] = defaultdict(list)
    for i, rec in enumerate(records):
        fam = rec["language_family"]
        if fam:
            family_data[fam].append({
                "rank_shift": float(rank_shifts[i]),
                "abs_rank_shift": float(abs_shifts[i]),
                "mean_sentence_length": rec["mean_sentence_length"],
                "case_richness": rec["case_richness"],
            })

    family_shifts = []
    for fam, items in family_data.items():
        family_shifts.append({
            "family": fam,
            "n": len(items),
            "mean_rank_shift": float(np.mean([x["rank_shift"] for x in items])),
            "mean_abs_rank_shift": float(np.mean([x["abs_rank_shift"] for x in items])),
            "mean_sent_len": float(np.mean([x["mean_sentence_length"] for x in items])),
            "mean_case_richness": float(np.mean([x["case_richness"] for x in items])),
        })
    family_shifts.sort(key=lambda x: x["mean_abs_rank_shift"], reverse=True)
    logger.info(f"Top 5 most-shifted families: {[f['family'] for f in family_shifts[:5]]}")

    # ------------------------------------------------------------------
    # Step 7: Linking mechanism correlations
    # ------------------------------------------------------------------
    # Use only treebanks with complete metadata
    complete = [
        i for i, r in enumerate(records)
        if r["language_family"] and r["word_order"] and r["case_richness"] is not None
    ]
    if len(complete) > 10:
        cr = np.array([records[i]["case_richness"] for i in complete])
        sl = np.array([records[i]["mean_sentence_length"] for i in complete])
        rs = np.array([abs_shifts[i] for i in complete])
        corr_case_sentlen = float(stats.spearmanr(cr, sl).statistic)
        corr_case_rankshift = float(stats.spearmanr(cr, rs).statistic)
        corr_sentlen_rankshift = float(stats.spearmanr(sl, rs).statistic)
    else:
        corr_case_sentlen = corr_case_rankshift = corr_sentlen_rankshift = None

    logger.info(
        f"Linking: case~sentlen={corr_case_sentlen}, case~shift={corr_case_rankshift}, "
        f"sentlen~shift={corr_sentlen_rankshift}"
    )

    # ------------------------------------------------------------------
    # Step 8: Sensitivity — alternative reference distributions
    # ------------------------------------------------------------------
    sensitivity = {}

    # Alt 1: Median treebank distribution
    sent_lens = [r["mean_sentence_length"] for r in records]
    median_sl = float(np.median(sent_lens))
    # Find treebank closest to median
    median_tb = min(records, key=lambda r: abs(r["mean_sentence_length"] - median_sl))
    median_ref = median_tb["sentence_length_distribution"]
    logger.info(f"Median ref treebank: {median_tb['treebank_id']} (mean_sl={median_tb['mean_sentence_length']:.1f})")

    median_std = np.array([compute_std_ddm(r["ddm_by_length"], median_ref) for r in records])
    median_ranks = rank_descending(median_std)
    rho_median_naive = float(stats.spearmanr(naive_ranks, median_ranks).statistic)
    rho_median_pooled = float(stats.spearmanr(std_ranks, median_ranks).statistic)
    sensitivity["median_ref"] = {
        "reference_treebank": median_tb["treebank_id"],
        "spearman_rho_vs_naive": rho_median_naive,
        "spearman_rho_vs_pooled_std": rho_median_pooled,
    }

    # Alt 2: Written-only (exclude "spoken" in treebank_id)
    written_ref = build_pooled_ref(records, filter_fn=lambda r: "spoken" not in r["treebank_id"].lower())
    written_std = np.array([compute_std_ddm(r["ddm_by_length"], written_ref) for r in records])
    written_ranks = rank_descending(written_std)
    rho_written_naive = float(stats.spearmanr(naive_ranks, written_ranks).statistic)
    rho_written_pooled = float(stats.spearmanr(std_ranks, written_ranks).statistic)
    sensitivity["written_ref"] = {
        "spearman_rho_vs_naive": rho_written_naive,
        "spearman_rho_vs_pooled_std": rho_written_pooled,
    }

    # Alt 3: Uniform distribution over n=3..40
    uniform_ref = {n: 1.0 / 38 for n in range(3, 41)}
    uniform_std = np.array([compute_std_ddm(r["ddm_by_length"], uniform_ref) for r in records])
    uniform_ranks = rank_descending(uniform_std)
    rho_uniform_naive = float(stats.spearmanr(naive_ranks, uniform_ranks).statistic)
    rho_uniform_pooled = float(stats.spearmanr(std_ranks, uniform_ranks).statistic)
    sensitivity["uniform_ref"] = {
        "spearman_rho_vs_naive": rho_uniform_naive,
        "spearman_rho_vs_pooled_std": rho_uniform_pooled,
    }

    logger.info(f"Sensitivity rhos vs naive: median={rho_median_naive:.4f}, written={rho_written_naive:.4f}, uniform={rho_uniform_naive:.4f}")

    # ------------------------------------------------------------------
    # Step 9: Word-order and case subgroup analysis
    # ------------------------------------------------------------------
    wo_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        wo = r["word_order"] or "unknown"
        wo_groups[wo].append(i)

    word_order_results = {}
    for wo, idxs in wo_groups.items():
        word_order_results[wo] = {
            "n": len(idxs),
            "mean_naive_ddm": float(np.mean([naive_vals[i] for i in idxs])),
            "mean_std_ddm": float(np.mean([std_vals[i] for i in idxs])),
            "mean_rank_shift": float(np.mean([rank_shifts[i] for i in idxs])),
            "mean_abs_rank_shift": float(np.mean([abs_shifts[i] for i in idxs])),
        }

    # Case richness bins: 0, 1-3, 4-6, 7+
    def case_bin(c: int) -> str:
        if c == 0:
            return "0"
        elif c <= 3:
            return "1-3"
        elif c <= 6:
            return "4-6"
        else:
            return "7+"

    case_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        case_groups[case_bin(r["case_richness"])].append(i)

    case_richness_results = {}
    for cb, idxs in case_groups.items():
        case_richness_results[cb] = {
            "n": len(idxs),
            "mean_naive_ddm": float(np.mean([naive_vals[i] for i in idxs])),
            "mean_std_ddm": float(np.mean([std_vals[i] for i in idxs])),
            "mean_rank_shift": float(np.mean([rank_shifts[i] for i in idxs])),
            "mean_abs_rank_shift": float(np.mean([abs_shifts[i] for i in idxs])),
        }

    # ------------------------------------------------------------------
    # Step 10: Build all_treebank_rankings
    # ------------------------------------------------------------------
    all_rankings = []
    for i, r in enumerate(records):
        all_rankings.append({
            "treebank_id": r["treebank_id"],
            "naive_ddm": r["naive_ddm"],
            "std_ddm": r["standardized_ddm"],
            "naive_rank": int(naive_ranks[i]),
            "std_rank": int(std_ranks[i]),
            "rank_shift": int(rank_shifts[i]),
            "language_family": r["language_family"],
            "word_order": r["word_order"],
            "case_richness": r["case_richness"],
            "mean_sentence_length": r["mean_sentence_length"],
        })

    # ------------------------------------------------------------------
    # Build output in exp_gen_sol_out schema format
    # ------------------------------------------------------------------
    method_metadata = {
        "method_name": "Standardized DDM Rankings Analysis",
        "description": "Compares naive vs sentence-length-standardized DDM rankings across 314 UD treebanks",
        "primary_results": {
            "spearman_rho": round(rho, 4),
            "spearman_p": float(rho_p),
            "cohens_d": round(cohens_d, 4),
            "n_treebanks": n,
            "mean_abs_rank_shift": round(mean_abs_shift, 2),
            "max_rank_shift": round(max_shift, 1),
        },
        "sensitivity": sensitivity,
        "family_shifts_top10": family_shifts[:10],
        "linking_mechanism": {
            "corr_case_sentlen": corr_case_sentlen,
            "corr_case_rankshift": corr_case_rankshift,
            "corr_sentlen_rankshift": corr_sentlen_rankshift,
        },
        "word_order_groups": word_order_results,
        "case_richness_groups": case_richness_results,
    }

    # Build examples: one per treebank with input=treebank info, output=ranking result
    examples = []
    for i, r in enumerate(records):
        inp_obj = {
            "treebank_id": r["treebank_id"],
            "language_family": r["language_family"],
            "word_order": r["word_order"],
            "case_richness": r["case_richness"],
            "n_sentences": r["n_sentences"],
            "mean_sentence_length": r["mean_sentence_length"],
        }
        out_obj = {
            "naive_ddm": r["naive_ddm"],
            "standardized_ddm": r["standardized_ddm"],
            "naive_rank": int(naive_ranks[i]),
            "std_rank": int(std_ranks[i]),
            "rank_shift": int(rank_shifts[i]),
            "median_ref_std_ddm": float(median_std[i]),
            "written_ref_std_ddm": float(written_std[i]),
            "uniform_ref_std_ddm": float(uniform_std[i]),
        }
        examples.append({
            "input": json.dumps(inp_obj),
            "output": json.dumps(out_obj),
            "predict_standardized_ddm": json.dumps({
                "std_rank": int(std_ranks[i]),
                "standardized_ddm": r["standardized_ddm"],
                "rank_shift": int(rank_shifts[i]),
            }),
            "predict_naive_ddm": json.dumps({
                "naive_rank": int(naive_ranks[i]),
                "naive_ddm": r["naive_ddm"],
            }),
            "metadata_treebank_id": r["treebank_id"],
            "metadata_naive_rank": int(naive_ranks[i]),
            "metadata_std_rank": int(std_ranks[i]),
            "metadata_rank_shift": int(rank_shifts[i]),
            "metadata_language_family": r["language_family"],
            "metadata_word_order": r["word_order"],
        })

    output = {
        "metadata": method_metadata,
        "datasets": [
            {
                "dataset": "ud_treebank_summaries",
                "examples": examples,
            }
        ],
    }

    out_path = WORK_DIR / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
