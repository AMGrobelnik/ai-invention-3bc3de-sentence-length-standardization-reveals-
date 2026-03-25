#!/usr/bin/env python3
"""Comprehensive Results Synthesis with Bootstrap CIs.

Loads UD DDM datasets, runs three hypothesis tests (standardization ranking divergence,
Cox model for case vs word order, linking mechanism), bootstraps 1000 resamples for CIs,
and produces a pass/fail summary table against all success criteria.
"""

import gc
import glob
import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from loguru import logger
from scipy import stats

# === Logging ===
logger.remove()
LOG_FMT = "\033[92m{time:HH:mm:ss}\033[0m|{level:<7}|\033[96m{function}\033[0m| {message}"
logger.add(sys.stdout, level="INFO", format=LOG_FMT)
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG", format=LOG_FMT)

# === Hardware Detection ===
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

# RAM limit: 14GB (leaving headroom on 30GB system)
RAM_BUDGET = int(14 * 1024**3)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, {AVAILABLE_RAM_GB:.1f}GB available")

# === Paths ===
WORKSPACE = Path(__file__).parent
_ITER1_ART = WORKSPACE.parents[2] / "iter_1" / "gen_art"
DATA5_DIR = _ITER1_ART / "data_id5_it1__opus"
DATA4_DIR = _ITER1_ART / "data_id4_it1__opus"

N_BOOTSTRAP = 1000
RNG = np.random.RandomState(42)


def load_data5() -> tuple[list[dict], list[dict]]:
    """Load ud_treebank_summaries and ud_core_argument_deps from data_id5 split files."""
    logger.info("Loading data_id5 (3 split files)...")
    treebank_summaries = []
    core_arg_deps = []

    for fpath in sorted(glob.glob(str(DATA5_DIR / "full_data_out" / "full_data_out_*.json"))):
        logger.info(f"  Loading {Path(fpath).name}...")
        raw = json.loads(Path(fpath).read_text())
        for ds in raw["datasets"]:
            if ds["dataset"] == "ud_treebank_summaries":
                treebank_summaries.extend(ds["examples"])
            elif ds["dataset"] == "ud_core_argument_deps":
                core_arg_deps.extend(ds["examples"])
        del raw
        gc.collect()

    logger.info(f"Loaded {len(treebank_summaries)} treebank summaries, {len(core_arg_deps)} core arg deps")
    return treebank_summaries, core_arg_deps


def load_data4() -> list[dict]:
    """Load typological profiles from data_id4."""
    logger.info("Loading data_id4...")
    raw = json.loads((DATA4_DIR / "full_data_out.json").read_text())
    profiles = []
    for ds in raw["datasets"]:
        if ds["dataset"] == "full_typological_profiles":
            profiles.extend(ds["examples"])
        elif ds["dataset"] == "word_order_prediction":
            profiles.extend(ds["examples"])
    del raw
    gc.collect()
    logger.info(f"Loaded {len(profiles)} typological profiles")
    return profiles


def parse_treebank_summaries(examples: list[dict]) -> pd.DataFrame:
    """Parse treebank summary examples into a DataFrame."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        rows.append({
            "treebank_id": inp["treebank_id"],
            "iso_code": inp.get("iso_code"),
            "language_family": inp.get("language_family"),
            "word_order": inp.get("word_order"),
            "case_richness": inp.get("case_richness", 0),
            "n_sentences": inp.get("n_sentences", 0),
            "mean_sentence_length": inp.get("mean_sentence_length"),
            "naive_ddm": out["naive_ddm"],
            "standardized_ddm": out["standardized_ddm"],
        })
    return pd.DataFrame(rows)


def parse_core_arg_deps(examples: list[dict]) -> pd.DataFrame:
    """Parse core argument dep examples into a DataFrame."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        rows.append({
            "treebank_id": inp["treebank_id"],
            "deprel": inp.get("deprel"),
            "sentence_length": inp.get("sentence_length"),
            "case_value": inp.get("case_value"),
            "word_order": inp.get("word_order"),
            "language_family": inp.get("language_family"),
            "distance": out["distance"],
        })
    return pd.DataFrame(rows)


def parse_typological_profiles(examples: list[dict]) -> pd.DataFrame:
    """Parse typological profiles for word_order and language_family by iso_code."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        # word_order_prediction dataset has output as word order string
        wo = None
        if ex.get("metadata_task_type") == "classification":
            wo = ex.get("output")
        # full_typological_profiles might have word_order in input
        if wo is None and "word_order_81A" in inp:
            wo = inp.get("word_order_81A")

        rows.append({
            "iso_code": ex.get("metadata_language_code_ud", inp.get("language_name")),
            "language_family_typo": inp.get("language_family"),
            "word_order_typo": wo,
        })
    df = pd.DataFrame(rows)
    # Keep the most informative row per iso_code (prefer non-null word_order)
    df = df.sort_values("word_order_typo", na_position="last").drop_duplicates("iso_code", keep="first")
    return df


def enrich_treebank_df(df_tb: pd.DataFrame, df_typo: pd.DataFrame) -> pd.DataFrame:
    """Join typological profiles onto treebank summaries to fill missing word_order/family."""
    df = df_tb.merge(df_typo, on="iso_code", how="left")
    # Fill missing values from typological data
    df["word_order"] = df["word_order"].fillna(df["word_order_typo"])
    df["language_family"] = df["language_family"].fillna(df["language_family_typo"])
    df.drop(columns=["word_order_typo", "language_family_typo"], inplace=True)
    return df


# === Analysis 2a: Standardization Ranking Divergence ===
def analysis_standardization(df_tb: pd.DataFrame) -> dict:
    """Test if standardization changes treebank rankings (rho < 0.85, Cohen's d > 0.5)."""
    logger.info("=== Analysis 2a: Standardization Ranking Divergence ===")
    df = df_tb.dropna(subset=["naive_ddm", "standardized_ddm"]).copy()
    logger.info(f"  Using {len(df)} treebanks with both DDM values")

    naive_ranks = df["naive_ddm"].rank(ascending=False)
    std_ranks = df["standardized_ddm"].rank(ascending=False)
    rank_shifts = naive_ranks - std_ranks

    rho, p_rho = stats.spearmanr(df["naive_ddm"], df["standardized_ddm"])
    cohens_d = abs(rank_shifts.mean()) / rank_shifts.std() if rank_shifts.std() > 0 else 0.0

    # Also compute mean absolute rank shift
    mean_abs_shift = rank_shifts.abs().mean()

    logger.info(f"  Spearman rho={rho:.4f} (p={p_rho:.2e}), Cohen's d={cohens_d:.4f}, mean|shift|={mean_abs_shift:.1f}")

    # Bootstrap
    rho_boots, d_boots = [], []
    n = len(df)
    for _ in range(N_BOOTSTRAP):
        idx = RNG.choice(n, size=n, replace=True)
        s = df.iloc[idx]
        r, _ = stats.spearmanr(s["naive_ddm"], s["standardized_ddm"])
        nr = s["naive_ddm"].rank(ascending=False)
        sr = s["standardized_ddm"].rank(ascending=False)
        rs = nr.values - sr.values
        d = abs(rs.mean()) / rs.std() if rs.std() > 0 else 0.0
        rho_boots.append(r)
        d_boots.append(d)

    rho_ci = [float(np.percentile(rho_boots, 2.5)), float(np.percentile(rho_boots, 97.5))]
    d_ci = [float(np.percentile(d_boots, 2.5)), float(np.percentile(d_boots, 97.5))]

    logger.info(f"  Bootstrap CIs: rho={rho_ci}, d={d_ci}")

    return {
        "criterion_1_rho": {
            "value": float(rho), "p_value": float(p_rho),
            "ci_95": rho_ci, "threshold": "< 0.85",
            "pass": bool(rho < 0.85),
        },
        "criterion_1_cohens_d": {
            "value": float(cohens_d),
            "ci_95": d_ci, "threshold": "> 0.5",
            "pass": bool(cohens_d > 0.5),
        },
        "mean_absolute_rank_shift": float(mean_abs_shift),
        "n_treebanks": int(len(df)),
    }


# === Analysis 2b: Cox Proportional Hazards ===
def analysis_cox(df_deps: pd.DataFrame, df_tb: pd.DataFrame) -> dict:
    """Cox PH model: case marking vs word order as predictors of dependency distance."""
    logger.info("=== Analysis 2b: Cox Proportional Hazards ===")

    # Merge case_richness from treebank summaries
    tb_case = df_tb[["treebank_id", "case_richness"]].drop_duplicates("treebank_id")
    df = df_deps.merge(tb_case, on="treebank_id", how="left", suffixes=("", "_tb"))
    if "case_richness_tb" in df.columns:
        df["case_richness"] = df["case_richness"].fillna(df["case_richness_tb"])
        df.drop(columns=["case_richness_tb"], inplace=True)
    elif "case_richness" not in df.columns:
        df["case_richness"] = 0

    df["case_richness"] = df["case_richness"].fillna(0)

    # Filter to rows with known word_order
    df_wo = df[df["word_order"].notna() & (df["word_order"] != "")].copy()
    logger.info(f"  Rows with word_order: {len(df_wo)} / {len(df)}")

    if len(df_wo) < 100:
        logger.warning("  Too few rows with word_order, using case_richness only")
        return _cox_case_only(df)

    # Create word_order dummies
    top_orders = df_wo["word_order"].value_counts().head(3).index.tolist()
    for wo in top_orders:
        df_wo[f"wo_{wo}"] = (df_wo["word_order"] == wo).astype(int)

    # Create deprel dummies
    deprels = df_wo["deprel"].unique().tolist()
    for dep in deprels[1:]:  # drop first as reference
        df_wo[f"dep_{dep}"] = (df_wo["deprel"] == dep).astype(int)

    # Subsample for Cox (500k is too large, use 50k)
    if len(df_wo) > 50000:
        df_cox = df_wo.sample(n=50000, random_state=42).copy()
        logger.info(f"  Subsampled to {len(df_cox)} for Cox model")
    else:
        df_cox = df_wo.copy()

    df_cox["event"] = 1  # all observed
    df_cox["distance"] = df_cox["distance"].clip(lower=1)  # Cox needs positive durations

    covariates = ["case_richness"] + [c for c in df_cox.columns if c.startswith("wo_") or c.startswith("dep_")]

    try:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter(penalizer=0.01)
        cox_df = df_cox[["distance", "event"] + covariates].dropna()
        logger.info(f"  Fitting Cox model on {len(cox_df)} rows, {len(covariates)} covariates")
        cph.fit(cox_df, duration_col="distance", event_col="event")

        summary = cph.summary
        logger.info(f"  Cox summary:\n{summary.to_string()[:2000]}")

        # Extract HRs
        hr_case = float(np.exp(summary.loc["case_richness", "coef"]))
        p_case = float(summary.loc["case_richness", "p"])

        wo_cols = [c for c in covariates if c.startswith("wo_")]
        if wo_cols:
            hr_wo_vals = {c: float(np.exp(summary.loc[c, "coef"])) for c in wo_cols}
            hr_wo_max_col = max(hr_wo_vals, key=lambda k: abs(hr_wo_vals[k] - 1))
            hr_wo = hr_wo_vals[hr_wo_max_col]
            p_wo = float(summary.loc[hr_wo_max_col, "p"])
        else:
            hr_wo = 1.0
            p_wo = 1.0

        logger.info(f"  HR_case={hr_case:.4f} (p={p_case:.2e}), HR_wo_max={hr_wo:.4f} (p={p_wo:.2e})")

        # Bootstrap CIs for HRs
        hr_case_boots, hr_wo_boots = [], []
        n_cox = len(cox_df)
        for b in range(N_BOOTSTRAP):
            try:
                idx = RNG.choice(n_cox, size=min(n_cox, 10000), replace=True)
                bs = cox_df.iloc[idx].copy()
                cph_b = CoxPHFitter(penalizer=0.01)
                cph_b.fit(bs, duration_col="distance", event_col="event")
                hr_case_boots.append(float(np.exp(cph_b.summary.loc["case_richness", "coef"])))
                if wo_cols:
                    hr_wo_boots.append(float(np.exp(cph_b.summary.loc[hr_wo_max_col, "coef"])))
            except Exception:
                continue

        hr_case_ci = [float(np.percentile(hr_case_boots, 2.5)), float(np.percentile(hr_case_boots, 97.5))] if hr_case_boots else [hr_case, hr_case]
        hr_wo_ci = [float(np.percentile(hr_wo_boots, 2.5)), float(np.percentile(hr_wo_boots, 97.5))] if hr_wo_boots else [hr_wo, hr_wo]

        # PH assumption check
        ph_ok = True
        try:
            cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
        except Exception as e:
            ph_ok = False
            logger.warning(f"  PH assumption check: {str(e)[:200]}")

        case_effect_larger = abs(hr_case - 1) > abs(hr_wo - 1)

        return {
            "criterion_2_case_hr": {
                "value": hr_case, "ci_95": hr_case_ci, "p_value": p_case,
            },
            "criterion_2_wo_max_hr": {
                "value": hr_wo, "ci_95": hr_wo_ci, "p_value": p_wo,
                "word_order_variable": hr_wo_max_col if wo_cols else None,
            },
            "criterion_2_case_gt_wo": {
                "case_effect_larger": bool(case_effect_larger),
                "case_p_lt_001": bool(p_case < 0.01),
                "pass": bool(case_effect_larger and p_case < 0.01),
            },
            "ph_assumption_holds": bool(ph_ok),
            "n_observations": int(len(cox_df)),
            "n_bootstrap_successful": len(hr_case_boots),
            "model": "Cox PH (penalizer=0.01)",
        }

    except Exception as e:
        logger.error(f"Cox model failed: {e}")
        return _cox_fallback_logistic(df_wo, covariates=["case_richness"] + [c for c in df_wo.columns if c.startswith("wo_")])


def _cox_case_only(df: pd.DataFrame) -> dict:
    """Fallback: only use case_richness without word order comparison."""
    logger.warning("  Fallback: case-only analysis (no word order)")
    corr, p = stats.spearmanr(df["case_richness"], df["distance"])
    return {
        "criterion_2_case_hr": {"value": float(corr), "p_value": float(p)},
        "criterion_2_wo_max_hr": {"value": None, "p_value": None},
        "criterion_2_case_gt_wo": {"pass": False, "note": "Insufficient word_order data"},
        "model": "Spearman correlation (fallback)",
    }


def _cox_fallback_logistic(df: pd.DataFrame, covariates: list[str]) -> dict:
    """Fallback: logistic regression on median-split distance."""
    from sklearn.linear_model import LogisticRegression
    logger.warning("  Fallback: logistic regression")

    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    median_dist = df["distance"].median()
    y = (df["distance"] > median_dist).astype(int)
    X = df[covariates].fillna(0)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, y)

    coefs = dict(zip(covariates, lr.coef_[0]))
    return {
        "criterion_2_case_hr": {"value": float(np.exp(coefs.get("case_richness", 0))), "p_value": None},
        "criterion_2_wo_max_hr": {"value": None, "p_value": None},
        "criterion_2_case_gt_wo": {"pass": False, "note": "Logistic regression fallback"},
        "model": "Logistic regression (fallback)",
    }


# === Analysis 2c: Linking Mechanism ===
def analysis_linking(df_tb: pd.DataFrame) -> dict:
    """Test linking mechanism: case-rich languages → longer sentences → larger rank shifts."""
    logger.info("=== Analysis 2c: Linking Mechanism ===")

    # Per-language aggregation (use iso_code to avoid treebank duplication)
    df = df_tb.dropna(subset=["naive_ddm", "standardized_ddm"]).copy()
    naive_ranks = df["naive_ddm"].rank(ascending=False)
    std_ranks = df["standardized_ddm"].rank(ascending=False)
    df["rank_shift"] = (naive_ranks - std_ranks).abs()

    # Aggregate by iso_code
    lang = df.groupby("iso_code").agg({
        "case_richness": "first",
        "mean_sentence_length": "mean",
        "rank_shift": "mean",
    }).dropna()

    logger.info(f"  {len(lang)} languages for linking analysis")

    if len(lang) < 10:
        logger.warning("  Too few languages for linking analysis")
        return {"criterion_3_linking": {"pass": False, "note": "Too few languages"}}

    corr_cs, p_cs = stats.spearmanr(lang["case_richness"], lang["mean_sentence_length"])
    corr_cr, p_cr = stats.spearmanr(lang["case_richness"], lang["rank_shift"])
    corr_sr, p_sr = stats.spearmanr(lang["mean_sentence_length"], lang["rank_shift"])

    logger.info(f"  case~sentlen: r={corr_cs:.3f} (p={p_cs:.2e})")
    logger.info(f"  case~rankshift: r={corr_cr:.3f} (p={p_cr:.2e})")
    logger.info(f"  sentlen~rankshift: r={corr_sr:.3f} (p={p_sr:.2e})")

    # Bootstrap CIs
    boot_cs, boot_cr, boot_sr = [], [], []
    n = len(lang)
    for _ in range(N_BOOTSTRAP):
        idx = RNG.choice(n, size=n, replace=True)
        s = lang.iloc[idx]
        r1, _ = stats.spearmanr(s["case_richness"], s["mean_sentence_length"])
        r2, _ = stats.spearmanr(s["case_richness"], s["rank_shift"])
        r3, _ = stats.spearmanr(s["mean_sentence_length"], s["rank_shift"])
        boot_cs.append(r1)
        boot_cr.append(r2)
        boot_sr.append(r3)

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    # Check: all positive and at least one significant
    all_positive = corr_cs > 0 and corr_cr > 0 and corr_sr > 0
    any_sig = p_cs < 0.05 or p_cr < 0.05 or p_sr < 0.05

    return {
        "criterion_3_linking": {
            "corr_case_sentlen": {"value": float(corr_cs), "p_value": float(p_cs), "ci_95": ci(boot_cs)},
            "corr_case_rankshift": {"value": float(corr_cr), "p_value": float(p_cr), "ci_95": ci(boot_cr)},
            "corr_sentlen_rankshift": {"value": float(corr_sr), "p_value": float(p_sr), "ci_95": ci(boot_sr)},
            "all_positive": bool(all_positive),
            "any_significant": bool(any_sig),
            "pass": bool(all_positive and any_sig),
        },
        "n_languages": int(len(lang)),
    }


# === Baseline: Naive Ranking (no standardization) ===
def baseline_naive_ranking(df_tb: pd.DataFrame) -> dict:
    """Baseline: just use naive DDM rankings without any standardization."""
    logger.info("=== Baseline: Naive DDM Ranking ===")
    df = df_tb.dropna(subset=["naive_ddm"]).copy()
    # Baseline assumes naive and standardized are the same → rho=1, d=0
    # We report descriptive stats of naive DDM
    return {
        "method": "naive_ddm_ranking",
        "description": "Baseline using naive (unstandardized) DDM — assumes sentence length has no confounding effect",
        "mean_naive_ddm": float(df["naive_ddm"].mean()),
        "std_naive_ddm": float(df["naive_ddm"].std()),
        "n_treebanks": int(len(df)),
    }


def format_output(
    results_std: dict,
    results_cox: dict,
    results_linking: dict,
    baseline: dict,
    df_tb: pd.DataFrame,
) -> dict:
    """Format results as exp_gen_sol_out.json schema."""
    # Overall pass
    pass_1 = results_std.get("criterion_1_rho", {}).get("pass", False) and results_std.get("criterion_1_cohens_d", {}).get("pass", False)
    pass_2 = results_cox.get("criterion_2_case_gt_wo", {}).get("pass", False)
    pass_3 = results_linking.get("criterion_3_linking", {}).get("pass", False)
    overall = pass_1 and pass_2 and pass_3

    summary = {
        "criterion_1_standardization": pass_1,
        "criterion_2_cox_case_vs_wo": pass_2,
        "criterion_3_linking_mechanism": pass_3,
        "overall_confirmed": overall,
    }

    full_results = {
        "summary": summary,
        "standardization_analysis": results_std,
        "cox_analysis": results_cox,
        "linking_analysis": results_linking,
        "baseline": baseline,
    }

    # Build per-treebank examples for the output schema
    examples = []
    df = df_tb.dropna(subset=["naive_ddm", "standardized_ddm"]).copy()
    naive_ranks = df["naive_ddm"].rank(ascending=False)
    std_ranks = df["standardized_ddm"].rank(ascending=False)

    for i, (_, row) in enumerate(df.iterrows()):
        inp = json.dumps({
            "treebank_id": row["treebank_id"],
            "iso_code": row.get("iso_code"),
            "word_order": row.get("word_order"),
            "case_richness": int(row.get("case_richness", 0)),
            "naive_ddm": float(row["naive_ddm"]),
            "standardized_ddm": float(row["standardized_ddm"]),
        })
        out = json.dumps({
            "naive_rank": int(naive_ranks.iloc[i]),
            "standardized_rank": int(std_ranks.iloc[i]),
            "rank_shift": int(naive_ranks.iloc[i] - std_ranks.iloc[i]),
        })
        examples.append({
            "input": inp,
            "output": out,
            "predict_baseline": json.dumps({"rank": int(naive_ranks.iloc[i])}),
            "predict_our_method": json.dumps({"rank": int(std_ranks.iloc[i])}),
            "metadata_treebank_id": row["treebank_id"],
            "metadata_word_order": row.get("word_order"),
            "metadata_case_richness": int(row.get("case_richness", 0)),
        })

    return {
        "metadata": {
            "method_name": "comprehensive_results_synthesis",
            "description": "Bootstrap CI analysis of DDM standardization, Cox PH for case vs word order, and linking mechanism",
            "n_bootstrap": N_BOOTSTRAP,
            "results": full_results,
        },
        "datasets": [
            {
                "dataset": "ud_treebank_summaries",
                "examples": examples,
            }
        ],
    }


@logger.catch
def main():
    logger.info("Starting comprehensive results synthesis...")

    # Load data
    tb_examples, dep_examples = load_data5()
    typo_examples = load_data4()

    # Parse into DataFrames
    logger.info("Parsing DataFrames...")
    df_tb = parse_treebank_summaries(tb_examples)
    del tb_examples
    gc.collect()

    df_deps = parse_core_arg_deps(dep_examples)
    del dep_examples
    gc.collect()

    df_typo = parse_typological_profiles(typo_examples)
    del typo_examples
    gc.collect()

    logger.info(f"df_tb: {df_tb.shape}, df_deps: {df_deps.shape}, df_typo: {df_typo.shape}")

    # Enrich treebank data with typological profiles
    df_tb = enrich_treebank_df(df_tb, df_typo)
    del df_typo
    gc.collect()

    wo_coverage = df_tb["word_order"].notna().mean()
    logger.info(f"Word order coverage after enrichment: {wo_coverage:.1%}")

    # Also enrich deps with word_order from treebank data
    tb_wo = df_tb[["treebank_id", "word_order", "case_richness"]].drop_duplicates("treebank_id")
    df_deps = df_deps.merge(tb_wo, on="treebank_id", how="left", suffixes=("", "_tb"))
    if "word_order_tb" in df_deps.columns:
        df_deps["word_order"] = df_deps["word_order"].fillna(df_deps["word_order_tb"])
        df_deps.drop(columns=["word_order_tb"], inplace=True)
    if "case_richness_tb" in df_deps.columns:
        df_deps["case_richness"] = df_deps.get("case_richness", pd.Series(dtype=float)).fillna(df_deps["case_richness_tb"])
        df_deps.drop(columns=["case_richness_tb"], inplace=True)

    # Run analyses
    results_std = analysis_standardization(df_tb)
    results_cox = analysis_cox(df_deps, df_tb)

    del df_deps
    gc.collect()

    results_linking = analysis_linking(df_tb)
    baseline = baseline_naive_ranking(df_tb)

    # Format and save
    output = format_output(results_std, results_cox, results_linking, baseline, df_tb)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {out_path}")

    # Log summary
    summary = output["metadata"]["results"]["summary"]
    logger.info(f"=== SUMMARY ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {'PASS' if v else 'FAIL'}")

    return output


if __name__ == "__main__":
    main()
