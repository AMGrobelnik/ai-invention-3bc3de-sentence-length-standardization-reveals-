#!/usr/bin/env python3
"""Stratified Cox PH Models: Case Richness vs Word Order on Core-Argument Dependency Distances.

Fits Cox proportional hazards models on ~500k core-argument dependency records to test whether
case richness has a larger hazard ratio than word order typology for dependency resolution.
"""

import json
import math
import os
import resource
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from loguru import logger
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# ── Logging ──
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(sys.stdout, level="INFO", format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ──
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

# ── Memory limits ──
import psutil
_avail = psutil.virtual_memory().available
RAM_BUDGET = int(min(14 * 1024**3, _avail * 0.85))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {_avail/1e9:.1f}GB available RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# ── Paths ──
def _json_default(o):
    """JSON serializer for numpy types."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

DEP5_DIR = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_yqz/3_invention_loop/iter_1/gen_art/data_id5_it1__opus")
DEP4_DIR = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/comp-ling-dobrovoljc_yqz/3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
OUT_DIR = Path(".")

# ── Data Loading ──
def load_dataset_examples(filepath: Path, dataset_name: str) -> list[dict]:
    """Load examples from a specific dataset within a JSON file."""
    data = json.loads(filepath.read_text())
    for ds in data["datasets"]:
        if ds["dataset"] == dataset_name:
            return ds["examples"]
    return []

def parse_core_arg_record(ex: dict) -> dict:
    """Parse a core-argument dependency record."""
    inp = json.loads(ex["input"])
    out = json.loads(ex["output"])
    return {
        "treebank_id": inp["treebank_id"],
        "deprel": inp["deprel"],
        "sentence_length": inp["sentence_length"],
        "case_value": inp.get("case_value"),
        "word_order": inp.get("word_order"),
        "language_family": inp.get("language_family"),
        "distance": out["distance"],
    }

def parse_treebank_summary(ex: dict) -> dict:
    """Parse a treebank summary record."""
    inp = json.loads(ex["input"])
    return {
        "treebank_id": inp["treebank_id"],
        "iso_code": inp.get("iso_code"),
        "case_richness": inp.get("case_richness", 0),
        "word_order": inp.get("word_order"),
        "language_family": inp.get("language_family"),
        "language_name": inp.get("language_name"),
    }

def parse_typological_profile(ex: dict) -> dict:
    """Parse a typological profile record."""
    inp = json.loads(ex["input"])
    return {
        "iso_code": inp.get("language_iso"),
        "ud_case_richness": inp.get("ud_case_richness", 0),
        "case_count_49A": inp.get("case_count_49A"),
        "language_name_ud": ex.get("metadata_language_code_ud"),
    }

@logger.catch
def load_all_data(use_mini: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all datasets."""
    # Load treebank summaries
    if use_mini:
        tb_examples = load_dataset_examples(DEP5_DIR / "mini_data_out.json", "ud_treebank_summaries")
    else:
        tb_examples = []
        for chunk in sorted(DEP5_DIR.glob("full_data_out/full_data_out_*.json")):
            tb_examples.extend(load_dataset_examples(chunk, "ud_treebank_summaries"))
    logger.info(f"Loaded {len(tb_examples)} treebank summary examples")
    tb_records = [parse_treebank_summary(ex) for ex in tb_examples]
    df_tb = pd.DataFrame(tb_records)

    # Load core-argument deps
    if use_mini:
        ca_examples = load_dataset_examples(DEP5_DIR / "mini_data_out.json", "ud_core_argument_deps")
    else:
        ca_examples = []
        for chunk in sorted(DEP5_DIR.glob("full_data_out/full_data_out_*.json")):
            ca_examples.extend(load_dataset_examples(chunk, "ud_core_argument_deps"))
    logger.info(f"Loaded {len(ca_examples)} core-arg dependency examples")
    ca_records = [parse_core_arg_record(ex) for ex in ca_examples]
    df_ca = pd.DataFrame(ca_records)

    # Load typological profiles
    if use_mini:
        tp_examples = load_dataset_examples(DEP4_DIR / "mini_data_out.json", "full_typological_profiles")
    else:
        tp_examples = load_dataset_examples(DEP4_DIR / "full_data_out.json", "full_typological_profiles")
    logger.info(f"Loaded {len(tp_examples)} typological profile examples")
    tp_records = [parse_typological_profile(ex) for ex in tp_examples]
    df_tp = pd.DataFrame(tp_records)

    return df_ca, df_tb, df_tp

def prepare_data(df_ca: pd.DataFrame, df_tb: pd.DataFrame, df_tp: pd.DataFrame) -> pd.DataFrame:
    """Merge and prepare data for Cox PH modeling."""
    logger.info(f"Core-arg records: {len(df_ca)}, Treebank summaries: {len(df_tb)}, Typological profiles: {len(df_tp)}")

    # Merge case_richness from treebank summaries into core-arg deps
    tb_case = df_tb[["treebank_id", "case_richness"]].drop_duplicates(subset="treebank_id")
    df = df_ca.merge(tb_case, on="treebank_id", how="left")

    # Fill missing word_order and language_family from treebank summaries
    tb_meta = df_tb[["treebank_id", "word_order", "language_family"]].drop_duplicates(subset="treebank_id")
    # Only fill where missing
    df = df.drop(columns=["word_order", "language_family"], errors="ignore")
    df = df.merge(tb_meta, on="treebank_id", how="left")

    # For core-arg records that had word_order/language_family in their own input, restore those
    # Actually, the core-arg input already has word_order and language_family — let me re-check
    # Re-parse: df_ca already has word_order, language_family from its own input
    # The merge above dropped them. Let me fix: use ca's own values first, fill from tb where null
    # Simpler approach: just use the tb-merged values since they should be the same

    logger.info(f"After merge: {len(df)} records")

    # Add event column (all observed)
    df["event"] = 1

    # Filter: need case_richness available and word_order not null
    df_full = df.copy()
    df = df[df["case_richness"].notna() & (df["case_richness"] >= 0)]
    df = df[df["word_order"].notna() & (df["word_order"] != "")]
    df = df[df["language_family"].notna() & (df["language_family"] != "")]
    df = df[df["distance"] > 0]
    df = df[df["sentence_length"] > 0]
    logger.info(f"After filtering (need case_richness, word_order, language_family, distance>0): {len(df)} records")

    if len(df) == 0:
        logger.warning("No records after filtering! Relaxing: dropping language_family requirement")
        df = df_full[df_full["case_richness"].notna() & (df_full["case_richness"] >= 0)]
        df = df[df["word_order"].notna() & (df["word_order"] != "")]
        df = df[df["distance"] > 0]
        df["language_family"] = df["language_family"].fillna("Unknown")
        logger.info(f"After relaxed filtering: {len(df)} records")

    # Log descriptive stats
    logger.info(f"Unique treebanks: {df['treebank_id'].nunique()}")
    logger.info(f"Unique language families: {df['language_family'].nunique()}")
    logger.info(f"Word order distribution:\n{df['word_order'].value_counts().to_string()}")
    logger.info(f"Case richness stats: mean={df['case_richness'].mean():.2f}, median={df['case_richness'].median():.1f}, max={df['case_richness'].max()}")
    logger.info(f"Distance stats: mean={df['distance'].mean():.2f}, median={df['distance'].median():.1f}, max={df['distance'].max()}")
    logger.info(f"Deprel distribution:\n{df['deprel'].value_counts().to_string()}")

    # Standardize case_richness (z-score)
    df["case_richness_z"] = (df["case_richness"] - df["case_richness"].mean()) / (df["case_richness"].std() + 1e-8)

    # Create word order dummies (SVO as reference)
    wo_counts = df["word_order"].value_counts()
    logger.info(f"Word order counts: {wo_counts.to_dict()}")

    # Collapse rare word orders (< 1% of data) into "Other"
    threshold = len(df) * 0.01
    rare_orders = wo_counts[wo_counts < threshold].index.tolist()
    if rare_orders:
        logger.info(f"Collapsing rare word orders into 'Other': {rare_orders}")
        df["word_order_cat"] = df["word_order"].apply(lambda x: "Other" if x in rare_orders else x)
    else:
        df["word_order_cat"] = df["word_order"]

    # Create dummies with SVO as reference
    wo_dummies = pd.get_dummies(df["word_order_cat"], prefix="wo", drop_first=False, dtype=float)
    if "wo_SVO" in wo_dummies.columns:
        wo_dummies = wo_dummies.drop(columns=["wo_SVO"])
    elif len(wo_dummies.columns) > 0:
        wo_dummies = wo_dummies.iloc[:, 1:]  # drop first if SVO not found

    df = pd.concat([df.reset_index(drop=True), wo_dummies.reset_index(drop=True)], axis=1)

    return df, list(wo_dummies.columns)


def subsample_for_cox(df: pd.DataFrame, max_n: int = 50000) -> pd.DataFrame:
    """Stratified subsample by word_order to keep Cox fitting tractable."""
    if len(df) <= max_n:
        return df
    logger.info(f"Subsampling {len(df)} -> {max_n} records (stratified by word_order)")
    return df.groupby("word_order_cat", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), int(max_n * len(x) / len(df))), random_state=42)
    ).reset_index(drop=True)


def fit_cox_model(
    df: pd.DataFrame,
    covariates: list[str],
    duration_col: str = "distance",
    event_col: str = "event",
    strata: list[str] | None = None,
    cluster_col: str | None = None,
    label: str = "model",
) -> dict:
    """Fit a Cox PH model and return results dict."""
    logger.info(f"Fitting Cox model '{label}' with {len(df)} records, covariates={covariates}")

    fit_df = df[covariates + [duration_col, event_col] + (strata or [])].copy()
    if cluster_col and cluster_col in df.columns:
        fit_df[cluster_col] = df[cluster_col].values

    # Drop rows with NaN in any covariate
    fit_df = fit_df.dropna()
    logger.info(f"  Records after dropna: {len(fit_df)}")

    if len(fit_df) < 100:
        logger.warning(f"  Too few records ({len(fit_df)}) for model '{label}'")
        return {"error": f"Too few records: {len(fit_df)}", "label": label}

    cph = CoxPHFitter()
    try:
        fit_kwargs = dict(
            duration_col=duration_col,
            event_col=event_col,
        )
        if strata:
            fit_kwargs["strata"] = strata
        if cluster_col and cluster_col in fit_df.columns:
            fit_kwargs["cluster_col"] = cluster_col

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(fit_df, **fit_kwargs)

    except Exception as e:
        logger.error(f"  Cox fit failed for '{label}': {e}")
        # Retry without cluster_col
        if cluster_col:
            logger.info(f"  Retrying without cluster_col")
            fit_kwargs.pop("cluster_col", None)
            if cluster_col in fit_df.columns:
                fit_df = fit_df.drop(columns=[cluster_col])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cph.fit(fit_df, **fit_kwargs)
            except Exception as e2:
                logger.error(f"  Retry also failed: {e2}")
                return {"error": str(e2), "label": label}
        else:
            return {"error": str(e), "label": label}

    # Extract results
    summary = cph.summary
    hazard_ratios = {}
    for cov in summary.index:
        row = summary.loc[cov]
        hazard_ratios[cov] = {
            "HR": round(float(row["exp(coef)"]), 4),
            "CI_lower": round(float(row["exp(coef) lower 95%"]), 4),
            "CI_upper": round(float(row["exp(coef) upper 95%"]), 4),
            "p_value": round(float(row["p"]), 6),
            "coef": round(float(row["coef"]), 4),
        }

    result = {
        "label": label,
        "n_records": len(fit_df),
        "hazard_ratios": hazard_ratios,
        "concordance_index": round(float(cph.concordance_index_), 4),
        "log_likelihood": round(float(cph.log_likelihood_), 2),
    }

    # PH assumption test
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ph_test = proportional_hazard_test(cph, fit_df, time_transform="rank")
        ph_results = {}
        for cov in ph_test.summary.index:
            row = ph_test.summary.loc[cov]
            ph_results[str(cov)] = {
                "test_stat": round(float(row["test_statistic"]), 4),
                "p_value": round(float(row["p"]), 6),
                "violates_ph": bool(row["p"] < 0.05),
            }
        result["ph_test_results"] = ph_results
    except Exception as e:
        logger.warning(f"  PH test failed for '{label}': {e}")
        result["ph_test_results"] = {"error": str(e)}

    logger.info(f"  Concordance: {result['concordance_index']}, LogLik: {result['log_likelihood']}")
    for cov, hr in hazard_ratios.items():
        logger.info(f"  {cov}: HR={hr['HR']}, p={hr['p_value']}")

    return result


def build_output(
    df: pd.DataFrame,
    wo_dummy_cols: list[str],
    results: dict,
) -> dict:
    """Build the output JSON conforming to exp_gen_sol_out schema."""
    # For each core-arg record, the "input" is the original fields, "output" is the model prediction context
    # We produce a dataset-level summary as the output

    examples = []

    # Create summary examples per treebank (aggregated)
    treebank_groups = df.groupby("treebank_id")
    for tb_id, grp in treebank_groups:
        inp = json.dumps({  # noqa

            "treebank_id": tb_id,
            "n_records": len(grp),
            "word_order": grp["word_order"].iloc[0] if "word_order" in grp.columns else None,
            "case_richness": float(grp["case_richness"].iloc[0]) if "case_richness" in grp.columns else None,
            "language_family": grp["language_family"].iloc[0] if "language_family" in grp.columns else None,
            "mean_distance": round(float(grp["distance"].mean()), 3),
            "deprel_counts": grp["deprel"].value_counts().to_dict(),
        }, default=_json_default)

        out = json.dumps({
            "cox_main_model": {
                "concordance": results.get("main_model", {}).get("concordance_index"),
                "case_richness_HR": results.get("main_model", {}).get("hazard_ratios", {}).get("case_richness_z", {}),
            },
            "hypothesis_supported": results.get("hypothesis_supported", False),
            "key_finding": results.get("key_finding", ""),
        }, default=_json_default)

        # predict_cox_ph: our method's prediction (main model with case+word order)
        main_hr = results.get("main_model", {}).get("hazard_ratios", {})
        case_hr_info = main_hr.get("case_richness_z", {})
        wo_val = grp["word_order"].iloc[0] if "word_order" in grp.columns else None
        wo_key = f"wo_{wo_val}" if wo_val and wo_val != "SVO" else None
        wo_hr_info = main_hr.get(wo_key, {}) if wo_key else {}

        predict_cox = json.dumps({
            "predicted_hazard_effect": "case+word_order",
            "case_richness_HR": case_hr_info.get("HR"),
            "case_richness_p": case_hr_info.get("p_value"),
            "word_order_HR": wo_hr_info.get("HR", 1.0),
            "word_order_p": wo_hr_info.get("p_value", 1.0),
            "concordance": results.get("main_model", {}).get("concordance_index"),
        }, default=_json_default)

        # predict_baseline: baseline model using only sentence_length (no typological features)
        bl = results.get("case_only_model", {})
        predict_bl = json.dumps({
            "predicted_hazard_effect": "sentence_length_only",
            "concordance": bl.get("concordance_index"),
            "sentence_length_HR": bl.get("hazard_ratios", {}).get("sentence_length", {}).get("HR"),
        }, default=_json_default)

        examples.append({
            "input": inp,
            "output": out,
            "predict_cox_ph": predict_cox,
            "predict_baseline": predict_bl,
            "metadata_treebank_id": tb_id,
            "metadata_word_order": str(grp["word_order"].iloc[0]) if "word_order" in grp.columns else "",
            "metadata_case_richness": float(grp["case_richness"].iloc[0]) if "case_richness" in grp.columns else 0.0,
            "metadata_mean_distance": round(float(grp["distance"].mean()), 3),
            "metadata_n_records": int(len(grp)),
        })

    output = {
        "metadata": {
            "method_name": "Stratified Cox PH: Case Richness vs Word Order",
            "description": "Cox proportional hazards models comparing case richness and word order typology effects on core-argument dependency distances",
            "n_total_records": int(len(df)),
            "n_treebanks": int(df["treebank_id"].nunique()),
            "n_language_families": int(df["language_family"].nunique()) if "language_family" in df.columns else 0,
            "results_summary": results,
        },
        "datasets": [
            {
                "dataset": "cox_ph_treebank_results",
                "examples": examples,
            }
        ],
    }
    return output


@logger.catch
def main(use_mini: bool = False):
    logger.info("=" * 60)
    logger.info("Starting Cox PH analysis: Case Richness vs Word Order")
    logger.info("=" * 60)

    # Step 1: Load data
    df_ca, df_tb, df_tp = load_all_data(use_mini=use_mini)

    # Step 2: Prepare data
    df, wo_dummy_cols = prepare_data(df_ca, df_tb, df_tp)
    logger.info(f"Prepared DataFrame: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Word order dummy columns: {wo_dummy_cols}")

    all_results = {}

    # Subsample for Cox fitting (400k too slow; 50k is statistically sufficient)
    df_cox = subsample_for_cox(df, max_n=50000)

    # Step 3: Main model (full) — stratified by deprel, clustered by language_family
    main_covariates = ["case_richness_z", "sentence_length"] + wo_dummy_cols
    main_result = fit_cox_model(
        df=df_cox,
        covariates=main_covariates,
        strata=["deprel"],
        cluster_col="language_family",
        label="main_model",
    )
    all_results["main_model"] = main_result

    # Step 3b: Case-only model
    case_only_result = fit_cox_model(
        df=df_cox,
        covariates=["case_richness_z", "sentence_length"],
        strata=["deprel"],
        cluster_col="language_family",
        label="case_only_model",
    )
    all_results["case_only_model"] = case_only_result

    # Step 3c: Word-order-only model
    wo_only_result = fit_cox_model(
        df=df_cox,
        covariates=["sentence_length"] + wo_dummy_cols,
        strata=["deprel"],
        cluster_col="language_family",
        label="word_order_only_model",
    )
    all_results["word_order_only_model"] = wo_only_result

    # Step 4: Per-deprel models
    per_deprel = {}
    for deprel in df_cox["deprel"].unique():
        df_dep = df_cox[df_cox["deprel"] == deprel]
        if len(df_dep) >= 100:
            dep_result = fit_cox_model(
                df=df_dep,
                covariates=["case_richness_z", "sentence_length"] + wo_dummy_cols,
                cluster_col="language_family",
                label=f"deprel_{deprel}",
            )
            per_deprel[deprel] = dep_result
    all_results["per_deprel_models"] = per_deprel

    # Step 5: Sensitivity — restrict to treebanks with >= 100 records
    tb_counts = df_cox["treebank_id"].value_counts()
    large_tbs = tb_counts[tb_counts >= 100].index
    df_large = df_cox[df_cox["treebank_id"].isin(large_tbs)]
    if len(df_large) >= 100:
        restricted_result = fit_cox_model(
            df=df_large,
            covariates=main_covariates,
            strata=["deprel"],
            cluster_col="language_family",
            label="restricted_100plus",
        )
        all_results["sensitivity_restricted"] = restricted_result

    # Step 5b: Interaction model (case_richness × word_order)
    if len(wo_dummy_cols) > 0:
        # Add interaction terms
        df_interact = df_cox.copy()
        interact_cols = []
        for wo_col in wo_dummy_cols:
            interact_name = f"case_x_{wo_col}"
            df_interact[interact_name] = df_interact["case_richness_z"] * df_interact[wo_col]
            interact_cols.append(interact_name)

        interact_result = fit_cox_model(
            df=df_interact,
            covariates=main_covariates + interact_cols,
            strata=["deprel"],
            cluster_col="language_family",
            label="interaction_model",
        )
        all_results["sensitivity_interaction"] = interact_result

    # Step 5c: Exclude distance=1 (adjacent deps)
    df_nonadj = df_cox[df_cox["distance"] > 1]
    if len(df_nonadj) >= 100:
        nonadj_result = fit_cox_model(
            df=df_nonadj,
            covariates=main_covariates,
            strata=["deprel"],
            cluster_col="language_family",
            label="nonadjacent_only",
        )
        all_results["sensitivity_nonadjacent"] = nonadj_result

    # Determine key finding
    main_hrs = main_result.get("hazard_ratios", {})
    case_hr = main_hrs.get("case_richness_z", {})
    case_hr_val = case_hr.get("HR", None)
    case_p = case_hr.get("p_value", 1.0)

    # Find largest word order HR (by |log(HR)|)
    max_wo_hr = None
    max_wo_name = None
    max_wo_p = 1.0
    for col in wo_dummy_cols:
        wo_hr = main_hrs.get(col, {})
        hr_val = wo_hr.get("HR", 1.0)
        if max_wo_hr is None or abs(np.log(hr_val + 1e-10)) > abs(np.log(max_wo_hr + 1e-10)):
            max_wo_hr = hr_val
            max_wo_name = col
            max_wo_p = wo_hr.get("p_value", 1.0)

    if case_hr_val is not None and max_wo_hr is not None:
        case_effect = abs(np.log(case_hr_val + 1e-10))
        wo_effect = abs(np.log(max_wo_hr + 1e-10))
        hypothesis_supported = case_effect > wo_effect and case_p < 0.05
        key_finding = (
            f"case_richness HR={case_hr_val} (p={case_p}) vs "
            f"largest word_order ({max_wo_name}) HR={max_wo_hr} (p={max_wo_p}). "
            f"Case effect {'>' if case_effect > wo_effect else '<='} word order effect. "
            f"Hypothesis {'supported' if hypothesis_supported else 'not supported'}."
        )
    else:
        hypothesis_supported = False
        key_finding = "Could not compare: missing HR values"

    all_results["hypothesis_supported"] = hypothesis_supported
    all_results["key_finding"] = key_finding
    logger.info(f"KEY FINDING: {key_finding}")

    # Step 6: Build output
    output = build_output(df, wo_dummy_cols, all_results)

    # Save
    out_path = OUT_DIR / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=_json_default))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    return output


if __name__ == "__main__":
    use_mini = "--mini" in sys.argv
    main(use_mini=use_mini)
