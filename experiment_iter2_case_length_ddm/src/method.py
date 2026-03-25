#!/usr/bin/env python3
"""Linking Mechanism: Case Richness → Sentence Length → DDM Rank Shifts.

Tests whether case-rich languages have longer sentences AND larger rank shifts
between naive/standardized DDM, establishing the linking mechanism.
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
from loguru import logger
from scipy import stats
from scipy.stats import bootstrap, kruskal, pearsonr, spearmanr
import statsmodels.api as sm

# --- Logging ---
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
FMT = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=FMT)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# --- Hardware ---
NUM_CPUS = min(len(os.sched_getaffinity(0)), 3)
RAM_BUDGET = int(14 * 1024**3)  # 14 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

WORKSPACE = Path(__file__).parent
N_BOOTSTRAP = 10000


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


def load_treebank_summaries(limit: int | None = None) -> pd.DataFrame:
    """Load ud_treebank_summaries from data_id5 split files."""
    records = []
    for fpath in sorted(glob.glob(str(WORKSPACE / "data_id5" / "full_data_out_*.json"))):
        logger.info(f"Loading {fpath}")
        data = json.loads(Path(fpath).read_text())
        for ds in data["datasets"]:
            if ds["dataset"] == "ud_treebank_summaries":
                for ex in ds["examples"]:
                    inp = json.loads(ex["input"])
                    out = json.loads(ex["output"])
                    rec = {**inp, **out}
                    records.append(rec)
        del data
        gc.collect()
        if limit and len(records) >= limit:
            records = records[:limit]
            break
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} treebank summaries")
    return df


def load_typological_profiles() -> pd.DataFrame:
    """Load full_typological_profiles from data_id4."""
    data = json.loads((WORKSPACE / "data_id4" / "full_data_out.json").read_text())
    records = []
    for ds in data["datasets"]:
        if ds["dataset"] == "full_typological_profiles":
            for ex in ds["examples"]:
                inp = json.loads(ex["input"])
                records.append(inp)
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} typological profiles")
    return df


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_func, n_resamples: int = N_BOOTSTRAP) -> tuple:
    """Compute 95% bootstrap CI for a correlation statistic."""
    rng = np.random.default_rng(42)
    n = len(x)
    stats_vals = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            stats_vals[i] = stat_func(x[idx], y[idx])
        except Exception:
            stats_vals[i] = np.nan
    lo, hi = np.nanpercentile(stats_vals, [2.5, 97.5])
    return float(lo), float(hi)


def compute_correlation(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Compute Spearman + Pearson with bootstrap CIs."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return {"label": label, "n": n, "error": "too few observations"}

    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)

    rho_ci = bootstrap_ci(x, y, lambda a, b: spearmanr(a, b).statistic)
    r_ci = bootstrap_ci(x, y, lambda a, b: pearsonr(a, b).statistic)

    return {
        "label": label,
        "n": int(n),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rho_p),
        "spearman_ci95": [round(rho_ci[0], 4), round(rho_ci[1], 4)],
        "pearson_r": round(float(r), 4),
        "pearson_p": float(r_p),
        "pearson_ci95": [round(r_ci[0], 4), round(r_ci[1], 4)],
    }


def run_ols(df: pd.DataFrame, y_col: str, x_cols: list[str], label: str) -> dict:
    """Run OLS regression and return summary dict."""
    sub = df[x_cols + [y_col]].dropna()
    if len(sub) < 10:
        return {"label": label, "error": "too few obs", "n": len(sub)}
    X = sm.add_constant(sub[x_cols].values)
    y = sub[y_col].values
    model = sm.OLS(y, X).fit()
    coefs = {}
    names = ["const"] + x_cols
    for i, name in enumerate(names):
        coefs[name] = {
            "coef": round(float(model.params[i]), 6),
            "se": round(float(model.bse[i]), 6),
            "t": round(float(model.tvalues[i]), 4),
            "p": float(model.pvalues[i]),
        }
    return {
        "label": label,
        "n": int(len(sub)),
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_stat": round(float(model.fvalue), 4),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": coefs,
    }


@logger.catch
def main(limit: int | None = None):
    logger.info("=== Starting Linking Mechanism Analysis ===")

    # 1. Load data
    df = load_treebank_summaries(limit=limit)
    typo = load_typological_profiles()

    # Parse input fields if needed - ensure we have key columns
    required = ["treebank_id", "iso_code", "case_richness", "naive_ddm",
                 "standardized_ddm", "mean_sentence_length", "n_sentences"]
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")

    # 2. Fill missing case_richness from typological profiles
    # Map iso_code in data_id4 (metadata_language_code_ud) to ud_case_richness
    typo_case = typo[["language_iso", "ud_case_richness"]].copy()
    # language_iso in typo is 3-letter, iso_code in df is 2-letter sometimes
    # We need the ud language code - check preview: metadata_language_code_ud is 2-letter
    # Actually typo has language_name which is the ud code. Let's check:
    # From preview: language_iso=afr, language_name=af => language_name is the UD code
    typo["ud_iso"] = typo["language_name"]  # language_name holds the UD iso code

    missing_case = df["case_richness"].isna() | (df["case_richness"] == 0)
    logger.info(f"Treebanks with case_richness=0 or NaN: {missing_case.sum()}/{len(df)}")

    # Merge to fill
    typo_map = typo.set_index("ud_iso")["ud_case_richness"].to_dict()
    df["case_richness_filled"] = df.apply(
        lambda r: typo_map.get(r["iso_code"], r["case_richness"])
        if pd.isna(r["case_richness"]) or r["case_richness"] == 0
        else r["case_richness"],
        axis=1
    )

    # Also fill word_order from typo if missing
    typo_wo = {}
    for ds in json.loads((WORKSPACE / "data_id4" / "full_data_out.json").read_text())["datasets"]:
        if ds["dataset"] == "word_order_prediction":
            for ex in ds["examples"]:
                inp = json.loads(ex["input"])
                code = ex.get("metadata_language_code_ud", inp.get("language_iso", ""))
                typo_wo[code] = ex["output"]
    df["word_order_filled"] = df["word_order"].fillna(df["iso_code"].map(typo_wo))

    # Use case_richness_filled for analysis
    df["cr"] = df["case_richness_filled"].fillna(0).astype(float)
    df["msl"] = pd.to_numeric(df.get("mean_sentence_length"), errors="coerce")
    df["naive"] = pd.to_numeric(df["naive_ddm"], errors="coerce")
    df["std"] = pd.to_numeric(df["standardized_ddm"], errors="coerce")
    df["n_sent"] = pd.to_numeric(df["n_sentences"], errors="coerce")

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"case_richness stats: mean={df['cr'].mean():.2f}, nonzero={( df['cr'] > 0).sum()}")
    logger.info(f"mean_sentence_length non-null: {df['msl'].notna().sum()}")

    # 3. Compute rank shifts
    df["naive_rank"] = df["naive"].rank(ascending=False, method="average")
    df["std_rank"] = df["std"].rank(ascending=False, method="average")
    df["abs_rank_shift"] = (df["naive_rank"] - df["std_rank"]).abs()
    df["signed_rank_shift"] = df["naive_rank"] - df["std_rank"]

    # 4. Core correlations
    logger.info("Computing correlations...")
    correlations = []
    pairs = [
        ("cr", "msl", "case_richness vs mean_sentence_length"),
        ("cr", "abs_rank_shift", "case_richness vs abs_rank_shift"),
        ("msl", "abs_rank_shift", "mean_sentence_length vs abs_rank_shift"),
        ("cr", "signed_rank_shift", "case_richness vs signed_rank_shift"),
        ("msl", "signed_rank_shift", "mean_sentence_length vs signed_rank_shift"),
        ("cr", "naive", "case_richness vs naive_ddm"),
        ("cr", "std", "case_richness vs standardized_ddm"),
        ("msl", "naive", "mean_sentence_length vs naive_ddm"),
        ("msl", "std", "mean_sentence_length vs standardized_ddm"),
    ]
    for x_col, y_col, label in pairs:
        corr = compute_correlation(df[x_col].values, df[y_col].values, label)
        correlations.append(corr)
        logger.info(f"  {label}: rho={corr.get('spearman_rho','N/A')}, p={corr.get('spearman_p','N/A')}")

    # 5. Mediation-style regressions
    logger.info("Running OLS regressions...")
    regressions = []

    # Path c: rank_shift ~ case_richness
    regressions.append(run_ols(df, "abs_rank_shift", ["cr"], "abs_rank_shift ~ case_richness"))
    # Path c': rank_shift ~ case_richness + mean_sentence_length
    regressions.append(run_ols(df, "abs_rank_shift", ["cr", "msl"], "abs_rank_shift ~ case_richness + mean_sentence_length"))
    # Path a: mean_sentence_length ~ case_richness
    regressions.append(run_ols(df, "msl", ["cr"], "mean_sentence_length ~ case_richness"))
    # Path b: rank_shift ~ mean_sentence_length
    regressions.append(run_ols(df, "abs_rank_shift", ["msl"], "abs_rank_shift ~ mean_sentence_length"))
    # Additional: signed rank shift
    regressions.append(run_ols(df, "signed_rank_shift", ["cr", "msl"], "signed_rank_shift ~ case_richness + mean_sentence_length"))

    for reg in regressions:
        logger.info(f"  {reg['label']}: R²={reg.get('r_squared','N/A')}, n={reg.get('n','N/A')}")

    # 6. Group-level analysis by case richness bins
    logger.info("Group-level analysis by case richness bins...")
    df["case_bin"] = pd.cut(df["cr"], bins=[-0.1, 0, 3, 6, 100],
                            labels=["none(0)", "low(1-3)", "medium(4-6)", "high(7+)"])
    group_stats = []
    for bin_label, grp in df.groupby("case_bin", observed=True):
        group_stats.append({
            "case_bin": str(bin_label),
            "n_treebanks": int(len(grp)),
            "mean_sentence_length": round(float(grp["msl"].mean()), 2) if grp["msl"].notna().any() else None,
            "mean_abs_rank_shift": round(float(grp["abs_rank_shift"].mean()), 2),
            "mean_signed_rank_shift": round(float(grp["signed_rank_shift"].mean()), 2),
            "mean_naive_ddm": round(float(grp["naive"].mean()), 4),
            "mean_std_ddm": round(float(grp["std"].mean()), 4),
            "mean_case_richness": round(float(grp["cr"].mean()), 2),
        })
        logger.info(f"  {bin_label}: n={len(grp)}, msl={grp['msl'].mean():.1f}, abs_shift={grp['abs_rank_shift'].mean():.1f}")

    # Kruskal-Wallis tests across bins
    kw_tests = {}
    for var in ["msl", "abs_rank_shift", "signed_rank_shift"]:
        groups = [grp[var].dropna().values for _, grp in df.groupby("case_bin", observed=True) if len(grp[var].dropna()) >= 2]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            kw_tests[var] = {"H": round(float(stat), 4), "p": float(p), "n_groups": len(groups)}
            logger.info(f"  Kruskal-Wallis {var}: H={stat:.4f}, p={p:.4e}")

    # 7. Language family analysis
    logger.info("Language family analysis...")
    fam_col = "language_family"
    if fam_col not in df.columns:
        fam_col = "language_family"
    family_stats = []
    fam_counts = df[fam_col].value_counts()
    for fam in fam_counts[fam_counts >= 3].index:
        if pd.isna(fam):
            continue
        grp = df[df[fam_col] == fam]
        family_stats.append({
            "family": str(fam),
            "n_treebanks": int(len(grp)),
            "mean_case_richness": round(float(grp["cr"].mean()), 2),
            "mean_sentence_length": round(float(grp["msl"].mean()), 2) if grp["msl"].notna().any() else None,
            "mean_abs_rank_shift": round(float(grp["abs_rank_shift"].mean()), 2),
            "mean_signed_rank_shift": round(float(grp["signed_rank_shift"].mean()), 2),
            "mean_naive_ddm": round(float(grp["naive"].mean()), 4),
            "mean_std_ddm": round(float(grp["std"].mean()), 4),
        })
    family_stats.sort(key=lambda x: x["mean_abs_rank_shift"], reverse=True)
    for fs in family_stats[:5]:
        logger.info(f"  {fs['family']}: cr={fs['mean_case_richness']}, shift={fs['mean_abs_rank_shift']}")

    # 8. Robustness check: filter to well-attested treebanks (n_sentences >= 100)
    logger.info("Robustness check: n_sentences >= 100...")
    df_robust = df[df["n_sent"] >= 100].copy()
    logger.info(f"  Robust subset: {len(df_robust)} treebanks")
    robust_correlations = []
    for x_col, y_col, label in pairs[:4]:  # Core 4 correlations
        corr = compute_correlation(df_robust[x_col].values, df_robust[y_col].values, f"[robust] {label}")
        robust_correlations.append(corr)
        logger.info(f"  {label}: rho={corr.get('spearman_rho','N/A')}, p={corr.get('spearman_p','N/A')}")

    # Additional robustness: log-transformed case richness
    df["log_cr"] = np.log1p(df["cr"])
    log_correlations = []
    for y_col, label in [("msl", "log_case_richness vs msl"), ("abs_rank_shift", "log_case_richness vs abs_rank_shift")]:
        corr = compute_correlation(df["log_cr"].values, df[y_col].values, label)
        log_correlations.append(corr)

    # Binary case: 0 vs >0
    df["has_case"] = (df["cr"] > 0).astype(float)
    binary_tests = {}
    for var in ["msl", "abs_rank_shift", "signed_rank_shift"]:
        g0 = df[df["has_case"] == 0][var].dropna()
        g1 = df[df["has_case"] == 1][var].dropna()
        if len(g0) >= 3 and len(g1) >= 3:
            u_stat, u_p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            binary_tests[var] = {
                "no_case_mean": round(float(g0.mean()), 4),
                "has_case_mean": round(float(g1.mean()), 4),
                "n_no_case": int(len(g0)),
                "n_has_case": int(len(g1)),
                "mann_whitney_U": float(u_stat),
                "mann_whitney_p": float(u_p),
            }

    # 9. Determine verdict
    cr_msl = correlations[0]  # case_richness vs msl
    cr_shift = correlations[1]  # case_richness vs abs_rank_shift
    msl_shift = correlations[2]  # msl vs abs_rank_shift

    mechanism_holds = (
        cr_msl.get("spearman_p", 1) < 0.05
        and msl_shift.get("spearman_p", 1) < 0.05
    )

    # Check mediation: does cr effect drop when controlling for msl?
    reg_c = regressions[0]  # abs_rank_shift ~ cr
    reg_cprime = regressions[1]  # abs_rank_shift ~ cr + msl
    mediation_evidence = False
    if "coefficients" in reg_c and "coefficients" in reg_cprime:
        cr_coef_c = abs(reg_c["coefficients"].get("cr", {}).get("coef", 0))
        cr_coef_cprime = abs(reg_cprime["coefficients"].get("cr", {}).get("coef", 0))
        if cr_coef_c > 0:
            mediation_evidence = cr_coef_cprime < cr_coef_c * 0.8  # 20%+ reduction

    verdict = {
        "linking_mechanism_supported": mechanism_holds,
        "mediation_evidence": mediation_evidence,
        "summary": (
            f"Case richness → sentence length: rho={cr_msl.get('spearman_rho','N/A')}, p={cr_msl.get('spearman_p','N/A')}. "
            f"Sentence length → rank shift: rho={msl_shift.get('spearman_rho','N/A')}, p={msl_shift.get('spearman_p','N/A')}. "
            f"Mediation evidence: {'yes' if mediation_evidence else 'no'}. "
            f"Overall: {'Mechanism supported' if mechanism_holds else 'Mechanism NOT supported'}."
        ),
    }
    logger.info(f"Verdict: {verdict['summary']}")

    # Build output in exp_gen_sol_out schema format
    # Each treebank becomes an example with input/output and predictions
    examples = []
    for _, row in df.iterrows():
        inp = json.dumps({
            "treebank_id": row.get("treebank_id"),
            "iso_code": row.get("iso_code"),
            "case_richness": int(row["cr"]) if pd.notna(row["cr"]) else None,
            "mean_sentence_length": round(float(row["msl"]), 2) if pd.notna(row["msl"]) else None,
            "naive_ddm": round(float(row["naive"]), 4) if pd.notna(row["naive"]) else None,
            "standardized_ddm": round(float(row["std"]), 4) if pd.notna(row["std"]) else None,
        })
        out = json.dumps({
            "naive_rank": int(row["naive_rank"]) if pd.notna(row["naive_rank"]) else None,
            "std_rank": int(row["std_rank"]) if pd.notna(row["std_rank"]) else None,
            "abs_rank_shift": round(float(row["abs_rank_shift"]), 2) if pd.notna(row["abs_rank_shift"]) else None,
            "signed_rank_shift": round(float(row["signed_rank_shift"]), 2) if pd.notna(row["signed_rank_shift"]) else None,
        })
        ex = {
            "input": inp,
            "output": out,
            "metadata_treebank_id": row.get("treebank_id"),
            "metadata_iso_code": row.get("iso_code"),
            "metadata_case_richness": int(row["cr"]) if pd.notna(row["cr"]) else None,
            "metadata_abs_rank_shift": round(float(row["abs_rank_shift"]), 2) if pd.notna(row["abs_rank_shift"]) else None,
            "predict_linking_mechanism": json.dumps({
                "case_bin": str(row.get("case_bin", "")),
                "has_case": bool(row.get("has_case", False)),
            }),
        }
        examples.append(ex)

    output = {
        "metadata": {
            "method_name": "linking_mechanism_case_richness_sentence_length_ddm_rank_shifts",
            "description": "Tests whether case-rich languages have longer sentences AND larger rank shifts between naive/standardized DDM",
            "n_treebanks": len(df),
            "correlations": correlations,
            "regressions": regressions,
            "group_analysis": group_stats,
            "kruskal_wallis_tests": kw_tests,
            "family_analysis": family_stats,
            "robustness_n100": robust_correlations,
            "robustness_log_cr": log_correlations,
            "binary_case_tests": binary_tests,
            "verdict": verdict,
        },
        "datasets": [
            {
                "dataset": "ud_treebank_summaries",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    return output


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
