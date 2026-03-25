#!/usr/bin/env python3
"""Compile typological metadata for UD languages: WALS (81A, 49A) + UD case richness."""

import csv
import io
import json
import resource
import sys
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import requests
from loguru import logger

# --- Setup ---
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# Memory limit
_avail = psutil.virtual_memory().available
RAM_BUDGET = min(4 * 1024**3, int(_avail * 0.7))  # 4GB or 70% available
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

WALS_BASE = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf"
UD_LANGS_URL = "https://raw.githubusercontent.com/UniversalDependencies/docs/pages-source/_data/languages.json"
UD_TREEBANKS_URL = "https://raw.githubusercontent.com/UniversalDependencies/docs/pages-source/_data/treebanks.json"


def fetch_csv(url: str) -> list[dict]:
    """Fetch a CSV from URL and return list of dicts."""
    logger.info(f"Fetching {url.split('/')[-1]}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return list(reader)


def fetch_json(url: str) -> dict | list:
    """Fetch JSON from URL."""
    logger.info(f"Fetching {url.split('/')[-1]}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


@logger.catch
def build_wals_data() -> dict:
    """Build WALS lookup: iso_code -> {word_order_81A, case_count_49A, wals_code, family, ...}."""
    # Fetch all needed CSVs in parallel
    urls = {
        "values": f"{WALS_BASE}/values.csv",
        "codes": f"{WALS_BASE}/codes.csv",
        "languages": f"{WALS_BASE}/languages.csv",
    }
    fetched = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_csv, url): name for name, url in urls.items()}
        for fut in as_completed(futures):
            fetched[futures[fut]] = fut.result()

    # Build code label lookup: code_id -> label
    code_labels = {}
    for row in fetched["codes"]:
        code_labels[row["ID"]] = row["Name"]

    # Build language lookup: wals_id -> {iso, name, family, lat, lon}
    lang_lookup = {}
    for row in fetched["languages"]:
        lang_lookup[row["ID"]] = {
            "iso_code": row.get("ISO639P3code", ""),
            "name": row.get("Name", ""),
            "family": row.get("Family", ""),
            "latitude": row.get("Latitude", ""),
            "longitude": row.get("Longitude", ""),
            "glottocode": row.get("Glottocode", ""),
        }

    # Filter values for 81A and 49A
    wals_by_lang = defaultdict(dict)  # wals_id -> {81A: ..., 49A: ...}
    for row in fetched["values"]:
        param = row.get("Parameter_ID", "")
        if param in ("81A", "49A"):
            lang_id = row["Language_ID"]
            code_id = row.get("Code_ID", "")
            label = code_labels.get(code_id, code_id)
            wals_by_lang[lang_id][param] = label

    # Build ISO-indexed result
    result = {}  # iso -> {...}
    for wals_id, features in wals_by_lang.items():
        lang_info = lang_lookup.get(wals_id, {})
        iso = lang_info.get("iso_code", "")
        if not iso:
            continue
        if iso not in result:
            result[iso] = {
                "wals_code": wals_id,
                "wals_name": lang_info.get("name", ""),
                "family_wals": lang_info.get("family", ""),
                "latitude": lang_info.get("latitude", ""),
                "longitude": lang_info.get("longitude", ""),
                "glottocode": lang_info.get("glottocode", ""),
                "word_order_81A": None,
                "case_count_49A": None,
            }
        if "81A" in features:
            result[iso]["word_order_81A"] = features["81A"]
        if "49A" in features:
            result[iso]["case_count_49A"] = features["49A"]

    logger.info(f"WALS: {len(result)} languages with ISO codes (81A or 49A)")
    count_81a = sum(1 for v in result.values() if v["word_order_81A"])
    count_49a = sum(1 for v in result.values() if v["case_count_49A"])
    logger.info(f"  81A coverage: {count_81a}, 49A coverage: {count_49a}")
    return result


@logger.catch
def build_ud_case_richness() -> dict:
    """Extract case richness from UD treebanks via HuggingFace streaming API.
    Returns: iso -> {ud_case_richness, case_values, treebank_ids}
    """
    from datasets import load_dataset

    # Get list of UD treebank configs - use commul version (parquet, no trust_remote_code needed)
    logger.info("Loading UD treebank config list from commul/universal_dependencies...")
    from datasets import get_dataset_config_names
    configs = get_dataset_config_names("commul/universal_dependencies")
    configs = [c for c in configs if c != "default"]

    logger.info(f"Found {len(configs)} UD treebank configs")

    # Map treebank config name to ISO code
    # Config names are like "en_ewt", "de_gsd", etc. - first part before _ is language code
    lang_cases = defaultdict(set)  # lang_code -> set of Case values
    lang_treebanks = defaultdict(list)  # lang_code -> list of treebank names

    def process_treebank(config: str) -> tuple[str, set, str]:
        """Process one treebank, return (lang_code, case_values, config_name)."""
        lang_code = config.split("_")[0]
        cases = set()
        try:
            ds = load_dataset(
                "commul/universal_dependencies",
                config,
                split="train",
                streaming=True,
            )
            count = 0
            for example in ds:
                feats = example.get("feats") or []
                if isinstance(feats, list):
                    for feat_str in feats:
                        if feat_str and "Case=" in str(feat_str):
                            for part in str(feat_str).split("|"):
                                if part.startswith("Case="):
                                    cases.add(part.split("=")[1])
                count += 1
                if count >= 5000:
                    break
        except Exception as e:
            logger.debug(f"Error processing {config}: {e}")
        return lang_code, cases, config

    # Process treebanks with thread pool (I/O bound - streaming from HF)
    logger.info("Processing UD treebanks for Case features (streaming, 5k sentences each)...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_treebank, c): c for c in configs}
        done = 0
        for fut in as_completed(futures):
            lang_code, cases, config = fut.result()
            lang_cases[lang_code].update(cases)
            lang_treebanks[lang_code].append(config)
            done += 1
            if done % 25 == 0:
                logger.info(f"  Processed {done}/{len(configs)} treebanks")

    logger.info(f"UD: {len(lang_cases)} languages processed")

    # Build result
    result = {}
    for lang_code, cases in lang_cases.items():
        result[lang_code] = {
            "ud_case_richness": len(cases),
            "ud_case_values": sorted(cases) if cases else [],
            "ud_treebank_ids": sorted(lang_treebanks[lang_code]),
        }

    langs_with_case = sum(1 for v in result.values() if v["ud_case_richness"] > 0)
    logger.info(f"  Languages with Case features: {langs_with_case}")
    return result


@logger.catch
def build_ud_metadata() -> dict:
    """Fetch UD language metadata (ISO codes, families) from UD docs repo."""
    try:
        data = fetch_json(UD_LANGS_URL)
        logger.info(f"UD metadata: {len(data)} languages")
        # data is a dict or list with language info
        result = {}
        if isinstance(data, dict):
            for code, info in data.items():
                iso3 = info.get("iso3", "") or info.get("lcode", code)
                result[code] = {
                    "iso3": iso3,
                    "name": info.get("name", info.get("lname", "")),
                    "family": info.get("family", ""),
                }
        elif isinstance(data, list):
            for info in data:
                code = info.get("lcode", "")
                result[code] = {
                    "iso3": info.get("iso3", code),
                    "name": info.get("lname", ""),
                    "family": info.get("family", ""),
                }
        return result
    except Exception as e:
        logger.warning(f"Could not fetch UD metadata: {e}")
        return {}


# ISO 639-1 to 639-3 mapping for common UD languages
ISO1_TO_ISO3 = {
    "af": "afr", "am": "amh", "ar": "ara", "be": "bel", "bg": "bul",
    "bm": "bam", "bn": "ben", "br": "bre", "ca": "cat", "cs": "ces",
    "cu": "chu", "cy": "cym", "da": "dan", "de": "deu", "el": "ell",
    "en": "eng", "es": "spa", "et": "est", "eu": "eus", "fa": "fas",
    "fi": "fin", "fo": "fao", "fr": "fra", "fy": "fry", "ga": "gle",
    "gd": "gla", "gl": "glg", "gn": "grn", "gv": "glv", "he": "heb",
    "hi": "hin", "hr": "hrv", "hu": "hun", "hy": "hye", "id": "ind",
    "is": "isl", "it": "ita", "ja": "jpn", "ka": "kat", "kk": "kaz",
    "km": "khm", "kn": "kan", "ko": "kor", "ku": "kur", "la": "lat",
    "lt": "lit", "lv": "lav", "mk": "mkd", "ml": "mal", "mn": "mon",
    "mr": "mar", "mt": "mlt", "my": "mya", "nb": "nob", "nl": "nld",
    "nn": "nno", "no": "nor", "or": "ori", "pa": "pan", "pl": "pol",
    "ps": "pus", "pt": "por", "ro": "ron", "ru": "rus", "sa": "san",
    "si": "sin", "sk": "slk", "sl": "slv", "sq": "sqi", "sr": "srp",
    "sv": "swe", "sw": "swa", "ta": "tam", "te": "tel", "th": "tha",
    "tl": "tgl", "tr": "tur", "ug": "uig", "uk": "ukr", "ur": "urd",
    "uz": "uzb", "vi": "vie", "wo": "wol", "yo": "yor", "zh": "zho",
    "cop": "cop", "got": "got", "grc": "grc", "orv": "orv", "fro": "fro",
    "sme": "sme", "hsb": "hsb", "kmr": "kmr", "myv": "myv", "pcm": "pcm",
    "qtd": "qtd", "swl": "swl", "tpn": "tpn", "wbp": "wbp", "yue": "yue",
    "lzh": "lzh", "hbo": "hbo", "hit": "hit", "akk": "akk", "sux": "sux",
    "qhe": "qhe", "bho": "bho", "mag": "mag", "aii": "aii", "bxr": "bxr",
    "ckt": "ckt", "ess": "ess", "kfm": "kfm", "koi": "koi", "kpv": "kpv",
    "krl": "krl", "mdf": "mdf", "mpu": "mpu", "nds": "nds", "olo": "olo",
    "sms": "sms", "gsw": "gsw", "qfn": "qfn", "aqz": "aqz", "arr": "arr",
    "bej": "bej", "gub": "gub", "gun": "gun", "kaa": "kaa", "myu": "myu",
    "nyq": "nyq", "shp": "shp", "tgk": "tgk", "tuk": "tuk", "xav": "xav",
}


@logger.catch
def main():
    # Phase 1: Fetch all data sources in parallel
    logger.info("=== Phase 1: Fetching data sources ===")

    with ThreadPoolExecutor(max_workers=3) as pool:
        wals_future = pool.submit(build_wals_data)
        ud_meta_future = pool.submit(build_ud_metadata)
        ud_case_future = pool.submit(build_ud_case_richness)

        wals_data = wals_future.result() or {}
        ud_meta = ud_meta_future.result() or {}
        ud_case = ud_case_future.result() or {}

    # Phase 2: Merge everything
    logger.info("=== Phase 2: Merging datasets ===")

    # Collect all UD language codes
    all_ud_langs = set(ud_case.keys())
    logger.info(f"Total UD language codes: {len(all_ud_langs)}")

    # Build output records
    records = []
    for ud_code in sorted(all_ud_langs):
        # Map UD code (ISO 639-1 or 639-3) to ISO 639-3
        iso3 = ISO1_TO_ISO3.get(ud_code, ud_code)

        # Try to get UD metadata
        ud_info = ud_meta.get(ud_code, {})
        if ud_info.get("iso3"):
            iso3 = ud_info["iso3"]

        # Get WALS data by ISO3
        wals = wals_data.get(iso3, {})

        # Get UD case data
        ud = ud_case.get(ud_code, {})

        # Determine language name and family
        lang_name = wals.get("wals_name", "") or ud_info.get("name", ud_code)
        family = wals.get("family_wals", "") or ud_info.get("family", "")

        record = {
            "language_iso": iso3,
            "language_code_ud": ud_code,
            "language_name": lang_name,
            "ud_treebank_ids": ud.get("ud_treebank_ids", []),
            "wals_code": wals.get("wals_code", None),
            "word_order_81A": wals.get("word_order_81A", None),
            "case_count_49A": wals.get("case_count_49A", None),
            "ud_case_richness": ud.get("ud_case_richness", 0),
            "ud_case_values": ud.get("ud_case_values", []),
            "language_family": family if family else None,
            "glottocode": wals.get("glottocode", None),
            "latitude": float(wals["latitude"]) if wals.get("latitude") else None,
            "longitude": float(wals["longitude"]) if wals.get("longitude") else None,
        }
        records.append(record)

    logger.info(f"Total merged records: {len(records)}")
    matched_wals = sum(1 for r in records if r["wals_code"])
    has_81a = sum(1 for r in records if r["word_order_81A"])
    has_49a = sum(1 for r in records if r["case_count_49A"])
    has_case = sum(1 for r in records if r["ud_case_richness"] > 0)
    logger.info(f"  WALS matched: {matched_wals}")
    logger.info(f"  81A word order: {has_81a}")
    logger.info(f"  49A case count: {has_49a}")
    logger.info(f"  UD case richness > 0: {has_case}")

    # Save output
    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    logger.info(f"Saved {len(records)} records to {out_path}")

    # Save temp copies too
    temp_dir = WORKSPACE / "temp" / "datasets"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "full_typological_metadata.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"Dataset compiled: {len(records)} UD languages")
    print(f"  WALS 81A (word order): {has_81a} languages")
    print(f"  WALS 49A (case count): {has_49a} languages")
    print(f"  UD case richness > 0:  {has_case} languages")
    print(f"  Language families:     {len(set(r['language_family'] for r in records if r['language_family']))}")
    print(f"Output: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
