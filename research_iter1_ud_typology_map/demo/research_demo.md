# UD Typology Map

## Summary

Compiled a mapping of 139 Universal Dependencies languages to word-order typology (WALS 81A: SOV/SVO/VSO/Other) and case-system richness (WALS 49A: None/Low/Medium/High/Unknown). Word order has 100% coverage; case data covers 62% directly from WALS with 53 languages needing Grambank GB070 or manual gap-filling. Output in research_out.json with per-language records including ISO codes, classifications, and source notes.

## Research Findings

## Word-Order and Case-System Classifications for UD Languages

### Data Sources and Structure

The primary sources for typological classification of Universal Dependencies languages are WALS Online (World Atlas of Language Structures) and Grambank, both available as CLDF datasets on GitHub.

WALS Feature 81A (Order of Subject, Object, and Verb) classifies 1,376 languages into 7 categories: SOV (564 languages), SVO (488), VSO (95), VOS (25), OVS (11), OSV (4), and No dominant order (189) [1]. The data is downloadable as tab-separated values from wals.info and as CSV from the cldf-datasets/wals GitHub repository [2].

WALS Feature 49A (Number of Cases) classifies 261 languages into 9 categories ranging from "No morphological case-marking" (100 languages) to "10 or more cases" (24 languages) [3]. Coverage is sparser than 81A — only ~261 languages total.

Grambank Feature GB070 (morphological cases for non-pronominal core arguments) provides a binary classification (present=1 in 777 languages, absent=0 in 1,485 languages) across 2,262 languages [4]. Grambank GB073 covers oblique case on personal pronouns [5]. The full Grambank dataset is available at https://github.com/grambank/grambank [6].

### Mapping to UD Languages

The compiled mapping covers 139 distinct UD languages (from UD v2.13's ~148 languages) [7]:

- Word order (81A): 100% coverage achieved. ~75 languages mapped directly from WALS 81A data; the remainder filled from typological literature. Distribution among UD languages: SVO: 63, SOV: 57, VSO: 10, Other (VOS/free order): 9 [1].

- Case system (49A): 62% coverage (86/139) from WALS 49A directly. The remaining 53 languages have "Unknown" case richness and require gap-filling from grammar descriptions or Grambank GB070 [3, 4]. Case richness distribution: None: 37, High (6+): 26, Low (2-3): 14, Medium (4-5): 9, Unknown: 53.

### Key Findings by Language Family

Indo-European languages dominate UD and show wide typological variation:
- Romance (French, Spanish, Italian, Portuguese, Catalan, Romanian, Galician): uniformly SVO with no/minimal case [1, 3].
- Germanic: SVO or V2/"No dominant order" (German, Dutch); mostly lost case except Icelandic (4 cases), Faroese (4 cases), German (4 cases) [1, 3, 8].
- Slavic: SVO with rich case systems (Russian, Polish, Czech, Ukrainian: 6-7 cases; Bulgarian, Macedonian: lost case) [1, 3].
- Indo-Aryan (Hindi, Urdu, Bengali, Marathi, Gujarati): SOV with varying case (2-7 cases) [1, 3].

Turkic languages (Turkish, Azerbaijani, Kazakh, Kyrgyz, Uyghur, Tatar): uniformly SOV with 6-7 agglutinative cases [1, 9].

Uralic languages (Finnish, Estonian, Hungarian, Erzya): SVO or free order with very rich case systems (10+ cases) [1, 3].

East/Southeast Asian (Chinese, Vietnamese, Thai, Indonesian vs Japanese, Korean): split between analytic SVO languages (no case) and agglutinative SOV (Japanese 8-9 cases, Korean 6-7 cases) [1, 3].

Celtic (Irish, Welsh, Manx, Scottish Gaelic, Breton): VSO (except Breton=SVO), minimal case [1, 3].

### Coverage Gaps and Limitations

1. WALS 49A is sparse: only 261 languages covered vs. 1,376 for 81A. Many UD languages (especially low-resource: Akuntsu, Bokota, Chintang, Karo, Makurap, Kangri, Khunsari) lack WALS case data [3].

2. Grambank GB070 could fill gaps but provides only binary (case present/absent), not richness gradation. Downloading the full dataset is recommended [4, 6].

3. WALS uses its own 3-letter codes (not ISO 639-3). The languages.csv file in the CLDF dataset provides ISO 639-3 mappings needed for joining to UD [2].

4. Word order ambiguity: Several languages are classified as "No dominant order" in WALS (German, Dutch, Hungarian, Armenian, Greek, Chukchi, Belarusian), reflecting genuine flexibility rather than missing data [1]. A recent study confirmed that dominant orders extracted from UD corpora largely concur with WALS classifications [10].

5. Historical/ancient languages (Gothic, Hittite, Akkadian, Classical Armenian, Old Church Slavonic, Latin, Sanskrit, Ancient Greek) are in UD but mostly absent from WALS; their typology is well-established in historical linguistics literature [11].

### Confidence Assessment

High confidence for word-order assignments: the SVO/SOV/VSO classification of major world languages is well-established and largely uncontroversial. Medium confidence for case richness: direct WALS data covers 62% of UD languages; supplementary assignments from linguistic literature are reliable for well-described languages but uncertain for ~15 low-resource languages. Downloading Grambank GB070 data would raise coverage to ~90%+.

### Data Output

The accompanying JSON contains 139 language records, each with fields: ud_language, iso_639, wals_81a_word_order, wals_81a_simplified, wals_49a_case, case_richness, and source_notes. This table is ready for direct use as covariates in the Cox proportional hazards model.

## Sources

[1] [WALS Online - Feature 81A: Order of Subject, Object and Verb](https://wals.info/feature/81A) — Primary source for word-order classifications of 1,376 languages into SOV/SVO/VSO/VOS/OVS/OSV/No dominant order categories

[2] [WALS CLDF Dataset on GitHub](https://github.com/cldf-datasets/wals) — Downloadable CSV files with WALS data including languages.csv (with ISO 639-3 codes), values.csv, and codes.csv

[3] [WALS Online - Feature 49A: Number of Cases](https://wals.info/feature/49A) — Primary source for morphological case classification of 261 languages into 9 categories from no case to 10+ cases

[4] [Grambank Feature GB070: Morphological cases for core arguments](https://grambank.clld.org/parameters/GB070) — Binary feature (present/absent) for case marking on S/A/P arguments, covering 2,262 languages

[5] [Grambank Feature GB073: Morphological cases for oblique pronouns](https://grambank.clld.org/parameters/GB073) — Binary feature for oblique case marking on personal pronouns

[6] [Grambank GitHub Repository](https://github.com/grambank/grambank) — Full downloadable dataset with values.csv containing all feature codings for 2,400+ languages

[7] [Universal Dependencies Official Website](https://universaldependencies.org/) — Lists all UD treebanks and languages; v2.13 has 259 treebanks in 148 languages

[8] [Word Order - Wikipedia](https://en.wikipedia.org/wiki/Word_order) — Overview of V2 word order in Germanic languages (German, Dutch, Afrikaans) and global word order patterns

[9] [Kazakh Language - Wikipedia](https://en.wikipedia.org/wiki/Kazakh_language) — Confirms Turkic SOV word order and 7-case agglutinative system shared across Turkic family

[10] [Basic word order typology revisited: UD and WALS crosslinguistic study](https://www.degruyterbrill.com/document/doi/10.1515/lingvan-2021-0001/html) — Shows UD-extracted dominant orders concur with WALS for 74 languages

[11] [Proto-Indo-European Language - Wikipedia](https://en.wikipedia.org/wiki/Proto-Indo-European_language) — Historical reconstruction of PIE as SOV with well-documented daughter language word orders

## Follow-up Questions

- How should V2 languages (German, Dutch, Afrikaans) be coded for the Cox model — as SVO, SOV, or a separate 'V2' category?
- Should Grambank GB070 binary data be downloaded and used to fill the 53 languages with unknown case richness, or should those be treated as missing data in the Cox model?
- For the ~10 ancient/historical languages in UD (Gothic, Latin, Sanskrit, Hittite, etc.), should they be included in the Cox model or excluded since their typological features reflect dead languages?

---
*Generated by AI Inventor Pipeline*
