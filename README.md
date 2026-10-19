# Consumer complaints: language, template reuse, and outcomes

**Work in progress.** This project studies how the language and composition of consumer complaints change over time, with a focus on credit-reporting disputes in the Consumer Financial Protection Bureau (CFPB) database. It combines text measurement, template-reuse indicators, distributional comparisons, and exploratory models of company response categories.

The central question is whether changes in complaint volume and reported outcomes also reflect changes in how complaints are written and submitted. Credit-repair-related language is one part of that investigation. A separate benchmark panel covers several other financial-product groups.

The project began as DS202W coursework and has continued as an independent analysis. Thanks to **Dr Ghita Berrada** for feedback on the research.

## Motivation and research questions

A complaint database records both financial problems and the process through which people describe those problems. Two consumers can face similar disputes but use very different language: one writes a personal account, while another adapts a shared template or uses formal legal phrasing. If the balance between those styles changes, a rise in complaints or a change in response rates can be difficult to interpret from headline counts alone.

This project uses applied text analysis to investigate that measurement problem. Its main questions are:

- **How is complaint language changing?** Track narrative length, vocabulary, legal phrasing, and shared wording over time, alongside changes in products and issues.
- **How much do repeated templates shape the observed patterns?** Compare the full sample with deduplicated and non-templated subsets to see which patterns depend on shared text.
- **Do language patterns differ across companies, issues, and financial products?** Use these comparisons to distinguish a broad shift in writing from a change concentrated in a particular part of the database.
- **How are text features associated with recorded company responses?** Examine whether those associations persist after accounting for observed complaint characteristics and calendar categories.

The motivation is to make administrative text more useful for understanding consumer experiences and informing further research. Automated writing tools and credit-repair templates are possible parts of the story, but the observable object here is the text itself. The project does not treat a writing style as a reliable identifier of who wrote a complaint or which tool they used.

## From narratives to interpretable evidence

The workflow starts with complaint records and their published narratives. It converts each narrative into a set of numerical features, groups exact matches after text normalization, and attaches the product, issue, company, date, and response category. This creates a feature store that can support both aggregate comparisons and statistical models without repeatedly processing the raw text.

A **template flag** combines several signals rather than relying on one phrase. Repeated wording captures direct reuse; legal and template phrases describe style; a weighted score brings several text characteristics together. These signals are useful for examining composition, but they remain imperfect measurement choices. Comparing alternative flags and perturbing the score weights shows how much the conclusions would depend on those choices.

The temporal analysis then asks whether the language distribution is moving, rather than merely counting more complaints. Jensen–Shannon divergence and Hellinger distance summarize how different a month's vocabulary or style distribution is from the chosen reference group. A larger distance means a larger distributional difference; it does not explain the cause of that difference. Comparisons of full and deduplicated data help reveal whether repeated text is contributing to the shift.

Finally, logistic regression relates the measured features to response categories such as monetary or non-monetary relief. An odds ratio describes an association with the odds of a recorded response, conditional on the included controls. For example, an association between template use and relief could reflect differences in the underlying disputes, who submits them, or company processing, as well as wording. Establishing whether changing the wording itself changes a consumer's outcome would require a separate causal design.

## Implemented analysis

| Component | Implementation |
| --- | --- |
| Data preparation | Read a CFPB CSV in chunks, retain records with narratives, and select the credit-reporting family using product and issue patterns. Build a separate product benchmark feature store. |
| Text features | Measure length, lexical diversity, first-person language, uppercase and digit shares, placeholders, legal phrases, template phrases, and credit-repair-related markers. |
| Template reuse | Hash normalized text to identify repeated narratives; combine reuse clusters, a legalistic-language rule, and a weighted feature score into a broad template flag. |
| Temporal composition | Compare monthly template shares, issue mix, company concentration, repeated-submission proxies, full versus deduplicated records, and period-specific summaries. |
| Distributional change | Compare unigram and style-feature distributions with a reference subset using Jensen–Shannon divergence and Hellinger distance. |
| Sensitivity analysis | Perturb the weighted score's coefficients in 200 Monte Carlo draws while preserving total positive and negative weight mass. |
| Outcome association | Fit logistic models for monetary relief, non-monetary relief, combined relief, and flat closure, with text controls and company, issue, product, and time categories. |
| Supporting summaries | Summarize credit-repair-related themes, tagged groups, and cross-product comparisons. |

The score threshold is the 95th percentile among records dated through 2021, falling back to the whole sample if that subset is empty. Feature standardization uses the full scored sample. Exact normalized-text clusters contain at least three records; the legalistic rule combines at least two legal-pattern matches, at least one template-pattern match, and at least 60 tokens.

The code retains exploratory labels such as `possible_ai_consumer_template` and `likely_credit_repair_firm_drafted`. These are rule-based groupings of textual features. They do not establish AI use, identify an author, or verify commercial assistance. Shared text can arise from many sources.

## Data

Download a CSV from the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) and keep it locally, for example at `data/complaints.csv`. The repository contains source code and documentation; complaint records, narrative excerpts, review documents, and generated outputs are excluded.

The main script accepts the CFPB display-name columns below or their snake_case equivalents:

```text
Date received
Product
Sub-product
Issue
Sub-issue
Company
Tags
Company response to consumer
Consumer complaint narrative
```

The reduced-data utility expects the display-name field `Consumer complaint narrative`. It retains rows with a nonmissing value in that field; it does not otherwise validate or anonymize the text.

## Usage

Use Python 3.10 or newer. Dependencies are listed in [requirements.txt](requirements.txt); `pyarrow` supplies the Parquet engine used by the cached feature stores.

```powershell
python -m pip install -r requirements.txt
python cfpb_ai_analysis.py --csv-path data/complaints.csv --output-dir analysis_outputs --chunksize 100000 --model-sample-size 200000 --random-state 42
```

For text and composition analysis without the outcome regressions:

```powershell
python cfpb_ai_analysis.py --csv-path data/complaints.csv --output-dir analysis_outputs_text --skip-outcome-models
```

To rebuild the scored caches after changing the input data or feature rules:

```powershell
python cfpb_ai_analysis.py --csv-path data/complaints.csv --output-dir analysis_outputs --refresh-cache
```

To create a smaller CSV containing records with narratives:

```powershell
python refresh_cfpb_data.py --source-csv data/complaints.csv --target-csv data/complaints_with_narratives.csv --chunksize 100000
```

The reduced-data utility replaces an existing target file. Keep its source and target paths distinct, and create the target directory first.

The main pipeline reads raw data in chunks but retains the resulting feature stores in memory. `--chunksize` therefore does not bound total memory use. `--model-sample-size` affects the regression sampling stage only: flagged records are retained and unflagged records are sampled, so the final model dataset can exceed that number. It does not cap the full scoring pass.

Existing scored Parquet files are reused automatically. Cache reuse does not check whether the input CSV or analysis rules have changed. Use a fresh output directory or `--refresh-cache` when those change. Dependency versions are not pinned to a recorded execution environment.

## Outputs

The main script writes CSV tables, PNG figures, JSON summaries, and Parquet caches under the selected output directory, plus a `report_package` copy of report artifacts. Examples of aggregate tables include:

- `template_monthly_panel.csv`: template shares and response categories over time.
- `dataset_comparison_summary.csv`: full, deduplicated, and non-templated comparisons.
- `text_drift_jsd_panel.csv`: distributional distance measures by month.
- `template_weight_sensitivity_summary.csv`: score stability under weight perturbations.
- `template_outcome_model.csv`: selected model coefficients and odds ratios.
- `cross_product_benchmark_panel.csv`: broader product comparisons.

The pipeline also writes narrative-containing extracts and examples, including `template_examples.csv`, and copies those CSVs into `report_package`. These are local analysis artifacts; they are excluded from version control along with the source data and caches. No saved result or narrative excerpt is bundled here.

## Interpretation and current scope

The primary contribution is measurement of complaint-language patterns and dataset composition. Outcome models describe associations with recorded company response categories; they do not identify the causal effect of writing style or measure consumer satisfaction. The main model uses statsmodels with HC1 covariance estimates; if fitting fails, the code falls back to regularized scikit-learn logistic regression and leaves p-values missing.

The CFPB explains that its database is not a representative sample of all consumers' experiences. Complaint volumes also require context about the population, product, and company involved. See the [CFPB database guidance](https://www.consumerfinance.gov/data-research/consumer-complaints/).

Text flags depend on hand-chosen rules, normalization, and thresholds. Repeated text does not identify repeated consumers, and deduplication can remove distinct complaints that share wording. The reference corpus is selected using the same heuristic features used elsewhere in the analysis. Calendar groupings, including the 2021 baseline boundary and later-period labels, are analytical conventions rather than a verified record of tool adoption.

Further work includes manually reviewing the text measures, assessing alternative reference corpora and thresholds, and refining the relationship between dataset composition and outcome associations.

## Files

- [cfpb_ai_analysis.py](cfpb_ai_analysis.py): analysis pipeline and command-line interface.
- [refresh_cfpb_data.py](refresh_cfpb_data.py): narrative-only CSV utility.
- [requirements.txt](requirements.txt): third-party dependencies.
- [.gitignore](.gitignore): local data, generated artifacts, environments, and credential exclusions.
