from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


DEFAULT_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company",
    "Tags",
    "Company response to consumer",
    "Consumer complaint narrative",
]

ALT_COLUMNS = [
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "tags",
    "company_response_to_consumer",
    "consumer_complaint_narrative",
]

POSITIVE_OUTCOMES = {
    "Closed with monetary relief",
    "Closed with non-monetary relief",
}

MONETARY_RELIEF_OUTCOMES = {"Closed with monetary relief"}
NON_MONETARY_RELIEF_OUTCOMES = {"Closed with non-monetary relief"}
FLAT_CLOSURE_OUTCOMES = {"Closed with explanation", "Closed"}

LEGAL_PATTERNS = [
    r"\bfair credit reporting act\b",
    r"\bfcra\b",
    r"\bfdcpa\b",
    r"\b15\s+u\.?s\.?c\.?\b",
    r"\bsection\s+\d{3,4}\b",
    r"\bpursuant to\b",
    r"\bin accordance with\b",
    r"\bconsumer reporting agenc(?:y|ies)\b",
    r"\bpermissible purpose\b",
    r"\bcivil liabilit(?:y|ies)\b",
    r"\bnegligent noncompliance\b",
    r"\bverify(?:ing|ication)? or delete\b",
]

TEMPLATE_PATTERNS = [
    r"\bi am writing to formally\b",
    r"\bto whom it may concern\b",
    r"\bi kindly request\b",
    r"\bplease investigate and delete\b",
    r"\bi request a written response\b",
    r"\bthis information is inaccurate\b",
    r"\bremove this item from my credit report\b",
    r"\bwithout my knowledge(?: or consent)?\b",
    r"\bunauthorized (?:account|accounts|inquir(?:y|ies))\b",
    r"\bidentity theft\b",
]

CREDIT_REPAIR_PATTERNS = [
    r"\bcredit repair\b",
    r"\bcredit repair (?:company|firm|service|agency)\b",
    r"\bcredit restoration\b",
    r"\bcredit specialist\b",
    r"\bcredit sweep\b",
    r"\bsection 609\b",
    r"\bon behalf of\b",
    r"\bprepared by\b",
    r"\bour client\b",
    r"\bclient file\b",
    r"\bauthorized representative\b",
    r"\bpower of attorney\b",
    r"\bfile prepared\b",
    r"\bcredit consultant\b",
]

COMPLAINT_ABOUT_CREDIT_REPAIR_PATTERNS = [
    r"\bused (?:a |the )?credit repair\b",
    r"\btheir credit repair services\b",
    r"\bpaid .*credit repair\b",
    r"\bsigned up for .*credit repair\b",
    r"\btelemarketing\b",
    r"\bdidn'?t provide services promised\b",
    r"\bcharged upfront\b",
    r"\bunexpected fees\b",
]

THIRD_PARTY_PATTERNS = [
    r"\bon behalf of\b",
    r"\bour client\b",
    r"\bmy client\b",
    r"\bauthorized representative\b",
    r"\bpower of attorney\b",
    r"\bprepared by\b",
    r"\bfile prepared\b",
]

ANTI_CREDIT_REPAIR_PATTERNS = [
    r"\bnot a credit repair (?:company|firm|service|agency)\b",
    r"\bi am not a credit repair (?:company|firm|service|agency)\b",
    r"\bnot working with a credit repair\b",
    r"\bwithout a credit repair company\b",
]

KNOWN_CREDIT_REPAIR_FIRMS = [
    "lexington law",
    "creditrepair.com",
    "ovation credit",
    "sky blue credit",
    "the credit people",
    "msi credit solutions",
    "credit saint",
    "safeport law",
    "credit versio",
]

TEMPLATE_SCORE_WEIGHTS = {
    "exact_template_group_size_z": 0.32,
    "legal_hits_z": 0.20,
    "template_hits_z": 0.18,
    "placeholder_hits_z": 0.10,
    "credit_repair_hits_z": 0.08,
    "known_firm_hits_z": 0.07,
    "third_party_hits_z": 0.06,
    "token_count_z": 0.08,
    "uppercase_share_z": 0.05,
    "digit_share_z": 0.05,
    "type_token_ratio": -0.14,
    "first_person_share": -0.10,
}
PRE_AI_END_YEAR = 2021
AI_COMMON_USE_START_YEAR = 2023
TEMPLATE_SCORE_PRE_AI_QUANTILE = 0.95
MONTE_CARLO_DRAWS = 200
MONTE_CARLO_WEIGHT_SIGMA = 0.20
MIN_DRIFT_MONTHLY_COMPLAINTS = 25
JSD_TOP_VOCAB_SIZE = 500
CREDIT_REPORTING_PRODUCT_PATTERN = r"credit reporting|consumer reports|credit repair services"
CREDIT_REPORTING_ISSUE_PATTERN = (
    r"incorrect information on your report|incorrect information on credit report|"
    r"improper use of your report|improper use of my credit report|"
    r"investigation into an existing problem|investigation into an existing issue|"
    r"credit reporting company'?s investigation|unable to get your credit report or credit score|"
    r"unable to get credit report/credit score|fraud alerts|security freezes|"
    r"credit monitoring|identity theft protection|problem with credit report or credit score|privacy"
)
PRODUCT_BENCHMARK_PATTERNS = {
    "Credit reporting": r"credit reporting|consumer reports|credit repair services",
    "Debt collection": r"debt collection",
    "Credit card": r"credit card|prepaid card",
    "Mortgage": r"mortgage",
    "Student loan": r"student loan",
    "Bank account": r"checking|savings|bank account",
    "Personal loan": r"payday|title loan|personal loan|consumer loan|vehicle loan|lease",
}

FIRST_PERSON_PATTERN = re.compile(r"\b(i|me|my|mine|we|our|ours)\b", flags=re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z']+")
PLACEHOLDER_PATTERN = re.compile(r"\bx{2,}\b", flags=re.IGNORECASE)
LEGAL_REGEX = re.compile("|".join(LEGAL_PATTERNS), flags=re.IGNORECASE)
TEMPLATE_REGEX = re.compile("|".join(TEMPLATE_PATTERNS), flags=re.IGNORECASE)
CREDIT_REPAIR_REGEX = re.compile("|".join(CREDIT_REPAIR_PATTERNS), flags=re.IGNORECASE)
COMPLAINT_ABOUT_CREDIT_REPAIR_REGEX = re.compile("|".join(COMPLAINT_ABOUT_CREDIT_REPAIR_PATTERNS), flags=re.IGNORECASE)
THIRD_PARTY_REGEX = re.compile("|".join(THIRD_PARTY_PATTERNS), flags=re.IGNORECASE)
ANTI_CREDIT_REPAIR_REGEX = re.compile("|".join(ANTI_CREDIT_REPAIR_PATTERNS), flags=re.IGNORECASE)


@dataclass
class RowFeatures:
    date_received: str
    month: str
    product: str
    issue: str
    company: str
    tags: str
    company_response: str
    positive_outcome: bool
    monetary_relief: bool
    non_monetary_relief: bool
    flat_closure: bool
    normalized_hash: str
    token_count: int
    type_token_ratio: float
    legal_hits: int
    template_hits: int
    placeholder_hits: int
    credit_repair_hits: int
    known_firm_hits: int
    company_known_credit_repair: int
    complaint_about_credit_repair_hits: int
    third_party_hits: int
    anti_credit_repair_hits: int
    first_person_share: float
    uppercase_share: float
    digit_share: float
    narrative_excerpt: str


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = (
        renamed.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    return renamed


def normalize_text(text: str) -> str:
    text = str(text)
    text = PLACEHOLDER_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalized_hash(text: str) -> str:
    normalized = (
        normalize_text(text)
        .lower()
        .replace("\n", " ")
    )
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def lexical_metrics(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    token_count = len(tokens)
    char_count = len(text)
    if token_count == 0:
        return {
            "token_count": 0,
            "type_token_ratio": 0.0,
            "legal_hits": 0,
            "template_hits": 0,
            "placeholder_hits": 0,
            "credit_repair_hits": 0,
            "known_firm_hits": 0,
            "company_known_credit_repair": 0,
            "complaint_about_credit_repair_hits": 0,
            "third_party_hits": 0,
            "anti_credit_repair_hits": 0,
            "first_person_share": 0.0,
            "uppercase_share": 0.0,
            "digit_share": 0.0,
        }

    first_person_hits = len(FIRST_PERSON_PATTERN.findall(text))
    return {
        "token_count": token_count,
        "type_token_ratio": len(set(tokens)) / token_count,
        "legal_hits": len(LEGAL_REGEX.findall(text)),
        "template_hits": len(TEMPLATE_REGEX.findall(text)),
        "placeholder_hits": len(PLACEHOLDER_PATTERN.findall(text)),
        "credit_repair_hits": len(CREDIT_REPAIR_REGEX.findall(text)),
        "known_firm_hits": sum(name in text.lower() for name in KNOWN_CREDIT_REPAIR_FIRMS),
        "company_known_credit_repair": 0,
        "complaint_about_credit_repair_hits": len(COMPLAINT_ABOUT_CREDIT_REPAIR_REGEX.findall(text)),
        "third_party_hits": len(THIRD_PARTY_REGEX.findall(text)),
        "anti_credit_repair_hits": len(ANTI_CREDIT_REPAIR_REGEX.findall(text)),
        "first_person_share": first_person_hits / token_count,
        "uppercase_share": sum(ch.isupper() for ch in text) / max(char_count, 1),
        "digit_share": sum(ch.isdigit() for ch in text) / max(char_count, 1),
    }


def focus_credit_subset(df: pd.DataFrame) -> pd.DataFrame:
    product_mask = df["product"].fillna("").str.contains(CREDIT_REPORTING_PRODUCT_PATTERN, case=False, regex=True)
    issue_mask = df["issue"].fillna("").str.contains(CREDIT_REPORTING_ISSUE_PATTERN, case=False, regex=True)
    return df.loc[product_mask | issue_mask].copy()


def focus_product_benchmark_subset(df: pd.DataFrame) -> pd.DataFrame:
    product_mask = pd.Series(False, index=df.index)
    for pattern in PRODUCT_BENCHMARK_PATTERNS.values():
        product_mask = product_mask | df["product"].fillna("").str.contains(pattern, case=False, regex=True)
    return df.loc[product_mask].copy()


def assign_product_benchmark_bucket(product: str) -> str:
    product_text = str(product)
    for label, pattern in PRODUCT_BENCHMARK_PATTERNS.items():
        if re.search(pattern, product_text, flags=re.IGNORECASE):
            return label
    return "Other"


def iter_credit_chunks(csv_path: Path, chunksize: int) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    if set(DEFAULT_COLUMNS).issubset(header.columns):
        usecols = DEFAULT_COLUMNS
    elif set(ALT_COLUMNS).issubset(header.columns):
        usecols = ALT_COLUMNS
    else:
        raise ValueError(f"CSV columns do not match the expected CFPB schema in {csv_path}.")

    for chunk_number, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        chunk = snake_case_columns(chunk)
        chunk = chunk.dropna(subset=["consumer_complaint_narrative"])
        if chunk.empty:
            print(f"[{timestamp()}] Chunk {chunk_number}: no narratives kept")
            continue
        chunk = focus_credit_subset(chunk)
        if chunk.empty:
            print(f"[{timestamp()}] Chunk {chunk_number}: no credit-related complaints kept")
            continue
        print(f"[{timestamp()}] Chunk {chunk_number}: kept {len(chunk):,} credit-related narrative complaints")
        yield chunk


def iter_product_benchmark_chunks(csv_path: Path, chunksize: int) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    if set(DEFAULT_COLUMNS).issubset(header.columns):
        usecols = DEFAULT_COLUMNS
    elif set(ALT_COLUMNS).issubset(header.columns):
        usecols = ALT_COLUMNS
    else:
        raise ValueError(f"CSV columns do not match the expected CFPB schema in {csv_path}.")

    for chunk_number, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        chunk = snake_case_columns(chunk)
        chunk = chunk.dropna(subset=["consumer_complaint_narrative"])
        if chunk.empty:
            continue
        chunk = focus_product_benchmark_subset(chunk)
        if chunk.empty:
            continue
        print(f"[{timestamp()}] Benchmark chunk {chunk_number}: kept {len(chunk):,} narrative complaints")
        yield chunk


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_row_features(row: dict[str, object]) -> RowFeatures:
    narrative = normalize_text(row["consumer_complaint_narrative"])
    metrics = lexical_metrics(narrative)
    date_received = pd.to_datetime(row["date_received"], errors="coerce")
    company_text = str(row.get("company", ""))
    company_response = str(row.get("company_response_to_consumer", ""))
    company_lower = company_text.lower()
    issue_text = str(row.get("issue", "")).lower()
    company_known_credit_repair = int(any(name in company_lower for name in KNOWN_CREDIT_REPAIR_FIRMS))
    complaint_about_credit_repair_hits = int(metrics["complaint_about_credit_repair_hits"])
    if company_known_credit_repair:
        complaint_about_credit_repair_hits += 1
    if (
        ("credit repair" in narrative.lower() or int(metrics["known_firm_hits"]) > 0)
        and any(term in issue_text for term in ["service", "fee", "promised", "charged", "unexpected fees"])
    ):
        complaint_about_credit_repair_hits += 1
    month = ""
    date_string = ""
    if pd.notna(date_received):
        month = str(date_received.to_period("M"))
        date_string = date_received.strftime("%Y-%m-%d")

    return RowFeatures(
        date_received=date_string,
        month=month,
        product=str(row.get("product", "")),
        issue=str(row.get("issue", "")),
        company=company_text,
        tags=str(row.get("tags", "")),
        company_response=company_response,
        positive_outcome=company_response in POSITIVE_OUTCOMES,
        monetary_relief=company_response in MONETARY_RELIEF_OUTCOMES,
        non_monetary_relief=company_response in NON_MONETARY_RELIEF_OUTCOMES,
        flat_closure=company_response in FLAT_CLOSURE_OUTCOMES,
        normalized_hash=normalized_hash(narrative),
        token_count=int(metrics["token_count"]),
        type_token_ratio=float(metrics["type_token_ratio"]),
        legal_hits=int(metrics["legal_hits"]),
        template_hits=int(metrics["template_hits"]),
        placeholder_hits=int(metrics["placeholder_hits"]),
        credit_repair_hits=int(metrics["credit_repair_hits"]),
        known_firm_hits=int(metrics["known_firm_hits"]),
        company_known_credit_repair=company_known_credit_repair,
        complaint_about_credit_repair_hits=complaint_about_credit_repair_hits,
        third_party_hits=int(metrics["third_party_hits"]),
        anti_credit_repair_hits=int(metrics["anti_credit_repair_hits"]),
        first_person_share=float(metrics["first_person_share"]),
        uppercase_share=float(metrics["uppercase_share"]),
        digit_share=float(metrics["digit_share"]),
        narrative_excerpt=narrative[:800],
    )


def build_full_feature_store(csv_path: Path, chunksize: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    total_rows = 0
    for chunk in iter_credit_chunks(csv_path, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            features = build_row_features(row._asdict())
            records.append(asdict(features))
        total_rows += len(chunk)
        print(f"[{timestamp()}] Running total retained for scoring: {total_rows:,}")
    if not records:
        raise ValueError("No credit-related narrative complaints were found.")
    df = pd.DataFrame.from_records(records)
    for column in ["product", "issue", "company", "month"]:
        df[column] = df[column].astype("category")
    print(f"[{timestamp()}] Feature store built with {len(df):,} rows")
    return df


def build_product_benchmark_feature_store(csv_path: Path, chunksize: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    total_rows = 0
    for chunk in iter_product_benchmark_chunks(csv_path, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            features = build_row_features(row._asdict())
            record = asdict(features)
            record["benchmark_product_bucket"] = assign_product_benchmark_bucket(record["product"])
            records.append(record)
        total_rows += len(chunk)
        print(f"[{timestamp()}] Benchmark running total retained for scoring: {total_rows:,}")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    for column in ["product", "issue", "company", "month", "benchmark_product_bucket"]:
        df[column] = df[column].astype("category")
    print(f"[{timestamp()}] Benchmark feature store built with {len(df):,} rows")
    return df


def safe_zscore(series: pd.Series) -> pd.Series:
    centered = series - series.mean()
    scaled = series.std(ddof=0)
    if scaled == 0 or np.isnan(scaled):
        return pd.Series(0.0, index=series.index)
    return centered / scaled


def build_template_score(scored: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=scored.index)
    for column, weight in weights.items():
        score = score + weight * scored[column]
    return score


def draw_weight_vectors(base_weights: dict[str, float], draws: int, random_state: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(random_state)
    varied_weights: list[dict[str, float]] = []
    positive_columns = [column for column, weight in base_weights.items() if weight > 0]
    negative_columns = [column for column, weight in base_weights.items() if weight < 0]
    positive_total = sum(base_weights[column] for column in positive_columns)
    negative_total = -sum(base_weights[column] for column in negative_columns)

    for _ in range(draws):
        sampled: dict[str, float] = {}
        if positive_columns:
            pos_base = np.array([base_weights[column] for column in positive_columns], dtype=float)
            pos_draw = pos_base * rng.lognormal(mean=0.0, sigma=MONTE_CARLO_WEIGHT_SIGMA, size=len(positive_columns))
            pos_draw = positive_total * pos_draw / pos_draw.sum()
            sampled.update(dict(zip(positive_columns, pos_draw)))
        if negative_columns:
            neg_base = np.array([-base_weights[column] for column in negative_columns], dtype=float)
            neg_draw = neg_base * rng.lognormal(mean=0.0, sigma=MONTE_CARLO_WEIGHT_SIGMA, size=len(negative_columns))
            neg_draw = negative_total * neg_draw / neg_draw.sum()
            sampled.update(dict(zip(negative_columns, -neg_draw)))
        varied_weights.append(sampled)
    return varied_weights


def compute_template_flags(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["exact_template_group_size"] = scored.groupby("normalized_hash")["normalized_hash"].transform("size")
    scored["date_dt"] = pd.to_datetime(scored["date_received"], errors="coerce")
    scored["year_received"] = scored["date_dt"].dt.year
    scored["pre_ai_period"] = scored["year_received"].fillna(0).astype(int) <= PRE_AI_END_YEAR
    scored["post_ai_period"] = scored["year_received"].fillna(0).astype(int) >= AI_COMMON_USE_START_YEAR

    zscore_columns = [
        "legal_hits",
        "template_hits",
        "placeholder_hits",
        "credit_repair_hits",
        "known_firm_hits",
        "third_party_hits",
        "token_count",
        "exact_template_group_size",
        "uppercase_share",
        "digit_share",
    ]
    for column in zscore_columns:
        scored[f"{column}_z"] = safe_zscore(scored[column])

    scored["template_score"] = build_template_score(scored, TEMPLATE_SCORE_WEIGHTS)
    pre_ai_scores = scored.loc[scored["pre_ai_period"], "template_score"]
    if pre_ai_scores.empty:
        pre_ai_scores = scored["template_score"]
    score_threshold = float(pre_ai_scores.quantile(TEMPLATE_SCORE_PRE_AI_QUANTILE))
    scored["template_score_threshold_pre_ai"] = score_threshold

    scored["template_cluster"] = scored["exact_template_group_size"] >= 3
    scored["legalistic_template"] = (
        (scored["legal_hits"] >= 2)
        & (scored["template_hits"] >= 1)
        & (scored["token_count"] >= 60)
    )
    scored["score_based_template"] = scored["template_score"] >= score_threshold
    scored["is_templated"] = scored["template_cluster"] | scored["legalistic_template"] | scored["score_based_template"]

    scored["is_templated"] = (
        scored["template_cluster"]
        | scored["legalistic_template"]
        | scored["score_based_template"]
    )
    scored["complaint_about_credit_repair_firm"] = (
        (scored["company_known_credit_repair"] > 0)
        | (scored["complaint_about_credit_repair_hits"] >= 2)
        | (
            (scored["known_firm_hits"] > 0)
            & (scored["anti_credit_repair_hits"] == 0)
            & (scored["third_party_hits"] == 0)
        )
    )
    scored["credit_repair_signature"] = (
        (scored["credit_repair_hits"] > 0).astype(int)
        + (scored["known_firm_hits"] > 0).astype(int)
        + (scored["third_party_hits"] > 0).astype(int)
        + (scored["legal_hits"] >= 3).astype(int)
    )
    scored["likely_credit_repair_firm_drafted"] = (
        scored["is_templated"]
        & ~scored["complaint_about_credit_repair_firm"]
        & (scored["anti_credit_repair_hits"] == 0)
        & (scored["credit_repair_signature"] >= 2)
        & (
            scored["template_cluster"]
            | ((scored["third_party_hits"] > 0) & (scored["legal_hits"] >= 2))
            | ((scored["credit_repair_hits"] > 0) & (scored["legal_hits"] >= 3) & (scored["first_person_share"] < 0.05))
        )
    )
    scored["possible_ai_consumer_template"] = (
        scored["is_templated"]
        & ~scored["complaint_about_credit_repair_firm"]
        & ~scored["likely_credit_repair_firm_drafted"]
        & scored["post_ai_period"]
        & ~scored["template_cluster"]
        & (scored["third_party_hits"] == 0)
        & (scored["known_firm_hits"] == 0)
        & (scored["template_hits"] >= 1)
        & (scored["legal_hits"] >= 2)
        & (scored["first_person_share"] >= 0.02)
    )
    scored["possible_consumer_side_automation"] = (
        scored["is_templated"]
        & ~scored["complaint_about_credit_repair_firm"]
        & ~scored["likely_credit_repair_firm_drafted"]
        & ~scored["possible_ai_consumer_template"]
    )
    scored["credit_repair_exposure"] = (
        scored["complaint_about_credit_repair_firm"]
        | scored["likely_credit_repair_firm_drafted"]
        | (scored["credit_repair_hits"] > 0)
        | (scored["known_firm_hits"] > 0)
        | (scored["third_party_hits"] > 0)
    )
    scored["pre_ai_template_cluster"] = scored["template_cluster"] & scored["pre_ai_period"]
    scored["pre_ai_credit_repair_style_template"] = (
        scored["pre_ai_template_cluster"]
        & (
            scored["likely_credit_repair_firm_drafted"]
            | (scored["credit_repair_signature"] >= 2)
            | ((scored["credit_repair_hits"] > 0) & (scored["legal_hits"] >= 3))
            | ((scored["third_party_hits"] > 0) & (scored["legal_hits"] >= 2))
        )
    )
    scored["template_origin"] = np.select(
        [
            scored["complaint_about_credit_repair_firm"],
            scored["likely_credit_repair_firm_drafted"],
            scored["possible_ai_consumer_template"],
            scored["possible_consumer_side_automation"],
            scored["is_templated"],
        ],
        [
            "Complaint about credit repair firm",
            "Likely credit-repair-firm drafted",
            "Possible consumer AI-assisted template",
            "Other templated / consumer-side automation",
            "Other templated / automated style",
        ],
        default="Non-templated",
    )
    scored["period_bin"] = pd.cut(
        pd.to_datetime(scored["date_received"], errors="coerce").dt.year,
        bins=[2014, 2019, 2021, 2023, 2026],
        labels=["2015-2019", "2020-2021", "2022-2023", "2024-2026"],
        include_lowest=True,
    ).astype(str)
    return scored


def build_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("is_templated")
        .agg(
            complaints=("narrative_excerpt", "size"),
            avg_tokens=("token_count", "mean"),
            avg_ttr=("type_token_ratio", "mean"),
            duplicate_rate=("exact_template_group_size", lambda s: (s > 1).mean()),
            positive_outcome_rate=("positive_outcome", "mean"),
            monetary_relief_rate=("monetary_relief", "mean"),
            non_monetary_relief_rate=("non_monetary_relief", "mean"),
            flat_closure_rate=("flat_closure", "mean"),
        )
        .reset_index()
    )
    summary["group"] = np.where(summary["is_templated"], "Templated / AI-style", "Non-templated")
    return summary[
        [
            "group",
            "complaints",
            "avg_tokens",
            "avg_ttr",
            "duplicate_rate",
            "positive_outcome_rate",
            "monetary_relief_rate",
            "non_monetary_relief_rate",
            "flat_closure_rate",
        ]
    ]


def build_origin_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("template_origin", observed=False)
        .agg(
            complaints=("narrative_excerpt", "size"),
            positive_outcome_rate=("positive_outcome", "mean"),
            duplicate_rate=("exact_template_group_size", lambda s: (s > 1).mean()),
            direct_credit_repair_mentions=("credit_repair_hits", lambda s: (s > 0).mean()),
            known_firm_mentions=("known_firm_hits", lambda s: (s > 0).mean()),
            third_party_representation_mentions=("third_party_hits", lambda s: (s > 0).mean()),
        )
        .reset_index()
        .sort_values("complaints", ascending=False)
    )
    summary["share_of_dataset"] = summary["complaints"] / len(df)
    return summary


def build_credit_repair_complaints(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date_received",
        "company",
        "issue",
        "product",
        "positive_outcome",
        "credit_repair_hits",
        "known_firm_hits",
        "company_known_credit_repair",
        "complaint_about_credit_repair_hits",
        "narrative_excerpt",
    ]
    return df.loc[df["complaint_about_credit_repair_firm"], columns].sort_values("date_received").reset_index(drop=True)


def build_credit_repair_firm_drafted(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date_received",
        "company",
        "issue",
        "product",
        "positive_outcome",
        "template_score",
        "exact_template_group_size",
        "credit_repair_hits",
        "known_firm_hits",
        "third_party_hits",
        "legal_hits",
        "template_hits",
        "narrative_excerpt",
    ]
    return df.loc[df["likely_credit_repair_firm_drafted"], columns].sort_values(
        ["exact_template_group_size", "template_score"], ascending=False
    ).reset_index(drop=True)


def build_credit_repair_theme_summary(complaints_about_df: pd.DataFrame) -> pd.DataFrame:
    theme_map = {
        "fees_and_billing": ["fee", "fees", "charged", "billing", "refund", "money", "thousand", "payment"],
        "services_not_delivered": ["promised", "service", "services", "helping", "remove", "deleted", "improve"],
        "telemarketing_or_sales": ["telemarketing", "sales", "live-transferred", "marketing", "called"],
        "score_harm_or_no_improvement": ["score", "credit remained", "worse", "dropped", "improvement"],
        "identity_theft_and_disputes": ["identity theft", "dispute", "investigation", "inquiry", "report"],
    }
    rows = []
    narratives = complaints_about_df["narrative_excerpt"].fillna("").str.lower()
    for theme, terms in theme_map.items():
        pattern = "|".join(re.escape(term) for term in terms)
        count = int(narratives.str.contains(pattern, regex=True).sum())
        rows.append({"theme": theme, "complaints": count, "share_with_theme": count / max(len(complaints_about_df), 1)})
    return pd.DataFrame(rows).sort_values("complaints", ascending=False).reset_index(drop=True)


def build_credit_repair_phrase_table(complaints_about_df: pd.DataFrame) -> pd.DataFrame:
    stopwords = {
        "the", "and", "that", "this", "with", "have", "from", "they", "were", "been", "their", "them", "then",
        "when", "your", "would", "could", "should", "into", "about", "there", "because", "which", "after", "before",
        "still", "than", "just", "very", "also", "have", "had", "for", "not", "but", "are", "was", "did", "didnt",
    }
    unigram_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()
    for text in complaints_about_df["narrative_excerpt"].fillna("").astype(str):
        tokens = [tok for tok in tokenize(text) if len(tok) > 2 and tok not in stopwords]
        unigram_counter.update(tokens)
        bigram_counter.update(" ".join(pair) for pair in zip(tokens, tokens[1:]))

    top_unigrams = [{"feature_type": "unigram", "feature": term, "count": count} for term, count in unigram_counter.most_common(20)]
    top_bigrams = [{"feature_type": "bigram", "feature": term, "count": count} for term, count in bigram_counter.most_common(20)]
    return pd.DataFrame(top_unigrams + top_bigrams)


def build_credit_repair_accessibility_summary(df: pd.DataFrame) -> pd.DataFrame:
    tag_groups = {
        "Servicemember": r"servicemember",
        "Older American": r"older american",
    }
    rows = []
    for label, pattern in tag_groups.items():
        mask = df["tags"].fillna("").str.contains(pattern, case=False, regex=True)
        tagged = df.loc[mask].copy()
        if tagged.empty:
            continue
        rows.append(
            {
                "group": label,
                "complaints": int(len(tagged)),
                "share_of_dataset": len(tagged) / len(df),
                "templated_share": tagged["is_templated"].mean(),
                "possible_ai_consumer_template_share": tagged["possible_ai_consumer_template"].mean(),
                "pre_ai_template_cluster_share": tagged["pre_ai_template_cluster"].mean(),
                "pre_ai_credit_repair_style_template_share": tagged["pre_ai_credit_repair_style_template"].mean(),
                "share_complaint_about_credit_repair_firm": tagged["complaint_about_credit_repair_firm"].mean(),
                "share_likely_credit_repair_firm_drafted": tagged["likely_credit_repair_firm_drafted"].mean(),
                "share_any_credit_repair_exposure": tagged["credit_repair_exposure"].mean(),
                "positive_outcome_rate": tagged["positive_outcome"].mean(),
            }
        )

    untagged = df.loc[~df["tags"].fillna("").str.contains("|".join(tag_groups.values()), case=False, regex=True)].copy()
    if not untagged.empty:
        rows.append(
            {
                "group": "No CFPB tag",
                "complaints": int(len(untagged)),
                "share_of_dataset": len(untagged) / len(df),
                "templated_share": untagged["is_templated"].mean(),
                "possible_ai_consumer_template_share": untagged["possible_ai_consumer_template"].mean(),
                "pre_ai_template_cluster_share": untagged["pre_ai_template_cluster"].mean(),
                "pre_ai_credit_repair_style_template_share": untagged["pre_ai_credit_repair_style_template"].mean(),
                "share_complaint_about_credit_repair_firm": untagged["complaint_about_credit_repair_firm"].mean(),
                "share_likely_credit_repair_firm_drafted": untagged["likely_credit_repair_firm_drafted"].mean(),
                "share_any_credit_repair_exposure": untagged["credit_repair_exposure"].mean(),
                "positive_outcome_rate": untagged["positive_outcome"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_tag_similarity_summary(df: pd.DataFrame) -> pd.DataFrame:
    tag_groups = {
        "Servicemember": r"servicemember",
        "Older American": r"older american",
    }
    rows: list[dict[str, object]] = []
    clean_reference = df.loc[(~df["is_templated"]) & (df["token_count"] > 0)].copy()
    if clean_reference.empty:
        clean_reference = df.loc[df["token_count"] > 0].copy()
    reference_tokens = Counter()
    for text in clean_reference["narrative_excerpt"].fillna(""):
        reference_tokens.update(tokenize(str(text)))
    vocabulary = [token for token, _ in reference_tokens.most_common(300)]
    if not vocabulary:
        vocabulary = ["credit"]
    reference_unigram = build_unigram_distribution(clean_reference["narrative_excerpt"], vocabulary)

    group_frames: dict[str, pd.DataFrame] = {}
    for label, pattern in tag_groups.items():
        group_frames[label] = df.loc[df["tags"].fillna("").str.contains(pattern, case=False, regex=True)].copy()
    group_frames["No CFPB tag"] = df.loc[
        ~df["tags"].fillna("").str.contains("|".join(tag_groups.values()), case=False, regex=True)
    ].copy()

    for label, frame in group_frames.items():
        if frame.empty:
            continue
        pre_ai = frame.loc[frame["pre_ai_period"]]
        post_ai = frame.loc[frame["post_ai_period"]]
        rows.append(
            {
                "group": label,
                "complaints": int(len(frame)),
                "templated_share": float(frame["is_templated"].mean()),
                "possible_ai_consumer_template_share": float(frame["possible_ai_consumer_template"].mean()),
                "pre_ai_template_cluster_share": float(pre_ai["pre_ai_template_cluster"].mean()) if not pre_ai.empty else np.nan,
                "post_ai_template_share": float(post_ai["is_templated"].mean()) if not post_ai.empty else np.nan,
                "jsd_group_vs_clean_unigram": jensen_shannon_divergence(
                    build_unigram_distribution(frame["narrative_excerpt"], vocabulary),
                    reference_unigram,
                ),
                "jsd_pre_ai_vs_post_ai_unigram": jensen_shannon_divergence(
                    build_unigram_distribution(pre_ai["narrative_excerpt"], vocabulary) if not pre_ai.empty else np.ones(len(vocabulary)),
                    build_unigram_distribution(post_ai["narrative_excerpt"], vocabulary) if not post_ai.empty else np.ones(len(vocabulary)),
                ) if (not pre_ai.empty and not post_ai.empty) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_cross_product_benchmark_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "benchmark_product_bucket" not in df.columns:
        return pd.DataFrame()
    panel = (
        df[df["month"] != ""]
        .groupby(["benchmark_product_bucket", "month"], observed=False)
        .agg(
            complaints=("narrative_excerpt", "size"),
            templated_share=("is_templated", "mean"),
            template_cluster_share=("template_cluster", "mean"),
            possible_ai_template_share=("possible_ai_consumer_template", "mean"),
        )
        .reset_index()
        .sort_values(["benchmark_product_bucket", "month"])
    )
    panel["year"] = panel["month"].str.slice(0, 4).astype(int)
    panel["templated_share_rolling_3m"] = (
        panel.groupby("benchmark_product_bucket", observed=False)["templated_share"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    return panel


def build_deduplicated_dataset(df: pd.DataFrame) -> pd.DataFrame:
    deduped = (
        df.sort_values("date_received")
        .drop_duplicates(subset=["normalized_hash"], keep="first")
        .reset_index(drop=True)
    )
    return deduped


def build_dataset_comparison_summary(full_df: pd.DataFrame, dedup_df: pd.DataFrame) -> pd.DataFrame:
    versions = {
        "full_dataset": full_df,
        "deduplicated_dataset": dedup_df,
        "non_templated_subset": full_df.loc[~full_df["is_templated"]].copy(),
    }
    rows = []
    for version, frame in versions.items():
        rows.append(
            {
                "dataset_version": version,
                "complaints": int(len(frame)),
                "templated_share": frame["is_templated"].mean() if "is_templated" in frame else np.nan,
                "positive_outcome_rate": frame["positive_outcome"].mean(),
                "monetary_relief_rate": frame["monetary_relief"].mean(),
                "non_monetary_relief_rate": frame["non_monetary_relief"].mean(),
                "flat_closure_rate": frame["flat_closure"].mean(),
                "duplicate_rate": (frame["exact_template_group_size"] > 1).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_issue_counterfactual(df: pd.DataFrame, dedup_df: pd.DataFrame) -> pd.DataFrame:
    def issue_share(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        out = frame["issue"].value_counts(normalize=True).rename(label).reset_index()
        out.columns = ["issue", label]
        return out

    merged = issue_share(df, "raw_share")
    merged = merged.merge(issue_share(df.loc[~df["is_templated"]], "non_templated_share"), on="issue", how="outer")
    merged = merged.merge(issue_share(dedup_df, "deduplicated_share"), on="issue", how="outer")
    for column in ["raw_share", "non_templated_share", "deduplicated_share"]:
        merged[column] = merged[column].fillna(0)
    merged["raw_minus_deduplicated_pp"] = 100 * (merged["raw_share"] - merged["deduplicated_share"])
    merged["raw_minus_non_templated_pp"] = 100 * (merged["raw_share"] - merged["non_templated_share"])
    return merged.sort_values("raw_minus_deduplicated_pp", ascending=False).reset_index(drop=True)


def build_company_concentration_summary(df: pd.DataFrame, dedup_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {
        "templated_full": df.loc[df["is_templated"]],
        "non_templated_full": df.loc[~df["is_templated"]],
        "full_deduplicated": dedup_df,
    }
    for name, frame in subsets.items():
        if frame.empty:
            continue
        shares = frame["company"].value_counts(normalize=True)
        rows.append(
            {
                "subset": name,
                "complaints": int(len(frame)),
                "company_hhi": float((shares**2).sum()),
                "top_5_company_share": float(shares.head(5).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_repeat_submission_summary(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["date_dt"] = pd.to_datetime(frame["date_received"], errors="coerce")
    repeat_groups = (
        frame.groupby(["company", "issue", "normalized_hash"], observed=True)["date_dt"]
        .agg(["size", "min", "max"])
        .reset_index()
        .rename(columns={"size": "complaints", "min": "first_date", "max": "last_date"})
    )
    repeat_groups["days_span"] = (repeat_groups["last_date"] - repeat_groups["first_date"]).dt.days.fillna(0)
    repeat_groups["repeat_within_90_days"] = (repeat_groups["complaints"] > 1) & (repeat_groups["days_span"] <= 90)
    return repeat_groups.sort_values(["repeat_within_90_days", "complaints"], ascending=[False, False]).reset_index(drop=True)


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = (
        df[df["month"] != ""]
        .groupby("month", observed=False)
        .agg(
            complaints=("narrative_excerpt", "size"),
            templated_share=("is_templated", "mean"),
            template_cluster_share=("template_cluster", "mean"),
            legalistic_template_share=("legalistic_template", "mean"),
            score_based_template_share=("score_based_template", "mean"),
            likely_credit_repair_share=("likely_credit_repair_firm_drafted", "mean"),
            possible_ai_template_share=("possible_ai_consumer_template", "mean"),
            credit_repair_exposure_share=("credit_repair_exposure", "mean"),
            positive_outcome_rate=("positive_outcome", "mean"),
            avg_template_score=("template_score", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    panel["year"] = panel["month"].str.slice(0, 4).astype(int)
    panel["templated_share_rolling_3m"] = panel["templated_share"].rolling(3, min_periods=1).mean()
    return panel


def build_monthly_spike_decomposition(df: pd.DataFrame, top_n_hashes: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    month_df = df[df["month"] != ""].copy()
    for month, month_frame in month_df.groupby("month", observed=False):
        templated = month_frame.loc[month_frame["is_templated"]].copy()
        hash_counts = templated["normalized_hash"].value_counts()
        top_hashes = hash_counts.head(top_n_hashes)
        top_issue = templated["issue"].mode().iloc[0] if not templated.empty else ""
        top_company = templated["company"].mode().iloc[0] if not templated.empty else ""
        rows.append(
            {
                "month": month,
                "complaints": int(len(month_frame)),
                "templated_complaints": int(len(templated)),
                "templated_share": float(month_frame["is_templated"].mean()),
                "template_cluster_share": float(month_frame["template_cluster"].mean()),
                "legalistic_template_share": float(month_frame["legalistic_template"].mean()),
                "score_based_template_share": float(month_frame["score_based_template"].mean()),
                "likely_credit_repair_share": float(month_frame["likely_credit_repair_firm_drafted"].mean()),
                "possible_ai_template_share": float(month_frame["possible_ai_consumer_template"].mean()),
                "duplicate_share": float((month_frame["exact_template_group_size"] > 1).mean()),
                "max_template_group_size": int(month_frame["exact_template_group_size"].max()),
                "top_hash_1": top_hashes.index[0] if len(top_hashes) > 0 else "",
                "top_hash_1_count": int(top_hashes.iloc[0]) if len(top_hashes) > 0 else 0,
                "top_hash_2": top_hashes.index[1] if len(top_hashes) > 1 else "",
                "top_hash_2_count": int(top_hashes.iloc[1]) if len(top_hashes) > 1 else 0,
                "top_hash_3": top_hashes.index[2] if len(top_hashes) > 2 else "",
                "top_hash_3_count": int(top_hashes.iloc[2]) if len(top_hashes) > 2 else 0,
                "top_templated_issue": top_issue,
                "top_templated_company": top_company,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    kl_pm = np.sum(p[p_mask] * np.log2(p[p_mask] / m[p_mask]))
    kl_qm = np.sum(q[q_mask] * np.log2(q[q_mask] / m[q_mask]))
    return float(0.5 * (kl_pm + kl_qm))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))


def build_unigram_distribution(texts: pd.Series, vocabulary: list[str]) -> np.ndarray:
    counts = Counter()
    for text in texts.fillna(""):
        counts.update(tokenize(str(text)))
    vector = np.array([counts.get(token, 0) for token in vocabulary], dtype=float)
    if vector.sum() == 0:
        return np.ones(len(vocabulary), dtype=float)
    return vector


def feature_histogram(frame: pd.DataFrame, column: str, bins: np.ndarray) -> np.ndarray:
    values = frame[column].fillna(0).to_numpy(dtype=float)
    hist, _ = np.histogram(values, bins=bins)
    return hist.astype(float) + 1e-9


def build_drift_reference(df: pd.DataFrame) -> tuple[list[str], list[tuple[str, np.ndarray]], np.ndarray, np.ndarray]:
    reference_tokens = Counter()
    for text in df["narrative_excerpt"].fillna(""):
        reference_tokens.update(tokenize(str(text)))
    vocabulary = [token for token, _ in reference_tokens.most_common(JSD_TOP_VOCAB_SIZE)]
    if not vocabulary:
        vocabulary = ["credit"]
    reference_unigram = build_unigram_distribution(df["narrative_excerpt"], vocabulary)
    style_columns = [
        ("token_count", np.linspace(0, max(df["token_count"].max(), 1), 13)),
        ("type_token_ratio", np.linspace(0, 1, 13)),
        ("legal_hits", np.linspace(0, max(df["legal_hits"].max(), 1), 13)),
        ("template_hits", np.linspace(0, max(df["template_hits"].max(), 1), 13)),
        ("first_person_share", np.linspace(0, max(df["first_person_share"].max(), 0.2), 13)),
    ]
    reference_style = np.concatenate([feature_histogram(df, column, bins) for column, bins in style_columns])
    return vocabulary, style_columns, reference_unigram, reference_style


def build_text_drift_panel(df: pd.DataFrame) -> pd.DataFrame:
    clean_reference = df.loc[
        (~df["credit_repair_exposure"])
        & (~df["template_cluster"])
        & (~df["is_templated"])
        & (df["token_count"] > 0)
    ].copy()
    if clean_reference.empty:
        clean_reference = df.loc[df["token_count"] > 0].copy()

    vocabulary, style_columns, reference_unigram, reference_style = build_drift_reference(clean_reference)

    rows: list[dict[str, object]] = []
    for month, month_frame in df[df["month"] != ""].groupby("month", observed=False):
        if len(month_frame) < MIN_DRIFT_MONTHLY_COMPLAINTS:
            continue
        exposure_frame = month_frame.loc[month_frame["credit_repair_exposure"]]
        full_unigram = build_unigram_distribution(month_frame["narrative_excerpt"], vocabulary)
        full_style = np.concatenate([feature_histogram(month_frame, column, bins) for column, bins in style_columns])
        if exposure_frame.empty:
            exposure_unigram = np.ones(len(vocabulary), dtype=float)
            exposure_style = np.ones_like(reference_style)
        else:
            exposure_unigram = build_unigram_distribution(exposure_frame["narrative_excerpt"], vocabulary)
            exposure_style = np.concatenate([feature_histogram(exposure_frame, column, bins) for column, bins in style_columns])
        rows.append(
            {
                "month": month,
                "complaints": int(len(month_frame)),
                "credit_repair_exposure_complaints": int(len(exposure_frame)),
                "jsd_unigram_full_vs_clean": jensen_shannon_divergence(full_unigram, reference_unigram),
                "jsd_style_full_vs_clean": jensen_shannon_divergence(full_style, reference_style),
                "hellinger_unigram_full_vs_clean": hellinger_distance(full_unigram, reference_unigram),
                "hellinger_style_full_vs_clean": hellinger_distance(full_style, reference_style),
                "jsd_unigram_credit_repair_vs_clean": jensen_shannon_divergence(exposure_unigram, reference_unigram) if not exposure_frame.empty else np.nan,
                "jsd_style_credit_repair_vs_clean": jensen_shannon_divergence(exposure_style, reference_style) if not exposure_frame.empty else np.nan,
                "hellinger_unigram_credit_repair_vs_clean": hellinger_distance(exposure_unigram, reference_unigram) if not exposure_frame.empty else np.nan,
                "hellinger_style_credit_repair_vs_clean": hellinger_distance(exposure_style, reference_style) if not exposure_frame.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_templated_drift_monthly(df: pd.DataFrame) -> pd.DataFrame:
    non_templated_reference = df.loc[(~df["is_templated"]) & (df["token_count"] > 0)].copy()
    if non_templated_reference.empty:
        non_templated_reference = df.loc[df["token_count"] > 0].copy()
    vocabulary, style_columns, reference_unigram, reference_style = build_drift_reference(non_templated_reference)

    rows: list[dict[str, object]] = []
    for month, month_frame in df[df["month"] != ""].groupby("month", observed=False):
        if len(month_frame) < MIN_DRIFT_MONTHLY_COMPLAINTS:
            continue
        templated_frame = month_frame.loc[month_frame["is_templated"]]
        full_unigram = build_unigram_distribution(month_frame["narrative_excerpt"], vocabulary)
        full_style = np.concatenate([feature_histogram(month_frame, column, bins) for column, bins in style_columns])
        if templated_frame.empty:
            templated_unigram = np.ones(len(vocabulary), dtype=float)
            templated_style = np.ones_like(reference_style)
        else:
            templated_unigram = build_unigram_distribution(templated_frame["narrative_excerpt"], vocabulary)
            templated_style = np.concatenate([feature_histogram(templated_frame, column, bins) for column, bins in style_columns])
        rows.append(
            {
                "month": month,
                "complaints": int(len(month_frame)),
                "templated_complaints": int(len(templated_frame)),
                "templated_share": float(month_frame["is_templated"].mean()),
                "jsd_unigram_full_vs_non_templated": jensen_shannon_divergence(full_unigram, reference_unigram),
                "jsd_style_full_vs_non_templated": jensen_shannon_divergence(full_style, reference_style),
                "hellinger_unigram_full_vs_non_templated": hellinger_distance(full_unigram, reference_unigram),
                "hellinger_style_full_vs_non_templated": hellinger_distance(full_style, reference_style),
                "jsd_unigram_templated_vs_non_templated": jensen_shannon_divergence(templated_unigram, reference_unigram) if not templated_frame.empty else np.nan,
                "jsd_style_templated_vs_non_templated": jensen_shannon_divergence(templated_style, reference_style) if not templated_frame.empty else np.nan,
                "hellinger_unigram_templated_vs_non_templated": hellinger_distance(templated_unigram, reference_unigram) if not templated_frame.empty else np.nan,
                "hellinger_style_templated_vs_non_templated": hellinger_distance(templated_style, reference_style) if not templated_frame.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_templated_drift_period_panel(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["period_block"] = pd.cut(
        frame["year_received"],
        bins=[2010, 2014, 2019, 2021, 2023, 2026],
        labels=["2011-2014", "2015-2019", "2020-2021", "2022-2023", "2024-2026"],
        include_lowest=True,
    ).astype(str)
    non_templated_reference = frame.loc[(~frame["is_templated"]) & (frame["token_count"] > 0)].copy()
    if non_templated_reference.empty:
        non_templated_reference = frame.loc[frame["token_count"] > 0].copy()
    vocabulary, style_columns, reference_unigram, reference_style = build_drift_reference(non_templated_reference)

    rows: list[dict[str, object]] = []
    for period_block, period_frame in frame.groupby("period_block", observed=False):
        if period_block == "nan" or period_frame.empty:
            continue
        templated_frame = period_frame.loc[period_frame["is_templated"]]
        full_unigram = build_unigram_distribution(period_frame["narrative_excerpt"], vocabulary)
        full_style = np.concatenate([feature_histogram(period_frame, column, bins) for column, bins in style_columns])
        templated_unigram = build_unigram_distribution(templated_frame["narrative_excerpt"], vocabulary) if not templated_frame.empty else np.ones(len(vocabulary), dtype=float)
        templated_style = np.concatenate([feature_histogram(templated_frame, column, bins) for column, bins in style_columns]) if not templated_frame.empty else np.ones_like(reference_style)
        rows.append(
            {
                "period_block": period_block,
                "complaints": int(len(period_frame)),
                "templated_share": float(period_frame["is_templated"].mean()),
                "jsd_unigram_full_vs_non_templated": jensen_shannon_divergence(full_unigram, reference_unigram),
                "jsd_style_full_vs_non_templated": jensen_shannon_divergence(full_style, reference_style),
                "jsd_unigram_templated_vs_non_templated": jensen_shannon_divergence(templated_unigram, reference_unigram),
                "jsd_style_templated_vs_non_templated": jensen_shannon_divergence(templated_style, reference_style),
                "hellinger_unigram_full_vs_non_templated": hellinger_distance(full_unigram, reference_unigram),
                "hellinger_style_full_vs_non_templated": hellinger_distance(full_style, reference_style),
                "hellinger_unigram_templated_vs_non_templated": hellinger_distance(templated_unigram, reference_unigram),
                "hellinger_style_templated_vs_non_templated": hellinger_distance(templated_style, reference_style),
            }
        )
    order = ["2011-2014", "2015-2019", "2020-2021", "2022-2023", "2024-2026"]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["period_block"] = pd.Categorical(out["period_block"], categories=order, ordered=True)
    return out.sort_values("period_block").reset_index(drop=True)


def build_template_weight_sensitivity(df: pd.DataFrame, monthly_panel: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    varied_weights = draw_weight_vectors(TEMPLATE_SCORE_WEIGHTS, MONTE_CARLO_DRAWS, random_state)
    month_lookup = {month: [] for month in monthly_panel["month"]}
    base_flag = df["score_based_template"].astype(int).to_numpy()
    agreement_rates = []

    for weight_set in varied_weights:
        varied_score = build_template_score(df, weight_set)
        pre_ai_scores = varied_score.loc[df["pre_ai_period"]]
        if pre_ai_scores.empty:
            pre_ai_scores = varied_score
        threshold = float(pre_ai_scores.quantile(TEMPLATE_SCORE_PRE_AI_QUANTILE))
        varied_flag = (varied_score >= threshold).astype(int)
        agreement_rates.append(float((varied_flag == base_flag).mean()))
        month_share = (
            pd.DataFrame({"month": df["month"], "flag": varied_flag})
            .loc[lambda frame: frame["month"] != ""]
            .groupby("month", observed=False)["flag"]
            .mean()
        )
        for month, share in month_share.items():
            if month in month_lookup:
                month_lookup[month].append(float(share))

    monthly_rows = []
    for month in monthly_panel["month"]:
        draws = month_lookup.get(month, [])
        base_share = monthly_panel.loc[monthly_panel["month"] == month, "score_based_template_share"].iloc[0]
        monthly_rows.append(
            {
                "month": month,
                "base_score_based_template_share": float(base_share),
                "monte_carlo_mean_share": float(np.mean(draws)) if draws else np.nan,
                "monte_carlo_std_share": float(np.std(draws)) if draws else np.nan,
                "monte_carlo_p10_share": float(np.quantile(draws, 0.10)) if draws else np.nan,
                "monte_carlo_p90_share": float(np.quantile(draws, 0.90)) if draws else np.nan,
            }
        )

    summary = pd.DataFrame(
        [
            {
                "draws": MONTE_CARLO_DRAWS,
                "weight_sigma": MONTE_CARLO_WEIGHT_SIGMA,
                "mean_agreement_with_base_score_flag": float(np.mean(agreement_rates)) if agreement_rates else np.nan,
                "min_agreement_with_base_score_flag": float(np.min(agreement_rates)) if agreement_rates else np.nan,
                "max_agreement_with_base_score_flag": float(np.max(agreement_rates)) if agreement_rates else np.nan,
                "pre_ai_threshold_quantile": TEMPLATE_SCORE_PRE_AI_QUANTILE,
            }
        ]
    )
    return summary, pd.DataFrame(monthly_rows)


def package_report_outputs(output_dir: Path) -> Path:
    package_dir = output_dir / "report_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    for artifact in output_dir.iterdir():
        if artifact.name == package_dir.name:
            continue
        if artifact.is_file() and artifact.suffix.lower() in {".csv", ".json", ".png", ".md"}:
            shutil.copy2(artifact, package_dir / artifact.name)
    return package_dir


def build_period_effects(df: pd.DataFrame) -> pd.DataFrame:
    period_summary = (
        df.groupby(["period_bin", "is_templated"], observed=False)
        .agg(
            complaints=("narrative_excerpt", "size"),
            positive_outcome_rate=("positive_outcome", "mean"),
        )
        .reset_index()
    )
    pivoted = period_summary.pivot(index="period_bin", columns="is_templated", values="positive_outcome_rate").reset_index()
    pivoted.columns = ["period_bin", "non_templated_rate", "templated_rate"]
    pivoted["templated_advantage_pp"] = 100 * (pivoted["templated_rate"] - pivoted["non_templated_rate"])
    return pivoted


def build_issue_shift(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["is_templated", "issue"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("is_templated")["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    pivoted = counts.pivot(index="issue", columns="is_templated", values="share").fillna(0)
    pivoted.columns = ["non_templated_share", "templated_share"]
    pivoted["share_gap"] = pivoted["templated_share"] - pivoted["non_templated_share"]
    return pivoted.sort_values("share_gap", ascending=False).reset_index().head(15)


def build_examples(df: pd.DataFrame, n_examples: int) -> pd.DataFrame:
    columns = [
        "date_received",
        "company",
        "product",
        "issue",
        "template_score",
        "template_origin",
        "exact_template_group_size",
        "legal_hits",
        "template_hits",
        "credit_repair_hits",
        "known_firm_hits",
        "third_party_hits",
        "narrative_excerpt",
    ]
    return df.sort_values("template_score", ascending=False).head(n_examples)[columns].reset_index(drop=True)


def fit_binary_outcome_model(
    df: pd.DataFrame,
    target_column: str,
    main_flag: str,
    interaction_prefix: str,
    model_sample_size: int,
    random_state: int,
    extra_flag_columns: list[str] | None = None,
) -> pd.DataFrame:
    model_df = df.copy()
    if len(model_df) > model_sample_size:
        keep_target = model_df.loc[model_df[main_flag]].copy()
        remainder = model_df.loc[~model_df[main_flag]].sample(
            n=min(len(model_df.loc[~model_df[main_flag]]), model_sample_size),
            random_state=random_state,
        )
        model_df = pd.concat([keep_target, remainder], ignore_index=True).drop_duplicates()

    company_counts = model_df["company"].value_counts()
    issue_counts = model_df["issue"].value_counts()
    model_df["company_bucket"] = np.where(
        model_df["company"].isin(company_counts.head(30).index),
        model_df["company"],
        "Other companies",
    )
    model_df["issue_bucket"] = np.where(
        model_df["issue"].isin(issue_counts.head(20).index),
        model_df["issue"],
        "Other issues",
    )
    model_df["token_count_sq"] = model_df["token_count"] ** 2
    period_dummies = pd.get_dummies(model_df["period_bin"], prefix="period", drop_first=True, dtype=float)
    for column in period_dummies.columns:
        model_df[f"{interaction_prefix}_{column}"] = model_df[main_flag].astype(float) * period_dummies[column]

    extra_flag_columns = extra_flag_columns or []
    model_columns = [
        main_flag,
        "token_count",
        "token_count_sq",
        "type_token_ratio",
        "legal_hits",
        "template_hits",
        "product",
        "month",
        "company_bucket",
        "issue_bucket",
        "period_bin",
    ] + extra_flag_columns + [col for col in model_df.columns if col.startswith(f"{interaction_prefix}_period_")]

    design = pd.get_dummies(
        model_df[model_columns],
        columns=["product", "month", "company_bucket", "issue_bucket", "period_bin"],
        drop_first=True,
        dtype=float,
    )
    for col in [main_flag, *extra_flag_columns, "token_count", "token_count_sq", "type_token_ratio", "legal_hits", "template_hits"]:
        design[col] = design[col].astype(float)
    design = sm.add_constant(design, has_constant="add")
    design = design.astype(float)
    try:
        model = sm.Logit(model_df[target_column].astype(int), design)
        fitted = model.fit(disp=False, maxiter=200, cov_type="HC1")
        result = pd.DataFrame(
            {
                "term": fitted.params.index,
                "coef": fitted.params.values,
                "odds_ratio": np.exp(fitted.params.values),
                "p_value": fitted.pvalues.values,
            }
        )
    except Exception:
        fallback_design = design.drop(columns=["const"], errors="ignore")
        fallback = LogisticRegression(max_iter=3000, solver="liblinear", random_state=random_state)
        fallback.fit(fallback_design, model_df[target_column].astype(int))
        coeffs = np.r_[fallback.intercept_, fallback.coef_.ravel()]
        terms = ["const", *fallback_design.columns.tolist()]
        result = pd.DataFrame(
            {
                "term": terms,
                "coef": coeffs[: len(terms)],
                "odds_ratio": np.exp(coeffs[: len(terms)]),
                "p_value": np.nan,
            }
        )
    keep_terms = [main_flag, *extra_flag_columns, "token_count", "token_count_sq", "type_token_ratio", "legal_hits", "template_hits"]
    keep_terms.extend([term for term in result["term"] if term.startswith(f"{interaction_prefix}_period_")])
    return result[result["term"].isin(keep_terms)].reset_index(drop=True)


def fit_outcome_model(df: pd.DataFrame, model_sample_size: int, random_state: int, target_column: str = "positive_outcome") -> pd.DataFrame:
    return fit_binary_outcome_model(
        df=df,
        target_column=target_column,
        main_flag="is_templated",
        interaction_prefix="templated_x",
        model_sample_size=model_sample_size,
        random_state=random_state,
    )


def fit_credit_repair_outcome_model(df: pd.DataFrame, model_sample_size: int, random_state: int) -> pd.DataFrame:
    return fit_binary_outcome_model(
        df=df,
        target_column="positive_outcome",
        main_flag="likely_credit_repair_firm_drafted",
        interaction_prefix="credit_repair_x",
        model_sample_size=model_sample_size,
        random_state=random_state,
        extra_flag_columns=["complaint_about_credit_repair_firm"],
    )


def save_figures(
    summary: pd.DataFrame,
    origin_summary: pd.DataFrame,
    monthly_panel: pd.DataFrame,
    issue_shift: pd.DataFrame,
    period_effects: pd.DataFrame,
    spike_decomposition: pd.DataFrame,
    text_drift_panel: pd.DataFrame,
    templated_drift_monthly: pd.DataFrame,
    templated_drift_period: pd.DataFrame,
    weight_sensitivity_monthly: pd.DataFrame,
    cross_product_benchmark_panel: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["group"], summary["duplicate_rate"], color=["#6c7a89", "#c44e52"])
    ax.set_title("Duplicate Rate by Complaint Type")
    ax.set_ylabel("Share of complaints with repeated text")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_duplicate_rate.png", dpi=200)
    plt.close(fig)

    chart_panel = monthly_panel[monthly_panel["year"] < 2026].copy()
    year_ticks = chart_panel.groupby("year", observed=False).head(1).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(chart_panel.index, chart_panel["templated_share"], color="#d98f5c", linewidth=1.5, alpha=0.35, label="Monthly share")
    ax.plot(chart_panel.index, chart_panel["templated_share_rolling_3m"], color="#c44e52", linewidth=2.5, label="3-month average")
    ax.set_title("Templated / AI-style Share of Credit Complaints Over Time")
    ax.set_ylabel("Share of complaints")
    ax.set_xlabel("Year")
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(chart_panel.loc[year_ticks, "year"].astype(str))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_templated_share_over_time.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    spike_chart = spike_decomposition[spike_decomposition["month"].str.slice(0, 4).astype(int) < 2026].copy()
    ax.plot(spike_chart.index, spike_chart["template_cluster_share"], label="Exact-repeat cluster share", color="#4c72b0", linewidth=2)
    ax.plot(spike_chart.index, spike_chart["legalistic_template_share"], label="Legalistic template share", color="#dd8452", linewidth=2)
    ax.plot(spike_chart.index, spike_chart["possible_ai_template_share"], label="Possible consumer AI template share", color="#55a868", linewidth=2)
    ax.set_title("What Drives Monthly Templated Spikes")
    ax.set_ylabel("Share of complaints")
    ax.set_xlabel("Year")
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(chart_panel.loc[year_ticks, "year"].astype(str))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_spike_decomposition.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(origin_summary["template_origin"], origin_summary["share_of_dataset"], color=["#c44e52", "#dd8452", "#8172b2", "#6c7a89"][: len(origin_summary)])
    ax.set_title("Attribution of Complaint Styles")
    ax.set_ylabel("Share of analyzed complaints")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_template_origin_mix.png", dpi=200)
    plt.close(fig)

    top_issue_shift = issue_shift.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_issue_shift["issue"], top_issue_shift["share_gap"], color="#4c72b0")
    ax.set_title("Issues Most Overrepresented in Templated Complaints")
    ax.set_xlabel("Templated share minus non-templated share")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_issue_shift.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(period_effects["period_bin"], period_effects["templated_advantage_pp"], color="#4c72b0", marker="o", linewidth=2)
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_title("Templated Complaint Advantage in Positive Outcomes")
    ax.set_ylabel("Percentage-point advantage over non-templated")
    ax.set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_templated_advantage_by_period.png", dpi=200)
    plt.close(fig)

    if not text_drift_panel.empty:
        drift_chart = text_drift_panel[text_drift_panel["month"].str.slice(0, 4).astype(int) < 2026].copy()
        drift_ticks = drift_chart.groupby(drift_chart["month"].str.slice(0, 4), observed=False).head(1).index.tolist()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(drift_chart.index, drift_chart["jsd_unigram_full_vs_clean"], label="Full corpus unigram JSD", color="#4c72b0", linewidth=2)
        ax.plot(drift_chart.index, drift_chart["jsd_style_full_vs_clean"], label="Full corpus style JSD", color="#c44e52", linewidth=2)
        ax.plot(
            drift_chart.index,
            drift_chart["jsd_unigram_credit_repair_vs_clean"],
            label="Credit-repair exposure unigram JSD",
            color="#55a868",
            linewidth=2,
        )
        ax.set_title("Textual Drift Relative to Clean Complaint Baseline")
        ax.set_ylabel("Jensen-Shannon divergence")
        ax.set_xlabel("Year")
        ax.set_xticks(drift_ticks)
        ax.set_xticklabels(drift_chart.loc[drift_ticks, "month"].str.slice(0, 4))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_text_drift_jsd_over_time.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(drift_chart.index, drift_chart["hellinger_unigram_full_vs_clean"], label="Full corpus unigram Hellinger", color="#4c72b0", linewidth=2)
        ax.plot(drift_chart.index, drift_chart["hellinger_style_full_vs_clean"], label="Full corpus style Hellinger", color="#c44e52", linewidth=2)
        ax.plot(
            drift_chart.index,
            drift_chart["hellinger_unigram_credit_repair_vs_clean"],
            label="Credit-repair exposure unigram Hellinger",
            color="#55a868",
            linewidth=2,
        )
        ax.set_title("Textual Drift Relative to Clean Baseline: Hellinger Distance")
        ax.set_ylabel("Hellinger distance")
        ax.set_xlabel("Year")
        ax.set_xticks(drift_ticks)
        ax.set_xticklabels(drift_chart.loc[drift_ticks, "month"].str.slice(0, 4))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_text_drift_hellinger_over_time.png", dpi=200)
        plt.close(fig)

    if not templated_drift_monthly.empty:
        drift_chart = templated_drift_monthly[templated_drift_monthly["month"].str.slice(0, 4).astype(int) < 2026].copy()
        drift_ticks = drift_chart.groupby(drift_chart["month"].str.slice(0, 4), observed=False).head(1).index.tolist()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(drift_chart.index, drift_chart["jsd_unigram_full_vs_non_templated"], label="Full corpus vs non-templated baseline", color="#4c72b0", linewidth=2)
        ax.plot(drift_chart.index, drift_chart["jsd_unigram_templated_vs_non_templated"], label="Templated subset vs non-templated baseline", color="#c44e52", linewidth=2)
        ax.set_title("How Far the Dataset Shifts Toward Templated Language")
        ax.set_ylabel("Jensen-Shannon divergence")
        ax.set_xlabel("Year")
        ax.set_xticks(drift_ticks)
        ax.set_xticklabels(drift_chart.loc[drift_ticks, "month"].str.slice(0, 4))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_templated_vs_non_templated_drift_monthly.png", dpi=200)
        plt.close(fig)

    if not templated_drift_period.empty:
        period_chart = templated_drift_period.copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(period_chart))
        width = 0.38
        ax.bar(x - width / 2, period_chart["jsd_unigram_full_vs_non_templated"], width=width, label="Full corpus vs non-templated", color="#4c72b0")
        ax.bar(x + width / 2, period_chart["jsd_unigram_templated_vs_non_templated"], width=width, label="Templated vs non-templated", color="#c44e52")
        ax.set_title("Binned Drift Toward Templated Language")
        ax.set_ylabel("Jensen-Shannon divergence")
        ax.set_xlabel("Period")
        ax.set_xticks(x)
        ax.set_xticklabels(period_chart["period_block"].astype(str))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_templated_vs_non_templated_drift_periods.png", dpi=200)
        plt.close(fig)

    if not weight_sensitivity_monthly.empty:
        sensitivity_chart = weight_sensitivity_monthly[weight_sensitivity_monthly["month"].str.slice(0, 4).astype(int) < 2026].copy()
        sens_ticks = sensitivity_chart.groupby(sensitivity_chart["month"].str.slice(0, 4), observed=False).head(1).index.tolist()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(sensitivity_chart.index, sensitivity_chart["base_score_based_template_share"], color="#4c72b0", linewidth=2, label="Base share")
        ax.fill_between(
            sensitivity_chart.index,
            sensitivity_chart["monte_carlo_p10_share"],
            sensitivity_chart["monte_carlo_p90_share"],
            color="#4c72b0",
            alpha=0.2,
            label="Monte Carlo 10th-90th percentile",
        )
        ax.set_title("Sensitivity of Score-Based Template Share to Weight Perturbation")
        ax.set_ylabel("Share of complaints")
        ax.set_xlabel("Year")
        ax.set_xticks(sens_ticks)
        ax.set_xticklabels(sensitivity_chart.loc[sens_ticks, "month"].str.slice(0, 4))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_template_weight_sensitivity.png", dpi=200)
        plt.close(fig)

    if not cross_product_benchmark_panel.empty:
        benchmark_chart = cross_product_benchmark_panel[cross_product_benchmark_panel["year"] < 2026].copy()
        month_order = sorted(benchmark_chart["month"].unique().tolist())
        month_to_x = {month: idx for idx, month in enumerate(month_order)}
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, frame in benchmark_chart.groupby("benchmark_product_bucket", observed=False):
            if label == "Other" or frame.empty:
                continue
            x = frame["month"].map(month_to_x)
            ax.plot(x, frame["templated_share_rolling_3m"], linewidth=2, label=label)
        ax.set_title("Templated Complaint Growth Across CFPB Product Areas")
        ax.set_ylabel("3-month rolling templated share")
        ax.set_xlabel("Month")
        chosen = [month for month in month_order if month.endswith("-01")]
        if chosen:
            ax.set_xticks([month_to_x[month] for month in chosen])
            ax.set_xticklabels([month[:4] for month in chosen], rotation=0)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_cross_product_templated_growth.png", dpi=200)
        plt.close(fig)


def write_outputs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    outcome_model: pd.DataFrame,
    monetary_outcome_model: pd.DataFrame,
    non_monetary_outcome_model: pd.DataFrame,
    flat_closure_model: pd.DataFrame,
    deduplicated_outcome_model: pd.DataFrame,
    deduplicated_monetary_outcome_model: pd.DataFrame,
    credit_repair_outcome_model: pd.DataFrame,
    credit_repair_monetary_outcome_model: pd.DataFrame,
    monthly_panel: pd.DataFrame,
    issue_shift: pd.DataFrame,
    dataset_comparison_summary: pd.DataFrame,
    issue_counterfactual: pd.DataFrame,
    company_concentration_summary: pd.DataFrame,
    repeat_submission_summary: pd.DataFrame,
    origin_summary: pd.DataFrame,
    period_effects: pd.DataFrame,
    credit_repair_complaints: pd.DataFrame,
    credit_repair_firm_drafted: pd.DataFrame,
    credit_repair_theme_summary: pd.DataFrame,
    credit_repair_phrase_table: pd.DataFrame,
    credit_repair_accessibility_summary: pd.DataFrame,
    tag_similarity_summary: pd.DataFrame,
    examples: pd.DataFrame,
    spike_decomposition: pd.DataFrame,
    text_drift_panel: pd.DataFrame,
    weight_sensitivity_summary: pd.DataFrame,
    weight_sensitivity_monthly: pd.DataFrame,
    cross_product_benchmark_panel: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "template_quality_summary.csv", index=False)
    if not outcome_model.empty:
        outcome_model.to_csv(output_dir / "template_outcome_model.csv", index=False)
    if not monetary_outcome_model.empty:
        monetary_outcome_model.to_csv(output_dir / "template_outcome_model_monetary.csv", index=False)
    if not non_monetary_outcome_model.empty:
        non_monetary_outcome_model.to_csv(output_dir / "template_outcome_model_non_monetary.csv", index=False)
    if not flat_closure_model.empty:
        flat_closure_model.to_csv(output_dir / "template_outcome_model_flat_closure.csv", index=False)
    if not deduplicated_outcome_model.empty:
        deduplicated_outcome_model.to_csv(output_dir / "template_outcome_model_deduplicated.csv", index=False)
    if not deduplicated_monetary_outcome_model.empty:
        deduplicated_monetary_outcome_model.to_csv(output_dir / "template_outcome_model_monetary_deduplicated.csv", index=False)
    if not credit_repair_outcome_model.empty:
        credit_repair_outcome_model.to_csv(output_dir / "credit_repair_outcome_model.csv", index=False)
    if not credit_repair_monetary_outcome_model.empty:
        credit_repair_monetary_outcome_model.to_csv(output_dir / "credit_repair_outcome_model_monetary.csv", index=False)
    monthly_panel.to_csv(output_dir / "template_monthly_panel.csv", index=False)
    issue_shift.to_csv(output_dir / "template_issue_shift.csv", index=False)
    dataset_comparison_summary.to_csv(output_dir / "dataset_comparison_summary.csv", index=False)
    issue_counterfactual.to_csv(output_dir / "issue_counterfactual_summary.csv", index=False)
    company_concentration_summary.to_csv(output_dir / "company_concentration_summary.csv", index=False)
    repeat_submission_summary.to_csv(output_dir / "repeat_submission_summary.csv", index=False)
    origin_summary.to_csv(output_dir / "template_origin_summary.csv", index=False)
    period_effects.to_csv(output_dir / "template_period_effects.csv", index=False)
    credit_repair_complaints.to_csv(output_dir / "complaints_about_credit_repair_firms.csv", index=False)
    credit_repair_firm_drafted.to_csv(output_dir / "credit_repair_firm_drafted.csv", index=False)
    credit_repair_theme_summary.to_csv(output_dir / "credit_repair_bad_practice_themes.csv", index=False)
    credit_repair_phrase_table.to_csv(output_dir / "credit_repair_bad_practice_terms.csv", index=False)
    credit_repair_accessibility_summary.to_csv(output_dir / "credit_repair_accessibility_summary.csv", index=False)
    tag_similarity_summary.to_csv(output_dir / "tag_similarity_summary.csv", index=False)
    examples.to_csv(output_dir / "template_examples.csv", index=False)
    spike_decomposition.to_csv(output_dir / "template_spike_decomposition.csv", index=False)
    text_drift_panel.to_csv(output_dir / "text_drift_jsd_panel.csv", index=False)
    weight_sensitivity_summary.to_csv(output_dir / "template_weight_sensitivity_summary.csv", index=False)
    weight_sensitivity_monthly.to_csv(output_dir / "template_weight_sensitivity_monthly.csv", index=False)
    cross_product_benchmark_panel.to_csv(output_dir / "cross_product_benchmark_panel.csv", index=False)

    overview = {
        "complaints_analysed": int(len(df)),
        "templated_share": float(df["is_templated"].mean()),
        "template_cluster_share": float(df["template_cluster"].mean()),
        "legalistic_template_share": float(df["legalistic_template"].mean()),
        "score_based_template_share": float(df["score_based_template"].mean()),
        "pre_ai_template_cluster_share": float(df["pre_ai_template_cluster"].mean()),
        "pre_ai_credit_repair_style_template_share": float(df["pre_ai_credit_repair_style_template"].mean()),
        "complaint_about_credit_repair_firm_share": float(df["complaint_about_credit_repair_firm"].mean()),
        "credit_repair_firm_drafted_share": float(df["likely_credit_repair_firm_drafted"].mean()),
        "possible_ai_consumer_template_share": float(df["possible_ai_consumer_template"].mean()),
        "consumer_side_automation_share": float(df["possible_consumer_side_automation"].mean()),
        "credit_repair_exposure_share": float(df["credit_repair_exposure"].mean()),
        "direct_credit_repair_mention_share": float((df["credit_repair_hits"] > 0).mean()),
        "known_firm_mention_share": float((df["known_firm_hits"] > 0).mean()),
        "third_party_representation_share": float((df["third_party_hits"] > 0).mean()),
        "repeat_submission_share_90d": float(repeat_submission_summary["repeat_within_90_days"].mean()) if not repeat_submission_summary.empty else 0.0,
        "pre_ai_threshold_quantile": TEMPLATE_SCORE_PRE_AI_QUANTILE,
        "pre_ai_score_threshold": float(df["template_score_threshold_pre_ai"].iloc[0]) if not df.empty else np.nan,
        "latest_month": monthly_panel.iloc[-1].to_dict() if not monthly_panel.empty else {},
        "peak_month": spike_decomposition.sort_values("templated_share", ascending=False).head(1).to_dict(orient="records"),
        "weight_sensitivity_summary": weight_sensitivity_summary.to_dict(orient="records"),
        "model_terms": outcome_model.to_dict(orient="records") if not outcome_model.empty else [],
        "monetary_model_terms": monetary_outcome_model.to_dict(orient="records") if not monetary_outcome_model.empty else [],
        "credit_repair_model_terms": credit_repair_outcome_model.to_dict(orient="records") if not credit_repair_outcome_model.empty else [],
        "credit_repair_accessibility": credit_repair_accessibility_summary.to_dict(orient="records"),
        "tag_similarity_summary": tag_similarity_summary.to_dict(orient="records"),
        "cross_product_benchmark_available": not cross_product_benchmark_panel.empty,
    }
    (output_dir / "template_findings.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    save_figures(
        summary,
        origin_summary,
        monthly_panel,
        issue_shift,
        period_effects,
        spike_decomposition,
        text_drift_panel,
        weight_sensitivity_monthly,
        cross_product_benchmark_panel,
        output_dir,
    )
    package_dir = package_report_outputs(output_dir)
    print(f"[{timestamp()}] Packaged report artifacts: {package_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse templated or AI-style CFPB credit complaints.")
    parser.add_argument("--csv-path", default="complaints.csv", help="Path to the full CFPB CSV.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size when reading the CSV.")
    parser.add_argument(
        "--model-sample-size",
        type=int,
        default=200_000,
        help="Maximum rows used in the regression model after the full-dataset scoring pass.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory for tables and figures.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached scored data and rebuild from the raw complaints CSV.",
    )
    parser.add_argument(
        "--skip-outcome-models",
        action="store_true",
        help="Skip outcome regressions and focus on text, template, and drift outputs only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now()
    clock_start = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_cache_path = output_dir / "scored_credit_complaints_credit_repair_focus.parquet"
    benchmark_cache_path = output_dir / "scored_product_benchmark.parquet"
    print(f"[{timestamp()}] Analysis started")
    print(f"[{timestamp()}] Source CSV: {Path(args.csv_path).resolve()}")
    if scored_cache_path.exists() and not args.refresh_cache:
        print(f"[{timestamp()}] Loading cached scored data: {scored_cache_path}")
        scored = pd.read_parquet(scored_cache_path)
    else:
        feature_store = build_full_feature_store(Path(args.csv_path), chunksize=args.chunksize)
        print(f"[{timestamp()}] Computing template flags")
        scored = compute_template_flags(feature_store)
        print(f"[{timestamp()}] Saving scored cache: {scored_cache_path}")
        scored.to_parquet(scored_cache_path, index=False)
    if benchmark_cache_path.exists() and not args.refresh_cache:
        print(f"[{timestamp()}] Loading cached benchmark data: {benchmark_cache_path}")
        benchmark_scored = pd.read_parquet(benchmark_cache_path)
    else:
        benchmark_feature_store = build_product_benchmark_feature_store(Path(args.csv_path), chunksize=args.chunksize)
        benchmark_scored = compute_template_flags(benchmark_feature_store) if not benchmark_feature_store.empty else pd.DataFrame()
        if not benchmark_scored.empty:
            print(f"[{timestamp()}] Saving benchmark cache: {benchmark_cache_path}")
            benchmark_scored.to_parquet(benchmark_cache_path, index=False)
    print(f"[{timestamp()}] Building summary tables")
    deduped = build_deduplicated_dataset(scored)
    summary = build_quality_summary(scored)
    dataset_comparison_summary = build_dataset_comparison_summary(scored, deduped)
    issue_counterfactual = build_issue_counterfactual(scored, deduped)
    company_concentration_summary = build_company_concentration_summary(scored, deduped)
    repeat_submission_summary = build_repeat_submission_summary(scored)
    origin_summary = build_origin_summary(scored)
    monthly_panel = build_monthly_panel(scored)
    spike_decomposition = build_monthly_spike_decomposition(scored)
    text_drift_panel = build_text_drift_panel(scored)
    weight_sensitivity_summary, weight_sensitivity_monthly = build_template_weight_sensitivity(
        scored,
        monthly_panel,
        random_state=args.random_state,
    )
    issue_shift = build_issue_shift(scored)
    period_effects = build_period_effects(scored)
    credit_repair_complaints = build_credit_repair_complaints(scored)
    credit_repair_firm_drafted = build_credit_repair_firm_drafted(scored)
    credit_repair_theme_summary = build_credit_repair_theme_summary(credit_repair_complaints)
    credit_repair_phrase_table = build_credit_repair_phrase_table(credit_repair_complaints)
    credit_repair_accessibility_summary = build_credit_repair_accessibility_summary(scored)
    tag_similarity_summary = build_tag_similarity_summary(scored)
    cross_product_benchmark_panel = build_cross_product_benchmark_panel(benchmark_scored)
    examples = build_examples(scored, n_examples=20)
    if args.skip_outcome_models:
        print(f"[{timestamp()}] Skipping outcome models")
        outcome_model = pd.DataFrame()
        monetary_outcome_model = pd.DataFrame()
        non_monetary_outcome_model = pd.DataFrame()
        flat_closure_model = pd.DataFrame()
        deduplicated_outcome_model = pd.DataFrame()
        deduplicated_monetary_outcome_model = pd.DataFrame()
        credit_repair_outcome_model = pd.DataFrame()
        credit_repair_monetary_outcome_model = pd.DataFrame()
    else:
        print(f"[{timestamp()}] Fitting outcome model")
        outcome_model = fit_outcome_model(scored, model_sample_size=args.model_sample_size, random_state=args.random_state)
        monetary_outcome_model = fit_outcome_model(
            scored, model_sample_size=args.model_sample_size, random_state=args.random_state, target_column="monetary_relief"
        )
        non_monetary_outcome_model = fit_outcome_model(
            scored, model_sample_size=args.model_sample_size, random_state=args.random_state, target_column="non_monetary_relief"
        )
        flat_closure_model = fit_outcome_model(
            scored, model_sample_size=args.model_sample_size, random_state=args.random_state, target_column="flat_closure"
        )
        deduplicated_outcome_model = fit_outcome_model(
            deduped, model_sample_size=args.model_sample_size, random_state=args.random_state, target_column="positive_outcome"
        )
        deduplicated_monetary_outcome_model = fit_outcome_model(
            deduped, model_sample_size=args.model_sample_size, random_state=args.random_state, target_column="monetary_relief"
        )
        credit_repair_outcome_model = fit_credit_repair_outcome_model(
            scored, model_sample_size=args.model_sample_size, random_state=args.random_state
        )
        credit_repair_monetary_outcome_model = fit_binary_outcome_model(
            df=scored,
            target_column="monetary_relief",
            main_flag="likely_credit_repair_firm_drafted",
            interaction_prefix="credit_repair_x",
            model_sample_size=args.model_sample_size,
            random_state=args.random_state,
            extra_flag_columns=["complaint_about_credit_repair_firm"],
        )
    print(f"[{timestamp()}] Writing outputs")
    write_outputs(
        scored,
        summary,
        outcome_model,
        monetary_outcome_model,
        non_monetary_outcome_model,
        flat_closure_model,
        deduplicated_outcome_model,
        deduplicated_monetary_outcome_model,
        credit_repair_outcome_model,
        credit_repair_monetary_outcome_model,
        monthly_panel,
        issue_shift,
        dataset_comparison_summary,
        issue_counterfactual,
        company_concentration_summary,
        repeat_submission_summary,
        origin_summary,
        period_effects,
        credit_repair_complaints,
        credit_repair_firm_drafted,
        credit_repair_theme_summary,
        credit_repair_phrase_table,
        credit_repair_accessibility_summary,
        tag_similarity_summary,
        examples,
        spike_decomposition,
        text_drift_panel,
        weight_sensitivity_summary,
        weight_sensitivity_monthly,
        cross_product_benchmark_panel,
        output_dir,
    )
    elapsed_seconds = perf_counter() - clock_start
    finished_at = datetime.now()
    print(f"[{timestamp()}] Analysis finished")
    print(f"[{timestamp()}] Started at:  {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[{timestamp()}] Finished at: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[{timestamp()}] Total runtime: {elapsed_seconds / 60:.2f} minutes ({elapsed_seconds:.1f} seconds)")


if __name__ == "__main__":
    main()
