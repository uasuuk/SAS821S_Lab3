"""
SAS821S Lab 3 - Part B: Text Mining and Threat-Intelligence Prioritisation
Run top to bottom. Random seed fixed at 42 throughout for reproducibility.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

RANDOM_SEED = 42
DATA_DIR = "data/"

# ============================================================
# B1. TEXT PREPARATION AND EXPLORATORY ANALYSIS
# ============================================================

train_raw = pd.read_csv(DATA_DIR + "egs_text_training.csv")
invest_raw = pd.read_csv(DATA_DIR + "egs_text_investigation.csv")

train_df = train_raw.copy()
invest_df = invest_raw.copy()

print("=== B1: DATA QUALITY CHECK ===")
print(f"Training reports: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Investigation reports: {invest_df.shape[0]} rows, {invest_df.shape[1]} columns")

print("\nMissing values (training):")
print(train_df.isnull().sum())
print("\nMissing values (investigation):")
print(invest_df.isnull().sum())

print("\nDuplicate rows (training):", train_df.duplicated().sum())
print("Duplicate rows (investigation):", invest_df.duplicated().sum())
print("Duplicate report_id (training):", train_df["report_id"].duplicated().sum())
print("Duplicate report_id (investigation):", invest_df["report_id"].duplicated().sum())

print("\nSource distribution (training):")
print(train_df["source"].value_counts())

print("\nClass balance (relevant_label):")
print(train_df["relevant_label"].value_counts())
print(train_df["relevant_label"].value_counts(normalize=True).round(3))

train_df["published_at"] = pd.to_datetime(train_df["published_at"])
invest_df["published_at"] = pd.to_datetime(invest_df["published_at"])
print("\nTraining date range:", train_df["published_at"].min(), "to", train_df["published_at"].max())
print("Investigation date range:", invest_df["published_at"].min(), "to", invest_df["published_at"].max())


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation/digit noise, collapse whitespace.
    Dots and hyphens are kept so B2's IOC extraction can still find
    domains/IPs in the same underlying text; this cleaner is only used
    for the TF-IDF classifier input.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean_text"] = train_df["report_text"].apply(clean_text)
invest_df["clean_text"] = invest_df["report_text"].apply(clean_text)

# Tokenisation / stop-words / n-grams are handled inside TfidfVectorizer in B3
# (English stop-word list, unigrams+bigrams, min_df=2) so training and
# investigation text are processed identically.

# Table 1: relevance by source
source_relevance = pd.crosstab(train_df["source"], train_df["relevant_label"])
source_relevance.columns = ["not_relevant", "relevant"]
print("\n=== Table 1: Relevance by source ===")
print(source_relevance)

# Chart 1: relevant vs not-relevant reports by month
train_df["month"] = train_df["published_at"].dt.to_period("M").astype(str)
monthly = train_df.groupby(["month", "relevant_label"]).size().unstack(fill_value=0)
monthly.columns = ["not_relevant", "relevant"]

fig, ax = plt.subplots(figsize=(10, 4))
monthly.plot(kind="bar", stacked=True, ax=ax, color=["#8899aa", "#c0392b"])
ax.set_title("Figure 1: Relevant vs Not-Relevant Reports by Month (training set)")
ax.set_xlabel("Month published")
ax.set_ylabel("Number of reports")
plt.tight_layout()
plt.savefig("figure1_reports_by_month.png", dpi=150)
plt.close()
print("\nSaved figure1_reports_by_month.png")

# Chart 2: top terms in reports labelled relevant
cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=20)
relevant_texts = train_df.loc[train_df["relevant_label"] == 1, "clean_text"]
term_counts = cv.fit_transform(relevant_texts)
term_freq = pd.Series(
    np.asarray(term_counts.sum(axis=0)).ravel(), index=cv.get_feature_names_out()
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
term_freq.plot(kind="barh", ax=ax, color="#c0392b")
ax.set_title("Figure 2: Top 20 Terms in Reports Labelled Operationally Relevant")
ax.set_xlabel("Frequency")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("figure2_top_terms_relevant.png", dpi=150)
plt.close()
print("Saved figure2_top_terms_relevant.png")


# ============================================================
# B2. IOC / ENTITY EXTRACTION AND OPERATIONAL RELEVANCE
# ============================================================

print("\n=== B2: IOC/ENTITY EXTRACTION ===")

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SHA256_PATTERN = re.compile(r"\b[a-f0-9]{64}\b")
DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:example|com|net|org|info)\b"
)
DEFANGED_DOMAIN_PATTERN = re.compile(r"\b(?:[a-z0-9-]+\[\.\])+[a-z0-9-]+\b")

# Named actor/malware reference list - built by inspecting the supplied
# text and IOC data. Kept explicit and short so the extraction stays
# auditable rather than a black box.
KNOWN_ACTORS_MALWARE = [
    "kalahari jackal", "sandstorm loader", "dustrat", "desert-sync",
]


def defang_to_domain(match: str) -> str:
    return match.replace("[.]", ".")


def extract_entities(report_id: str, text: str) -> list[dict]:
    """Extract IOC-style entities from one report's raw text."""
    text_lower = text.lower()
    findings = []

    for ip in set(IP_PATTERN.findall(text)):
        findings.append({
            "report_id": report_id, "entity_type": "ipv4", "entity_value": ip,
            "analytical_relevance": "Possible C2 or scanning source/destination address"
        })

    for h in set(SHA256_PATTERN.findall(text_lower)):
        findings.append({
            "report_id": report_id, "entity_type": "sha256", "entity_value": h,
            "analytical_relevance": "Possible malware/file hash for endpoint matching"
        })

    domains = set(DOMAIN_PATTERN.findall(text_lower))
    for d in set(DEFANGED_DOMAIN_PATTERN.findall(text_lower)):
        domains.add(defang_to_domain(d))
    for dom in domains:
        findings.append({
            "report_id": report_id, "entity_type": "domain", "entity_value": dom,
            "analytical_relevance": "Possible C2/phishing infrastructure domain"
        })

    for name in KNOWN_ACTORS_MALWARE:
        if name in text_lower:
            entity_type = "malware" if name in ("sandstorm loader", "dustrat") else "threat_actor"
            findings.append({
                "report_id": report_id, "entity_type": entity_type,
                "entity_value": name.title(),
                "analytical_relevance": "Named actor/malware associated with energy-sector targeting"
            })

    return findings


all_reports = pd.concat([
    train_df[["report_id", "report_text"]],
    invest_df[["report_id", "report_text"]]
], ignore_index=True)

extracted_rows = []
for _, row in all_reports.iterrows():
    extracted_rows.extend(extract_entities(row["report_id"], row["report_text"]))

extracted_iocs = pd.DataFrame(extracted_rows)
print(f"Extracted {len(extracted_iocs)} entity mentions across {all_reports.shape[0]} reports")
print(extracted_iocs["entity_type"].value_counts())

ioc_feed = pd.read_csv(DATA_DIR + "egs_ioc_feed.csv")


def normalise_indicator(val: str) -> str:
    val = str(val).lower().strip()
    val = re.sub(r"^https?://", "", val)
    val = val.rstrip("/")
    return val


ioc_feed["indicator_norm"] = ioc_feed["indicator_value"].apply(normalise_indicator)
extracted_iocs["entity_value_norm"] = extracted_iocs["entity_value"].apply(normalise_indicator)

matched = extracted_iocs.merge(
    ioc_feed[["ioc_id", "indicator_norm", "confidence", "context", "tactic", "first_seen"]],
    left_on="entity_value_norm", right_on="indicator_norm", how="left"
)

n_matched = matched["ioc_id"].notna().sum()
print(f"\n{n_matched} of {len(extracted_iocs)} extracted entities matched a known IOC feed entry")

print("\n=== Sample matched findings (for report narrative) ===")
sample_matches = matched[matched["ioc_id"].notna()][
    ["report_id", "entity_type", "entity_value", "ioc_id", "confidence", "tactic", "first_seen"]
].head(10)
print(sample_matches.to_string(index=False))

extracted_iocs_export = extracted_iocs[
    ["report_id", "entity_type", "entity_value", "analytical_relevance"]
].drop_duplicates()
extracted_iocs_export.to_csv("extracted_iocs.csv", index=False)
print(f"\nSaved extracted_iocs.csv ({len(extracted_iocs_export)} rows)")


# ============================================================
# B3. TEXT CLASSIFIER AND ADVERSARIAL LANGUAGE
# ============================================================

print("\n=== B3: TEXT CLASSIFIER ===")

X = train_df["clean_text"]
y = train_df["relevant_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED
)

vectorizer = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), min_df=2, max_features=5000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
false_negative_rate = fn / (fn + tp)

print("Confusion matrix (rows=actual, cols=predicted):")
print(cm)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
print(f"False-negative rate: {false_negative_rate:.3f}  (fn={fn}, tp={tp})")
print("\n", classification_report(y_test, y_pred, target_names=["not_relevant", "relevant"]))

invest_tfidf = vectorizer.transform(invest_df["clean_text"])
invest_df["relevance_probability"] = clf.predict_proba(invest_tfidf)[:, 1]

top_15 = invest_df.sort_values("relevance_probability", ascending=False).head(15)
top_15_export = top_15[["report_id", "published_at", "source", "title", "relevance_probability"]]
top_15_export.to_csv("top_15_relevant_reports.csv", index=False)
print(f"\nSaved top_15_relevant_reports.csv")
print(top_15_export.to_string(index=False))

adv = pd.read_csv(DATA_DIR + "egs_adversarial_text_cases.csv")
adv["orig_clean"] = adv["original_text"].apply(clean_text)
adv["mod_clean"] = adv["modified_text"].apply(clean_text)

orig_tfidf = vectorizer.transform(adv["orig_clean"])
mod_tfidf = vectorizer.transform(adv["mod_clean"])

adv["original_probability"] = clf.predict_proba(orig_tfidf)[:, 1]
adv["modified_probability"] = clf.predict_proba(mod_tfidf)[:, 1]
adv["probability_change"] = adv["modified_probability"] - adv["original_probability"]

print("\n=== Adversarial evasion test ===")
print(adv[["case_id", "evasion_technique", "original_probability",
           "modified_probability", "probability_change"]].to_string(index=False))

print(f"\nMean probability drop after evasion rewording: {adv['probability_change'].mean():.3f}")
print("\nDone. Files produced: figure1_reports_by_month.png, figure2_top_terms_relevant.png, "
      "extracted_iocs.csv, top_15_relevant_reports.csv")
