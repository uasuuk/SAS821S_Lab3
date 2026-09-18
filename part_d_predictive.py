"""
SAS821S Lab 3 - Part D: Predictive Security Intelligence and Adversarial ML
Run top to bottom.  Needs: pandas, numpy, matplotlib, scikit-learn
Data files must sit in a 'data' folder next to this script.
Outputs: top_10_risk_days.csv, incident_timeline.csv, adversarial_risk_scores.csv,
         figure6_risk_over_time.png, figure7_adversarial_risk.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

SEED = 42
TARGET_RECALL = 0.90          # operating-point goal: catch at least 90% of pre-incident days
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)

print("Versions: python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__,
      "| scikit-learn", sklearn.__version__, "| matplotlib", matplotlib.__version__, "| seed", SEED)

# ============================================================ D1. PREDICTIVE RISK MODEL
train = pd.read_csv(DATA / "egs_daily_risk_training.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
invest = pd.read_csv(DATA / "egs_daily_risk_investigation.csv", parse_dates=["date"])
FEATURES = [c for c in train.columns if c not in ("date", "incident_within_7d")]   # target never used as a feature
TARGET = "incident_within_7d"

print("\n=== D1: DATA CHECK ===")
print(f"Training days: {len(train)} ({train['date'].min().date()} to {train['date'].max().date()}); "
      f"investigation days: {len(invest)} ({invest['date'].min().date()} to {invest['date'].max().date()})")
print("Missing values (train):", int(train[FEATURES + [TARGET]].isnull().sum().sum()))
print("Class balance:", train[TARGET].value_counts().to_dict(), "| positive rate", round(train[TARGET].mean(), 3))

X, y = train[FEATURES], train[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

candidates = {
    "Logistic regression (scaled)": make_pipeline(StandardScaler(),
                                                   LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
    "Gradient boosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_auc = {}
for name, mdl in candidates.items():
    cv_auc[name] = cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
print("\n5-fold CV ROC-AUC (training split only):")
for k, v in cv_auc.items():
    print(f"  {k:32s} {v:.3f}")
best_auc = max(cv_auc.values())
lr_name = "Logistic regression (scaled)"
# prefer the interpretable model unless another is clearly better (>0.01 AUC)
chosen_name = lr_name if cv_auc[lr_name] >= best_auc - 0.01 else max(cv_auc, key=cv_auc.get)
model = candidates[chosen_name]
print("Chosen model:", chosen_name)

# operating threshold chosen on out-of-fold training predictions (test set stays untouched)
oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
rows = []
for t in grid:
    p = (oof >= t).astype(int)
    rows.append((t, recall_score(y_tr, p), precision_score(y_tr, p, zero_division=0), p.mean()))
thr_tbl = pd.DataFrame(rows, columns=["threshold", "recall", "precision", "alert_rate"])
ok = thr_tbl[thr_tbl["recall"] >= TARGET_RECALL]
THRESHOLD = float(ok["threshold"].max()) if len(ok) else 0.5
print(f"\nOperating threshold = {THRESHOLD:.2f} (highest threshold with out-of-fold recall >= {TARGET_RECALL:.0%})")

model.fit(X_tr, y_tr)
p_te = model.predict_proba(X_te)[:, 1]


def report(y_true, prob, thr, label):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"\n--- {label} (threshold {thr:.2f}) ---")
    print("Confusion matrix (rows=actual, cols=predicted):\n", np.array([[tn, fp], [fn, tp]]))
    print(f"Accuracy {accuracy_score(y_true, pred):.3f} | Precision {precision_score(y_true, pred):.3f} | "
          f"Recall {recall_score(y_true, pred):.3f} | F1 {f1_score(y_true, pred):.3f} | "
          f"False-negative rate {fn / (fn + tp):.3f} (fn={fn}, tp={tp}) | false alarms={fp} of {len(y_true)} days "
          f"({pred.mean():.1%} of days alerted)")


print("\n=== D1: TEST-SET EVALUATION (stratified 75/25 split) ===")
report(y_te, p_te, 0.50, "Default threshold")
report(y_te, p_te, THRESHOLD, "Chosen recall-oriented threshold")

# sanity check: chronological split (train on earlier days, test on later days) - guards against optimistic random-split results
cut = int(len(train) * 0.75)
m2 = clone(model).fit(X.iloc[:cut], y.iloc[:cut])
report(y.iloc[cut:], m2.predict_proba(X.iloc[cut:])[:, 1], THRESHOLD, "Chronological hold-out check (last 25% of days)")

# which indicators drive the score
if hasattr(model[-1] if hasattr(model, "steps") else model, "coef_"):
    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values(ascending=False)
else:
    coefs = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature influence (standardised coefficients or importances):\n", coefs.round(3).to_string())

# ============================================================ D2. RISK ESCALATION AND CORRELATION
invest["risk_probability"] = model.predict_proba(invest[FEATURES])[:, 1]
invest["above_threshold"] = invest["risk_probability"] >= THRESHOLD

top10 = invest.sort_values("risk_probability", ascending=False).head(10)
top10_out = top10[["date", "risk_probability", "above_threshold"] + FEATURES].copy()
top10_out["date"] = top10_out["date"].dt.date
top10_out.round(4).to_csv(BASE / "top_10_risk_days.csv", index=False)
print("\n=== D2: top_10_risk_days.csv ===")
print(top10_out[["date", "risk_probability", "above_threshold", "failed_remote_logins", "suspicious_dns_queries",
                 "ot_scan_attempts", "high_relevance_threat_reports"]].round(3).to_string(index=False))

# earliest defensible escalation: first day of 2 consecutive days at/above threshold
inv_s = invest.sort_values("date").reset_index(drop=True)
first_any = inv_s.loc[inv_s["above_threshold"], "date"].min()
sustained = inv_s["above_threshold"] & inv_s["above_threshold"].shift(-1, fill_value=False)
first_sustained = inv_s.loc[sustained, "date"].min()
print(f"\nDays above threshold: {int(inv_s['above_threshold'].sum())} of {len(inv_s)}")
print("First day above threshold:", None if pd.isna(first_any) else first_any.date())
print("Earliest defensible escalation (2 consecutive days above threshold):",
      None if pd.isna(first_sustained) else first_sustained.date())

# --- timeline from event logs + report/IOC evidence (all IDs read from the supplied files)
events = pd.read_csv(DATA / "egs_security_event_logs.csv", parse_dates=["timestamp"])
reports = pd.read_csv(DATA / "egs_text_investigation.csv", parse_dates=["published_at"])
iocs = pd.read_csv(DATA / "egs_ioc_feed.csv")
key_events = events[events["severity"].isin(["Medium", "High", "Critical"])].copy()
tl = [{"timestamp": r.timestamp, "id_type": "event_id", "id": r.event_id,
       "description": f"{r.source_type}: {r.event_type} {r.source_asset}->{r.destination_asset} [{r.action}, {r.severity}] {r.details}"}
      for r in key_events.itertuples()]
for r in reports[reports["report_id"].str.startswith("RPT")].itertuples():
    tl.append({"timestamp": r.published_at, "id_type": "report_id", "id": r.report_id, "description": f"{r.source}: {r.title}"})
rpt_text = " ".join(reports.loc[reports["report_id"].str.startswith("RPT"), "report_text"]).lower()
log_ind = set(events["indicator"].dropna().astype(str).str.lower())
iocs_rel = iocs[iocs["indicator_value"].astype(str).str.lower().apply(lambda v: v in log_ind or v in rpt_text)]
print(f"IOC feed rows: {len(iocs)}; kept {len(iocs_rel)} that appear in the event logs or the incident-window reports")
for r in iocs_rel.itertuples():
    tl.append({"timestamp": pd.to_datetime(r.first_seen), "id_type": "ioc_id", "id": r.ioc_id,
               "description": f"IOC first_seen: {r.indicator_value} ({r.confidence} confidence, {r.tactic}) - {r.recommended_action}"})
timeline = pd.DataFrame(tl).sort_values("timestamp").reset_index(drop=True)
timeline.to_csv(BASE / "incident_timeline.csv", index=False)
print(f"\n=== Incident timeline (incident_timeline.csv): {timeline['id_type'].value_counts().to_dict()} ===")
print(timeline.to_string(index=False))

# --- figure 6: risk over time
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(inv_s["date"], inv_s["risk_probability"], marker="o", color="#2c3e50", label="Predicted risk (incident within 7 days)")
ax.axhline(THRESHOLD, color="#c0392b", ls="--", label=f"Alert threshold ({THRESHOLD:.2f})")
ax.scatter(inv_s.loc[inv_s["above_threshold"], "date"], inv_s.loc[inv_s["above_threshold"], "risk_probability"],
           color="#c0392b", zorder=3, label="Above threshold")
ax.axvline(pd.Timestamp("2026-10-17"), color="#e67e22", ls=":", label="Attack chain in logs (EVT01574-EVT01590)")
if not pd.isna(first_sustained):
    ax.axvline(first_sustained, color="#27ae60", ls="-.", label=f"Earliest defensible escalation ({first_sustained.date()})")
ax.set_ylim(0, 1.02); ax.set_xlabel("Date (Namibia local time)"); ax.set_ylabel("Predicted probability")
ax.set_title("Figure 6: Daily predicted risk of an incident within 7 days (investigation period)")
ax.legend(fontsize=8, loc="upper left"); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(BASE / "figure6_risk_over_time.png", dpi=150); plt.close()

# ============================================================ D3. ADVERSARIAL ROBUSTNESS
adv = pd.read_csv(DATA / "egs_adversarial_risk_cases.csv")
adv["risk_probability"] = model.predict_proba(adv[FEATURES])[:, 1]
adv["risk_logit"] = model.decision_function(adv[FEATURES])       # log-odds: does not saturate at 1.0
adv["flagged_at_threshold"] = adv["risk_probability"] >= THRESHOLD
BEHAVIOUR = ["failed_remote_logins", "new_external_ips", "suspicious_dns_queries", "high_severity_edr_alerts",
             "ot_scan_attempts", "high_relevance_threat_reports", "after_hours_admin_actions"]
margins = []
for _, row in adv.iterrows():
    keep = 1.0
    for f in np.arange(1.0, -0.001, -0.01):          # shrink behavioural signals until the alert would be missed
        v = row[FEATURES].copy().astype(float); v[BEHAVIOUR] = v[BEHAVIOUR] * f
        if model.predict_proba(pd.DataFrame([v], columns=FEATURES))[:, 1][0] < THRESHOLD:
            break
        keep = f
    margins.append(round(keep, 2))
adv["min_signal_fraction_still_flagged"] = margins       # e.g. 0.30 = signals could shrink to 30% before evasion succeeds
base_p = adv.loc[adv["variant"] == "baseline_attack", "risk_probability"].iloc[0]
adv["change_vs_baseline"] = adv["risk_probability"] - base_p
adv.round(4).to_csv(BASE / "adversarial_risk_scores.csv", index=False)
print("\n=== D3: adversarial risk cases ===")
print(adv[["case_id", "variant", "risk_probability", "risk_logit", "change_vs_baseline", "flagged_at_threshold", "min_signal_fraction_still_flagged"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#c0392b" if f else "#8899aa" for f in adv["flagged_at_threshold"]]
bars = ax.bar(adv["variant"].str.replace("_", "\n"), adv["risk_probability"], color=colors)
ax.bar_label(bars, fmt="%.2f", fontsize=9)
ax.axhline(THRESHOLD, color="black", ls="--", label=f"Alert threshold ({THRESHOLD:.2f})")
ax.set_ylim(0, 1.05); ax.set_ylabel("Predicted probability")
ax.set_title("Figure 7: Model risk score for baseline vs evasive attack variants (red = alerted)")
ax.legend(); plt.tight_layout(); plt.savefig(BASE / "figure7_adversarial_risk.png", dpi=150); plt.close()
print("\nSaved: top_10_risk_days.csv, incident_timeline.csv, adversarial_risk_scores.csv, figure6_risk_over_time.png, figure7_adversarial_risk.png")

print("\nTraining-set maximum values (to judge how far outside normal the attack cases are):")
print(train[FEATURES].max().to_string())
print("\nDaily investigation risk series:")
print(inv_s.assign(risk_probability=inv_s["risk_probability"].round(3))[["date", "risk_probability", "above_threshold"]].to_string(index=False))

