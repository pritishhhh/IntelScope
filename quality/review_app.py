"""Local human annotation review. Run: streamlit run quality/review_app.py."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality.workflow import connect_reviews, export_reviews, read_json


st.set_page_config(page_title="IntentScope review", layout="wide")
st.title("IntentScope annotation review")
st.caption("Human judgments are saved locally. Source labels appear only after the first saved judgment.")

root = Path(st.sidebar.text_input("Audit directory", "quality-output"))
if not (root / "audit.json").exists() or not (root / "records.json").exists():
    st.info("Run the audit and sampling commands before opening the review app.")
    st.stop()
batch_paths = sorted(root.glob("batch*.json"))
if not batch_paths:
    st.info("No review batch found. Run the sampling command.")
    st.stop()
batch_path = st.sidebar.selectbox("Review batch", batch_paths, format_func=lambda p: p.name)
audit = read_json(root / "audit.json")
batch = read_json(batch_path)
records = {r["record_id"]: r for r in read_json(root / "records.json")}
if any(item["record_id"] not in records or records[item["record_id"]]["source_split"] != "official_train"
       for item in batch["items"]):
    st.error("Batch contains an ineligible or unknown record.")
    st.stop()
conn = connect_reviews(root / "reviews.sqlite")
completed = {row["record_id"] for row in conn.execute("SELECT record_id FROM reviews WHERE demo=0")}
st.sidebar.metric("Progress", f"{len(completed.intersection(item['record_id'] for item in batch['items']))}/{len(batch['items'])}")
only_pending = st.sidebar.checkbox("Only pending")
items = [item for item in batch["items"] if not only_pending or item["record_id"] not in completed]
if not items:
    st.success("No pending records in this view.")
    st.stop()
selected = st.sidebar.selectbox("Record", items, format_func=lambda x: x["record_id"] + (" ✓" if x["record_id"] in completed else ""))
record = records[selected["record_id"]]
prior = conn.execute("SELECT * FROM reviews WHERE record_id=? AND demo=0", (record["record_id"],)).fetchone()
st.subheader(f"Query {record['record_id']}")
st.write(record["text"] if record["text"] is not None else "[missing text]")
st.caption(f"Review reason: {', '.join(selected['reason'])}; audit flags: {', '.join(record['flags']) or 'none'}")

labels = audit["labels"]
dispositions = ["labeled", "ambiguous", "needs_context"]
issues = ["none", "possible_source_label", "ambiguous_text", "insufficient_context", "data_quality", "taxonomy_overlap", "other"]
with st.form(f"review-{record['record_id']}"):
    disposition = st.selectbox("Judgment", dispositions,
                               index=dispositions.index(prior["disposition"]) if prior else 0)
    choices = ["— no label —"] + labels
    current = prior["reviewer_label"] if prior and prior["reviewer_label"] else choices[0]
    label = st.selectbox("Reviewer label (type to search)", choices, index=choices.index(current))
    confidence = st.slider("Confidence (1 low – 5 high)", 1, 5, prior["confidence"] if prior else 3)
    issue = st.selectbox("Issue category", issues, index=issues.index(prior["issue_category"]) if prior else 0)
    rationale = st.text_area("Short rationale; describe the reason for any revision", value=prior["rationale"] if prior else "")
    submitted = st.form_submit_button("Save judgment" if not prior else "Save documented revision")
if submitted:
    try:
        from quality.workflow import save_review
        revision = save_review(conn, record, labels, None if label == choices[0] else label,
                               disposition, confidence, rationale, issue)
        st.success(f"Saved revision {revision}.")
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

if prior:
    st.divider()
    st.subheader("Post-judgment comparison")
    st.write(f"Original dataset label: **{record['source_label']}**")
    st.write(f"Current reviewer judgment: **{prior['reviewer_label'] or prior['disposition']}**")
    if prior["disposition"] == "labeled" and prior["reviewer_label"] != record["source_label"]:
        st.warning("This is a source-label disagreement for review, not a confirmed labeling error.")
    st.caption("No model predictions are loaded in this workflow.")
    st.subheader("Decision history")
    history = [dict(row) for row in conn.execute(
        "SELECT revision, updated_at, reviewer_label, disposition, confidence, issue_category, rationale FROM review_history WHERE record_id=? AND demo=0 ORDER BY revision",
        (record["record_id"],))]
    st.dataframe(history, use_container_width=True, hide_index=True)

if st.sidebar.button("Export real reviews"):
    count = export_reviews(conn, root / "reviews.json")
    st.sidebar.success(f"Exported {count} reviews to {root / 'reviews.json'}")
conn.close()
