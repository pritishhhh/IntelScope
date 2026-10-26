from __future__ import annotations

import csv
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from sklearn.model_selection import train_test_split


DATASET = "PolyAI/banking77"
SEED = 42
VALIDATION_SIZE = 0.15


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_source(dataset_dir: Path):
    """Read only official training data. Never open the official test CSV."""
    categories = read_json(dataset_dir / "categories.json")
    if not isinstance(categories, list) or not all(isinstance(x, str) and x for x in categories):
        raise ValueError("categories.json must contain nonempty label names")
    with (dataset_dir / "train.csv").open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not {"text", "category"}.issubset(reader.fieldnames or []):
            raise ValueError("train.csv needs text and category columns")
        records = []
        for row_number, row in enumerate(reader, 1):
            digest = hashlib.sha256(json.dumps([row.get("text"), row.get("category")], ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
            records.append({"record_id": f"train-{row_number:06d}-{digest}", "source_split": "official_train",
                            "source_row": row_number, "text": row.get("text"), "source_label": row.get("category")})
    return categories, records


def assign_splits(records: list[dict], categories: list[str], seed: int = SEED, validation_size: float = VALIDATION_SIZE):
    """Mirror the notebook's stratified train_test_split over original CSV row positions."""
    labels = [r["source_label"] for r in records]
    counts = Counter(labels)
    n_val = int(len(records) * validation_size + 0.999999)
    eligible = (len(records) > 1 and all(label in categories for label in labels)
                and min(counts.values(), default=0) >= 2 and n_val >= len(counts)
                and len(records) - n_val >= len(counts))
    if eligible:
        label_to_id = {name: index for index, name in enumerate(categories)}
        train_idx, val_idx = train_test_split(list(range(len(records))), test_size=validation_size,
                                             random_state=seed, stratify=[label_to_id[label] for label in labels])
        val_set = set(val_idx)
        note = f"Stratified {validation_size:.0%} validation split using seed {seed} and original CSV row positions; notebook default is 15% and seed 42."
    else:
        val_set = set()
        note = "No reproducible stratified validation split possible for this input; official training rows only."
    for i, record in enumerate(records):
        record["split"] = "validation" if i in val_set else "train"
    return note


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _quantile(values: list[int], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def audit(dataset_dir: Path, output_dir: Path, seed: int = SEED, validation_size: float = VALIDATION_SIZE):
    categories, records = load_source(dataset_dir)
    mapping_duplicates = sorted(name for name, count in Counter(categories).items() if count > 1)
    split_note = assign_splits(records, categories, seed, validation_size)
    flags = defaultdict(set)
    exact = defaultdict(list)
    norm = defaultdict(list)
    lengths = []
    for record in records:
        rid, value, label = record["record_id"], record["text"], record["source_label"]
        exact[value].append(record)
        if value is None or not value.strip():
            flags[rid].add("missing_or_blank_text")
        else:
            norm[normalized(value)].append(record)
            lengths.append(len(value.split()))
        if label is None or not label.strip():
            flags[rid].add("missing_label")
        elif label not in categories:
            flags[rid].add("invalid_label")
    q1, q3 = _quantile(lengths, .25), _quantile(lengths, .75)
    lower, upper = max(0, q1 - 1.5 * (q3 - q1)), q3 + 1.5 * (q3 - q1)
    for record in records:
        value = record["text"]
        if value and value.strip() and not lower <= len(value.split()) <= upper:
            flags[record["record_id"]].add("length_outlier")
    groups = {key: [] for key in ("exact_duplicate", "normalized_match", "conflicting_labels", "train_validation_overlap")}
    for group in exact.values():
        by_label = defaultdict(list)
        for record in group:
            by_label[record["source_label"]].append(record)
        for same_label in by_label.values():
            if len(same_label) > 1:
                groups["exact_duplicate"].append([r["record_id"] for r in same_label])
        if len(by_label) > 1:
            groups["conflicting_labels"].append([r["record_id"] for r in group])
        if len({r["split"] for r in group}) > 1:
            groups["train_validation_overlap"].append([r["record_id"] for r in group])
    for group in norm.values():
        if len(group) > 1 and len({r["text"] for r in group}) > 1:
            groups["normalized_match"].append([r["record_id"] for r in group])
        if len({r["source_label"] for r in group}) > 1 and len({r["text"] for r in group}) > 1:
            # A normalized conflict is a candidate, not a confirmed source-label error.
            for record in group:
                flags[record["record_id"]].add("normalized_label_conflict")
    for key, items in groups.items():
        for group in items:
            for rid in group:
                flags[rid].add(key)
    for record in records:
        record["flags"] = sorted(flags[record["record_id"]])
    flag_counts = dict(sorted(Counter(flag for record in records for flag in record["flags"]).items()))
    distribution = {split: dict(sorted(Counter(r["source_label"] for r in records if r["split"] == split).items()))
                    for split in ("train", "validation")}
    summary = {"dataset": DATASET, "source": str(dataset_dir.resolve()), "audited_source_splits": ["official_train"],
               "held_out_test_excluded": True, "seed": seed, "validation_size": validation_size,
               "split_note": split_note, "labels": list(dict.fromkeys(categories)),
               "label_mapping_issues": {"duplicate_category_names": mapping_duplicates},
               "record_count": len(records), "split_counts": dict(Counter(r["split"] for r in records)),
               "finding_record_counts": flag_counts,
               "finding_group_counts": {key: len(items) for key, items in groups.items()},
               "finding_groups": groups, "class_distribution": distribution,
               "text_length_words": {"count": len(lengths), "min": min(lengths, default=0),
                                     "median": median(lengths) if lengths else 0,
                                     "mean": round(mean(lengths), 2) if lengths else 0,
                                     "p95": _quantile(lengths, .95), "max": max(lengths, default=0),
                                     "outlier_lower": lower, "outlier_upper": upper}}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "audit.json", summary)
    write_json(output_dir / "records.json", records)
    lines = ["# Dataset audit", "", f"Dataset: {DATASET}", f"Audited: {len(records)} official training rows; official test excluded.",
             split_note, "", "## Findings (affected records)", ""]
    lines.extend(f"- {key}: {value}" for key, value in flag_counts.items())
    lines.append(f"- Duplicate taxonomy names: {len(mapping_duplicates)}")
    lines += ["", "Normalization matches, conflicting labels, and length outliers require human interpretation.",
              "No records or source labels were changed."]
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, records


def sample_batch(audit_dir: Path, target: int = 200, flagged_fraction: float = .25, seed: int = SEED,
                 output_path: Path | None = None):
    if target < 1 or not 0 <= flagged_fraction <= 1:
        raise ValueError("target must be positive and flagged_fraction within 0..1")
    records = read_json(audit_dir / "records.json")
    if any(r["source_split"] != "official_train" or r["split"] not in ("train", "validation") for r in records):
        raise ValueError("Batch source contains an ineligible split")
    rng = random.Random(seed)
    flagged = [r for r in records if r["flags"]]
    rng.shuffle(flagged)
    flagged_target = min(len(flagged), round(min(target, len(records)) * flagged_fraction))
    chosen = {r["record_id"]: {"record_id": r["record_id"], "reason": ["audit_flag"]} for r in flagged[:flagged_target]}
    remaining = max(0, min(target, len(records)) - len(chosen))
    buckets = defaultdict(list)
    for record in records:
        if record["record_id"] not in chosen:
            buckets[record["source_label"] or "<missing>"].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = sorted(buckets)
    rng.shuffle(labels)
    while remaining and any(buckets.values()):
        for label in labels:
            if remaining and buckets[label]:
                record = buckets[label].pop()
                chosen[record["record_id"]] = {"record_id": record["record_id"], "reason": ["stratified"]}
                remaining -= 1
    batch = {"dataset": DATASET, "seed": seed, "target": target, "flagged_fraction": flagged_fraction,
             "eligible_splits": ["train", "validation"], "count": len(chosen),
             "sampling_counts": dict(Counter(reason for item in chosen.values() for reason in item["reason"])),
             "items": list(chosen.values())}
    write_json(output_path or audit_dir / "batch.json", batch)
    return batch


def connect_reviews(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS reviews (
        record_id TEXT PRIMARY KEY, source_label TEXT, reviewer_label TEXT, disposition TEXT NOT NULL,
        confidence INTEGER NOT NULL, rationale TEXT NOT NULL, issue_category TEXT NOT NULL,
        demo INTEGER NOT NULL DEFAULT 0, revision INTEGER NOT NULL, updated_at TEXT NOT NULL)""")
    connection.execute("""CREATE TABLE IF NOT EXISTS review_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, source_label TEXT,
        reviewer_label TEXT, disposition TEXT NOT NULL, confidence INTEGER NOT NULL,
        rationale TEXT NOT NULL, issue_category TEXT NOT NULL, demo INTEGER NOT NULL,
        revision INTEGER NOT NULL, updated_at TEXT NOT NULL)""")
    connection.commit()
    return connection


def save_review(connection, record: dict, labels: list[str], reviewer_label: str | None,
                disposition: str, confidence: int, rationale: str, issue_category: str,
                demo: bool = False):
    if record["source_split"] != "official_train" or record["split"] not in ("train", "validation"):
        raise ValueError("Only eligible training/validation records may be reviewed")
    if disposition not in ("labeled", "ambiguous", "needs_context"):
        raise ValueError("Invalid disposition")
    if reviewer_label is not None and reviewer_label not in labels:
        raise ValueError("Unknown reviewer label")
    if disposition == "labeled" and reviewer_label is None:
        raise ValueError("Labeled review needs a label")
    if not isinstance(confidence, int) or not 1 <= confidence <= 5 or not rationale.strip():
        raise ValueError("Confidence must be 1..5 and rationale required")
    if issue_category not in ("none", "possible_source_label", "ambiguous_text", "insufficient_context", "data_quality", "taxonomy_overlap", "other"):
        raise ValueError("Invalid issue category")
    previous = connection.execute("SELECT * FROM reviews WHERE record_id=?", (record["record_id"],)).fetchone()
    revision = 1 if previous is None else previous["revision"] + 1
    if previous and not rationale.strip():
        raise ValueError("Revision rationale required")
    now = datetime.now(timezone.utc).isoformat()
    fields = (record["record_id"], record["source_label"], reviewer_label, disposition, confidence,
              rationale.strip(), issue_category, int(demo), revision, now)
    with connection:
        connection.execute("""INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(record_id) DO UPDATE SET source_label=excluded.source_label,
            reviewer_label=excluded.reviewer_label, disposition=excluded.disposition,
            confidence=excluded.confidence, rationale=excluded.rationale,
            issue_category=excluded.issue_category, demo=excluded.demo,
            revision=excluded.revision, updated_at=excluded.updated_at""", fields)
        connection.execute("INSERT INTO review_history (record_id,source_label,reviewer_label,disposition,confidence,rationale,issue_category,demo,revision,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)", fields)
    return revision


def export_reviews(connection, path: Path):
    rows = [dict(row) for row in connection.execute("SELECT * FROM reviews WHERE demo=0 ORDER BY record_id")]
    write_json(path, rows)
    return len(rows)


def clean_candidates(audit_dir: Path, output_dir: Path, trim_outer_whitespace: bool = False):
    records = read_json(audit_dir / "records.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    changes = []
    with (output_dir / "candidate.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["record_id", "source_split", "split", "source_row", "original_text", "candidate_text", "original_label", "unresolved_flags"])
        writer.writeheader()
        for record in records:
            original = record["text"]
            candidate = original.strip() if trim_outer_whitespace and original is not None else original
            if candidate != original:
                changes.append({"record_id": record["record_id"], "rule": "trim_outer_whitespace",
                                "original_value": original, "resulting_value": candidate})
            writer.writerow({"record_id": record["record_id"], "source_split": record["source_split"],
                             "split": record["split"], "source_row": record["source_row"],
                             "original_text": original, "candidate_text": candidate,
                             "original_label": record["source_label"], "unresolved_flags": ";".join(record["flags"])})
    write_json(output_dir / "changes.json", changes)
    return changes


def generate_report(audit_dir: Path, db_path: Path, cleaning_dir: Path, output_path: Path):
    audit_result = read_json(audit_dir / "audit.json")
    records = read_json(audit_dir / "records.json")
    batch = read_json(audit_dir / "batch.json") if (audit_dir / "batch.json").exists() else None
    connection = connect_reviews(db_path)
    eligible_ids = {r["record_id"] for r in records}
    reviews = [dict(row) for row in connection.execute("SELECT * FROM reviews WHERE demo=0 ORDER BY record_id")
               if row["record_id"] in eligible_ids]
    connection.close()
    changes = read_json(cleaning_dir / "changes.json") if (cleaning_dir / "changes.json").exists() else []
    agreement = sum(r["disposition"] == "labeled" and r["reviewer_label"] == r["source_label"] for r in reviews)
    disagreement = sum(r["disposition"] == "labeled" and r["reviewer_label"] != r["source_label"] for r in reviews)
    ambiguity = sum(r["disposition"] in ("ambiguous", "needs_context") for r in reviews)
    examples = []
    by_id = {r["record_id"]: r for r in records}
    for review in reviews:
        if review["record_id"] in by_id and (review["disposition"] != "labeled" or review["reviewer_label"] != review["source_label"]):
            examples.append(f"- `{review['record_id']}`: {by_id[review['record_id']]['text']!r}; reviewer: {review['reviewer_label'] or review['disposition']}; rationale: {review['rationale']}")
        if len(examples) == 5:
            break
    distribution = audit_result["class_distribution"]
    length = audit_result["text_length_words"]
    lines = ["# IntentScope dataset quality and human review", "", f"Dataset: {audit_result['dataset']}.",
             f"Audited splits: train and validation derived from official training data. Official test excluded. {audit_result['split_note']}",
             f"Records: {audit_result['record_count']} ({audit_result['split_counts']}).", "", "## Audit findings", "",
             "Counts are affected records, not confirmed defects."]
    lines += [f"- {key}: {value}" for key, value in audit_result["finding_record_counts"].items()]
    if not audit_result["finding_record_counts"]:
        lines.append("- No flagged records under the configured checks.")
    candidates = [r for r in records if r["flags"]]
    candidates.sort(key=lambda r: ("normalized_match" not in r["flags"], r["source_row"]))
    lines += ["", "Representative audit candidates (interpretation pending):"]
    lines += [f"- `{r['record_id']}` ({', '.join(r['flags'])}): {r['text']!r}" for r in candidates[:5]]
    lines += ["", "## Distribution and length", "",
              f"Class counts: train min/max {min(distribution['train'].values(), default=0)}/{max(distribution['train'].values(), default=0)}; validation min/max {min(distribution['validation'].values(), default=0)}/{max(distribution['validation'].values(), default=0)}. Full counts in audit.json.",
              f"Word length: median {length['median']}, mean {length['mean']}, p95 {length['p95']}, range {length['min']}–{length['max']}; IQR outlier bounds {length['outlier_lower']}–{length['outlier_upper']}.",
              "", "## Human review", "", f"{len(reviews)} reviewed; source-label agreement {agreement}; disagreement/review candidates {disagreement}; ambiguous or needs context {ambiguity}."]
    if batch:
        batch_ids = {item["record_id"] for item in batch["items"]}
        reviewed_batch = sum(r["record_id"] in batch_ids for r in reviews)
        lines.append(f"Batch progress: {reviewed_batch}/{batch['count']} unique records; seed {batch['seed']}, target {batch['target']}; {batch['sampling_counts']}. Stratified label coverage plus audit-flag supplement.")
    else:
        lines.append("Batch not generated yet.")
    if not reviews:
        lines.append("Representative reviewer reasoning and human-validated conclusions await review. No agreement rate is reported.")
    else:
        lines += ["", "Representative review candidates (not confirmed source errors):", *examples]
    lines += ["", "## Cleaning and limitations", "", f"Explicit cleaning changes: {len(changes)}.",
              "Raw text and original labels remain in the source CSV and audit records; findings remain unresolved until human adjudication.",
              "Audit flags do not establish labeling errors. Review decisions do not establish model errors. No test records, model predictions, retraining, or performance claims are part of this workflow."]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"reviewed": len(reviews), "agreement": agreement, "disagreement": disagreement, "ambiguity": ambiguity, "cleaning_changes": len(changes)}
