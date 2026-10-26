import csv
import json
from pathlib import Path

import pytest
from sklearn.model_selection import train_test_split

from quality.workflow import (audit, clean_candidates, connect_reviews, export_reviews,
                              generate_report, read_json, reviewer_directory, sample_batch, save_review)


def fixture_dataset(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "categories.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    rows = [
        ("Same query", "a"), ("Same query", "a"), ("Same query", "b"),
        ("  SAME   QUERY ", "a"), ("Another question", "b"),
        ("More words here", "a"), ("One more example", "b"),
        ("Different", "a"), ("Test-like row", "b"), ("Last", "a"),
        ("Other", "b"), ("Final text", "b"),
    ]
    with (source / "train.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["text", "category"])
        writer.writerows(rows)
    # Deliberately malformed: audit must not read official test data.
    (source / "test.csv").write_text("SECRET HELD OUT\n", encoding="utf-8")
    return source


def test_audit_ids_duplicates_conflicts_and_test_exclusion(tmp_path):
    source = fixture_dataset(tmp_path)
    first, records = audit(source, tmp_path / "audit")
    second, again = audit(source, tmp_path / "audit2")
    assert first["finding_group_counts"]["exact_duplicate"] == 1
    assert first["finding_group_counts"]["conflicting_labels"] == 1
    assert first["finding_group_counts"]["normalized_match"] == 1
    assert records == again
    assert records[0]["record_id"] != records[1]["record_id"]
    assert records[0]["source_row"] == 1
    assert all(r["source_split"] == "official_train" and "SECRET" not in str(r) for r in records)
    assert first["held_out_test_excluded"] is True
    names = ["a", "b"]
    numeric_labels = [names.index(r["source_label"]) for r in records]
    _, expected_validation = train_test_split(list(range(len(records))), test_size=.15,
                                              random_state=42, stratify=numeric_labels)
    assert {i for i, r in enumerate(records) if r["split"] == "validation"} == set(expected_validation)


def test_mapping_issue_is_reported(tmp_path):
    source = fixture_dataset(tmp_path)
    (source / "categories.json").write_text(json.dumps(["a", "b", "a"]), encoding="utf-8")
    summary, _ = audit(source, tmp_path / "audit")
    assert summary["label_mapping_issues"]["duplicate_category_names"] == ["a"]


def test_sampling_stable_unique_and_small(tmp_path):
    audit(fixture_dataset(tmp_path), tmp_path / "audit")
    first = sample_batch(tmp_path / "audit", target=200)
    second = sample_batch(tmp_path / "audit", target=200)
    ids = [item["record_id"] for item in first["items"]]
    assert first == second
    assert len(ids) == len(set(ids)) == 12
    assert set(first["sampling_counts"]) <= {"audit_flag", "stratified"}


def test_review_persistence_history_export_and_report(tmp_path):
    audit(fixture_dataset(tmp_path), tmp_path / "audit")
    sample_batch(tmp_path / "audit", target=4)
    record = read_json(tmp_path / "audit" / "records.json")[0]
    db = tmp_path / "reviews.sqlite"
    conn = connect_reviews(db)
    assert save_review(conn, record, ["a", "b"], "a", "labeled", 4, "Initial choice", "none") == 1
    assert save_review(conn, record, ["a", "b"], "b", "labeled", 2, "Reconsidered wording", "possible_source_label") == 2
    assert save_review(conn, read_json(tmp_path / "audit" / "records.json")[1], ["a", "b"],
                       None, "ambiguous", 2, "Fixture only", "other", demo=True) == 1
    with pytest.raises(ValueError):
        save_review(conn, {**record, "source_split": "official_test"}, ["a", "b"], "a", "labeled", 4, "No", "none")
    conn.close()
    conn = connect_reviews(db)
    assert conn.execute("SELECT COUNT(*) FROM review_history WHERE record_id=?", (record["record_id"],)).fetchone()[0] == 2
    assert export_reviews(conn, tmp_path / "export.json") == 1
    conn.close()
    assert len(read_json(tmp_path / "export.json")) == 1
    counts = generate_report(tmp_path / "audit", db, tmp_path / "cleaned", tmp_path / "report.md")
    assert counts == {"reviewed": 1, "agreement": 0, "disagreement": 1, "ambiguity": 0, "cleaning_changes": 0}
    assert "Reconsidered wording" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_reviewer_workspaces_keep_judgments_separate(tmp_path):
    audit(fixture_dataset(tmp_path), tmp_path / "audit")
    sample_batch(tmp_path / "audit", target=4)
    record = read_json(tmp_path / "audit" / "records.json")[0]
    alice_dir = reviewer_directory(tmp_path / "audit", "Alice")
    bob_dir = reviewer_directory(tmp_path / "audit", "bob")
    assert alice_dir == reviewer_directory(tmp_path / "audit", " alice ")
    assert alice_dir != bob_dir
    assert alice_dir.is_relative_to(tmp_path / "audit" / "reviewers")
    with pytest.raises(ValueError):
        reviewer_directory(tmp_path / "audit", "   ")
    alice = connect_reviews(alice_dir / "reviews.sqlite")
    save_review(alice, record, ["a", "b"], "a", "labeled", 5, "Clear", "none")
    alice.close()
    bob = connect_reviews(bob_dir / "reviews.sqlite")
    assert bob.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
    save_review(bob, record, ["a", "b"], None, "ambiguous", 2, "Both possible", "taxonomy_overlap")
    bob.close()
    alice_count = generate_report(tmp_path / "audit", alice_dir / "reviews.sqlite", tmp_path / "cleaned", alice_dir / "report.md")
    bob_count = generate_report(tmp_path / "audit", bob_dir / "reviews.sqlite", tmp_path / "cleaned", bob_dir / "report.md")
    assert alice_count["agreement"] == 1 and alice_count["ambiguity"] == 0
    assert bob_count["agreement"] == 0 and bob_count["ambiguity"] == 1


def test_cleaning_preserves_source_and_records_changes(tmp_path):
    source = fixture_dataset(tmp_path)
    original = (source / "train.csv").read_bytes()
    audit(source, tmp_path / "audit")
    changes = clean_candidates(tmp_path / "audit", tmp_path / "cleaned", trim_outer_whitespace=True)
    assert len(changes) == 1
    assert changes[0]["original_value"] == "  SAME   QUERY "
    assert changes[0]["resulting_value"] == "SAME   QUERY"
    assert (source / "train.csv").read_bytes() == original
    assert read_json(tmp_path / "audit" / "records.json")[3]["text"] == "  SAME   QUERY "
