"""Run from repository root: python scripts/quality_cli.py <command>."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality.workflow import audit, clean_candidates, export_reviews, generate_report, sample_batch, connect_reviews, reviewer_directory


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("audit")
    a.add_argument("--dataset-dir", type=Path, default=Path("intentscope-offline-assets/banking77"))
    a.add_argument("--output-dir", type=Path, default=Path("quality-output"))
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--validation-size", type=float, default=.15)
    s = commands.add_parser("sample")
    s.add_argument("--audit-dir", type=Path, default=Path("quality-output"))
    s.add_argument("--target", type=int, default=200)
    s.add_argument("--flagged-fraction", type=float, default=.25)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--output", type=Path, default=None, help="Batch JSON path; defaults to audit-dir/batch.json")
    c = commands.add_parser("clean")
    c.add_argument("--audit-dir", type=Path, default=Path("quality-output"))
    c.add_argument("--output-dir", type=Path, default=Path("quality-output/cleaned"))
    c.add_argument("--trim-outer-whitespace", action="store_true")
    e = commands.add_parser("export")
    e.add_argument("--audit-dir", type=Path, default=Path("quality-output"))
    e.add_argument("--reviewer-id", type=str, default=None)
    e.add_argument("--db", type=Path, default=None)
    e.add_argument("--output", type=Path, default=None)
    r = commands.add_parser("report")
    r.add_argument("--audit-dir", type=Path, default=Path("quality-output"))
    r.add_argument("--reviewer-id", type=str, default=None)
    r.add_argument("--db", type=Path, default=None)
    r.add_argument("--cleaning-dir", type=Path, default=Path("quality-output/cleaned"))
    r.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "audit":
        result, _ = audit(args.dataset_dir, args.output_dir, args.seed, args.validation_size)
        print(f"Audited {result['record_count']} eligible records; results in {args.output_dir}")
    elif args.command == "sample":
        result = sample_batch(args.audit_dir, args.target, args.flagged_fraction, args.seed, args.output)
        print(f"Sampled {result['count']} unique records; reasons: {result['sampling_counts']}")
    elif args.command == "clean":
        result = clean_candidates(args.audit_dir, args.output_dir, args.trim_outer_whitespace)
        print(f"Exported candidates with {len(result)} explicit changes")
    elif args.command == "export":
        reviewer_dir = reviewer_directory(args.audit_dir, args.reviewer_id) if args.reviewer_id else args.audit_dir
        db = args.db or reviewer_dir / "reviews.sqlite"
        output = args.output or reviewer_dir / "reviews.json"
        conn = connect_reviews(db)
        print(f"Exported {export_reviews(conn, output)} real reviews to {output}")
        conn.close()
    elif args.command == "report":
        reviewer_dir = reviewer_directory(args.audit_dir, args.reviewer_id) if args.reviewer_id else args.audit_dir
        db = args.db or reviewer_dir / "reviews.sqlite"
        output = args.output or reviewer_dir / "report.md"
        print(generate_report(args.audit_dir, db, args.cleaning_dir, output))


if __name__ == "__main__":
    main()
