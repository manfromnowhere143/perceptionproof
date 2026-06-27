"""Command-line entry: `perceptionproof run` and `perceptionproof verify`.

run    -> executes the mission DAG (ingest -> run_models -> signals -> score ->
          receipt) over the slice, against a chosen backend.
verify -> recomputes content hashes and checks the receipt chain + signatures.

The `synthetic` backend runs the full machine end-to-end on planted data (proof of
plumbing). The `local`/`aweb` backends run real models and land at P2+.
"""

from __future__ import annotations

import argparse
import json
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perceptionproof")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the study over the slice")
    run.add_argument("--backend", choices=["synthetic", "local", "aweb"], default="synthetic")
    run.add_argument("--slice", default="protocol/slices.json")
    run.add_argument("--out", default="results")

    ver = sub.add_parser("verify", help="verify a receipt chain + signatures")
    ver.add_argument("receipts_file", help="path to a *_receipts.jsonl file")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        if args.backend != "synthetic":
            print(f"backend '{args.backend}' lands at P2+ (real models). See docs/CONTINUITY.md.")
            return 2
        from .runner import default_synthetic_slice, run_synthetic

        ids = default_synthetic_slice()
        if os.path.exists(args.slice):
            with open(args.slice) as f:
                data = json.load(f)
            if data.get("segment_ids"):
                ids = data["segment_ids"]
        report, receipts, _ = run_synthetic(ids, out_dir=args.out)
        print(json.dumps(report, indent=2))
        print(f"\n{len(receipts)} receipts written to {args.out}/synthetic_receipts.jsonl (chain verified).")
        return 0

    if args.cmd == "verify":
        from .runner import verify_receipts_file

        ok = verify_receipts_file(args.receipts_file)
        print("VERIFIED" if ok else "FAILED")
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
