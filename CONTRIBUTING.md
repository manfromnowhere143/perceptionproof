# Contributing

PerceptionProof is a research artifact whose value is reproducibility and integrity.
Contributions must preserve both.

## Ground rules

1. **Pre-registration is binding.** Hypotheses, thresholds, and the slice are frozen in
   `PREREGISTRATION.md` before results are computed. Changes to analysis after seeing
   results are added as dated amendments, never silent overwrites.
2. **No dataset media.** Never commit dataset frames, point clouds, or labels — only
   segment ids and derived outputs/receipts. See `DATA_LICENSES.md`. `.gitignore`
   blocks common dataset extensions.
3. **Every estimator ships with a known-answer test.** Each signal and metric must be
   validated on synthetic data whose answer is known before it touches real labels.
4. **Honesty over polish.** Negative results are reported unmodified. The README and any
   report may not claim more than the evidence supports.
5. **Determinism.** Results must reproduce from content-addressed inputs and the fixed
   seed in `PREREGISTRATION.md`.

## Developer setup

```bash
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/ruff check .
./.venv/bin/pytest
```

CI runs `ruff check` and `pytest` on every push and pull request; both must pass.

## Scope

The open repository holds the science (signals, metrics, statistics), the runner, and the
receipt verifier. A separate governed orchestration backend is out of scope for this repo.
