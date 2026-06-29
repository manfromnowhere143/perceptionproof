# Project Status & Roadmap

The durable state of PerceptionProof: what is built, what is verified, and what is next.
Append dated entries; do not rewrite history.

## What this project is

A reproducible study and harness testing whether cheap, label-free signals (model
disagreement, temporal inconsistency, occupancy conflict, VLA reasoning self-consistency)
predict human-rated long-tail driving failure (Waymo's Rater Feedback Score), and whether
they recover the safety ranking on cases where open-loop planning metrics are known to
mis-rank closed-loop driving. It is an evaluation layer **around** the driving stack — not
a controller. The science is open and checkable; orchestration is pluggable via a backend
interface (an open deterministic backend ships here; a separate governed backend is the
commercial half — open-core).

## Contribution, stated honestly

- **Not novel, and not claimed:** disagreement-as-uncertainty (Deep Ensembles, 2017).
- **The contribution:** (1) bridging cheap label-free signals to the 2026 metric-validity
  result that open-loop metrics mis-rank closed-loop safety; (2) a falsifiable, multi-signal
  study under one pre-registered protocol; (3) auditable, signed provenance for the whole
  evaluation. The study is publishable even if its result is negative.

## Source-of-truth documents

- Mathematics (every signal + validity metric): `docs/MATHEMATICS.md`
- Architecture (backend interface, receipts, mission DAG): `docs/ARCHITECTURE.md`
- Frozen analysis (hypotheses, thresholds, seed): `PREREGISTRATION.md`
- Data licensing & compliance: `DATA_LICENSES.md`

## Phase ledger

| Phase | Gate | State |
|---|---|---|
| P0 Plan | design approved | DONE |
| P1 Scaffold | hypotheses frozen pre-result | DONE — repo skeleton, math, architecture, pre-registration |
| P2 Data & models | inputs reproducible from receipts | IN PROGRESS — production `LocalBackend` + `DatasetAdapter`/`ModelRunner` seam BUILT + tested on fixtures; real WOD-E2E/NAVSIM adapters + model runners remain (GPU/dataset-bound). Runbook: `docs/P2_SETUP.md` |
| P3 Signals | tests green; deterministic | DONE — all four signals implemented + tested on known-answer inputs (S1 MMD disagreement, S2 SE(2)-aligned temporal flicker, S3 corridor occupancy-conflict, S4 semantic-entropy clustering) |
| P4 Study | each finding adversarially verified | core DONE — all validity statistics implemented + tested |
| P2a real (open-loop) | real frames, receipted | DONE — NAVSIM ensemble, ρ=0.699; leave-one-out independent outcome ρ=0.683 (retires coupling caveat). `results/navsim_p2a_report.md`, `experiments/navsim_p2a/` |
| P2b closed-loop (PDMS) | independent closed-loop score | DONE — NULL result: disagreement (S1) does NOT transfer to PDMS (ρ=−0.074 [−0.396,0.285], AUROC≈0.53, 400 scenes/8 drives) while open-loop ρ=0.70. On-thesis (reproduces the open-loop↔closed-loop gap). Pre-registered null, published. `results/navsim_p2b_report.md`, `experiments/navsim_p2b/`. Pipeline solved: symlink data→`navsim_logs/<split>/`, metric-cache via run_metric_caching, score via `navsim.evaluate.pdm_score` with native-arch ensemble (weights assigned into EgoStatusMLPAgent._mlp) |
| P2c gate-events | label-free signal vs PDMS gates | DONE — DECISIVE: label-free signals predict the binary NC/DAC gate events at AUROC 0.77–0.83 (CIs exclude chance, 1317 scenes / 55 drives) where the same signal was chance vs the PDMS scalar (P2b). HONEST NULL: collision-geometry vs disagreement inconclusive on both gates (paired Δ CIs include 0). `results/navsim_p2c_report.md`, `experiments/navsim_p2c/`. Engineering: parallel PDM scoring (free loader pre-fork + maxtasksperchild to avoid OOM), drive-stratified tokens, n2-standard-16 (32-core hit capacity) |
| P2d.1 leave-one-out NC | remove coupling | DONE — disagreement{1,2,3} predicts held-out member-0 collision at AUROC 0.821 [0.775,0.865] (~ coupled 0.829) → not an artifact. `experiments/navsim_p2c/leave_one_out_nc.py` |
| P2d.2 TransFuser (real sensor planner) | real collisions | BLOCKED on data consistency. Pipeline VALIDATED end-to-end (3 pretrained TransFuser ckpts load, metric-cache + sensor-loader + pdm_score all execute). 446 GB navtrain sensors downloaded. BUT all scenes fail FileNotFoundError on specific CAM_F0 frames: our `openscene_metadata_trainval` (1310 logs, only 7.2% navtrain coverage) does NOT frame-align with the navtrain sensor blobs — partial/mismatched metadata version. FIX next session: re-download the FULL/correct trainval metadata to match the sensors (Option A), or use the self-consistent `mini` split (Option B). Scripts: scratchpad tf_score.py, gen_navtrain_tokens.py, vm_tf_run.sh. VM engineering-node-02 is n2-standard-16, 727 GB disk |
| P2d.2b TransFuser on mini | real sensor planner | DONE (pipeline validated, underpowered). 3-seed pretrained TransFuser runs end-to-end on the frame-consistent `mini` split (396 scenes, 52 drives, 0 err; drives = unique log id). But strong planner → only 3 collisions / 12 off-road → can't measure failure-AUROC; any-gate disagreement 0.600 [0.511,0.695] borderline; all paired tests inconclusive. `results/navsim_p2d_transfuser_report.md`. Mini fix: symlink `navsim_logs/mini`→`mini_navsim_logs/mini` (pkls), data via download_mini (sensor URL has `/openscene_sensor_mini_<kind>/` subdir). For POWER: need larger frame-consistent sensors (full trainval ~TBs, or fix navtrain version mismatch). Scripts: scratchpad vm_mini_data.sh, vm_mini_run.sh, tf_score.py (LOGS_DIR/SENSOR_DIR/METRIC_CACHE env) |
| P2e WOD-E2E RFS (human raters) | label-free signal vs Waymo Rater Feedback Score | DONE — H1 NOT MET (published negative). 479 rater frames / 93 drives (all 93 val shards, 243 GB, parsed parallel; 4-seed ego-status MLP ensemble; official `get_rater_feedback_score`). Label-free disagreement vs neg-RFS ρ = +0.151 [0.063, 0.237], BH q<0.05 — real but below the pre-registered 0.30 bar. Oracle anchor ADE (needs the human label) ρ = +0.395 [0.326, 0.458] → RFS *is* predictable; the ego-only ensemble carries too little scene info (disagreement↔ADE only 0.19 here vs 0.70 in P2a). Triage AUROC: disagreement 0.629 / ADE 0.734. `results/wod_e2e_rfs_report.md`, `experiments/wod_e2e_rfs/`. Next power lever = perception-grounded ensemble, not a different statistic |
| P2f perception-grounded RFS (PRELIMINARY) | does scene perception help? | SUPERSEDED by P2g. A single instantiation suggested ego 0.10 → vision 0.22, paired Δρ=+0.122 [+0.003,+0.242] (CI cleared 0 by only +0.003). NOT replicated — retracted. `results/wod_e2e_rfs_vision_report.md` (correction banner) |
| P2g surround + STABILITY (the correction) | is the perception lift robust? | DONE — ROBUST NULL. Pushed to all 8 cameras (PCA-256); single run INVERTED the ranking (ego 0.29 > front 0.16 > surround 0.10). Stability study over 20 seed-sets (paired, same seeds per rung, off the cached embeddings): mean ρ ego +0.176 (sd .048) / front +0.161 / surround +0.124; paired Δ intervals ALL straddle 0 (front−ego [−0.092,+0.076], surr−ego [−0.133,+0.033]); P(vision>ego) ≈ 0.10–0.30. Frozen-encoder perception grounding does NOT robustly improve RFS prediction; P2f lift was seed-noise. Robust human-RFS signal = ρ≈0.18, below 0.30 bar. LESSON: report distributions not single draws for small effects; cache (`wod_surround_emb.npz`) makes ablations cheap. NOT ruled out: jointly-trained E2E vision/LiDAR planner (GPU). `results/wod_e2e_rfs_surround_report.md`, `experiments/wod_e2e_rfs/{run_rfs_surround,stability_study,analyze_surround}.py`. ENGINEERING: pin BLAS threads (OMP/OPENBLAS/MKL/NUMEXPR=1) before numpy or 12 workers × multithread BLAS → load 190 wedges IAP-SSH → `instances reset` recovers (disk persists) |
| P2h jointly-trained vision (GPU) | does a driving-trained planner clear 0.30? | DONE — NO. The one thing P2g could not rule out: K=6 DINOv2-S planners FINE-TUNED end-to-end (front cam + ego, on an L4 GPU). Disagreement vs neg-RFS ρ=+0.202 [0.123,0.280] (CI upper 0.28 < 0.30); member-bootstrap stability mean +0.193 sd 0.027. Joint training gave a TIGHTER signal (sd .027 vs ego .05) and a BETTER planner (ADE-vs-RFS 0.458, highest in project) but the disagreement-RFS correlation still does not clear 0.30 and overlaps ego (~0.18). CEILING confirmed across ego/frozen-image/surround/jointly-trained (ρ≈0.12–0.20): label-free disagreement is a strong proxy for open-loop error + binary safety gates, weak for fine-grained human ratings. `results/wod_e2e_rfs_jointtrained_report.md`, `experiments/wod_e2e_rfs/{run_parse_pixels,train_vision,rfs_eval,analyze_p2h}.py`. INFRA: L4 g2-standard-8 DLVM (torch 2.9 cu129); data via SNAPSHOT of engineering-node-02 boot disk → disk-from-snapshot → attach ro-mount /mnt/wod; parse with mounted `wod` env, train with DLVM torch+timm |
| P5 Report & repo | reproduce-in-one-command verified | pending |

## What runs today (CPU, no GPU/dataset)

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pytest                                            # 33 tests green
./.venv/bin/python -m harness.cli run --backend synthetic     # full mission, writes results/ + receipts
./.venv/bin/python -m harness.cli verify results/synthetic_receipts.jsonl   # -> VERIFIED
```

Implemented + tested: the receipt layer (hash-chain + Ed25519 sign/verify), all four
signals, all validity statistics (Spearman + cluster bootstrap + permutation null,
failure-mining AUROC/AP, risk-coverage/AURC, Kendall-inversion recovery, mutual
information, BH-FDR), a synthetic end-to-end run, AND the production `LocalBackend`
(DatasetAdapter + ModelRunner composition) validated end-to-end on fixtures.

The synthetic report is a **plumbing test** (its correlation is planted by construction)
— it confirms the machine threads signals → labels → receipts correctly. It is not a
scientific finding.

## Next (P2) — see docs/P2_SETUP.md for the full researched runbook

The harness, signals, statistics, receipts, and the production `LocalBackend` are built and
tested. P2 only fills the real adapters/runners (genuinely GPU/dataset-bound):

1. Track A (fastest): NAVSIM — implement `NavsimAdapter` + `NavsimAgentRunner`; run >= 2
   baseline agents over navtest; test disagreement vs PDMS.
2. Track B (the H1 result): WOD-E2E val — implement `WodE2EAdapter` (cameras+ego+routing+RFS);
   ensemble of E2E planner + occupancy + VLA runners; signals S1–S4 vs RFS.
3. Freeze the `TBD@P2` pre-registration values from the real RFS/PDMS distribution BEFORE scoring.

Gate to leave P2: a clean run produces receipted `ModelOutput` for the slice on real frames,
the real numbers are computed with CIs, the chain verifies, and re-running is byte-identical.

## Published

Public repository: https://github.com/manfromnowhere143/perceptionproof (Apache-2.0).
CI (ruff + pytest) runs on every push and pull request.

## First real measurement (P2a)

On 788 held-out real NAVSIM scenes (disjoint by-log split, 4-member ego-status MLP ensemble,
CPU on GCP), label-free disagreement predicts open-loop error: Spearman ρ=0.699 [0.599, 0.750],
p=0.0005; failure-mining AUROC=0.855 / AP=0.846 (prec@50=0.98); E-AURC=0.180. Full writeup +
honest caveats (open-loop not closed-loop; weak ego-status models; structural coupling; own split
not official navtest): `results/navsim_p2a_report.md`. Next decisive test = P2b (NAVSIM PDMS) and
WOD-E2E RFS. VM bootstrap scripts: scratchpad (vm_setup.sh, vm_data.sh, pp_experiment.py).

## Log

- **2026-06-29** — P2h: closed the perception question on a GPU. Trained K=6 DINOv2 planners
  end-to-end (jointly, not frozen) on an L4. Disagreement-vs-RFS ρ = 0.20 [0.123, 0.280];
  tighter signal + better planner (ADE-vs-RFS 0.46), but still under 0.30 and overlapping ego.
  The human-RFS ceiling (ρ ≈ 0.12–0.20) does not move with perception or end-to-end training —
  a genuine ceiling, not a tuning problem. Data shared to the GPU box via disk snapshot+mount.
- **2026-06-29** — P2g + correction: pushed perception grounding to all 8 cameras and ran a
  20-seed-set **stability study**. It overturned P2f: the apparent vision lift was seed-noise
  (mean ρ ego 0.18 / front 0.16 / surround 0.12; paired Δ intervals straddle 0). Robust
  finding: frozen-encoder perception grounding does NOT improve label-free RFS prediction;
  the human-RFS signal is ρ ≈ 0.18, below the 0.3 bar. P2f claim retracted (correction banner).
  Lesson logged: report distributions, not single draws, for small effects.
- **2026-06-29** — P2f (PRELIMINARY, later corrected): a single instantiation suggested a
  frozen-DINOv2 front-camera lift (ego 0.10 → vision 0.22, paired Δρ CI cleared 0 by +0.003).
  torch/timm installed in the `wod` env. Superseded by P2g.
- **2026-06-29** — P2e: human-rated benchmark. All 93 WOD-E2E val shards (243 GB) parsed
  parallel on the CPU VM; 4-seed ego-status MLP ensemble; 479 rater frames scored with
  Waymo's official RFS. Label-free disagreement vs RFS ρ = 0.15 (real, BH q<0.05, but
  below the pre-registered 0.30 bar → H1 not met, published as a negative); oracle ADE
  anchor ρ = 0.40 confirms RFS is predictable. The cheap signal is strongest on open-loop
  error and weak against human judgment with an ego-only planner.
- **2026-06-28** — P2a: first REAL number on real NAVSIM data (ρ=0.70, AUROC=0.855). NAVSIM env +
  minimal data (maps + trainval logs, no sensors) stood up on GCP CPU VM engineering-node-02;
  4-member ego-status MLP ensemble trained; disagreement-vs-ADE scored by the tested code.
- **2026-06-27** — P0/P1 complete (thesis, math, architecture, pre-registration). CPU core
  built and green: receipts, all four signals, all validity statistics, and a synthetic
  end-to-end harness + CLI that runs the mission and verifies the receipt chain (31 tests).
  Remaining work is P2 (real models on real frames).
