# PerceptionProof — Mathematical Formalism

This document defines every signal, fusion rule, validity metric, and statistical test used by PerceptionProof. It is written to be falsifiable and reproducible: each construct has an exact equation, a stated estimator, and a pre-registered decision rule. Nothing here claims a signal *works*; it defines how we would *prove or disprove* that it does.

Math renders via GitHub's MathJax. Notation is fixed once in §1 and reused throughout.

---

## 1. Setup and notation

- Dataset of driving segments $\mathcal{S}$, with $n = |\mathcal{S}|$. Each segment $s$ carries multi-camera observations $o_s$, routing/ego state, and a drive identifier $\mathrm{drive}(s)$ (for leakage-aware resampling).
- A frozen, content-hashed set of $M \ge 3$ models/planners $\{f_1,\dots,f_M\}$. Each $f_m$ maps $o_s$ to a predicted future trajectory $\tau_m(s) \in \mathbb{R}^{T \times 2}$ (T BEV waypoints, ego frame), and — where the model supports it — a multimodal set, an occupancy field, or a reasoning trace.
- Human ground truth: the Waymo **Rater Feedback Score** $y^{\mathrm{RFS}}_m(s) \in [0,10]$ for method $m$'s trajectory on segment $s$ (higher = more human-acceptable).
- **Failure event** (pre-registered threshold $\theta_{\mathrm{RFS}}$): $\,Y_m(s) = \mathbb{1}\!\left[\,y^{\mathrm{RFS}}_m(s) < \theta_{\mathrm{RFS}}\,\right]$.
- A **label-free signal** is a function $g_j : s \mapsto \mathbb{R}_{\ge 0}$, $j \in \{1,2,3,4\}$, computed without any RFS label. Convention: larger $g_j$ means more predicted risk/uncertainty.

A trajectory distance, used throughout, with optional horizon discount $\gamma \in (0,1]$:

$$
d(\tau, \tau') \;=\; \frac{\sum_{t=1}^{T} \gamma^{\,t}\,\lVert \tau_t - \tau'_t \rVert_2}{\sum_{t=1}^{T} \gamma^{\,t}} .
$$

---

## 2. The four label-free signals

Each signal is tied to a *verified* failure mechanism from the thesis (see `AWEB_CV_AUTONOMOUS_DRIVING_THESIS_2026-06-27.md`), not invented.

### 2.1 S1 — Ensemble trajectory disagreement
*Mechanism: model committees disagree most where the scene is ambiguous (ensembles, Lakshminarayanan 2017). We extend to be multimodality-aware, because averaging multimodal trajectories is ill-posed.*

Unimodal form (one trajectory per model) — mean pairwise distance:

$$
g_1^{\text{pair}}(s) \;=\; \binom{M}{2}^{-1} \sum_{1 \le i < j \le M} d\!\left(\tau_i(s),\,\tau_j(s)\right).
$$

Multimodal form — each model emits a weighted mode set $\mathcal{M}_m(s)=\{(\tau_m^{(k)}, w_m^{(k)})\}$. Treat each model's prediction as a distribution and use the **Maximum Mean Discrepancy** with an RBF kernel $\kappa(\tau,\tau')=\exp\!\big(-d(\tau,\tau')^2/2\sigma^2\big)$:

$$
g_1(s) \;=\; \binom{M}{2}^{-1}\!\!\sum_{i<j} \mathrm{MMD}^2\!\big(\mathcal{M}_i(s),\mathcal{M}_j(s)\big),\qquad
\mathrm{MMD}^2(P,Q)=\mathbb{E}_{\tau,\tau'\sim P}\kappa - 2\,\mathbb{E}_{\tau\sim P,\tau'\sim Q}\kappa + \mathbb{E}_{\tau,\tau'\sim Q}\kappa .
$$

MMD is a proper, non-parametric divergence that does not collapse multimodal structure — the correct generalization of "disagreement" when planners are multimodal.

### 2.2 S2 — Temporal inconsistency (forecast flicker)
*Mechanism: compounding closed-loop error (verified 3-0). A stable model's forecast made at time $k$ should agree, on the overlapping horizon, with the forecast made at $k{+}1$ once both are mapped into a common world frame via ego motion $\mathbf{E}_{k\to k+1}$.*

For model $m$, with $\Pi$ the alignment-and-overlap operator:

$$
g_2(s) \;=\; \frac{1}{M}\sum_{m=1}^{M}\; \frac{1}{|\mathcal{K}_s|}\sum_{k\in\mathcal{K}_s}
d\!\Big(\Pi\big(\tau_m^{(k)}\big),\; \Pi\big(\mathbf{E}_{k\to k+1}\,\tau_m^{(k+1)}\big)\Big).
$$

High $g_2$ = the model keeps changing its mind frame-to-frame — an early indicator of the instability that snowballs in closed loop.

### 2.3 S3 — Occupancy conflict ("the scene pretending to be safe")
*Mechanism: occupancy perception captures vertical/occluded structure BEV cannot (verified 3-0). The dangerous case is when the planner's free-space confidence in the ego corridor conflicts with occupancy uncertainty there — the hidden-actor/occlusion risk.*

Let an occupancy model give per-voxel probability $p(v)\in[0,1]$ over voxels $v$, with binary entropy $H(p)=-p\log p-(1-p)\log(1-p)$. Restrict to the **ego corridor** $\mathcal{C}(s)$ (planned-path sweep + its occluded frustums). Define a corridor-restricted entropy and a planner-conflict term:

$$
H_{\mathcal C}(s)=\frac{1}{|\mathcal C|}\sum_{v\in\mathcal C} H\!\big(p(v)\big),
\qquad
\mathrm{Conf}(s)=\frac{1}{|\mathcal C|}\sum_{v\in\mathcal C}\underbrace{\mathbb{1}\!\big[p(v)>\theta_{\mathrm{occ}}\big]}_{\text{occupancy says risk}}\cdot\underbrace{c_{\text{free}}(s,v)}_{\text{planner says clear}} ,
$$

$$
g_3(s) \;=\; \alpha\,H_{\mathcal C}(s) + (1-\alpha)\,\mathrm{Conf}(s),\quad \alpha\in[0,1]\ \text{pre-registered}.
$$

### 2.4 S4 — VLA reasoning self-(in)consistency
*Mechanism: large-model hallucination is "a severe safety risk" (verified 3-0). The SOTA way to quantify it is semantic entropy (Kuhn et al. 2023), not token entropy.*

Sample $K$ stochastic reasoning rollouts from the VLA, yielding decisions/trajectories $\{a_1,\dots,a_K\}$. Cluster by semantic equivalence into clusters $\{\mathcal{c}_1,\dots,\mathcal{c}_L\}$ (meaning, not surface form), with empirical mass $\hat\pi_\ell=|\mathcal{c}_\ell|/K$. Semantic entropy:

$$
g_4(s) \;=\; -\sum_{\ell=1}^{L}\hat\pi_\ell \log \hat\pi_\ell .
$$

For continuous trajectory decisions, the cluster step is replaced by the S1 multimodal dispersion over the $K$ samples.

---

## 3. Fusion and calibration

We pre-register **two** fusion rules and report both (no post-hoc selection):

1. **Unsupervised (rank-average).** Per-signal rank-normalize $\tilde g_j(s)=\mathrm{rank}(g_j(s))/n \in (0,1]$, then $G(s)=\frac{1}{4}\sum_j \tilde g_j(s)$. No labels touched.
2. **Supervised (calibrated logistic), cross-validated.** Fit $\hat P(Y{=}1\mid s)=\sigma\!\big(\beta_0+\sum_j \beta_j\, z_j(s)\big)$ on $z$-scored signals, with $\beta$ estimated on training folds only (§5 cluster-aware CV). Probabilities are **Platt/temperature-calibrated** on a held-out fold.

**Calibration is reported, not assumed.** Expected Calibration Error over $B$ probability bins:

$$
\mathrm{ECE}=\sum_{b=1}^{B}\frac{|\mathcal{B}_b|}{n}\,\big|\,\mathrm{acc}(\mathcal{B}_b)-\mathrm{conf}(\mathcal{B}_b)\,\big|,
$$

with a reliability diagram in `results/`.

---

## 4. Validity metrics (the contribution)

These operationalize hypotheses H1–H3. Each metric maps to a deployable decision.

### 4.1 Predictive correlation (H1)
Spearman $\rho$ and Kendall $\tau$ between the signal and **negated** RFS (high signal should predict low RFS):

$$
\rho_j = \rho_{\text{Spearman}}\!\big(g_j(s),\,-\,y^{\mathrm{RFS}}(s)\big).
$$

Reported with cluster-bootstrap 95% CI and a permutation $p$-value (§5).

### 4.2 Failure mining (H3)
Treat $g$ (or $G$) as a failure detector against label $Y$. Report **AUROC**, **Average Precision** (AUPRC), and precision@k:

$$
\mathrm{prec}@k=\frac{1}{k}\sum_{s\in \mathrm{top}_k(g)} Y(s),\qquad \text{compared against base rate } \bar Y=\tfrac{1}{n}\sum_s Y(s).
$$

### 4.3 Selective prediction / risk–coverage (the deployable framing)
The product is "defer the riskiest segments to human review." Retain the coverage-$c$ fraction with **lowest** $g$; risk = mean failure (or mean $10-\mathrm{RFS}$) on the retained set:

$$
R(c)=\frac{\sum_{s} Y(s)\,\mathbb{1}[g(s)\le q_c]}{\sum_{s}\mathbb{1}[g(s)\le q_c]},\qquad
\mathrm{AURC}=\int_0^1 R(c)\,dc,\qquad
\text{E-AURC}=\mathrm{AURC}-\mathrm{AURC}^\star,
$$

where $q_c$ is the $c$-quantile of $g$ and $\mathrm{AURC}^\star$ is the oracle. Lower AURC = a better triage signal (Geifman & El-Yaniv 2017). This is the number an AV eval team actually cares about: *how much safer is the retained set per unit of human review spent.*

### 4.4 Ranking-inversion recovery (H2 — the centerpiece, ties to arXiv 2605.00066)
The field's open problem: an open-loop metric (PDMS) mis-ranks methods vs ground truth. Let $\hat r_{\text{open}}$ rank the $M$ methods by PDMS and $\hat r_{\text{true}}$ rank them by aggregate human ground truth. **Kendall tau distance** counts ranking inversions:

$$
K(r,r')=\big|\{(i,j): i<j,\ \mathrm{sgn}(r_i-r_j)\ne \mathrm{sgn}(r'_i-r'_j)\}\big|.
$$

Define a **signal-adjusted** method score that penalizes long-tail risk, $u_m=\overline{\mathrm{PDMS}}_m-\lambda\,\overline{g}_m$ ($\lambda$ pre-registered), re-rank to $\hat r_{\text{adj}}$. **H2 holds iff**

$$
K(\hat r_{\text{adj}},\hat r_{\text{true}}) \;<\; K(\hat r_{\text{open}},\hat r_{\text{true}})
$$

with a cluster-bootstrap CI on the difference excluding 0. In words: *the cheap label-free signal recovers the true safety ranking better than the field's open-loop metric.*

### 4.5 Information-theoretic effect size
As a model-free complement, estimate the mutual information between signal and failure, $I(g;Y)$, via the KSG estimator. A signal is informative only if $I(g;Y)$ is significantly above its permutation null.

---

## 5. Statistical protocol (fixed before results)

- **Resampling — cluster bootstrap.** Resample at the **drive** level ($\mathrm{drive}(s)$), not the segment level, so correlated segments from one drive do not inflate confidence. $B=10{,}000$ replicates for all CIs.
- **Null model — permutation.** For each $\rho_j$ and $I(g_j;Y)$, build the null by permuting the signal against the labels $P=10{,}000$ times; the $p$-value is the right-tail mass.
- **Multiple comparisons.** We test 4 signals across several metrics. Control the false discovery rate with **Benjamini–Hochberg** at $q=0.05$ across the full pre-registered test family.
- **Cross-validation.** The supervised fusion uses **drive-grouped** K-fold CV (GroupKFold) so no drive appears in both train and test.
- **Power / sample size.** Using the Fisher $z$ transform, the segments needed to detect a true Spearman $\rho$ at $\alpha=0.05$, power $1-\beta=0.8$:

$$
n \;\approx\; \left(\frac{z_{1-\alpha/2}+z_{1-\beta}}{\operatorname{arctanh}\rho}\right)^{2} + 3 .
$$

The chosen slice size $n$ and the minimum detectable $\rho$ are recorded in `PREREGISTRATION.md` *before* any result is computed.

---

## 6. Decision rules (pre-registered, binding)

| Hyp. | Test | Confirm if | Honest null outcome |
|---|---|---|---|
| **H1** | §4.1 Spearman $\rho_j$, BH-corrected | any $j$: $\rho_j \ge 0.3$, $q<0.05$ | report which signals carry no information |
| **H2** | §4.4 Kendall-distance difference, cluster bootstrap | CI of $K_{\text{open}}-K_{\text{adj}} > 0$ | open-loop metric is not improvable by these signals |
| **H3** | §4.2/§4.3 AP and E-AURC vs base rate / oracle | AP $>\bar Y$ and E-AURC $<$ random, $q<0.05$ | signal does not triage better than chance |

A null result on all three is a **publishable, honest** finding: it tells the field which cheap signals do *not* substitute for human raters on the long tail. We report it unmodified. The integrity is the product.

---

## 7. References (method provenance)

- Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
- Gal, Ghahramani. *Dropout as a Bayesian Approximation.* ICML 2016.
- Kuhn, Gal, Farquhar. *Semantic Uncertainty / Semantic Entropy.* ICLR 2023.
- Geifman, El-Yaniv. *Selective Classification (risk–coverage, AURC).* NeurIPS 2017.
- Kraskov, Stögbauer, Grassberger (KSG). *Estimating Mutual Information.* Phys. Rev. E 2004.
- Benjamini, Hochberg. *Controlling the False Discovery Rate.* JRSS-B 1995.
- WOD-E2E / Rater Feedback Score: arXiv 2510.26125 (CVPR 2026).
- Open-loop vs closed-loop ranking inversions: arXiv 2605.00066.
