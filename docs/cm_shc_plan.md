# CM-SHC implementation plan

**Project.** Algorithms for Big Data class project. Implement **CM-SHC** — a cross-modal extension of *Semantic Hash Centers* (Chen et al., 2025) on top of the existing DCMH scaffold — and benchmark it against the trained DCMH baseline on MIR-Flickr-25k.

**Baseline already in repo.** DCMH @ 128-bit on MIR-Flickr-25k: MAP I→T = **0.6647**, MAP T→I = **0.6602** (`experiments/results/eval_alexnet_mlp_e0500.json`).

---

## 1. Method design

### 1.1 What changes vs. DCMH

DCMH uses **pairwise** similarity `S_ij = 1[label_i · label_j > 0]` and learns `F` (image codes), `G` (text codes), `B` (shared binary) by maximizing `log P(S | F, G)` plus quantization and bit-balance. Complexity is O(N²) in similarity evaluations, and training signal is purely local.

CM-SHC replaces local pairwise supervision with a **global center signal**. Each class gets one hash center in `{-1,+1}^q`. A sample's target code is a label-weighted combination of its classes' centers. Both modalities regress to that target, and a cross-modal consistency term ties the two modalities together. Complexity becomes O(N · C). Training is also more stable because every sample sees the same global structure.

### 1.2 Three stages (direct port of SHC to cross-modal)

**Stage 1 — Class similarity matrix `S ∈ R^{C×C}`.**

MIR-Flickr-25k has 24 multi-label classes. Two drop-in options; I recommend implementing both so we can ablate:

- **(S-cooc) Label co-occurrence.** With `Y ∈ {0,1}^{N×C}` the training label matrix, let `S̃ = YᵀY / diag(YᵀY)^(1/2)` (cosine of column vectors) so `S̃ ∈ [0,1]^{C×C}`. Cheap, no extra model.
- **(S-clf) Classifier-based (SHC paper method).** Fine-tune a lightweight multi-label classifier (ResNet18 with sigmoid head) for ~5 epochs on the train split, run it over the training set, average per-class prediction vectors with the top-1 entry masked, then normalize + symmetrize. Captures "visually confusable" similarity, which is what SHC argues matters.

Either way we produce `S ∈ [0,1]^{C×C}` with `S_ii = 1` and `S_ij = S_ji`.

**Stage 2 — Semantic hash centers `H ∈ {-1,+1}^{q×C}`.**

One center per class. Let `R = 2S − 1 ∈ [-1, 1]` so that ideal inner products `hᵢᵀhⱼ / q` match `R_ij`. Solve:

```
min_H   (1/q)‖ R − (1/q) HᵀH ‖_F²           # semantic alignment
      + μ · Σ_{i≠j} (hᵢᵀhⱼ / q)²             # soft distance regularizer
s.t.   hᵢᵀhⱼ ≤ q − 2d    ∀ i ≠ j             # Gilbert–Varshamov bound
       hᵢ ∈ {−1, +1}^q
```

where `d` is the minimum Hamming distance implied by the GV bound for `(q, C)`. For `C = 24` classes at `q = 64`, the GV bound easily yields `d ≥ 16`.

**Solver.** Use alternating relaxation matching SHC Algorithm 1:

1. Relax `H` to `H̃ ∈ [-1, 1]^{q×C}`, penalize deviation from ±1 with `β‖H̃ − sign(H̃)‖²`.
2. Projected gradient descent on the unconstrained objective, projecting violating pairs back to the distance bound (clip inner product to `q − 2d`).
3. Round: `H = sign(H̃)`.

This is a one-time O(q·C²) optimization — fast, runs in seconds for C = 24.

**Stage 3 — Per-sample target codes & cross-modal training.**

For sample `i` with multi-hot label `y_i ∈ {0,1}^C`, compute the target code via **bit-wise majority vote** over its class centers (this is CSQ's multi-label rule and handles multi-label cleanly):

```
t_i = sign( H · y_i )   ∈ {−1, +1}^q       # tie-break randomly once, cached
```

Train two encoders `f_θ` (image) and `g_φ` (text) that each map to a continuous code in `R^q`. **Joint objective:**

```
L = λ_center · L_center
  + λ_Q      · L_quant
  + λ_CM     · L_cross_modal
  + λ_bal    · L_balance       (optional; DCMH bit-balance)
```

with

```
L_center      = BCE( σ(f_i), (t_i + 1)/2 ) + BCE( σ(g_i), (t_i + 1)/2 )      # per-sample central BCE (CSQ form)
L_quant       = Σ log cosh(| 2σ(f_i) − 1 | − 1) + same for g_i                # CSQ smooth quantization
L_cross_modal = ‖ tanh(f_i) − tanh(g_i) ‖²   (or DCMH pairwise NLL on batch)
L_balance     = ‖ f.sum(0) ‖² + ‖ g.sum(0) ‖²
```

Hashing codes at inference: `b_img = sign(f_θ(x))`, `b_txt = sign(g_φ(y))`. Because both modalities share a common center target, their codes are *directly comparable* — no `B` buffer, no alternating optimization, no O(N²) pairwise matrix. A single forward+backward pass per batch.

### 1.3 Why this is defensible as "Algorithms for Big Data" work

- **Complexity.** Drops training-time similarity evaluation from O(N²) (DCMH) to O(N·C) per epoch.
- **Information-theoretic structure.** Gilbert–Varshamov bound is a classical coding-theory result; discuss connection between hash codes and error-correcting codes in the write-up.
- **Discrete optimization.** Semantic center optimization is a discrete problem with combinatorial constraints; SHC's relaxation+projection is a non-trivial algorithm.
- **Empirical claim.** We expect MAP improvements over DCMH on MIR-Flickr-25k with the same backbones and the same bit budget.

---

## 2. File-level changes

### 2.1 New code (fill in existing stubs)

| Path | Role | Status |
|------|------|--------|
| `src/hashing/centers.py` | `build_class_similarity(labels, method)`, `optimize_semantic_centers(S, q, d, mu)`, `multi_label_target(Y, H)` | stub → implement |
| `src/hashing/gv_bound.py` *(new)* | `gilbert_varshamov_distance(q, C)` — returns largest feasible `d` | implement |
| `src/models/hashing/cm_shc.py` | `CMSHC` model: same backbones as DCMH, outputs `(f, g)` | stub → implement |
| `src/models/losses/semantic_center_loss.py` | Central BCE loss, quantization (log-cosh), cross-modal consistency | stub → implement |
| `src/core/trainer.py` | Add `CMSHCTrainer` class alongside `DCMHTrainer`; factor shared code into a base | extend |
| `src/pipelines/train.py` | `build_model` branches on `cfg.model.name` to return DCMH or CM-SHC; trainer selection likewise | extend |
| `configs/model/cm_shc.yaml` | Full config: bit_dim, lambda_* weights, mu, similarity_method, backbones | rewrite stub |
| `configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml` *(new)* | Points to `model: cm_shc` + `dataset: mirflickr25k` + 128-bit override | add |
| `configs/experiments/exp_cmshc_mirflickr25k_64bit.yaml` *(new)* | Same at 64 bits (ablation) | add |

### 2.2 Tests

- `tests/test_centers.py`: GV bound correctness on known `(q, C, d)` triples; center optimization returns a ±1 matrix whose pairwise distances meet the bound; `multi_label_target` shape and sign.
- `tests/test_cmshc_loss.py`: central BCE gradient pushes `σ(f)` toward `(t+1)/2`; quantization loss is zero when codes are saturated.

### 2.3 SLURM job scripts (`scripts/slurm/`)

One train + one eval script per experiment, mirroring `dcmh_mirflickr25k_128bit.sbatch` / `evaluate_dcmh_mirflickr25k_128bit.sbatch` (48h wall, 1 GPU, 128G RAM, `CM_SHC_OFFLINE=1`, `TORCH_HOME=model_cache/torch`). All take a `CONFIG=` env override so one script template can drive multiple configs when useful.

| Script | Drives |
|--------|--------|
| `cmshc_mirflickr25k_128bit.sbatch` | Train CM-SHC @ 128 bits (main run, S-clf centers). |
| `evaluate_cmshc_mirflickr25k_128bit.sbatch` | Evaluate the latest checkpoint for the 128-bit config. |
| `cmshc_mirflickr25k_64bit.sbatch` | Train CM-SHC @ 64 bits (bit-budget ablation). |
| `evaluate_cmshc_mirflickr25k_64bit.sbatch` | Eval counterpart to the 64-bit run. |
| `cmshc_mirflickr25k_128bit_cooc.sbatch` | Ablation run with S-cooc centers (override `CONFIG=` or add a dedicated experiment YAML). |
| `cmshc_mirflickr25k_128bit_csq.sbatch` | Ablation run with CSQ Hadamard centers. |
| `cmshc_mirflickr25k_128bit_nocm.sbatch` | Ablation dropping `L_cross_modal`. |
| `train_classifier_mirflickr25k.sbatch` *(optional, short run)* | Fits the ResNet18 multi-label classifier used to build `S-clf`; outputs `class_similarity.pt` under `experiments/`. Shorter wall (≤4h), same GPU spec. |

All scripts submit from the repo root (`cd "${SLURM_SUBMIT_DIR:-$PWD}"`), write logs to `experiments/logs/slurm_%j.out/err`, and keep the mail/module/conda blocks commented-out in the same place as the DCMH templates so cluster-specific lines stay consistent. Day 1 of the timeline already produces the first two scripts; the ablation scripts land on Day 9 alongside those runs.

### 2.4 Evaluation pipeline

`src/pipelines/evaluate.py` already encodes paired datasets and computes MAP under the DCMH protocol (`query_labels @ db_labels > 0` relevance). It should work with CM-SHC unchanged because the model exposes `encode_image` / `encode_text` returning continuous `q`-dim codes, then `binary_sign_codes` → `hamming_distance_matrix` → MAP. Just pass the CM-SHC config + checkpoint.

### 2.5 Keep the DCMH code untouched

We compare against the `epoch_0500.pt` checkpoint you already have. Don't retrain DCMH.

---

## 3. Training protocol

**Data.** MIR-Flickr-25k, same 2000 / 10000 / remainder split you're already using. Seed 42.

**Backbones.** To match DCMH apples-to-apples:
- Image: AlexNet (ImageNet-pretrained, final FC → `q`)
- Text: 2-layer MLP on 1386-dim BoW (same as DCMH)

**Bit budgets.** 128 (primary, matches baseline) and 64 (secondary, for bit-budget ablation).

**Hyperparameters (starting point, tune as needed).**
```yaml
model:
  name: cm_shc
  bit_dim: 128
  similarity_method: classifier       # or cooccurrence (ablation)
  lambda_center: 1.0
  lambda_quant:  0.1
  lambda_cross_modal: 1.0
  lambda_balance: 0.0                 # optional
  mu: 1.0                             # center optimization regularizer
  gv_d: auto                          # derived from (q, C)
  backbone:
    image: alexnet
    text:  MLP
  text_feature_dim: 1386

training:
  max_epochs: 120                     # much faster than DCMH's 500 — no alternating inner loops
  batch_size: 64
  lr_img: 0.0316
  lr_txt: 0.0316
  optimizer: SGD with cosine decay
```

**Expected wallclock.** ~3× faster per epoch than DCMH (single joint pass vs. two alternating passes). On the same cluster, 120 epochs of CM-SHC ≈ wallclock of ~80 epochs of DCMH.

---

## 4. Experiments and ablations

Minimum set for the report (all on MIR-Flickr-25k):

| # | Config | Purpose |
|---|--------|---------|
| 1 | DCMH @ 128 bits | Reported baseline (already run). |
| 2 | CM-SHC @ 128, S-cooc centers | Main comparison to DCMH. |
| 3 | CM-SHC @ 128, S-clf centers | Show semantic centers beat co-occurrence. |
| 4 | CM-SHC @ 128, CSQ centers (Hadamard) | Show semantic centers beat data-agnostic centers. |
| 5 | CM-SHC @ 64 | Bit-budget scaling. |
| 6 | CM-SHC @ 128 without `L_cross_modal` | Show CM term actually matters for cross-modal. |

**Metrics.** MAP I→T, MAP T→I, Recall@{1, 10, 100}, optional top-K precision curves. All under the existing DCMH evaluation protocol (label-overlap relevance). `src/pipelines/evaluate.py` already reports MAP; add Recall@K using `src/core/metrics.py::recall_at_k_hamming` if the report wants those plots.

---

## 5. Report outline (8–10 pages)

1. **Introduction** (≤1 page). Retrieval-over-huge-corpora motivation. Why hashing. Cross-modal angle. One paragraph on CM-SHC's contribution.
2. **Related work** (≤1 page). DCMH, CSQ, SHC, one line each on DAH / LCDH / MKDH for breadth.
3. **Background** (1 page). Hamming distance, binary codes as ECC, hash-center idea, Gilbert–Varshamov bound (cite the coding-theory result).
4. **Method** (2–3 pages).
   - Problem formulation.
   - Stage 1: class similarity.
   - Stage 2: semantic center optimization with GV constraint (pseudocode).
   - Stage 3: cross-modal training objective.
5. **Experiments** (2–3 pages). Dataset, setup, main table (DCMH vs. CM-SHC MAP @ 128/64 bits), ablations, representative retrieval examples (optional).
6. **Analysis** (1 page). Complexity comparison, convergence speed, qualitative inspection of center distances vs. class similarity.
7. **Conclusion + limitations** (≤½ page).

Deliver as a `.docx` per class convention. I'll generate it via the docx skill when the results are in.

---

## 6. Timeline (1–2 weeks)

| Day | Work |
|-----|------|
| 1 | `src/hashing/gv_bound.py`, `src/hashing/centers.py` scaffolding; unit tests. Add `scripts/slurm/cmshc_mirflickr25k_128bit.sbatch` + eval counterpart (even though we won't launch until Day 6). |
| 2 | `optimize_semantic_centers` solver; verify minimum-distance property; add both `S-cooc` and `S-clf` builders. |
| 3 | Multi-label classifier code + `scripts/slurm/train_classifier_mirflickr25k.sbatch` (~30 min run) producing `S-clf`. |
| 4 | `src/models/hashing/cm_shc.py` + `src/models/losses/semantic_center_loss.py`; wire to `train.py`. |
| 5 | `CMSHCTrainer` with full loss, debug on a small subset. |
| 6 | Launch run #2 (S-cooc @ 128) and #3 (S-clf @ 128) on HPC. |
| 7–8 | While runs execute: write Sections 1–3 of the report. |
| 9 | Add ablation SLURM scripts (`cmshc_mirflickr25k_128bit_cooc.sbatch`, `_csq.sbatch`, `_nocm.sbatch`, `cmshc_mirflickr25k_64bit.sbatch` + eval counterpart); launch ablations (#4 CSQ centers, #5 64-bit, #6 no-CM). |
| 10 | Collect results; fill main table and ablation tables. |
| 11 | Draft Section 4 (Method) and Section 5 (Experiments). |
| 12 | Draft Sections 6–7; generate retrieval figures. |
| 13 | Polish report; final `.docx` via docx skill. |
| 14 | Buffer for retries / last-minute bug fixes. |

**Two-week buffer budget.** Compute is the risk. If any HPC run fails or diverges, days 13–14 absorb the retry.

---

## 7. Open design questions (flag if you disagree)

1. **Similarity source for Stage 1.** I'm defaulting to classifier-based (S-clf) as the primary, with S-cooc as an ablation. If you want to skip training a classifier, we can flip them.
2. **Cross-modal term form.** `‖tanh(f) − tanh(g)‖²` is the simplest. Alternative: keep DCMH's pairwise NLL inside a batch. The NLL version is slightly more principled but reintroduces O(B²) per batch.
3. **Center re-optimization.** Fixed centers throughout training (paper default). An ablation could re-optimize centers every K epochs conditioning on learned features — mention as future work unless time permits.
4. **Bit budgets.** 128 (main) + 64 (ablation). Add 32 if you want a cleaner scaling plot.

Once you sign off on this plan I'll start with Stage 1 and Stage 2 code (days 1–3 above).
