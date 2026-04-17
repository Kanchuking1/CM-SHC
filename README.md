# Cross-modal semantic hashing (CM-SHC lab layout)

Research codebase with separated **data / models / losses / training / indexing**, config-driven experiments, and pluggable hashing methods. Two methods are supported:

- **DCMH** (Jiang et al., CVPR 2017) — [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.pdf). End-to-end cross-modal hashing with an **AlexNet** (CNN-F) image encoder and a **2-layer MLP** on 1386-dim bag-of-words vectors for text, alternating-optimization trainer. Fully implemented and used as the baseline.
- **CM-SHC** (this project) — a cross-modal extension of *Semantic Hash Centers* (Chen et al., 2025) on top of the same backbones. Replaces DCMH's O(N²) pairwise similarity with data-driven semantic hash centers subject to the Gilbert-Varshamov bound. See [docs/cm_shc_plan.md](docs/cm_shc_plan.md) for the full design. **In progress** — Stages 1-2 (centers construction) are done; the joint trainer lands in the next iteration.

## Layout

See repository tree: `configs/` (experiments), `src/` (library), `data/` (not versioned), `experiments/` (logs, checkpoints, results), `scripts/`, `tests/`, `notebooks/`.

For how configs merge, where models and data load, and a training flow diagram, see [docs/architecture.md](docs/architecture.md).

## Data: MIR-Flickr-25k

Download and unpack the v3b archive (from [LIACS / Leiden](http://press.liacs.nl/mirflickr/)):

```bash
wget http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip
unzip mirflickr25k.zip
cd mirflickr
mkdir -p annotations
wget http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip
unzip mirflickr25k_annotations_v080.zip
```

This produces a `mirflickr/` directory containing 25,000 images (`im1.jpg` … `im25000.jpg`), `meta/tags/` with per-image Flickr tags, `doc/common_tags.txt` (the 1386-word BOW vocabulary), and `annotations/` with 24-class semantic labels.

Only images with at least one annotation are used during training (~24,581 of 25,000). The BOW text features are built automatically from `doc/common_tags.txt` + `meta/tags/tags{id}.txt`.

Point `dataset.root` in [`configs/dataset/mirflickr25k.yaml`](configs/dataset/mirflickr25k.yaml) at the unpacked directory. By default the config uses the relative path `../../Datasets/mirflickr`.

## Conda environment

Requires **Python ≥ 3.10** (see `pyproject.toml`). From the repository root:

```bash
# Create and activate (pick a name you like)
conda create -n cm-shc python=3.11 -y
conda activate cm-shc
```

Install **PyTorch** first so the wheel matches your **CPU or CUDA** driver. Use the selector at [pytorch.org](https://pytorch.org) and run the given `conda install ...` or `pip install ...` line (typical pattern):

```bash
# Example: CUDA 12.x on Linux (replace with the exact command from pytorch.org for your OS/CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install the rest of the project dependencies:

```bash
pip install -r requirements.txt
# Editable install (optional, uses pyproject.toml)
pip install -e .
```

Development tools (pytest):

```bash
pip install -e ".[dev]"
```

On **shared clusters**, prefer loading a **CUDA** module that matches the PyTorch build, then create this conda env in your home or project space and point SLURM jobs at `conda activate cm-shc`.

## Train (DCMH baseline)

From the repo root, with MIR-Flickr-25k data at the path configured in `configs/dataset/mirflickr25k.yaml`:

```bash
python -m src.pipelines.train --config configs/experiments/exp_dcmh_mirflickr25k.yaml
```

Or:

```bash
bash scripts/train.sh
```

Checkpoints and `run_config.json` are written under `experiments/checkpoints/<experiment_name>_.../`.

### Backbone selection

The image CNN and text encoder are **config-driven** via [`configs/model/dcmh.yaml`](configs/model/dcmh.yaml). The defaults match the paper (AlexNet + MLP on BOW):

```yaml
model:
  backbone:
    image: alexnet        # or resnet50
  text_feature_dim: 1386  # triggers MLP path; set null for HF transformer
```

To switch to a ResNet-50 image backbone with a HuggingFace transformer text encoder:

```yaml
model:
  backbone:
    image: resnet50
    text: huawei-noah/TinyBERT_General_4L_312D
  text_feature_dim: null
  freeze_text_encoder: false
```

When `text_feature_dim` is set (e.g. `1386`), the pipeline skips HuggingFace tokenizer loading entirely and feeds pre-computed BOW vectors through a 2-layer MLP (`1386 → 4096 (ReLU) → c`). When `text_feature_dim` is `null`, it loads the HF transformer specified in `backbone.text`.

## CM-SHC (in progress)

CM-SHC reuses DCMH's dataset / backbones / evaluation pipeline but swaps the training objective for a hash-center regression. The flow is:

1. **(Optional) Train a multi-label classifier** on the same MIR-Flickr training split. Used later to build the SHC-style "visually confusable" class similarity matrix (`S-clf`). Skip this step if you only want label co-occurrence (`S-cooc`) or data-agnostic CSQ centers.
2. **Build hash centers** — compute `S ∈ [0,1]^{C×C}`, solve for `H ∈ {-1,+1}^{q×C}` under the Gilbert-Varshamov constraint, and cache per-sample target codes.
3. **Train CM-SHC** (Day 4+) — image and text encoders jointly regress to the target codes plus a cross-modal consistency term.
4. **Evaluate** with the existing `src.pipelines.evaluate` (MAP protocol is unchanged).

### Stage 1 — Multi-label classifier (for `S-clf` only)

```bash
python -m src.pipelines.train_classifier \
    --config configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml \
    --epochs 10 \
    --output experiments/centers/mirflickr25k_classifier_probs.pt
```

Fine-tunes a ResNet18 (ImageNet init) with a 24-class sigmoid head, then dumps per-sample probabilities over the training split. Writes a `.json` summary alongside the `.pt` file for sanity-checking per-class prevalence.

### Stage 2 — Build the centers

Three similarity sources are supported; all three are covered by the ablation plan:

```bash
# Data-agnostic CSQ baseline (Hadamard rows; no classifier or labels needed)
python -m src.pipelines.build_centers \
    --config configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml \
    --method csq

# Label co-occurrence (no classifier needed)
python -m src.pipelines.build_centers --config ... --method cooccurrence

# SHC-style, driven by the classifier from Stage 1
python -m src.pipelines.build_centers --config ... \
    --method classifier \
    --classifier-probs experiments/centers/mirflickr25k_classifier_probs.pt
```

Outputs go to `experiments/centers/{dataset}_{method}_q{q}.pt` and contain `H` (the `(q, C)` ±1 centers), `T_train` (per-sample ±1 target codes via bit-wise majority vote), `S` (the similarity matrix, or None for CSQ), plus metadata. The Gilbert-Varshamov bound for `(q, C)` is computed via `src.hashing.gv_bound.gilbert_varshamov_distance`; for the paper settings `GV(64, 24) = 25` and `GV(128, 24) = 54`.

### Stage 3 — CM-SHC training (coming next)

Config stub: `configs/model/cm_shc.yaml` and an experiment YAML pointing at `model: cm_shc` + `dataset: mirflickr25k`. The `CMSHCTrainer` class and joint loss (central-BCE + log-cosh quantization + cross-modal consistency) land alongside a full 128-bit experiment config in the Day 4 iteration.

### SLURM (CM-SHC)

```bash
# 6h auxiliary run: train the classifier for S_clf
sbatch scripts/slurm/train_classifier_mirflickr25k.sbatch

# Main CM-SHC training and evaluation (once Stage 3 lands)
sbatch scripts/slurm/cmshc_mirflickr25k_128bit.sbatch
sbatch scripts/slurm/evaluate_cmshc_mirflickr25k_128bit.sbatch
```

All three mirror the existing DCMH sbatch templates (offline mode, `TORCH_HOME=model_cache/torch`, `CONFIG` env override).

## Evaluation and retrieval

Metrics assume a **paired** dataset: row `i` pairs one image with one text (annotations for MIR-Flickr-25k). Ground truth for query `i` is item `i` in the other modality. Hamming distance uses `sign` of the image and text embeddings.

**Evaluate** (Recall@K, MRR for image→text and text→image). Writes JSON under `experiments/results/` by default.

```bash
# Use the newest training checkpoint for this experiment config
python -m src.pipelines.evaluate --config configs/experiments/exp_dcmh_mirflickr25k.yaml --latest

# Or pass a checkpoint path explicitly
python -m src.pipelines.evaluate --config configs/experiments/exp_dcmh_mirflickr25k.yaml \
  --checkpoint experiments/checkpoints/<run_name>/epoch_0120.pt

# Optional: batch size, K list, custom JSON path
python -m src.pipelines.evaluate --config ... --latest --batch-size 64 --ks 1,5,10,100 --output my_metrics.json
```

Or: `bash scripts/eval.sh --config ... --latest`

**Retrieve** (print top-K matches for one query index):

```bash
python -m src.pipelines.retrieve --config configs/experiments/exp_dcmh_mirflickr25k.yaml --latest \
  --query-index 0 --top-k 5 --mode i2t

# Text (tags) query ranking images
python -m src.pipelines.retrieve --config ... --checkpoint path/to/epoch_0120.pt --query-index 3 --mode t2i
```

Use the same `model_cache` / offline setup as training if the cluster has no Hub access.

### Offline HPC (no internet on compute nodes)

1. On a machine **with** internet, from the repo root, download torchvision weights (AlexNet or ResNet-50, depending on your config) into `model_cache/` (see `paths.model_cache` in [`configs/base.yaml`](configs/base.yaml)). If using the HF transformer text path (`text_feature_dim: null`), also download Hugging Face snapshots:

   ```bash
   python -m src.pipelines.download_models --config configs/experiments/exp_dcmh_mirflickr25k.yaml
   ```

2. Copy the repo (including `model_cache/`) to the cluster, or store `model_cache` on shared filesystem and point `paths.model_cache` at it.

3. Set `paths.offline_mode: true` in config (default in base) or export `CM_SHC_OFFLINE=1` so training uses `local_files_only` and does not call the Hub.

4. Training sets `TORCH_HOME` to `model_cache/torch` for torchvision ImageNet weights. Optional: export the same in SLURM before `python -m src.pipelines.train`.

`model_cache/` can be large; it is listed in `.gitignore` by default.

> **Note:** With the paper defaults (AlexNet + BOW MLP), no HuggingFace downloads are needed — only the torchvision AlexNet weights.

### Resume training

If `training.resume` is true (see [`configs/base.yaml`](configs/base.yaml)), `train` loads the latest `epoch_*.pt` under the experiment run directory and continues. Use `--no-resume` to always start from pretrained weights only.

### SLURM (e.g. TAMU Grace)

From the repo root:

```bash
# DCMH (baseline)
sbatch scripts/slurm/dcmh_mirflickr25k_128bit.sbatch
sbatch scripts/slurm/evaluate_dcmh_mirflickr25k_128bit.sbatch

# CM-SHC auxiliary: fit the classifier used for S_clf (~6h wall)
sbatch scripts/slurm/train_classifier_mirflickr25k.sbatch

# CM-SHC main runs (training + evaluation at 128 bits)
sbatch scripts/slurm/cmshc_mirflickr25k_128bit.sbatch
sbatch scripts/slurm/evaluate_cmshc_mirflickr25k_128bit.sbatch
```

Edit `#SBATCH` lines in those files for your partition, wall time, and uncomment `module` / `conda` as needed. All three CM-SHC templates accept a `CONFIG=` environment override so they can drive ablation configs without duplication.

## Dependencies (summary)

- PyTorch + torchvision: install **before** `requirements.txt`, using [pytorch.org](https://pytorch.org) for your hardware.
- Everything else: `requirements.txt` or `pip install -e .`

## Tests

```bash
pytest
```

Covers the DCMH hashing utilities (`tests/test_hashing.py`, `tests/test_metrics.py`, `tests/test_models.py`, `tests/test_model_paths.py`) and the CM-SHC center machinery (`tests/test_centers.py` — GV bound, cosine / classifier similarity, SHC solver, CSQ Hadamard baseline, multi-label majority vote).
