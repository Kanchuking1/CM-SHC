# Cross-modal semantic hashing (CM-SHC lab layout)

Research codebase with separated **data / models / losses / training / indexing**, config-driven experiments, and pluggable hashing methods. The default DCMH baseline replicates the architecture from [Jiang et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.pdf): an **AlexNet** (CNN-F) image encoder and a **2-layer MLP** on 1386-dim bag-of-words vectors for text.

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
# Training
sbatch scripts/slurm/dcmh_mirflickr25k_128bit.sbatch

# Evaluation (uses --latest checkpoint)
sbatch scripts/slurm/evaluate_dcmh_mirflickr25k_128bit.sbatch
```

Edit `#SBATCH` lines in those files for your partition, wall time, and uncomment `module` / `conda` as needed.

## Dependencies (summary)

- PyTorch + torchvision: install **before** `requirements.txt`, using [pytorch.org](https://pytorch.org) for your hardware.
- Everything else: `requirements.txt` or `pip install -e .`

## Tests

```bash
pytest
```
