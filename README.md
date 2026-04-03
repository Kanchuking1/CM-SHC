# Cross-modal semantic hashing (CM-SHC lab layout)

Research codebase with separated **data / models / losses / training / indexing**, config-driven experiments, and pluggable hashing methods.

## Layout

See repository tree: `configs/` (experiments), `src/` (library), `data/` (not versioned), `experiments/` (logs, checkpoints, results), `scripts/`, `tests/`, `notebooks/`.

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

From the repo root, with Flickr8k at `data/raw/flickr8k` (`Images/` + `captions.txt`) or override paths in `configs/dataset/flickr8k.yaml`:

```bash
python -m src.pipelines.train --config configs/experiments/exp_dcmh_flickr8k.yaml
```

Or:

```bash
bash scripts/train.sh
```

Checkpoints and `run_config.json` are written under `experiments/checkpoints/<experiment_name>_.../`.

### Offline HPC (no internet on compute nodes)

1. On a machine **with** internet, from the repo root, download Hugging Face snapshots and torchvision ResNet-50 weights into `model_cache/` (see `paths.model_cache` in [`configs/base.yaml`](configs/base.yaml)):

   ```bash
   python -m src.pipelines.download_models --config configs/experiments/exp_dcmh_flickr8k.yaml
   ```

2. Copy the repo (including `model_cache/`) to the cluster, or store `model_cache` on shared filesystem and point `paths.model_cache` at it.

3. Set `paths.offline_mode: true` in config (default in base) or export `CM_SHC_OFFLINE=1` so training uses `local_files_only` and does not call the Hub.

4. Training sets `TORCH_HOME` to `model_cache/torch` for ResNet ImageNet weights. Optional: export the same in SLURM before `python -m src.pipelines.train`.

`model_cache/` can be large; it is listed in `.gitignore` by default.

### Resume training

If `training.resume` is true (see [`configs/base.yaml`](configs/base.yaml)), `train` loads the latest `epoch_*.pt` under the experiment run directory and continues. Use `--no-resume` to always start from pretrained weights only.

### SLURM (e.g. TAMU Grace)

From the repo root:

```bash
sbatch scripts/slurm/dcmh_flickr8k_128bit.sbatch
```

Edit `#SBATCH` lines in that file for your partition, wall time, and uncomment `module` / `conda` as needed.

## Dependencies (summary)

- PyTorch + torchvision: install **before** `requirements.txt`, using [pytorch.org](https://pytorch.org) for your hardware.
- Everything else: `requirements.txt` or `pip install -e .`

## Tests

```bash
pytest
```
