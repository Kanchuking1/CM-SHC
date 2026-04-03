# Cross-modal semantic hashing (CM-SHC lab layout)

Research codebase with separated **data / models / losses / training / indexing**, config-driven experiments, and pluggable hashing methods.

## Layout

See repository tree: `configs/` (experiments), `src/` (library), `data/` (not versioned), `experiments/` (logs, checkpoints, results), `scripts/`, `tests/`, `notebooks/`.

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

### SLURM (e.g. TAMU Grace)

From the repo root:

```bash
sbatch scripts/slurm/dcmh_flickr8k_128bit.sbatch
```

Edit `#SBATCH` lines in that file for your partition, wall time, and uncomment `module` / `conda` as needed.

## Dependencies

- `pip install -r requirements.txt` (install PyTorch from [pytorch.org](https://pytorch.org) if needed)
- Optional: `pip install -e ".[dev]"` for pytest

## Tests

```bash
pytest
```
