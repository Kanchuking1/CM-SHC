# SSHC — Submission Notes

This file describes how to regenerate the report's central figure (the I→T / T→I
symmetry scatter across the seven trained models) and how to run the per-image
hash + image-to-text demo locally.

The trained model checkpoints and the eval JSONs are not in the GitHub
repository. They are packaged into two zips on Google Drive:

At the drive location (https://drive.google.com/drive/folders/1ghr-GAW630ikniZd2wDDnI-IO0Q6nALj) the following zips can be found.

| Archive | Contents | Size | Need |
| --- | --- | --- | --- |
| `results.zip`     | The 22 eval JSONs | ~17KB  | Needed to regenerate graphs | 
| `checkpoints.zip` | The 7 model checkpoints used in the report (`epoch_XXXX.pt`) | ~3.2 GB | Needed to test retrieval |
| `centers.zip` | The centers used for different CSQ and SHC | ~1.8MB | Needed to test retrieval |

---

## 1. Repository setup

`pip install -e .` reads `pyproject.toml` and installs `torch`, `torchvision`,
`transformers`, `peft`, `omegaconf`, `huggingface_hub`, `numpy`, `pandas`,
`Pillow`, `matplotlib`. Python 3.10+ is required.

A CUDA GPU with ~6 GB of memory is enough for the demo (one CLIP+LoRA
model at a time). The plotting step is CPU-only.

---

## 2. Regenerate the symmetry scatter plot

The scatter plot reads from the eval JSONs only — no GPU or model loading
needed.

### 2a. Download and unzip results

After unzipping results.zip in ./experiments, the directory should contain:

```
experiments/results/
├── eval_alexnet_mlp_e0500.json
├── eval_cmshc_clip_frozen_mirflickr25k_128bit_e0200.json
├── eval_cmshc_clip_lora_mirflickr25k_128bit_e0200.json
├── eval_cmshc_mirflickr25k_128bit_clf_e0200.json
├── eval_dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30_e0200.json
├── eval_dcmh_clip_frozen_mirflickr25k_128bit_e0200.json
├── eval_dcmh_clip_lora_mirflickr25k_128bit_e0200.json
└── ... (full λc-sweep + diagnostic JSONs)
```

### 2b. Regenerate the figure

```bash
python scripts/plot_symmetry_scatter.py
```

Output: `experiments/results/symmetry_scatter_by_backbone.png`.

The same command also writes the figure used in the report (Figure 1).

---

## 3. Run the per-image hash + image-to-text demo

This loads each of the seven trained models, encodes a single query image,
also encodes the 24 MIR-Flickr-25k label names through each model's text
tower, and renders one matplotlib figure with all seven hashes plus the
top-3 nearest text labels per model.

### 3a. Download and unzip checkpoints

After unzipping checkpoints.zip and centers.zip in ./experiments, the directory should contain seven sub-folders under
`experiments/checkpoints/`:

```
experiments/checkpoints/
├── alexnet_mlp_dcmh_mirflickr25k_128bit/                                  epoch_0500.pt
├── cmshc_mirflickr25k_128bit_clf_cm_shc_mirflickr25k_128bit/              epoch_0200.pt
├── dcmh_clip_frozen_mirflickr25k_128bit_dcmh_mirflickr25k_128bit/         epoch_0200.pt
├── cmshc_clip_frozen_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit/      epoch_0200.pt
├── dcmh_clip_lora_mirflickr25k_128bit_dcmh_mirflickr25k_128bit/           epoch_0200.pt
├── cmshc_clip_lora_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit/        epoch_0200.pt
└── dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30_dcmh_anchored_mirflickr25k_128bit/  epoch_0200.pt
```

### 3b. Make sure MIR-Flickr-25k's BoW vocab is reachable

The two AlexNet+BoW models hash *text* through a 1386-word vocabulary. The
file `doc/common_tags.txt` from the MIR-Flickr-25k distribution must be
reachable at the path set in `configs/dataset/mirflickr25k.yaml` (or via
the `MIRFLICKR_ROOT` environment variable). If you only want to look at the
five CLIP-based models, this dependency does not apply to them.

### 3c. Run the demo

The first CUDA run downloads CLIP ViT-B/32 (~600 MB) into the standard
HuggingFace cache. Subsequent runs reuse it.



```bash
python scripts/retrieval_demo.py --image path/to/your_query.jpg
```

Flags:

* `--top-k 3`            (top-K nearest text labels per model; default 3)
* `--device cpu`         (force CPU; defaults to CUDA if available, Untested)
* `--output demo.png`    (also save the figure to disk)
* `--offline`            (use only a pre-populated `model_cache/` for CLIP weights;
                         omit on a machine with internet — the script will
                         download `openai/clip-vit-base-patch32` on first run
                         and cache it normally)

### 3d. What the figure shows

* The query image, full width across the top.
* Seven rows, one per model, each showing:
  * the model label and a 32-character hex digest of the 128-bit hash,
  * the hash visualised as a 4×32 black/white grid,
  * the top-3 candidate text labels ranked by Hamming distance to the
    image hash (lower is closer; 64 is random).

CM-SHC variants — especially CM-SHC + CLIP-frozen and Anchored-DCMH at
λ_c = 3 on CLIP+LoRA — should produce noticeably tighter Hamming distances
to relevant labels than the DCMH variants. That asymmetry is the
inference-time mirror of the I↔T symmetry gap reported in Figure 1.
