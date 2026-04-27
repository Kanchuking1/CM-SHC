# SOTA Cross-Modal Hashing — Reading List

A prioritized list of papers relevant to extending the CM-SHC / Anchored-DCMH
project. Grouped by tier of relevance to the current draft, with publication
venues and year tags so you can prioritize.

## Tier 1 — Direct competitive baselines (CLIP-era cross-modal hashing)

These are post-2022 methods using CLIP backbones on MIR-Flickr-25k / NUS-WIDE /
MS-COCO. A reviewer at any A-tier venue will expect comparison against several
of them.

| Acronym | Title | Venue | Year | Why it matters |
| ------- | ----- | ----- | ---- | -------------- |
| **DCMHT**     | Differentiable Cross-Modal Hashing via Multimodal Transformers | ACM MM | 2022 | First widely-cited CLIP-based cross-modal hashing baseline. **Read first.** |
| **MITH**      | Multi-Granularity Interactive Transformer Hashing | ACM MM | 2023 | CLIP backbone + transformer decoder + multi-granularity loss. Strong recent baseline. |
| **DSPH**      | Deep Semantic-Aware Proxy Hashing for Multi-Label Cross-Modal Retrieval | IEEE TCSVT | 2024 | Proxy/center-based on CLIP — **most directly comparable to CM-SHC.** |
| **DNpH / OUR-DNPH** | Deep Neighborhood-Preserving Hashing With Quadratic Spherical Mutual Information / Deep Neighborhood-aware Proxy Hashing | IEEE TMM / ACM TOMM | 2024 | Same lab as DSPH; proxy-based variants. |
| **DIMCH**     | Cross-modal Hashing via Diverse Instances Matching | — | 2024 | Recent CLIP-based instance-matching approach. |
| **VTPH**      | [Enhancing Cross-Modal Retrieval via Visual-Textual Prompt Hashing](https://www.ijcai.org/proceedings/2024/0069.pdf) | IJCAI | 2024 | Prompt-learning on CLIP. **High relevance to LoRA story.** |
| **EGATH**     | [An End-to-End Graph Attention Network Hashing for Cross-Modal Retrieval](https://proceedings.neurips.cc/paper_files/paper/2024/file/03e7eaa586f0990c633f8a8e57e08ca6-Paper-Conference.pdf) | NeurIPS | 2024 | Most prestigious recent venue in this space. |
| **Strong-baseline-CLIP** | [When CLIP meets cross-modal hashing retrieval: A new strong baseline](https://www.sciencedirect.com/science/article/pii/S1566253523002841) | Information Fusion | 2024 | The "did you try the obvious thing" paper. |
| **CTCH**      | [Contrastive Transformer Cross-Modal Hashing for Video-Text Retrieval](https://www.ijcai.org/proceedings/2024/0136.pdf) | IJCAI | 2024 | Video-text but transferable framing. |
| **LCDH**      | [Lightweight Contrastive Distilled Hashing for Online Cross-modal Retrieval](https://arxiv.org/html/2502.19751v1) | arXiv | Feb 2025 | CLIP feature fusion with attention; recent fast-moving target. |
| **CMH-CLIP**  | [CLIP Multi-modal Hashing for Multimedia Retrieval](https://arxiv.org/html/2410.07783v1) | arXiv | Oct 2024 | Direct CLIP+hashing study. |

**Practical tip.** The [kalenforn/clip-based-cross-modal-hash](https://github.com/kalenforn/clip-based-cross-modal-hash)
GitHub repo consolidates implementations of DCMHT, MITH, DSPH, DNPH, TwDH, and
DIMCH in a common framework. Cloning that and running the lot on your splits
would close most of the "no modern baselines" gap in one weekend of GPU time.

## Tier 2 — Center-based hashing (your direct methodological lineage)

Make sure your Related Work treats these carefully.

| Acronym | Title | Venue | Year |
| ------- | ----- | ----- | ---- |
| **CSQ**           | Central Similarity Quantization for Efficient Image and Video Retrieval | CVPR | 2020 |
| **OrthoHash**     | One Loss for All: Deep Hashing with a Single Cosine Similarity-based Learning Objective | NeurIPS | 2021 |
| **MDS**           | Maximum-Distance-Separated Hashing | — | 2023 |
| **SHC**           | [Deep Hashing with Semantic Hash Centers for Image Retrieval](https://arxiv.org/html/2507.08404) | ACM TOIS | 2025 |
| **CCDH**          | [Codebook-Centric Deep Hashing: End-to-End Joint Learning of Semantic Hash](https://www.arxiv.org/pdf/2511.12162) | arXiv | 2025 |
| **Learnable CSQ** | [Learnable Central Similarity Quantization for Efficient Image and Video Retrieval](https://pubmed.ncbi.nlm.nih.gov/38090871/) | IEEE TNNLS | 2023 |

## Tier 3 — Useful context / framing

| Type | Title | Venue | Year |
| ---- | ----- | ----- | ---- |
| Survey      | [Cross-Modal Retrieval: A Systematic Review of Methods and Future Directions](https://arxiv.org/html/2308.14263v3) | arXiv | 2023/24 |
| Survey      | Multi-Modal Hashing for Efficient Multimedia Retrieval | IEEE TKDE | 2024 |
| Survey      | A Survey on Deep Hashing Methods | ACM TKDD | 2022 |
| Method      | CLIP4Hashing: Unsupervised Deep Hashing for Cross-Modal Video-Text Retrieval | ICMR | 2022 |
| Method      | PC-CLIP: Finetuning CLIP to Reason about Pairwise Differences | TMLR | 2025 |
| Method      | Semantic decomposition and enhancement hashing for deep cross-modal retrieval | Pattern Recognition | 2025 |
| Method      | Proxy-Based Graph Convolutional Hashing | IEEE Big Data | 2024 |

## Tier 4 — PEFT / fine-tuning fragility (not hashing-specific)

Useful if you lean harder on the "LoRA distorts CLIP" framing in the Discussion.

| Acronym | Title | Venue | Year |
| ------- | ----- | ----- | ---- |
| **LP-FT**    | Fine-tuning can distort pretrained features and underperform out-of-distribution (Kumar et al.) | ICLR | 2022 |
| **FLYP**     | Finetune like you pretrain: Improved finetuning of zero-shot vision models (Goyal et al.) | CVPR | 2023 |
| **WiSE-FT**  | Robust fine-tuning of zero-shot models (Wortsman et al.) | CVPR | 2022 |
| **Surgical FT** | Surgical fine-tuning improves adaptation to distribution shifts (Lee et al.) | ICLR | 2023 |
| **LoRA**     | LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.) | ICLR | 2022 |

## Suggested reading order (one weekend)

1. **DSPH** — closest center-based CLIP hashing competitor; read with your CM-SHC method section open.
2. **VTPH** — closest "what to tune on CLIP" framing; adjacent to your LoRA story.
3. **EGATH** — NeurIPS 2024; sets the bar for what the field considers publishable.
4. **MITH** — transformer hashing baseline; appears in every recent comparison table.
5. **The 2023 cross-modal retrieval survey** — for canonical citations and to avoid missing methods.

## All sources

- [CLIP-based fusion-modal reconstructing hashing](https://link.springer.com/article/10.1007/s13735-023-00268-7)
- [When CLIP meets cross-modal hashing retrieval: A new strong baseline](https://www.sciencedirect.com/science/article/pii/S1566253523002841)
- [CCUH: CLIP-Based Clustering Method for Unsupervised Hashing](https://link.springer.com/content/pdf/10.1007/978-981-96-7005-5_7.pdf)
- [LCDH: Lightweight Contrastive Distilled Hashing](https://arxiv.org/html/2502.19751v1)
- [EGATH: End-To-End Graph Attention Network Hashing (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/03e7eaa586f0990c633f8a8e57e08ca6-Paper-Conference.pdf)
- [VTPH: Visual-Textual Prompt Hashing (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0069.pdf)
- [Contrastive Transformer Cross-Modal Hashing (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0136.pdf)
- [CLIP Multi-modal Hashing for Multimedia Retrieval](https://arxiv.org/html/2410.07783v1)
- [CLIP4Hashing (ICMR 2022)](https://dl.acm.org/doi/10.1145/3512527.3531381)
- [DSPH GitHub](https://github.com/QinLab-WFU/DSPH)
- [DNpH GitHub](https://github.com/QinLab-WFU/DNpH)
- [OUR-DNPH GitHub](https://github.com/QinLab-WFU/OUR-DNPH)
- [kalenforn CLIP cross-modal hash repo](https://github.com/kalenforn/clip-based-cross-modal-hash)
- [Deep Hashing with Semantic Hash Centers (SHC, TOIS 2025)](https://arxiv.org/html/2507.08404)
- [Codebook-Centric Deep Hashing (2025)](https://www.arxiv.org/pdf/2511.12162)
- [Learnable CSQ (TNNLS 2023)](https://pubmed.ncbi.nlm.nih.gov/38090871/)
- [Cross-Modal Retrieval Survey](https://arxiv.org/html/2308.14263v3)
- [A Survey on Deep Hashing Methods (ACM TKDD 2022)](https://dl.acm.org/doi/10.1145/3532624)
- [Original DCMH (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.pdf)
