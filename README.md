# STARE: Scene Text-Aware multimodal Retrieval for Everyday objects

Kento Tokura, Ryosuke Korekata, Daichi Yashima, Kei Katsumata, Kanon Amemiya, Miyu Goko, Komei Sugiura

Keio University

[[Project Page]](https://stare-projectpage.pages.dev/)

## Requirements

- CUDA 11.8
- uv (Python package manager)

**Note**: This project is configured for CUDA 11.8. If you are using a different CUDA version, please check the appropriate PyTorch index URL from [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) and update the `[[tool.uv.index]]` section in `pyproject.toml` accordingly.

## Setup
```sh
uv sync
```

## Prepare Dataset

Run the following script to download the dataset:

```sh
bash scripts/download_dataset.sh
```

Run the following script to download the pretrained checkpoint:

```sh
bash scripts/download_pretrained_model.sh
```

## Train

```sh
bash scripts/train.sh
```

## Test

```sh
bash scripts/test.sh
```

Expected results (mean ± standard deviation over five runs with different seed values) are as follows:
| [%] | Recall@5↑ | Recall@10↑ | Recall@20↑ |
| :--: | :--: | :--: | :--: |
| GoGetIt (RefText) | 90.2 ± 0.4 | 94.4 ± 0.4 | 96.9 ± 0.3 |
| GoGetIt (Instruction) | 89.8 ± 0.6 | 92.8 ± 0.6 | 95.6 ± 0.2 |
| TextCaps-test | 92.4 ± 1.3 | 95.4 ± 0.6 | 98.3 ± 0.4 |
| LTRRIE | 72.5 ± 0.8 | 86.6 ± 0.9 | 95.4 ± 0.9 |

## Citation

```bibtex
@inproceedings{tokura2026stare,
  title     = {{STARE}: Scene Text-Aware multimodal Retrieval for Everyday objects},
  author    = {Kento Tokura and Ryosuke Korekata and Daichi Yashima and Kei Katsumata and Kanon Amemiya and Miyu Goko and Komei Sugiura},
  booktitle = {ACCV 2026 Workshop on Multimodal AI for Science and the Real World (MM4SciReal)},
  year      = {2026},
  note      = {Accepted, to appear}
}
```
