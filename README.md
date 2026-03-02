# STARE

## Requirements

- CUDA 11.8
- uv (Python package manager)

**Note**: This project is configured for CUDA 11.8. If you are using a different CUDA version, please check the appropriate PyTorch index URL from [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) and update the `[[tool.uv.index]]` section in `pyproject.toml` accordingly.

## Setup
```sh
uv sync
```

## Prepare Dataset

Run the following script to download the dataset and checkpoint:

```sh
bash scripts/download_dataset.sh
```

## Train

```sh
bash scripts/train.sh
```

## Test

```sh
bash scripts/test.sh
```

Expected results (averaged over 5 runs with different seed values) are as follows:
| [%] | Recall@1↑ | Recall@5↑ | Recall@10↑ | Recall@20↑ |
| :--: | :--: | :--: | :--: | :--: |
| GoGetIt (RefText) | 71.6 | 90.2 | 94.4 | 96.9 |
| GoGetIt (Instruction) | 74.0 | 89.8 | 92.8 | 95.6 |
| TextCaps-test | 77.4 | 92.4 | 95.4 | 98.3 |
| LTRRIE | 25.8 | 72.5 | 86.6 | 95.4 |
