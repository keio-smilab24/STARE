<div align="center">

# STARE: Scene Text-Aware multimodal Retrieval for Everyday objects

**ACCV 2026 Workshop on Multimodal AI for Science and the Real World (MM4SciReal)**

[Kento Tokura](https://orcid.org/0009-0009-4524-9022),
[Ryosuke Korekata](https://orcid.org/0009-0008-0354-9070),
[Daichi Yashima](https://orcid.org/0009-0005-2087-2038),
[Kei Katsumata](https://orcid.org/0009-0009-7078-457X),
[Kanon Amemiya](https://orcid.org/0009-0001-9655-7561),
[Miyu Goko](https://orcid.org/0009-0009-9233-7121),
[Komei Sugiura](https://orcid.org/0000-0002-0261-0510)

Keio University

<p>
  <a href="https://stare-projectpage.pages.dev/"><img src="https://img.shields.io/badge/Project-Page-1f72b8?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv (coming soon)">
  <img src="https://img.shields.io/badge/Dataset-Coming%20Soon-f2a900?style=for-the-badge&logo=databricks&logoColor=white" alt="Dataset (coming soon)">
</p>
<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.6">
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white" alt="CUDA 11.8">
  <img src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white" alt="uv">
</p>

<img src="assets/teaser.jpg" width="90%" alt="STARE teaser">

</div>

## 📖 Overview

We focus on service robots that retrieve and transport everyday objects based on free-form instructions that may refer to visual features and **scene text**.
This is challenging because it requires a multimodal understanding of the visual features and spatial relationships of objects, as well as scene text embedded in the environment.

**STARE** leverages visual and textual cues in response to free-form instructions:

- 🔤 **Crosslingual Visual Prompt** enhances scene text interpretation.
- 🔗 STARE models the complex relationships between **named entities** in the instructions and their corresponding objects.
- 📊 We evaluate STARE on two novel benchmarks, **GoGetIt** and **TextCaps-test**, where it outperforms all baseline methods, and achieves an **80%** success rate in zero-shot physical experiments.

<div align="center">
  <img src="assets/overview.jpg" width="95%" alt="STARE architecture">
</div>

## 🛠️ Installation

**Requirements**: CUDA 11.8 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/keio-smilab24/STARE.git
cd STARE
uv sync
```

> [!NOTE]
> This project is configured for CUDA 11.8. If you are using a different CUDA version, check the appropriate PyTorch index URL in [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) and update the `[[tool.uv.index]]` section in `pyproject.toml` accordingly.

## 📦 Data and Checkpoint

Download the dataset:

```sh
bash scripts/download_dataset.sh
```

Download the pretrained checkpoint:

```sh
bash scripts/download_pretrained_model.sh
```

## 🚀 Usage

### Train

```sh
bash scripts/train.sh
```

### Test

```sh
bash scripts/test.sh
```

### Inference server (optional)

The `start_server` mode extracts features from new images and instructions with third-party APIs. Set the following variables in a `.env` file before you run it:

```sh
OPENAI_API_KEY=<your OpenAI API key>
GOOGLE_APPLICATION_CREDENTIALS=<path to your Google Cloud Vision credentials JSON>
```

## 📊 Results

Mean ± standard deviation over five runs with different seed values [%].

<div align="center">

| Benchmark | Recall@5 ↑ | Recall@10 ↑ | Recall@20 ↑ |
| :-- | :--: | :--: | :--: |
| GoGetIt (RefText) | 90.2 ± 0.4 | 94.4 ± 0.4 | 96.9 ± 0.3 |
| GoGetIt (Instruction) | 89.8 ± 0.6 | 92.8 ± 0.6 | 95.6 ± 0.2 |
| TextCaps-test | 92.4 ± 1.3 | 95.4 ± 0.6 | 98.3 ± 0.4 |
| LTRRIE | 72.5 ± 0.8 | 86.6 ± 0.9 | 95.4 ± 0.9 |

</div>

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tokura2026stare,
  title     = {{STARE}: Scene Text-Aware multimodal Retrieval for Everyday objects},
  author    = {Kento Tokura and Ryosuke Korekata and Daichi Yashima and Kei Katsumata and Kanon Amemiya and Miyu Goko and Komei Sugiura},
  booktitle = {ACCV 2026 Workshop on Multimodal AI for Science and the Real World (MM4SciReal)},
  year      = {2026},
  note      = {Accepted, to appear}
}
```
