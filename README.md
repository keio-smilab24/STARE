# STARE

## Setup

```sh
pyenv install 3.10.0
pyenv virtualenv 3.10.0 stare
pyenv activate stare
poetry install
```

## Prepare Dataset
Download from [here](https://storage.googleapis.com/corl_stare/data.tar)
```sh
aria2c -x10 -s10 -k1M https://storage.googleapis.com/corl_stare/data.tar
tar -xvf data.tar
```

## Train

```sh
poetry run python src/main.py train --epochs 6 --bs 128 --eval_metric r@10 --alpha 0.8 --lambda_neg 0.8 --gamma 0.5 --eval_start_epoch 0 --eval_freq 1
```

## Evaluation

- checkpoint is available at [here](https://storage.googleapis.com/corl_stare/stare.tar)

  ```sh
  aria2c -x10 -s10 -k1M https://storage.googleapis.com/corl_stare/stare.tar
  tar -xvf stare.tar
  ```

- test
  ```sh
  export CHECKPOINT_PATH=model/stare.pth
  poetry run python src/main.py test --infer_model_path ${CHECKPOINT_PATH}
  ```

Expected results are as follow:
| [%] | Recall@1↑ | Recall@5↑ | Recall@10↑ | Recall@20↑ |
| :--: | :--: | :--: | :--: | :--: |
| GoGetIt (RefText) | 71.3 | 91.1 | 95.0 | 97.3 |
| GoGetIt (Instruction) | 72.2 | 88.5 | 92.6 | 95.4 |
| TextCaps-test | 78.4 | 93.3 | 96.1 | 99.0 |
| LTRRIE | 26.6 | 72.1 | 87.1 | 94.8 |
