# Fully Connected Baseline

Transformer側の実装を変更せず、同じ処理済みデータから学習・評価する独立した比較モデルです。
EncoderとDecoderはLinear、LayerNorm、GELU、Dropoutだけで構成され、Attention、RNN、CNNは使用しません。

## モデル

1. `freq`、`IR`、`Raman` のpadding部分を0にして結合・flatten
2. Fully connected Encoderで固定長の潜在ベクトルへ変換
3. Fully connected Decoderで各SMILES位置の特徴量へ変換
4. 共有Linear層で各位置のトークンを独立に分類

非自己回帰モデルなので、全SMILES位置を同時に予測します。学習損失はTransformerと同じく、
paddingを無視したtoken-level cross entropyです。

## 学習

リポジトリルートから実行します。

```bash
pipenv run python -m fully_connected_model.train --epochs 100 --batch-size 64
```

既定の出力先は `results/fully_connected_baseline/` です。主な構造パラメータを変更できます。

```bash
pipenv run python -m fully_connected_model.train \
  --hidden-dimension 256 \
  --latent-dimension 128 \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --dropout 0.1
```

### Initial Learning Rate

初期Learning Rateは `--learning-rate` で指定できます。既定値は `1e-4` です。

```bash
pipenv run python -m fully_connected_model.train \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 5e-4
```

Learning Rateを比較するときは、チェックポイントや学習履歴が混ざらないように
`--output-dir` で実験ごとに出力先を分けてください。

```bash
pipenv run python -m fully_connected_model.train \
  --learning-rate 5e-4 \
  --output-dir results/fully_connected_baseline_lr5e-4
```

指定したLearning Rateは、出力先の `run_config.json` にも記録されます。

## 評価

```bash
pipenv run python -m fully_connected_model.evaluate --split test
```

Transformer版と同じcanonical Top-1/3/5 accuracy、valid SMILES率、cross entropyを出力します。
各位置の確率を組み合わせて、非自己回帰モデル用のTop-k系列を作成します。
結果は `results/fully_connected_baseline/evaluation_<split>.json` に自動保存されます。
保存先を変更する場合は `--output` を指定します。

## CSVから予測

```bash
pipenv run python -m fully_connected_model.predict \
  --input spectra.csv \
  --top-k 5 \
  --output results/fully_connected_predictions.csv
```

入力CSVの形式と前処理上の注意はTransformer版の `README.md` と同じです。
