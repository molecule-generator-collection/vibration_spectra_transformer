# Vibrational Spectra Transformer

IR・ラマンスペクトルから分子の SMILES を予測するTransformerモデルと、比較用の
Fully Connectedモデルです。パスはすべてリポジトリのルートを基準に解決されます。

## セットアップ

Python 3.12 を使用します。

Linux + NVIDIA GPU（CUDA 11.8）を対象にしています。`Pipfile` はCUDA 11.8版の
PyTorch配布元を使用します。

```bash
pipenv install --skip-lock
```

環境構築後は通常の `pipenv run` コマンドで実行できます。

```bash
pipenv run python scripts/prepare_data.py
pipenv run python ir_raman_freq/train.py --device cuda:0
```

## 1. 学習データの作成

同梱の `data_diretory/size1_all.pickle` ～ `size9_all.pickle` から、学習・検証・
テスト用テンソルを `data_diretory/processed/` に作成します。

```bash
pipenv run python scripts/prepare_data.py
```

動作確認用の小さなデータセットは次のように作れます。

```bash
pipenv run python scripts/prepare_data.py --max-samples 1000
```

入力・出力場所を変える場合も、相対パスはこのリポジトリのルート基準です。

```bash
pipenv run python scripts/prepare_data.py --input-dir GDB_pickle_all --output-dir data_diretory/my_dataset
```

## 2. 学習

```bash
pipenv run python ir_raman_freq/train.py --epochs 100 --batch-size 64
```

初期学習率とDropoutはコマンドラインから調整できます。Dropoutはencoder、decoder、
SMILES embeddingのすべてに適用されます。

Transformerモデルの既定値は次のとおりです。Frequency + IR + Raman、Frequency + IR、
Frequency + Ramanの3バージョンで共通です。

| 引数 | 内容 | 既定値 |
|---|---|---:|
| `--initial-learning-rate` (`--learning-rate`) | AdamWの初期Learning Rate | `1e-4` |
| `--dropout` | encoder、decoder、SMILES embeddingのDropout率 | `0.0` |

```bash
pipenv run python ir_raman_freq/train.py \
  --initial-learning-rate 5e-5 \
  --dropout 0.1 \
  --output-dir results/lr5e-5_dropout0.1
```

`--learning-rate` と `--initial-learning-rate` は同じ引数です。比較実験ではモデルを
上書きしないよう、設定ごとに異なる `--output-dir` を指定してください。実際に使用した
値は出力先の `run_config.json` と各チェックポイント内に保存されます。

既定では最良モデルが `results/ir_raman_transformer/best.pt` に保存されます。CUDAが
利用可能ならGPUが自動選択されます。GPUを明示する場合は `--device cuda:0` を指定します。

### 複数GPUでの学習

同一マシン上の複数GPUでは、PyTorchの`torchrun`を使います。例えばGPU 4枚で
Frequency + IR + Ramanモデルを学習する場合は次のとおりです。

```bash
pipenv run torchrun --standalone --nproc-per-node=4 \
  ir_raman_freq/train.py \
  --epochs 100 \
  --batch-size 32 \
  --device cuda
```

`--nproc-per-node`には使用するGPU枚数を指定します。`--batch-size`はGPU 1枚あたりの値で、
上の例の実効バッチサイズは`32 × 4 = 128`です。特定のGPUだけを使う場合は、可視GPUを
絞ってから起動します。

```bash
CUDA_VISIBLE_DEVICES=0,2 pipenv run torchrun --standalone --nproc-per-node=2 \
  ir_freq/train.py --batch-size 32 --device cuda
```

Frequency + RamanモデルとFully Connectedモデルも同じ方法で起動できます。

```bash
pipenv run torchrun --standalone --nproc-per-node=2 \
  raman_freq/train.py --batch-size 32 --device cuda

pipenv run torchrun --standalone --nproc-per-node=2 \
  -m fully_connected_model.train --batch-size 32 --device cuda
```

学習データはGPUごとに分割され、勾配は各stepで同期されます。検証と
チェックポイント保存はrank 0だけが実行します。チェックポイントの形式は単一GPU学習と
共通なので、既存の`evaluate.py`と`predict.py`をそのまま使用できます。なお、GPU枚数を
増やして実効バッチサイズが変わると最適なLearning Rateも変わる場合があります。

### Initial Learning RateとDropoutの自動最適化

4モデルへ同じハイパーパラメーターを与え、validation lossのモデル間平均が最小になる
1組をランダムサーチまたはグリッドサーチで探索できます。

グリッドサーチでは、Learning Rateを乗算刻み、Dropoutを加算刻みで指定します。例えば
Learning Rateを10倍刻み、Dropoutを0.1刻みで範囲全体を探索するには次のように実行します。

```bash
pipenv run python scripts/optimize_hyperparameters.py \
  --search-method grid \
  --models ir_raman ir raman \
  --epochs 20 \
  --learning-rate-min 1e-5 \
  --learning-rate-max 1e-3 \
  --learning-rate-grid-factor 10 \
  --dropout-min 0.0 \
  --dropout-max 0.3 \
  --dropout-grid-step 0.1 \
  --device cuda:0
```

この例のLearning Rateは `1e-5, 1e-4, 1e-3`、Dropoutは
`0.0, 0.1, 0.2, 0.3`となり、直積の12組をすべて試します。3モデルを指定しているため、
実際の学習回数は `12組 × 3モデル = 36回`です。範囲の上限が刻みと一致しない場合も、
上限値を最後の候補として含めます。グリッドサーチでは `--trials` は使用されません。

従来のランダムサーチを使う場合は次のように指定します。Learning Rateは対数一様分布、
Dropoutは一様分布から探索されます。

```bash
pipenv run python scripts/optimize_hyperparameters.py \
  --search-method random \
  --models ir_raman ir raman fully_connected \
  --trials 20 \
  --epochs 20
```

`--models`を省略した場合も4モデルすべてが対象です。各trialでは、各モデルの全epoch中で
最小のvalidation lossを取得し、その平均を共通の目的値にします。モデルを1つだけ指定すれば、
そのモデル専用の最適化としても利用できます。

```bash
pipenv run python scripts/optimize_hyperparameters.py \
  --models ir_raman \
  --trials 30 \
  --epochs 20 \
  --output-dir results/tuning_ir_raman
```

結果は既定で `results/hyperparameter_optimization/` に保存されます。

- `best_hyperparameters.json`: 最良のLearning Rate、Dropout、目的値
- `trials.csv` / `trials.json`: 全trialとモデル別validation loss
- `trial_XXXX/<model>/history.json`: epochごとの学習履歴
- `trial_XXXX/<model>/train.log`: 各学習プロセスのログ

探索では大量のファイル生成を避けるため、各trialの `best.pt` と `last.pt` は既定で削除します。
残す場合は `--keep-checkpoints` を指定してください。中断した同じ設定の探索は `--resume` で
再開できます。最適値を決めた後は、その値と本番用epoch数を各 `train.py` に指定して改めて
学習してください。

ある組み合わせでlossやgradientが非有限値になって学習が早期停止した場合、そのtrialは
`failed`として `trials.json` / `trials.csv` に理由とともに記録され、残りの探索は継続します。
詳細は該当trialの `train.log` で確認できます。

まずコマンドだけ確認する場合は学習を行わない `--dry-run` が利用できます。

```bash
pipenv run python scripts/optimize_hyperparameters.py --trials 2 --epochs 2 --dry-run
```

## 3. モデルの検証

```bash
pipenv run python ir_raman_freq/evaluate.py --split test
```

交差エントロピー損失、RDKitでcanonicalizeしたSMILESのTop-1・Top-3・Top-5
構造一致率、およびTop-1候補のvalid SMILES率を表示します。Top-3・Top-5候補は
ビームサーチで生成するため、Top-1のみの評価より時間がかかります。

評価結果は画面へ表示されるとともに、チェックポイントと同じディレクトリの
`evaluation_<split>.json` に自動保存されます。例えばtest splitの既定の保存先は
`results/ir_raman_transformer/evaluation_test.json` です。保存先を変更する場合は
`--output results/my_evaluation.json` を指定します。

### 最終モデルの分子特性・官能基別解析

通常のハイパーパラメーター探索用データと区別するため、最終解析時だけ
`prepare_data.py` に `--analysis` を指定します。各splitのテンソルと同じ行順で、元pickleの
位置、SMILES、分子特性を記録した `analysis_manifest.csv` が追加保存されます。

```bash
pipenv run python scripts/prepare_data.py --analysis
pipenv run python ir_raman_freq/evaluate.py --split test --analysis
pipenv run python scripts/analyze_molecular_performance.py \
  --input results/ir_raman_transformer/evaluation_test_molecules.csv
```

解析には `dipole[3]`、`vip[0]`、`vea[0]`、`homolumo[0]`、`polar_aniso`、
`polar_iso`、`deen` を使用します。さらにRDKit記述子とSMARTSによる官能基を計算し、
数値特性との相関、重原子数で調整したロジスティック回帰係数、官能基あり・なしの
正解率と95%信頼区間、Fisher検定、多重検定補正、図を出力します。

解析結果は既定で
`results/ir_raman_transformer/evaluation_test_molecules_analysis/` に保存されます。
分子の結合にはSMILESを使わず、`size*.pickle` のファイル名と元行番号を使います。
評価時にはマニフェストと正解SMILESのcanonical構造が一致することも検証します。

## 4. 未知スペクトルから分子を同定

入力 CSV には `freq`, `IR`, `Raman` の3列が必要です。各セルは同じ長さの数値配列です。

```csv
sample_id,freq,IR,Raman
sample-1,"[650.0, 712.3]","[0.2, 1.4]","[3.1, 0.8]"
```

```bash
pipenv run python ir_raman_freq/predict.py --input spectra.csv --top-k 5
```

`predicted_smiles` に第一候補、`candidates_json` に上位候補と平均対数確率を保存します。
学習時の前処理（単位、計算条件、強度のスケーリング）は推論入力にも揃えてください。

各コマンドの全オプションは `--help` で確認できます。

## 5. モデル別ディレクトリ

3つの入力構成を個別のディレクトリに整理しています。いずれもリポジトリルートから
実行し、共通のTransformer実装と `data_diretory/processed/` の学習データを使用します。
2成分版はencoderの入力次元も3から2へ変更されるため、未使用スペクトルをゼロ埋めする
方式ではありません。

| バージョン | 実行ディレクトリ | 既定のモデル保存先 |
|---|---|---|
| Frequency + IR + Raman | `ir_raman_freq/` | `results/ir_raman_transformer/` |
| Frequency + IR | `ir_freq/` | `results/ir_freq_transformer/` |
| Frequency + Raman | `raman_freq/` | `results/raman_freq_transformer/` |

```bash
# Frequency + IR + Raman
pipenv run python ir_raman_freq/train.py --epochs 100 --dropout 0.1
pipenv run python ir_raman_freq/evaluate.py --split test

# Frequency + IR
pipenv run python ir_freq/train.py --epochs 100 --dropout 0.1
pipenv run python ir_freq/evaluate.py --split test

# Frequency + Raman
pipenv run python raman_freq/train.py --epochs 100 --dropout 0.1
pipenv run python raman_freq/evaluate.py --split test
```

学習結果、履歴、設定、予測CSVはモデルごとに別々の保存先へ出力されます。`scripts/` は
データ準備と各モデルから共有される内部実装の置き場所です。従来の
`python scripts/train.py` なども後方互換のため引き続き実行できますが、通常は上記の
モデル別エントリーポイントを使用してください。

## 6. Fully Connected比較モデル

比較用の非自己回帰Fully Connectedモデルは
[`fully_connected_model/`](fully_connected_model/) にあります。Frequency、IR、Ramanの
3成分をpadding mask適用後に結合・flattenし、全結合encoderとdecoderでSMILESの全位置を
同時に予測します。Attention、RNN、CNNは使用しません。

Transformerモデルと同じ `data_diretory/processed/` を使用し、リポジトリルートから
次のように実行します。

```bash
# 学習
pipenv run python -m fully_connected_model.train \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --dropout 0.1

# canonical Top-1・Top-3・Top-5評価
pipenv run python -m fully_connected_model.evaluate --split test

# CSVから予測
pipenv run python -m fully_connected_model.predict \
  --input spectra.csv \
  --top-k 5
```

既定のモデル保存先は `results/fully_connected_baseline/` です。推論CSVには `freq`、
`IR`、`Raman` の3列が必要です。主な調整可能パラメータは次のとおりです。
評価結果は既定で `results/fully_connected_baseline/evaluation_<split>.json` に保存されます。

| 引数 | 内容 | 既定値 |
|---|---|---:|
| `--hidden-dimension` | 全結合隠れ層の次元 | 256 |
| `--latent-dimension` | 潜在ベクトルの次元 | 128 |
| `--encoder-layers` | encoder層数 | 3 |
| `--decoder-layers` | decoder層数 | 3 |
| `--dropout` | Dropout率 | 0.1 |
| `--learning-rate` | initial learning rate | 1e-4 |

評価ではTransformer版と同様に、RDKitでcanonicalizeしたSMILESのTop-1・Top-3・Top-5
構造一致率、Top-1 valid SMILES率、cross entropyを出力します。モデル構造や追加オプションの
詳細は [Fully ConnectedモデルのREADME](fully_connected_model/README.md) を参照してください。
