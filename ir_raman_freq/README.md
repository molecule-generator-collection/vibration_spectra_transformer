# Frequency + IR + Raman model

振動数、IR強度、Raman強度を積み重ねて入力する3成分版です。リポジトリルートから
次のように実行します。

```bash
pipenv run python ir_raman_freq/train.py --epochs 100 --batch-size 64
pipenv run python ir_raman_freq/evaluate.py --split test
pipenv run python ir_raman_freq/predict.py --input spectra.csv --top-k 5
```

評価結果は `results/ir_raman_transformer/evaluation_<split>.json` に自動保存されます。

学習結果は既定で `results/ir_raman_transformer/` に保存されます。推論CSVには
`freq`、`IR`、`Raman` の3列が必要です。

学習時の既定値はinitial learning rateが `1e-4`、Dropoutが `0.0` です。変更する場合は
`--initial-learning-rate` と `--dropout` を指定します。

共通の学習・評価・推論ロジックは `scripts/` にあり、このディレクトリの各ファイルは
3成分版の入力と保存先を固定するエントリーポイントです。

## MMP1%で重原子数外挿を評価する

`prepare_mmp_evaluation.py` は `MMP1%.zip` を直接読み、重原子数ごとの評価テンソルと
`analysis_manifest.csv` を作成します。チェックポイントを渡すと、入力スペクトル長、
SMILES長、語彙サイズをチェックポイントから取得します。

まず size 10 の100件だけで、データと実行環境を確認します。

```bash
pipenv run python ir_raman_freq/prepare_mmp_evaluation.py \
  --input /absolute/path/to/MMP1%.zip \
  --checkpoint /absolute/path/to/best.pt \
  --output-dir data_diretory/mmp_evaluation_smoke \
  --min-heavy-size 10 \
  --max-heavy-size 10 \
  --max-samples-per-size 100

pipenv run python ir_raman_freq/evaluation.py \
  --checkpoint /absolute/path/to/best.pt \
  --data-dir data_diretory/mmp_evaluation_smoke \
  --output-dir results/ir_raman_transformer/mmp_evaluation_smoke \
  --device cuda:0 \
  --batch-size 64 \
  --save-predictions
```

全データを評価する場合は次のように実行します。`truncate` はチェックポイントの入力長を
超えたスペクトルを先頭から入力長までに切り詰めます。manifestと評価結果には、切り詰めの
有無と割合が記録されます。

```bash
pipenv run python ir_raman_freq/prepare_mmp_evaluation.py \
  --input /absolute/path/to/MMP1%.zip \
  --checkpoint /absolute/path/to/best.pt \
  --output-dir data_diretory/mmp_evaluation_truncate \
  --min-heavy-size 10 \
  --max-heavy-size 70 \
  --overflow-policy truncate

pipenv run python ir_raman_freq/evaluation.py \
  --checkpoint /absolute/path/to/best.pt \
  --data-dir data_diretory/mmp_evaluation_truncate \
  --output-dir results/ir_raman_transformer/mmp_evaluation_truncate \
  --min-heavy-size 10 \
  --max-heavy-size 70 \
  --device cuda:0 \
  --batch-size 64 \
  --beam-size 5 \
  --save-predictions
```

完全なスペクトルがモデル入力長以下の分子だけで厳密比較する場合は、別の出力先を指定して
`--overflow-policy reject` で準備します。

```bash
pipenv run python ir_raman_freq/prepare_mmp_evaluation.py \
  --input /absolute/path/to/MMP1%.zip \
  --checkpoint /absolute/path/to/best.pt \
  --output-dir data_diretory/mmp_evaluation_strict \
  --min-heavy-size 10 \
  --max-heavy-size 70 \
  --overflow-policy reject
```

評価はsizeディレクトリを1個ずつロードするため、全評価データを同時にメモリへ載せません。
結果は以下に保存されます。

- `evaluation.json`: 全体および重原子数別の全指標
- `evaluation_by_heavy_size.csv`: 重原子数別の比較表
- `evaluation_molecules.csv`: 全sizeを結合した分子別解析用データ
- `predictions/sizeN.csv`: 分子ごとの候補と指標（`--save-predictions`指定時）

指標はcanonical SMILESのTop-1/3/5完全一致率、Valid SMILES率、Morgan fingerprintの
Top-1 Tanimoto類似度、分子式一致率、ValidなTop-1予測における重原子数の平均絶対誤差、
スペクトル切り詰め率です。
GPUメモリが不足する場合は `--batch-size` を小さくしてください。

### 官能基が予測できているかを解析する

`--save-predictions` で作成した `evaluation_molecules.csv` は、既存の
`analyze_molecular_performance.py`へ直接渡せます。

```bash
pipenv run python scripts/analyze_molecular_performance.py \
  --input results/ir_raman_transformer/mmp_evaluation_truncate/evaluation_molecules.csv \
  --output-dir results/ir_raman_transformer/mmp_evaluation_truncate/molecular_analysis \
  --target top1_correct \
  --min-group-size 20
```

通常の完全一致率に対する官能基の影響に加えて、以下を出力します。

- `functional_group_recovery.csv`: 官能基別のTop-1 precision/recall/F1とTop-3/5 recall
- `functional_group_recovery_by_heavy_size.csv`: 上記指標の重原子数別集計
- `functional_group_recovery.png`: 官能基別Top-1/3/5 recallの比較図
- `molecule_analysis.csv`: 正解分子と予測分子の官能基フラグ・個数を追加した全分子表

ここでTop-3/5 recallは、「正解分子にその官能基がある場合に、上位3/5候補の少なくとも
1つにも同じ官能基が含まれる割合」です。全構造の完全一致率が低くても、例えばcarbonylや
nitrileなどの局所構造をスペクトルから回収できるかを評価できます。
各recall/precisionには95% Wilson信頼区間と対象分子数も出力されます。出現数の少ない
官能基や、陰性例が大部分を占める `presence_accuracy` だけでは結論を出さず、precision、
recall、信頼区間を合わせて確認してください。
