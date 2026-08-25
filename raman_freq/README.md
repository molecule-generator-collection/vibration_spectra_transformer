# Frequency + Raman model

Raman強度と振動数のみを入力する2成分版です。学習結果は既定で
`results/raman_freq_transformer/` に保存され、3成分版やIR版を上書きしません。

```bash
pipenv run python raman_freq/train.py --epochs 100 --batch-size 64
pipenv run python raman_freq/evaluate.py --split test
pipenv run python raman_freq/predict.py --input spectra.csv --top-k 5
```

評価結果は `results/raman_freq_transformer/evaluation_<split>.json` に自動保存されます。

推論CSVに必要なスペクトル列は `freq` と `Raman` です。

学習時の既定値はinitial learning rateが `1e-4`、Dropoutが `0.0` です。変更する場合は
`--initial-learning-rate` と `--dropout` を指定します。
