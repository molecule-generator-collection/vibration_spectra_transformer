import pandas as pd
import torch
import ast
from constants import PROCESSED_DATA_DIRECTORY

df = pd.read_csv(PROCESSED_DATA_DIRECTORY / "molecule_data_raw.csv", index_col=0)


"""
lengthの分布を調べる
まずはまとめ
"""
freqs = [ast.literal_eval(i) for i in df["freqs"].to_list()]
heavy_size_list = df["heavy_size"].to_list()
length_list = []
for i in freqs:
    length_list.append(len(i))

# use matplot lib and save as png
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np


# num_of_more_than_225 = np.sum(np.array(length_list) > 225)

# plt.hist(length_list, bins=100)
# plt.title(f"length of freqs (225以上は{num_of_more_than_225}つ)")
# plt.xlabel("length")
# plt.ylabel("count")
# plt.savefig(PROCESSED_DATA_DIRECTORY / "length_of_freqs_all_5_25.png")


"""
5-25それぞれで
"""
from tqdm import tqdm 

for heavy_size in tqdm(range(5, 26)):
    # print(df[df["heavy_size"] == heavy_size])
    freqs = [ast.literal_eval(i) for i in df[df["heavy_size"] == heavy_size]["freqs"].to_list()]
    length_list = []
    for i in freqs:
        length_list.append(len(i))
    print()
    print("freqs")
    print(len(freqs))
    print(freqs)
    print("length")
    print(len(length_list))
    print(length_list)

    num_of_more_than_225 = np.sum(np.array(length_list) > 225)

    plt.hist(length_list, bins=100)
    plt.title(f"length of freqs (225以上は{num_of_more_than_225}つ)")
    plt.xlabel("length")
    plt.ylabel("count")
    plt.savefig(PROCESSED_DATA_DIRECTORY / f"length_of_freqs_{heavy_size}.png")



