import pandas as pd
import torch
import ast
from constants import ORIGINAL_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY

df = pd.read_csv(PROCESSED_DATA_DIRECTORY + "/molecule_data_raw.csv", index_col=0)
train_df = df[df["split"] == "train"]
valid_df = df[df["split"] == "valid"]
test_df = df[df["split"] == "test"]
train_freqs = [ast.literal_eval(i) for i in train_df["freqs"].to_list()]
valid_freqs = [ast.literal_eval(i) for i in valid_df["freqs"].to_list()]
test_freqs = [ast.literal_eval(i) for i in test_df["freqs"].to_list()]

print("freq")
Max = -1
for i in train_freqs:
    if len(i) > Max:
        Max = len(i)
print("train max length")
print(Max)

Max = -1
for i in valid_freqs:
    if len(i) > Max:
        Max = len(i)
print("valid max length")
print(Max)

Max = -1
for i in test_freqs:
    if len(i) > Max:
        Max = len(i)
print("test max length")
print(Max)
print()

train_IRs = [ast.literal_eval(i) for i in train_df["IRs"].to_list()]
valid_IRs = [ast.literal_eval(i) for i in valid_df["IRs"].to_list()]
test_IRs = [ast.literal_eval(i) for i in test_df["IRs"].to_list()]

print("IR")
Max = -1
for i in train_IRs:
    if len(i) > Max:
        Max = len(i)
print("train max length")
print(Max)

Max = -1
for i in valid_IRs:
    if len(i) > Max:
        Max = len(i)
print("valid max length")
print(Max)

Max = -1
for i in test_IRs:
    if len(i) > Max:
        Max = len(i)
print("test max length")
print(Max)
print()

print("Ramana")
train_Ramans = [ast.literal_eval(i) for i in train_df["Ramans"].to_list()]
valid_Ramans = [ast.literal_eval(i) for i in valid_df["Ramans"].to_list()]
test_Ramans = [ast.literal_eval(i) for i in test_df["Ramans"].to_list()]

Max = -1
for i in train_Ramans:
    if len(i) > Max:
        Max = len(i)
print("train max length")
print(Max)

Max = -1
for i in valid_Ramans:
    if len(i) > Max:
        Max = len(i)
print("valid max length")
print(Max)

Max = -1
for i in test_Ramans:
    if len(i) > Max:
        Max = len(i)
print("test max length")
print(Max)
