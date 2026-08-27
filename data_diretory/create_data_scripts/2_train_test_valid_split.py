from sklearn.model_selection import train_test_split
import pandas as pd
from constants import PROCESSED_DATA_DIRECTORY 

df = pd.read_csv(PROCESSED_DATA_DIRECTORY + "/molecule_data_raw.csv", index_col=0)
train_molecule_indices, test_molecule_indices = train_test_split(df.index.to_list(), test_size=0.1, random_state=42, stratify=df["heavy_size"])
train_molecule_indices, valid_molecule_indices = train_test_split(train_molecule_indices, test_size=0.1 *(10/9), random_state=42, stratify=df.loc[train_molecule_indices, "heavy_size"])
print(train_molecule_indices)
df["split"] = "unknow"
df.loc[train_molecule_indices, "split"] = "train"
df.loc[valid_molecule_indices, "split"] = "valid"
df.loc[test_molecule_indices, "split"] = "test"
df.to_csv(PROCESSED_DATA_DIRECTORY / "molecule_data_raw.csv")
