import pickle
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).resolve().parent.parent.parent)))
from utils.tokenizers import SPETokenizerWrapper
from collections import deque
from typing import List
from tqdm import tqdm
import pandas as pd
from constants import ORIGINAL_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY
from rdkit import Chem

MAX_SIZE = 9
tokenizer = SPETokenizerWrapper()

heavy_size = deque()
molecule_indices: List[int] = deque()
smiles: List[str] = deque()
freqs: List[List[float]] = deque()
IRs: List[List[float]] = deque()
Ramans: List[List[float]] = deque()

for i in tqdm(range(1, MAX_SIZE + 1)):
    path =  str((Path(__file__).resolve().parent.parent)) + f"/size{i}_all.pickle"
    print(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    for j in range(len(data)):
        try:
            molecule_indice = data[j]["index"]
            smile = Chem.MolToSmiles(Chem.MolFromSmiles(data[j]["smiles"]))
            freq = data[j]["freq"]
            Ir = data[j]["IR"]
            Raman = data[j]["Raman"]
        except Exception as e:
            print(e)
            print(molecule_indice)
            continue

        try:
            tokenizer([smile])
        except Exception as e:
            print("smiles can't be tokenized error")
            print(e)
            print(smile)
            continue

        if freq[0] < 0:
            print("Imaginary frequency detected!")
            print(molecule_indice)
            print("freq", freq)
        else:
            heavy_size.append(i)
            molecule_indices.append(molecule_indice)
            smiles.append(smile)
            freqs.append(freq)
            IRs.append(Ir)
            Ramans.append(Raman)

molecule_indices = list(molecule_indices)
smiles = list(smiles)
freqs = list(freqs)
IRs = list(IRs)
Ramans = list(Ramans)
heavy_size = list(heavy_size)
index_list = deque()
for i in range(len(molecule_indices)):
    index_list.append((heavy_size[i], int(molecule_indices[i])))

df = pd.DataFrame(
    {   
        "data_index": index_list,
        "molecule_indices": molecule_indices,
        "smiles": smiles,
        "freqs": freqs,
        "IRs": IRs,
        "Ramans": Ramans,
        "heavy_size": heavy_size,
    }
)
df.to_csv(PROCESSED_DATA_DIRECTORY + "/molecule_data_raw.csv", index=False)
