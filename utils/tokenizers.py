import copy
import logging
import pickle as pkl
from collections import deque
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem as Chem
from torch import nn
from tqdm import tqdm
from typing import List, Union
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.tokenizer import SPE_Tokenizer
from pathlib import Path
import yaml

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data_diretory"
ATOM_JSON_PATH = DATA_DIRECTORY / "atom_vocab.json"
SPE_VOCAB_FILE_PATH = DATA_DIRECTORY / "SPE_ChEMBL.txt"
SPE_MAX_TOKENIZED_LENGTH = 30 + 2  # +2 for [BOS] and [EOS]


class BatchEncoding:
    def __init__(self, init=None) -> None:
        if init:
            self.data = init
        else:
            self.data = dict()
        pass

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError


class SPETokenizerWrapper:

    def __init__(self, spe_vocab_file_path: Optional[str] = None, atoms_list: Optional[List[str]] = None, vocab_list = None) -> None:
        """
        spe_vocab_file_path: path to vocab file
        atoms_list: list of atoms to be included in vocab
        """

        if spe_vocab_file_path is None:
            spe_vocab_file_path = SPE_VOCAB_FILE_PATH
        with open(spe_vocab_file_path, "r") as f:
            self.spe_tokenizer = SPE_Tokenizer(f)
        
        if atoms_list is None:
            with ATOM_JSON_PATH.open(encoding="utf-8") as handle:
                atoms_dict = yaml.safe_load(handle)
            atoms_list = list(atoms_dict.keys())

        # vocab_list
        if vocab_list is None:
            vocab_list = list(self.spe_tokenizer.bpe_codes_reverse.keys())
            vocab_list.extend(atoms_list)
            vocab_list.insert(0, " ")
            vocab_list.append("[BOS]")
            vocab_list.append("[EOS]")

        # dicts for encode and decode
        self.VOCABS_INDICES = dict((c, i) for i, c in enumerate(vocab_list))
        self.INDICES_VOCABS = dict((i, c) for i, c in enumerate(vocab_list))

        """
        便利変数
        """
        # length of vocab_list
        self.vocab_size = len(vocab_list)
        self.vocab_list = vocab_list
        

    def tokenize(self, smiles_list: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(smiles_list, str):
            return self.spe_tokenizer.tokenize(smiles_list)
        else:
            tokenized_list = deque()
            for smile in smiles_list:
                smile = self.spe_tokenizer.tokenize(smile)
                smile = smile.split()
                tokenized_list.append(smile)
            return list(tokenized_list)

    def __call__(self, smiles_list: List[str]) -> BatchEncoding:
        encoded_smiles_list = deque()
        attention_mask_list = deque()
        tokenized_smiles_list = self.tokenize(smiles_list)
        for smile in tokenized_smiles_list:
            encoded_smile = torch.tensor([0] * (SPE_MAX_TOKENIZED_LENGTH))  # 0はpadding
            attention_mask = torch.tensor([0] * (SPE_MAX_TOKENIZED_LENGTH))  # 0はpadding
            encoded_smile[0] = self.VOCABS_INDICES["[BOS]"]
            attention_mask[0] = 1
            for i, t in enumerate(smile):
                encoded_smile[i + 1] = self.VOCABS_INDICES[t]
                attention_mask[i + 1] = 1
            encoded_smile[i + 2] = self.VOCABS_INDICES["[EOS]"]
            attention_mask[i + 2] = 1

            encoded_smiles_list.append(encoded_smile)
            attention_mask_list.append(attention_mask)
        encoded_smiles_tensor = torch.stack(list(encoded_smiles_list))
        attention_mask_tensor = torch.stack(list(attention_mask_list))

        encodings = BatchEncoding()
        encodings.data["input_ids"] = encoded_smiles_tensor
        encodings.data["attention_mask"] = attention_mask_tensor
        return encodings

    def decode(self, encoded_tensors) -> List[List[str]]:
        smiles_list = deque()

        # Only 1 molecule as iput
        if len(encoded_tensors.size()) == 1:
            smile = []
            for idx in encoded_tensors:
                if self.INDICES_VOCABS[int(idx)] == "[BOS]":
                    continue
                if self.INDICES_VOCABS[int(idx)] == "[EOS]":
                    break
                smile.append(self.INDICES_VOCABS[int(idx)])
            smiles_list.append(smile)
            return list(smiles_list)

        # Multiple molecules as input
        else:
            for encoded_smile in encoded_tensors:
                smile = []
                for idx in encoded_smile:
                    if self.INDICES_VOCABS[int(idx)] == "[BOS]":
                        continue
                    if self.INDICES_VOCABS[int(idx)] == "[EOS]":
                        break
                    smile.append(self.INDICES_VOCABS[int(idx)])
                smiles_list.append(smile)
            return list(smiles_list)

    def decode_for_moses(self, encoded_tensors) -> List["str"]:
        smiles_list = deque()
        for encoded_smile in encoded_tensors:
            smile = ""
            for idx in encoded_smile:
                if self.INDICES_VOCABS[int(idx)] == "[BOS]":
                    continue
                if self.INDICES_VOCABS[int(idx)] == "[EOS]":
                    break
                smile += self.INDICES_VOCABS[int(idx)]
            smiles_list.append(smile)
        return list(smiles_list)


class CharLevelTokenizer:
    def __init__(self, CHARS: List[str]=None, max_length=75 + 2) -> None:
        """
        max_length: +2 for [BOS] and [EOS]
        """
        if CHARS is None:
            CHARS = yaml.safe_load(open( "your_path_to/char_vocab.json"))
        self.CHARS = copy.deepcopy(CHARS)
        self.CHARS.insert(0, "[PAD]")
        self.CHARS.append("[BOS]")
        self.CHARS.append("[EOS]")
        self.vocab_size = len(self.CHARS)
        self.VOCABS_INDICES = dict((c, i) for i, c in enumerate(self.CHARS))
        self.INDICES_VOCABS = dict((i, c) for i, c in enumerate(self.CHARS))
        self.INDICES_VOCABS[0] = " "
        self.max_length = max_length

    def __call__(self, smiles_list) -> BatchEncoding:
        encoded_smiles_list = deque()
        attention_mask_list = deque()
        for smile in tqdm(smiles_list):
            smile = smile.strip()
            smile = smile.replace(" ", "")
            encoded_smile = torch.tensor([0] * (self.max_length))
            attention_mask = torch.tensor([0] * (self.max_length))
            encoded_smile[0] = self.VOCABS_INDICES["[BOS]"]
            attention_mask[0] = 1
            for i, t in enumerate(smile):
                encoded_smile[i + 1] = self.VOCABS_INDICES[t]
                attention_mask[i + 1] = 1
            encoded_smile[i + 2] = self.VOCABS_INDICES["[EOS]"]
            attention_mask[i + 2] = 1

            encoded_smiles_list.append(encoded_smile)
            attention_mask_list.append(attention_mask)
        encoded_smiles_tensor = torch.stack(list(encoded_smiles_list))
        attention_mask_tensor = torch.stack(list(attention_mask_list))

        encodings = BatchEncoding()
        encodings.data["input_ids"] = encoded_smiles_tensor
        encodings.data["attention_mask"] = attention_mask_tensor
        return encodings

    def decode(self, encoded_tensors: torch.Tensor) -> List[List[str]]:
        """
        encoded_tensors: [data_size, max_length] 
            max_length includes [BOS] and [EOS]

        Returns:
            smiles_list: [data_size, max_length] includes [BOS] and [EOS]
        """
        smiles_list = deque()
        if len(encoded_tensors.size()) == 1:
            smile = []
            for idx in encoded_tensors:
                if self.INDICES_VOCABS[int(idx)] == "[BOS]":
                    continue
                if self.INDICES_VOCABS[int(idx)] == "[EOS]":
                    break
                smile.append(self.INDICES_VOCABS[int(idx)])
            smiles_list.append(smile)
            return list(smiles_list)
        
        # Multiple molecules as input
        else:
            for encoded_smile in encoded_tensors:
                smile = []
                for idx in encoded_smile:
                    if self.INDICES_VOCABS[int(idx)] == "[BOS]":
                        continue
                    if self.INDICES_VOCABS[int(idx)] == "[EOS]":
                        break
                    smile.append(self.INDICES_VOCABS[int(idx)])
                smiles_list.append(smile)
            return list(smiles_list)

    def decode_for_moses(self, encoded_tensors: torch.Tensor) -> List[str]:
        """
        the list is composed of smiles strings

        Returns:
            smiles_list: [data_size,] includes [BOS] and [EOS]
        """
        smiles_list = deque()
        for encoded_smile in encoded_tensors:
            smile = ""
            for idx in encoded_smile:
                if self.INDICES_VOCABS[int(idx)] == "[EOS]":
                    break
                if self.INDICES_VOCABS[int(idx)] == "[BOS]":
                    continue
                smile += self.INDICES_VOCABS[int(idx)]
            smiles_list.append(smile)
        return list(smiles_list)


class AtomLevelTokenizer:
    def __init__(self, ATOMS) -> None:
        self.ATOMS = copy.deepcopy(ATOMS)
        self.ATOMS.insert(0, " ")
        self.ATOMS.append("[BOS]")
        self.ATOMS.append("[EOS]")
        self.ATOMS_INDICES = dict((c, i) for i, c in enumerate(self.ATOMS))
        self.INDICES_ATOMS = dict((i, c) for i, c in enumerate(self.ATOMS))
        print(self.INDICES_ATOMS)

    def __call__(self, smiles_list) -> BatchEncoding:
        encoded_smiles_list = deque()
        attention_mask_list = deque()
        for smile in tqdm(smiles_list):
            encoded_smile = torch.tensor([0] * (SMILE_LENGTH + 2))
            attention_mask = torch.tensor([0] * (SMILE_LENGTH + 2))
            encoded_smile[0] = self.ATOMS_INDICES["[BOS]"]
            tokenized_smile = atomwise_tokenizer(smile)
            attention_mask[0] = 1
            for i, t in enumerate(tokenized_smile):
                encoded_smile[i + 1] = self.ATOMS_INDICES[t]
                attention_mask[i + 1] = 1
            encoded_smile[i + 2] = self.ATOMS_INDICES["[EOS]"]
            attention_mask[i + 2] = 1

            encoded_smiles_list.append(encoded_smile)
            attention_mask_list.append(attention_mask)
        encoded_smiles_tensor = torch.stack(list(encoded_smiles_list))
        attention_mask_tensor = torch.stack(list(attention_mask_list))

        encodings = BatchEncoding()
        encodings.data["input_ids"] = encoded_smiles_tensor
        encodings.data["attention_mask"] = attention_mask_tensor
        return encodings

    def decode(self, encoded_tensors):
        smiles_list = deque()
        if len(encoded_tensors.size()) == 1:
            smile = ""
            for idx in encoded_tensors:
                smile += self.INDICES_ATOMS[int(idx)]
                if self.INDICES_ATOMS[int(idx)] == "[EOS]":
                    break
            smiles_list.append(smile)
            return smiles_list
        else:
            for encoded_smile in encoded_tensors:
                smile = ""
                for idx in encoded_smile:
                    smile += self.INDICES_ATOMS[int(idx)]
                    if self.INDICES_ATOMS[int(idx)] == "[EOS]":
                        break
                smiles_list.append(smile)
            return smiles_list

    def decode_for_moses(self, encoded_tensors):
        smiles_list = deque()
        for encoded_smile in encoded_tensors:
            smile = ""
            for idx in encoded_smile:
                if self.INDICES_ATOMS[int(idx)] == "[EOS]":
                    break
                if self.INDICES_ATOMS[int(idx)] == "[BOS]":
                    continue
                smile += self.INDICES_ATOMS[int(idx)]
            smiles_list.append(smile)
        return smiles_list
