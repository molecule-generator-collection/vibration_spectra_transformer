import os
import torch
from constants import ORIGINAL_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY
import os

HEAVY_SIZE = os.environ["HEAVY_SIZE"]
DATA_DIRECTORY = PROCESSED_DATA_DIRECTORY + f"/word_level_tokenized_{HEAVY_SIZE}"

src_train_freq = torch.load(os.path.join(DATA_DIRECTORY, "train/freqs.pt"), map_location="cpu")
src_val_freq = torch.load(os.path.join(DATA_DIRECTORY, "valid/freqs.pt"), map_location="cpu")
src_test_freq = torch.load(os.path.join(DATA_DIRECTORY, "test/freqs.pt"), map_location="cpu")
src_train_IR = torch.load(os.path.join(DATA_DIRECTORY, "train/IRs.pt"), map_location="cpu")
src_val_IR = torch.load(os.path.join(DATA_DIRECTORY, "valid/IRs.pt"), map_location="cpu")
src_test_IR = torch.load(os.path.join(DATA_DIRECTORY, "test/IRs.pt"), map_location="cpu")
src_train_Raman = torch.load(os.path.join(DATA_DIRECTORY, "train/Ramans.pt"), map_location="cpu")
src_val_Raman = torch.load(os.path.join(DATA_DIRECTORY, "valid/Ramans.pt"), map_location="cpu")
src_test_Raman = torch.load(os.path.join(DATA_DIRECTORY, "test/Ramans.pt"), map_location="cpu")

#train
train_attention_mask_list = []
for i in range(len(src_train_freq)):
    attention_mask = src_train_freq[i] != -1
    attention_mask = attention_mask.to(torch.int)
    train_attention_mask_list.append(attention_mask)
train_attention_mask_tensor = torch.stack(train_attention_mask_list)

#valid
val_attention_mask_list = []
for i in range(len(src_val_freq)):
    attention_mask = src_val_freq[i] != -1
    attention_mask = attention_mask.to(torch.int)
    val_attention_mask_list.append(attention_mask)
val_attention_mask_tensor = torch.stack(val_attention_mask_list)

#test
test_attention_mask_list = []
for i in range(len(src_test_freq)):
    attention_mask = src_test_freq[i] != -1
    attention_mask = attention_mask.to(torch.int)
    test_attention_mask_list.append(attention_mask)
test_attention_mask_tensor = torch.stack(test_attention_mask_list)

torch.save(train_attention_mask_tensor, os.path.join(DATA_DIRECTORY, "train/freq_attention_masks.pt"))
torch.save(val_attention_mask_tensor, os.path.join(DATA_DIRECTORY, "valid/freq_attention_masks.pt"))
torch.save(test_attention_mask_tensor, os.path.join(DATA_DIRECTORY, "test/freq_attention_masks.pt"))
