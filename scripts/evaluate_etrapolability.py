import os
import json
import sys
from pathlib import Path

import japanize_matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.tokenizers import SPETokenizerWrapper
from modules import models
from modules.models import SmilesPredictor3dimFreqIrRaman
from modules.trainer import SmilesTrainer3dimFreqIrRaman
import time
tokenizer = SPETokenizerWrapper()

def main(BATCH_SIZE, MODEL_NUMBER, atom_size, device):
    LR = 1e-5
    DATA_DIRECTORY = f"data_directory/MMP05percent_extractLikeGDB/word_level_tokenized_{atom_size}"
    SCRIPT_OVER_WRITE = False
    ANNEALING_START_STEP = 10
    LR_PLATEAU_FACTOR = 0.1

    model_params = {
        "encoder_num_layers":3,
        "smiles_max_length": 32,
        "spectrum_max_length": 4000 - 650 + 1,
        "encoder_hidden_dimention": 128,
        "encoder_n_heads": 4,
        "encoder_dropout_rate": 0,
        "decoder_hidden_dimention": 128,
        "decoder_n_heads": 4,
        "decoder_dropout_rate": 0,
        "decoder_num_layers": 6,
        "smiles_vocab_size": 3067, #tokenizer.vocab_size, #[EOS], [BOS], " ", 含む 3067,
        "embed_dimention": 128, #same as decoder
        "embed_dropout_rate": 0,
        "composition_max_length": 15 + 2,
        "composition_vocab_size": 15 + 2 + 1,
    }

    SAVE_DIRECTORY_NAME = f"{MODEL_NUMBER}"
    BASE_DIRECTORY = str(PROJECT_ROOT / "results")
    MODEL_DIRECTORY = "Trained_models"
    RESULT_DIRECTOROY = "Result_graphs"
    os.makedirs(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/", exist_ok=True)
    os.makedirs(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/scripts", exist_ok=True)
    os.makedirs(f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/", exist_ok=True)

    training_params = {
        "init_lr": LR,
        "wamup":False,
        "warmup_end_step":0,
        "warmup_start_factor":1 * (10 ** -2),
        "warmup_end_factor":1,
        # "lr_start_factor":1,
        # "lr_end_factor":LR_ANNEALING_STOP_FACTOR ,
        "lr_annealing_start_step":ANNEALING_START_STEP,
        # "lr_annealing_total_steps":ANNEALING_STOP_STEP,
        "lr_plateau_factor":LR_PLATEAU_FACTOR,
        "lr_plateau_patience":10,
        "lr_plateau_threshold":1e-4,
        "model_check_interval": 1,
        "clip_max_grad_norm": 1,
        "validation_step_interval_rate": 1, #validation interval during a training epoch
        "save_model_name": f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/model.pt",
        "loss_fig_name": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/loss.png",
        "small_train_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_train_loss.txt",
        "small_valid_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_valid_loss.txt",
        "train_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/train_loss.txt",
        "valid_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/valid_loss.txt",
        "small_valid_accuracy_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_valid_accuracy.txt",
        "small_accuracy_fig_name": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_accuracy.png",
        "big_accuracy_fig_name": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/big_accuracy.png",
        "title": f"Functional_{MODEL_NUMBER}",
        "num_epochs": 10000,
        "patience": 1000000000000, #almost infinite value to ignore
        "small_val_ratio": 0.01,
        "script_save_dirctory": f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/scripts",
        "train_script_path": __file__,
        "module_directory_path": str(PROJECT_ROOT / "modules"),
        "bos_indice": int(tokenizer.VOCABS_INDICES["[BOS]"]),
        "num_label": 1, 
        "script_over_write":SCRIPT_OVER_WRITE,
        "lr_record_path": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/lr_record.txt",
    }
    with open(training_params["lr_record_path"], "w") as f:
        pass

    print(int(tokenizer.VOCABS_INDICES["[BOS]"]))
    from torch.utils.data import DataLoader, TensorDataset
    src_test_freq = torch.load(os.path.join(DATA_DIRECTORY, "freqs.pt"), map_location="cpu")
    src_test_IR = torch.load(os.path.join(DATA_DIRECTORY, "IRs.pt"), map_location="cpu")
    src_test_Raman = torch.load(os.path.join(DATA_DIRECTORY, "Ramans.pt"), map_location="cpu")
    src_test_attention_masks = torch.load(os.path.join(DATA_DIRECTORY, "freq_attention_masks.pt"), map_location="cpu")



    smiles_test = torch.load(os.path.join(DATA_DIRECTORY, "smiles_ids.pt"), map_location="cpu")
    smiles_attention_mask_test = torch.load(os.path.join(DATA_DIRECTORY, "smiles_attention_masks.pt"), map_location="cpu")

    dataset_test = TensorDataset(src_test_freq, src_test_IR, src_test_Raman, src_test_attention_masks, smiles_test, smiles_attention_mask_test)
    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    
    model: SmilesPredictor3dimFreqIrRaman = torch.load(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/model.pt")

    model = model.to(device)
    trainer = SmilesTrainer3dimFreqIrRaman()
    with open(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/training_params.json") as f:
        training_params = json.load(f)
    training_params["tokenizer_obj"] = tokenizer
    start = time.time()
    model, big_valid_loss, big_valid_acc = trainer.eval_with_dataloader(model, training_params, test_dataloader, device)
    end = time.time()
    print()
    print("BATCH_SIZE")
    print(BATCH_SIZE)
    print("big_valid_loss")
    print(big_valid_loss)
    print("big_valid_acc")
    print(big_valid_acc)
    print("time")
    print(end - start)

    #save
    with open(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/evaluation_atomsize{atom_size}_result_filterLikeGDB.txt_{BATCH_SIZE}", "w") as f:
        f.write("big_valid_loss\n")
        f.write(str(big_valid_loss))
        f.write("\n")
        f.write("big_valid_acc\n")
        f.write(str(big_valid_acc))
        f.write("\n")
        f.write("time\n")
        f.write(str(end - start))

if __name__ == "__main__":
    for model_number in ["Drop0_LR4_1", "Drop0_LR4_2", "Drop0_LR4_3"]:
        for atom_size in range(10, 11):
            main(BATCH_SIZE=4096, MODEL_NUMBER = model_number, atom_size=atom_size, device = "cuda:0")
