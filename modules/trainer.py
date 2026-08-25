import os

from sympy import true
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from pathlib import Path


class FunctionalTrainer:
    def batch_train(self, model, params, batch, epoch, optimizer, device):
        model.train() 
        criterion = torch.nn.BCELoss()
        spectrum = batch[0].to(device)  # token id, output
        # [batch_size, max_length]
        labels = batch[1].to(device).to(torch.float32)
        # [batch_size, num_labels]

        predicts, predict_logits = model(spectrum)
        # [batch_size, 38, 122]
        loss = criterion(
            predict_logits, labels
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=params["clip_max_grad_norm"]
        )
        optimizer.step()

        return model, loss.item()

    def batch_validation(self, model, params, batch, epoch, device):
        criterion = torch.nn.BCELoss()

        with torch.no_grad(): 
            spectrum = batch[0].to(device)  # token id, output
            # [batch_size, max_length]
            labels = batch[1].to(device).to(torch.float32)
            # [batch_size, num_labels]
            batch_size = spectrum.size(0)

            predicts, predict_logits = model(spectrum)
            # predicts = [batch_size, num_label]
            loss = criterion(predict_logits, labels)
            acc_tesor = (predicts == labels).sum(0) / batch_size
            # [, num_labels]
        return (
            model,
            loss.item(),
            acc_tesor,
        )

    def predicts_predictlogits_labels_return(self, model, data_loader, device):
        """
        Return:
            model
            predicts: torch.Tensor
        """
        predicts_list = []
        predicts_logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in data_loader:
                spectrum = batch[0].to(device)
                labels = batch[1].to(device).to(torch.float32)
                # [batch_size, num_labels]
                predicts, predict_logits = model(spectrum)
                predicts_list.append(predicts)
                predicts_logits_list.append(predict_logits)
                labels_list.append(labels)

        predicts = torch.cat(predicts_list, dim=0)
        predicts_logits = torch.cat(predicts_logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        return model, predicts, predicts_logits, labels
    
    def train_model_loop(
        self,
        model,
        params,
        data_loader_train,
        data_loader_valid,
        optimizer,
        device,
    ):
        if not os.path.exists(Path(params["script_save_dirctory"]) / Path(params["train_script_path"]).name) or params["script_over_write"]:
            shutil.copy(params["train_script_path"], Path(params["script_save_dirctory"]) / Path(params["train_script_path"]).name) # copy2はメタデータをコピーしない
            def ignore_pycache(dir, contents):
                return ['__pycache__']
            shutil.copytree(params["module_directory_path"], Path(params["script_save_dirctory"])/ Path(params["module_directory_path"]).name, ignore=ignore_pycache, dirs_exist_ok=True) # すでに存在する場合はsourcveで上書き
        
        model = model.to(device)
        train_loss_list = (
            []
        )
        train_reconstruction_loss_list = []
        valid_loss_list = (
            []
        )
        valid_acc_list = []
        valid_reconstruction_loss_list = []
        small_train_loss_list = []
        small_train_reconstruction_loss_list = []
        small_valid_loss_list = []
        small_valid_acc_list = []
        small_valid_reconstruction_loss_list = []

        print("no first evaluation")
        print("training start")
        scheduler = None
        if params["wamup"]:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=params["warmup_start_factor"],
                end_factor=params["warmup_end_factor"],
                total_iters=params["warmup_end_step"],
            )
        # save_time_best_valid_loss = 9999999999
        save_time_best_valid_acc = -1
        big_train_loss = 0
        big_train_reconstruction_loss = 0
        small_train_loss = 0
        small_train_reconstruction_loss = 0
        small_train_kl_loss = (
            0
        )
        num_eval = 0
        total_steps = 0
        add_kl_flag = False
        kl_weight = 0
        small_num_eval_list = []
        num_eval_list = []
        for epoch in tqdm(range(1, params["num_epochs"] + 1)):
            # 初期化
            small_train_loss = 0
            small_train_reconstruction_loss = 0
            small_train_kl_loss = 0
            big_train_loss = 0
            big_train_reconstruction_loss = 0
            for step, batch in enumerate(tqdm(data_loader_train)):
                # total_steps = (epoch - 1) * len(data_loader_train) + step
                total_steps += 1
                if total_steps == params["warmup_end_step"] + 1:
                    scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1,
                        end_factor=1,
                        total_iters=params["lr_annealing_start_step"]
                        - params["warmup_end_step"],
                    )
                if total_steps == params["lr_annealing_start_step"]:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=params["lr_plateau_factor"], #default 0.1
                        patience=params["lr_plateau_patience"], #default 10
                        threshold=params["lr_plateau_threshold"], #default 1e-4
                    )

                model, train_loss = self.batch_train(
                    model, params, batch, epoch, optimizer, device
                )

                big_train_loss += train_loss
                small_train_loss += train_loss
                if (
                    step
                    % int(
                        params["validation_step_interval_rate"] * len(data_loader_train)
                    )
                    == 0
                ):
                    num_eval += 1
                    small_num_eval_list.append(num_eval)

                    loss_steps = (
                        1
                        if step == 0
                        else int(
                            params["validation_step_interval_rate"]
                            * len(data_loader_train)
                        )
                    ) 
                    print("loss_seps", loss_steps)
                    small_train_loss = (
                        small_train_loss / loss_steps
                    )
                    small_train_loss_list.append(small_train_loss)
                    # initialize
                    small_train_loss = 0

                    # small validation
                    small_valid_loss = 0
                    small_valid_acc = torch.zeros(params["num_label"]).to(device)
                    for val_step, batch in enumerate(tqdm(data_loader_valid)):
                        model, valid_loss, valid_acc_tensor = self.batch_validation(
                            model, params, batch, epoch, device
                        )
                        small_valid_loss += valid_loss
                        small_valid_acc += valid_acc_tensor
                        if (
                            val_step
                            >= len(data_loader_valid) * params["small_val_ratio"]
                        ):
                            small_valid_loss = small_valid_loss / (val_step + 1)
                            small_valid_acc: torch.Tensor = small_valid_acc / (
                                val_step + 1
                            )

                            small_valid_loss_list.append(small_valid_loss)
                            small_valid_acc_list.append(small_valid_acc)
                            break

                    print(f"epoch:{epoch}")
                    print(f"steps:{step}")
                    print(
                        f"accumulated stpes:{(epoch - 1) * len(data_loader_train) + step}"
                    )  # epoch count starts from 1
                    print(f"small_train_loss:{small_train_loss}")
                    print(f"small_valid_loss:{small_valid_loss}")
                    with open(params["small_train_loss_filename"], "w") as f:
                        f.write("\n".join([str(x) for x in small_train_loss_list]))
                    with open(params["small_valid_loss_filename"], "w") as f:
                        f.write("\n".join([str(x) for x in small_valid_loss_list]))
                    with open(params["small_valid_accuracy_filename"], "w") as f:
                        f.write("\n".join([str(x) for x in small_valid_acc_list]))

                    fig1 = plt.figure(figsize=(20, 10))
                    fig2 = plt.figure(figsize=(20, 10))
                    fig3 = plt.figure(figsize=(20, 10))
                    ax1 = fig1.add_subplot(
                        111,
                        xlabel="eval回数",
                        ylabel="loss",
                        title=f"{params['title']}_loss",
                    )
                    ax1.plot(
                        small_num_eval_list,
                        small_train_loss_list,
                        label="small_train_loss",
                    )
                    ax1.plot(
                        small_num_eval_list,
                        small_valid_loss_list,
                        label="small_valid_loss",
                    )
                    ax1.plot(
                        num_eval_list,  # stepは1epochあたりのeval回数
                        train_loss_list,
                        label="train_loss",
                    )
                    ax1.plot(
                        num_eval_list,
                        valid_loss_list,
                        label="valid_loss",
                    )
                    ax2 = fig2.add_subplot(
                        111,
                        xlabel="eval回数",
                        ylabel="accuracy rate",
                        title=f"{params['title']}_small_accuracy_rate",
                    )
                    small_valid_acc_for_graph: torch.Tensor = (
                        torch.stack(small_valid_acc_list, dim=1).detach().cpu()
                    )
                    # [num_label, num of eval]
                    small_valid_acc_ALL: torch.Tensor = small_valid_acc_for_graph.mean(
                        0
                    )
                    # [,num of eval]
                    for i in range(params["num_label"]):
                        ax2.plot(
                            small_num_eval_list,
                            small_valid_acc_for_graph[i],
                            label=params["label_list"][i],
                        )
                    ax2.plot(
                        small_num_eval_list,
                        small_valid_acc_ALL,
                        label="ALL",
                    )

                    ax3 = fig3.add_subplot(
                        111,
                        xlabel="eval回数",
                        ylabel="accuracy rate",
                        title=f"{params['title']}_big_accuracy_rate",
                    )
                    if len(valid_acc_list) == 0:
                        valid_acc_for_graph = [[]] * params["num_label"]
                        valid_acc_ALL: torch.Tensor = torch.tensor([])
                    else:
                        valid_acc_for_graph = (
                            torch.stack(valid_acc_list, dim=1).detach().cpu()
                        )
                        valid_acc_ALL: torch.Tensor = valid_acc_for_graph.mean(0)
                    # [num_label, num of eval]
                    # [, num of eval]
                    for i in range(params["num_label"]):
                        ax3.plot(
                            num_eval_list,
                            valid_acc_for_graph[i],
                            label=params["label_list"][i],
                        )
                    ax3.plot(
                        num_eval_list,
                        valid_acc_ALL,
                        label="ALL",
                    )
                    ax1.set_ylim(0, 10)
                    ax1.legend()
                    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
                    ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))
                    fig1.subplots_adjust(right=0.6)
                    fig2.subplots_adjust(right=0.6)
                    fig3.subplots_adjust(right=0.6)
                    fig1.savefig(params["loss_fig_name"])
                    fig2.savefig(params["small_accuracy_fig_name"])
                    fig3.savefig(params["big_accuracy_fig_name"])

                    # best valid
                    best_valid_loss_index = np.argmin(small_valid_loss_list)
                    best_valid_loss = small_valid_loss_list[best_valid_loss_index]

                    if total_steps % params["model_check_interval"] == 0:
                        if len(small_valid_acc_ALL) == 0:
                            print("model saving")
                            torch.save(model, params["save_model_name"])
                        if small_valid_acc_ALL[-1].item() >= save_time_best_valid_acc:
                            save_time_best_valid_acc = small_valid_acc_ALL[-1].item()
                            print("model saving")
                            torch.save(model, params["save_model_name"])

                    if scheduler is not None:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(metrics=small_valid_loss)
                        else:
                            scheduler.step()

            # big train
            num_eval_list.append(num_eval)
            big_train_loss = big_train_loss / len(data_loader_train)
            big_train_reconstruction_loss = big_train_reconstruction_loss / len(
                data_loader_train
            )
            train_loss_list.append(big_train_loss)
            train_reconstruction_loss_list.append(big_train_reconstruction_loss)

            # big validation
            big_valid_loss = 0
            big_valid_acc: torch.Tensor = torch.zeros(params["num_label"]).to(device)
            for val_step, batch in enumerate(tqdm(data_loader_valid)):
                model, valid_loss, valid_acc_tensor = (
                    self.batch_validation(  # kl_lossはbigでは使わない
                        model, params, batch, epoch, device
                    )
                )
                big_valid_loss += valid_loss
                big_valid_acc += valid_acc_tensor

            big_valid_loss = big_valid_loss / len(data_loader_valid)
            big_valid_acc = big_valid_acc / len(data_loader_valid)
            valid_loss_list.append(big_valid_loss)
            valid_acc_list.append(big_valid_acc)

            print("----------------------------------------------")
            print(f"epoch:{epoch}")
            print(
                f"accumulated stpes:{epoch * len(data_loader_train)}"
            )  # epoch counts starts from 1
            print(f"big_train_loss:{big_train_loss}")
            print(f"big_valid_loss:{big_valid_loss}")
            with open(params["train_loss_filename"], "w") as f:
                f.write("\n".join([str(x) for x in train_loss_list]))
            with open(params["valid_loss_filename"], "w") as f:
                f.write("\n".join([str(x) for x in valid_loss_list]))

            fig1 = plt.figure(figsize=(20, 10))
            fig2 = plt.figure(figsize=(20, 10))
            fig3 = plt.figure(figsize=(20, 10))
            ax1 = fig1.add_subplot(
                111, xlabel="eval回数", ylabel="loss", title=f"{params['title']}_loss"
            )
            ax1.plot(
                small_num_eval_list,
                small_train_loss_list,
                label="small_train_loss",
            )
            ax1.plot(
                small_num_eval_list,
                small_valid_loss_list,
                label="small_valid_loss",
            )
            ax1.plot(
                num_eval_list,
                train_loss_list,
                label="train_loss",
            )
            ax1.plot(
                num_eval_list,
                valid_loss_list,
                label="valid_loss",
            )
            ax2 = fig2.add_subplot(
                111,
                xlabel="eval回数",
                ylabel="accuracy rate",
                title=f"{params['title']}_small_accuracy_rate",
            )
            small_valid_acc_for_graph: torch.Tensor = (
                torch.stack(small_valid_acc_list, dim=1).detach().cpu()
            )
            # [num_label, num of eval]
            small_valid_acc_ALL: torch.Tensor = small_valid_acc_for_graph.mean(0)
            # [,num of eval]
            for i in range(params["num_label"]):
                ax2.plot(
                    small_num_eval_list,
                    small_valid_acc_for_graph[i],
                    label=params["label_list"][i],
                )
            ax2.plot(
                small_num_eval_list,
                small_valid_acc_ALL,
                label="ALL",
            )

            ax3 = fig3.add_subplot(
                111,
                xlabel="eval回数",
                ylabel="accuracy rate",
                title=f"{params['title']}_big_accuracy_rate",
            )
            valid_acc_for_graph = torch.stack(valid_acc_list, dim=1).detach().cpu()
            # [num_label, num of eval]
            valid_acc_ALL: torch.Tensor = valid_acc_for_graph.mean(0)
            # [,num of eval]
            for i in range(params["num_label"]):
                ax3.plot(
                    num_eval_list,
                    valid_acc_for_graph[i],
                    label=params["label_list"][i],
                )
            ax3.plot(
                num_eval_list,
                valid_acc_ALL,
                label="ALL",
            )
            ax1.set_ylim(0, 10)
            ax1.legend()
            ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
            ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))
            fig1.subplots_adjust(right=0.6)
            fig2.subplots_adjust(right=0.6)
            fig3.subplots_adjust(right=0.6)
            fig1.savefig(params["loss_fig_name"])
            fig2.savefig(params["small_accuracy_fig_name"])
            fig3.savefig(params["big_accuracy_fig_name"])
        return model, train_loss_list, valid_loss_list

class SmilesTrainer(FunctionalTrainer):
    # overwrite
    def batch_train(self, model, params, batch, epoch, optimizer, device):
        criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        model.train()
        freq = batch[0].to(device)
        # [batch_size, spectrum_length] spectrum_length = 100
        ir = batch[1].to(device)
        # [batch_size, spectrum_length] spectrum_length = 100
        raman = batch[2].to(device)
        # [batch_size, spectrum_length] spectrum_length = 100
        spectrum_attention = batch[3].to(device)
        # [batch_size, spectrum_length] spectrum_length = 100
        smiles_ids = batch[4].to(device)
        # [batch_size, smiles_max_length] smiles_max_length=32
        smiles_masks = batch[5].to(device)
        # [batch_size, smiles_max_length] smiles_max_length=32
        batch_size = smiles_ids.shape[0]

        """
        lossの計算
        """
        predict_logits = model(freq, ir, raman, spectrum_attention != spectrum_attention[0, 0], smiles_ids.to(torch.float32), smiles_masks != smiles_masks[0, 0])
        predict_logits = predict_logits.permute(0, 2, 1)
        loss = criterion(predict_logits[:, :, :-1], smiles_ids[:, 1:])
        if torch.isnan(loss):
            raise ValueError("loss is nan")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=params["clip_max_grad_norm"]
        ) 
        optimizer.step()

        return model, loss.item()

    def batch_validation(self, model, params, batch, epoch, device):
        """
        output is divided by batch_size. So, to get the overall average, you can simply divide by len(data_loader) outside.

        """
        criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

        with torch.no_grad():
            freq = batch[0].to(device)
            # [batch_size, spectrum_length] spectrum_length = 100
            ir = batch[1].to(device)
            # [batch_size, spectrum_length] spectrum_length = 100
            raman = batch[2].to(device)
            # [batch_size, spectrum_length] spectrum_length = 100
            spectrum_attention = batch[3].to(device)
            # [batch_size, spectrum_length] spectrum_length = 100
            smiles_ids = batch[4].to(device)
            # [batch_size, smiles_max_length] smiles_max_length=32
            smiles_masks = batch[5].to(device)
            # [batch_size, smiles_max_length] smiles_max_length=32
            batch_size = smiles_ids.shape[0]
            """
            lossの計算
            """
            predict_logits = model(freq, ir, raman, spectrum_attention != spectrum_attention[0, 0], smiles_ids.to(torch.float32), smiles_masks != smiles_masks[0, 0])
            predict_logits = predict_logits.permute(0, 2, 1)
            loss = criterion(predict_logits[:, :, :-1], smiles_ids[:, 1:])
            if torch.isnan(loss):
                raise ValueError("loss is nan")
            """
            reconstruction rateの計算
            """

            generated_smiles_ids, generated_logits = model.generate(freq, ir, raman, spectrum_attention != spectrum_attention[0, 0], smiles_masks != smiles_masks[0, 0], params["bos_indice"]) # greedy (non teacher forcing)
            # [batch_size, smiles_max_length]
            smiles:list[str] = params["tokenizer_obj"].decode_for_moses(smiles_ids)
            generated_smiles:list[str] = params["tokenizer_obj"].decode_for_moses(generated_smiles_ids)
            reconstruction_rate_tensor = sum([a == b for a, b in zip(smiles, generated_smiles)]) / len(smiles)
        return (
            model,
            loss.item(),
            reconstruction_rate_tensor,
        )

    def for_analysys(self, model,  params, data_loader, device):
        """
        あとで

        Return:
            model
            predicts: torch.Tensor
        """
        generated_smiles_list = []
        generated_smiles_ids_list = []
        generated_logits_list = []
        smiles_ids_list = []
        smiles_list = []
        true_false_by_smiles_list = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                freq = batch[0].to(device)
                # [batch_size, spectrum_length] spectrum_length = 4000 - 650 + 1
                ir = batch[1].to(device)
                # [batch_size, spectrum_length] spectrum_length = 4000 - 650 + 1
                raman = batch[2].to(device)
                # [batch_size, spectrum_length] spectrum_length = 4000 - 650 + 1
                spectrum_attention_mask = batch[3].to(device)
                # [batch_size, spectrum_length] spectrum_length = 4000 - 650 + 1
                smiles_ids = batch[4].to(device)
                # [batch_size, smiles_max_length] smiles_max_length=32
                smiles_masks = batch[5].to(device)
                # [batch_size, smiles_max_length] smiles_max_length=32
                generated_smiles_ids, generated_logits = model.generate(freq, ir, raman, spectrum_attention_mask != spectrum_attention_mask[0, 0], smiles_masks != smiles_masks[0, 0], params["bos_indice"]) # greedy (non teacher forcing)
                # [batch_size, smiles_max_length]
                generated_smiles_ids_list.append(generated_smiles_ids.to("cpu"))
                generated_logits_list.append(generated_logits.to("cpu"))
                smiles_ids_list.append(smiles_ids.to("cpu"))
                # print(generated_smiles_ids == smiles_ids)
                # print(torch.all(torch.eq(generated_smiles_ids, smiles_ids), dim=1))
                # print(torch.all(torch.eq(generated_smiles_ids, smiles_ids), dim=1).shape)
                # true_false_by_id = torch.all(torch.eq(generated_smiles_ids, smiles_ids), dim=1)
                # true_false_by_id_list.append(true_false_by_id)
                # print(true_false_by_id)
                # print(generated_smiles_ids)

                smiles_list.extend(params["tokenizer_obj"].decode_for_moses(smiles_ids))
                generated_smiles_list.extend(params["tokenizer_obj"].decode_for_moses(generated_smiles_ids))

        
        generated_smiles_ids_tensor = torch.cat(generated_smiles_ids_list, dim=0)
        generated_logits_tensor = torch.cat(generated_logits_list, dim=0)
        smiles_ids_tensor = torch.cat(smiles_ids_list, dim=0)
        true_false_by_smiles_list = []
        for smiles, generated_smiles in zip(smiles_list, generated_smiles_list):
            true_false_by_smiles_list.append(smiles == generated_smiles)

                

        return model, generated_smiles_list, generated_smiles_ids_tensor, generated_logits_tensor, smiles_ids_tensor, smiles_list, true_false_by_smiles_list
    
    def eval_topN(self, model, training_params, dataset_test, N, device):
        DATA_NUM = len(dataset_test)
        CORRECT_COUNT = 0

        tokenizer = training_params["tokenizer_obj"]
        smiles_max_length =  training_params["smiles_max_length"]
        bos_indice = tokenizer.VOCABS_INDICES["[BOS]"]
        eos_indice = tokenizer.VOCABS_INDICES["[EOS]"]

        for num, (freq, ir, raman, spectrum_attention_mask, smiles_ids, smiles_attention_mask) in tqdm(enumerate(dataset_test)):

            smiles_ids = smiles_ids.to(torch.long)
            freq = freq.to(device)
            ir = ir.to(device)
            raman = raman.to(device)
            # freq = freq.unsqueeze(0).repeat(N, 1)
            # ir = ir.unsqueeze(0).repeat(N, 1)
            # raman = raman.unsqueeze(0).repeat(N, 1)
            freq = freq.unsqueeze(0)
            ir = ir.unsqueeze(0)
            raman = raman.unsqueeze(0)
            spectrum_attention_mask = spectrum_attention_mask != spectrum_attention_mask[0]
            spectrum_attention_mask = spectrum_attention_mask.to(device)
            smiles_ids = smiles_ids.to(device)
            smiles_attention_mask = smiles_attention_mask != smiles_attention_mask[0]
            smiles_attention_mask = smiles_attention_mask.to(device)
            z = model.encoder.encode(freq, ir, raman, spectrum_attention_mask)
            #[3, 128]
            decoder_inputs = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                N, smiles_max_length
            )
            decoder_inputs[:, 0] = bos_indice
            logit_average_topN = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                N
            )
            save_smiles_list = []
            for i in range(1, smiles_max_length):
                decoder_embed_topN = model.smiles_emb(decoder_inputs)
                # [N, smils_max_length, 128]
                spectrum_attention_mask_expand = spectrum_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                smiles_attention_mask_expand = smiles_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                # z_expand = z.unsqueeze(0).repeat(decoder_inputs.shape[0], 1, 1)
                
                # print()
                # print("z")
                # print(z.shape)
                # print("decoder_embed_topN")
                # print(decoder_embed_topN.shape)
                # [N, vocab_size]
                K = N - len(save_smiles_list)
                z_expand = z.repeat(decoder_inputs.shape[0], 1, 1)
                # print("z_expand")
                # print(z_expand.shape)
                # print()
                logits_topN =  model.decoder(z_expand, decoder_embed_topN, spectrum_attention_mask_expand, smiles_attention_mask_expand)[:, i-1, :]
                if i == 1:
                    Next_decoder_inputs_list = []
                    topk_values, topk_indices = logits_topN[0].topk(K) #only 1 is OK
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        logit_average_topN[j] = topk_values[j]
                        decoder_inputs[j, i] = topk_indices[j]
                        if topk_indices[j] == eos_indice:
                            save_smiles_list.append(decoder_inputs[j, :])
                        else:
                            Next_decoder_inputs_list.append(decoder_inputs[j, :])

                else:
                    logit_average_topK_flatten = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                        K * K
                    ) #flatten because to take max
                    smiles_ids_topK_flaten = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                        K * K, smiles_max_length
                    ) 
                    for j in range(K):
                        topk_values, topk_indices = logits_topN[j].topk(K)
                        for l in range(K):
                            logit_average_topK_flatten[j * K + l]
                            topk_values[l]
                            logit_average_topK_flatten[j * K + l] = (logit_average_topN[j] * (i - 1) + topk_values[l]) / i
                            smiles_ids_topK_flaten[j * K + l, :i] = decoder_inputs[j, :i]
                            smiles_ids_topK_flaten[j * K + l, i] = topk_indices[l]
                        
                    logit_average_topN, logit_average_topN_flatten_indices = logit_average_topK_flatten.topk(K)
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        if smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], i] == eos_indice or i == smiles_max_length - 1:
                            save_smiles_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])
                        else:
                            Next_decoder_inputs_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])
                if len(save_smiles_list) == N:
                    break

                decoder_inputs = torch.stack(Next_decoder_inputs_list)

            
            #check
            try:
                predicts = tokenizer.decode_for_moses(torch.stack(save_smiles_list))
            except Exception as e:
                print(save_smiles_list)
                raise Exception(e)
            answer = tokenizer.decode_for_moses([smiles_ids])[0]
            for p in predicts:
                if answer == p:
                    CORRECT_COUNT += 1
                    break
        return CORRECT_COUNT / DATA_NUM

    def eval_topN_for_analysys(self, model, training_params, dataset_test, N, device):
        DATA_NUM = len(dataset_test)
        CORRECT_COUNT = 0

        tokenizer = training_params["tokenizer_obj"]
        smiles_max_length =  training_params["smiles_max_length"]
        bos_indice = tokenizer.VOCABS_INDICES["[BOS]"]
        eos_indice = tokenizer.VOCABS_INDICES["[EOS]"]

        predicts_list_list = []
        true_false_list = []
        original_smiles_list = []
        original_conf_num_list = []
        for num, (freq, ir, raman, spectrum_attention_mask, smiles_ids, smiles_attention_mask) in tqdm(enumerate(dataset_test)):
            smiles_ids = smiles_ids.to(torch.long)
            freq = freq.to(device)
            ir = ir.to(device)
            raman = raman.to(device)
            freq = freq.unsqueeze(0)
            ir = ir.unsqueeze(0)
            raman = raman.unsqueeze(0)
            spectrum_attention_mask = spectrum_attention_mask != spectrum_attention_mask[0]
            spectrum_attention_mask = spectrum_attention_mask.to(device)
            smiles_ids = smiles_ids.to(device)
            smiles_attention_mask = smiles_attention_mask != smiles_attention_mask[0]
            smiles_attention_mask = smiles_attention_mask.to(device)

            z = model.encoder.encode(freq, ir, raman, spectrum_attention_mask)
            #[3, 128]
            decoder_inputs = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                N, smiles_max_length
            )
            decoder_inputs[:, 0] = bos_indice

            logit_average_topN = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                N
            )
            save_smiles_list = []
            for i in range(1, smiles_max_length):
                decoder_embed_topN = model.smiles_emb(decoder_inputs)
                # [N, smils_max_length, 128]
                spectrum_attention_mask_expand = spectrum_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                smiles_attention_mask_expand = smiles_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                # [N, vocab_size]
                K = N - len(save_smiles_list)
                z_expand = z.repeat(decoder_inputs.shape[0], 1, 1)
                logits_topN =  model.decoder(z_expand, decoder_embed_topN, spectrum_attention_mask_expand, smiles_attention_mask_expand)[:, i-1, :]
                if i == 1:
                    Next_decoder_inputs_list = []
                    topk_values, topk_indices = logits_topN[0].topk(K)
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        logit_average_topN[j] = topk_values[j]
                        decoder_inputs[j, i] = topk_indices[j]
                        if topk_indices[j] == eos_indice:
                            save_smiles_list.append(decoder_inputs[j, :])
                        else:
                            Next_decoder_inputs_list.append(decoder_inputs[j, :])

                else:
                    logit_average_topK_flatten = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                        K * K
                    ) 
                    smiles_ids_topK_flaten = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                        K * K, smiles_max_length
                    )
                    for j in range(K):
                        topk_values, topk_indices = logits_topN[j].topk(K)
                        for l in range(K):
                            logit_average_topK_flatten[j * K + l]
                            topk_values[l]
                            logit_average_topK_flatten[j * K + l] = (logit_average_topN[j] * (i - 1) + topk_values[l]) / i
                            smiles_ids_topK_flaten[j * K + l, :i] = decoder_inputs[j, :i]
                            smiles_ids_topK_flaten[j * K + l, i] = topk_indices[l]
                        
                    logit_average_topN, logit_average_topN_flatten_indices = logit_average_topK_flatten.topk(K)
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        if smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], i] == eos_indice or i == smiles_max_length - 1:
                            save_smiles_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])
                        else:
                            Next_decoder_inputs_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])

                if len(save_smiles_list) == N:
                    break

                decoder_inputs = torch.stack(Next_decoder_inputs_list)
            
            #check
            try:
                predicts = tokenizer.decode_for_moses(torch.stack(save_smiles_list))
            except Exception as e:
                print(save_smiles_list)
                raise Exception(e)

            predicts_list_list.append(predicts)
            true_false_list.append(False)
            def get_conf_num(smiles: str):
                count_c = smiles.lower().count("c")
                count_N = smiles.lower().count("n")
                count_O = smiles.lower().count("o")
                count_F = smiles.lower().count("f")
                return count_c + count_N + count_O + count_F
            

            answer = tokenizer.decode_for_moses([smiles_ids])[0]
            original_smiles_list.append(answer)
            original_conf_num_list.append(get_conf_num(answer))
            for p in predicts:
                if answer == p:
                    CORRECT_COUNT += 1
                    true_false_list[-1] = True
                    break
            
        return CORRECT_COUNT / DATA_NUM, predicts_list_list, true_false_list, original_smiles_list, original_conf_num_list

    def eval_topN_with_CONFnum(self, model, training_params, dataset_test, N, device):
        DATA_NUM = len(dataset_test)
        CORRECT_COUNT = 0
        conf_num_list = []

        tokenizer = training_params["tokenizer_obj"]
        smiles_max_length =  training_params["smiles_max_length"]
        bos_indice = tokenizer.VOCABS_INDICES["[BOS]"]
        eos_indice = tokenizer.VOCABS_INDICES["[EOS]"]

        for num, (freq, ir, raman, spectrum_attention_mask, smiles_ids, smiles_attention_mask) in tqdm(enumerate(dataset_test)):

            smiles_ids = smiles_ids.to(torch.long)
            freq = freq.to(device)
            ir = ir.to(device)
            raman = raman.to(device)
            freq = freq.unsqueeze(0)
            ir = ir.unsqueeze(0)
            raman = raman.unsqueeze(0)
            spectrum_attention_mask = spectrum_attention_mask != spectrum_attention_mask[0]
            spectrum_attention_mask = spectrum_attention_mask.to(device)
            smiles_ids = smiles_ids.to(device)
            smiles_attention_mask = smiles_attention_mask != smiles_attention_mask[0]
            smiles_attention_mask = smiles_attention_mask.to(device)
            z = model.encoder.encode(freq, ir, raman, spectrum_attention_mask)
            #[3, 128]
            decoder_inputs = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                N, smiles_max_length
            )
            decoder_inputs[:, 0] = bos_indice
            logit_average_topN = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                N
            )
            save_smiles_list = []
            for i in range(1, smiles_max_length):
                decoder_embed_topN = model.smiles_emb(decoder_inputs)
                # [N, smils_max_length, 128]
                spectrum_attention_mask_expand = spectrum_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                smiles_attention_mask_expand = smiles_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                
                K = N - len(save_smiles_list)
                z_expand = z.repeat(decoder_inputs.shape[0], 1, 1)
                logits_topN =  model.decoder(z_expand, decoder_embed_topN, spectrum_attention_mask_expand, smiles_attention_mask_expand)[:, i-1, :]
                if i == 1:
                    Next_decoder_inputs_list = []
                    topk_values, topk_indices = logits_topN[0].topk(K)
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        logit_average_topN[j] = topk_values[j]
                        decoder_inputs[j, i] = topk_indices[j]
                        if topk_indices[j] == eos_indice:
                            save_smiles_list.append(decoder_inputs[j, :])
                        else:
                            Next_decoder_inputs_list.append(decoder_inputs[j, :])

                else:
                    logit_average_topK_flatten = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                        K * K
                    )
                    smiles_ids_topK_flaten = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                        K * K, smiles_max_length
                    )
                    for j in range(K):
                        topk_values, topk_indices = logits_topN[j].topk(K)
                        for l in range(K):
                            logit_average_topK_flatten[j * K + l]
                            topk_values[l]
                            logit_average_topK_flatten[j * K + l] = (logit_average_topN[j] * (i - 1) + topk_values[l]) / i
                            smiles_ids_topK_flaten[j * K + l, :i] = decoder_inputs[j, :i]
                            smiles_ids_topK_flaten[j * K + l, i] = topk_indices[l]
                        
                    logit_average_topN, logit_average_topN_flatten_indices = logit_average_topK_flatten.topk(K)
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        if smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], i] == eos_indice or i == smiles_max_length - 1:
                            save_smiles_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])
                        else:
                            Next_decoder_inputs_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])
                if len(save_smiles_list) == N:
                    break
                    
                decoder_inputs = torch.stack(Next_decoder_inputs_list)
            

            #check
            try:
                predicts = tokenizer.decode_for_moses(torch.stack(save_smiles_list))
            except Exception as e:
                print(save_smiles_list)
                raise Exception(e)
            
            #CONF_num
            def get_conf_num(smiles: str):
                count_c = smiles.lower().count("c")
                count_N = smiles.lower().count("n")
                count_O = smiles.lower().count("o")
                count_F = smiles.lower().count("f")
                return count_c + count_N + count_O + count_F
            
            temp_conf_num_list = [] #一つの分子に対するtopN個のconfのリスト
            for p in predicts:
                conf_num = get_conf_num(p)
                temp_conf_num_list.append(conf_num)
            conf_num_list.append(sum(temp_conf_num_list) / len(temp_conf_num_list))

            answer = tokenizer.decode_for_moses([smiles_ids])[0]
            for p in predicts:
                if answer == p:
                    CORRECT_COUNT += 1
                    break

        return CORRECT_COUNT / DATA_NUM, sum(conf_num_list) / len(conf_num_list)

    def topN_with_1sample(self, model, training_params, dataset_test, N, device):
        DATA_NUM = len(dataset_test)
        CORRECT_COUNT = 0

        tokenizer = training_params["tokenizer_obj"]
        smiles_max_length =  training_params["smiles_max_length"]
        bos_indice = tokenizer.VOCABS_INDICES["[BOS]"]
        eos_indice = tokenizer.VOCABS_INDICES["[EOS]"]

        for num, (freq, ir, raman, spectrum_attention_mask, smiles_ids, smiles_attention_mask) in tqdm(enumerate(dataset_test)):

            smiles_ids = smiles_ids.to(torch.long)
            freq = freq.to(device)
            ir = ir.to(device)
            raman = raman.to(device)
            freq = freq.unsqueeze(0)
            ir = ir.unsqueeze(0)
            raman = raman.unsqueeze(0)
            spectrum_attention_mask = spectrum_attention_mask != spectrum_attention_mask[0]
            spectrum_attention_mask = spectrum_attention_mask.to(device)
            smiles_ids = smiles_ids.to(device)
            smiles_attention_mask = smiles_attention_mask != smiles_attention_mask[0]
            smiles_attention_mask = smiles_attention_mask.to(device)
            z = model.encoder.encode(freq, ir, raman, spectrum_attention_mask)
            #[3, 128]
            decoder_inputs = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                N, smiles_max_length
            )
            decoder_inputs[:, 0] = bos_indice
            logit_average_topN = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                N
            )
            save_smiles_list = []
            for i in range(1, smiles_max_length):
                decoder_embed_topN = model.smiles_emb(decoder_inputs)
                # [N, smils_max_length, 128]
                spectrum_attention_mask_expand = spectrum_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                smiles_attention_mask_expand = smiles_attention_mask.unsqueeze(0).repeat(decoder_inputs.shape[0], 1)
                # z_expand = z.unsqueeze(0).repeat(decoder_inputs.shape[0], 1, 1)
                
                # print()
                # print("z")
                # print(z.shape)
                # print("decoder_embed_topN")
                # print(decoder_embed_topN.shape)
                # [N, vocab_size]
                K = N - len(save_smiles_list)
                z_expand = z.repeat(decoder_inputs.shape[0], 1, 1)
                # print("z_expand")
                # print(z_expand.shape)
                # print()
                logits_topN =  model.decoder(z_expand, decoder_embed_topN, spectrum_attention_mask_expand, smiles_attention_mask_expand)[:, i-1, :]
                if i == 1:
                    #logits_topNは完全に一緒なので最初だけ Next_decoder_inputs_listを普通にtop3で更新
                    Next_decoder_inputs_list = []
                    topk_values, topk_indices = logits_topN[0].topk(K)
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        logit_average_topN[j] = topk_values[j]
                        decoder_inputs[j, i] = topk_indices[j]
                        if topk_indices[j] == eos_indice:
                            save_smiles_list.append(decoder_inputs[j, :])
                        else:
                            Next_decoder_inputs_list.append(decoder_inputs[j, :])

                else:
                    logit_average_topK_flatten = torch.tensor([0], device=z.device, dtype=torch.float).repeat(
                        K * K
                    )
                    smiles_ids_topK_flaten = torch.tensor([0], device=z.device, dtype=torch.int).repeat(
                        K * K, smiles_max_length
                    )
                    for j in range(K):
                        topk_values, topk_indices = logits_topN[j].topk(K)
                        for l in range(K):
                            logit_average_topK_flatten[j * K + l]
                            topk_values[l]
                            logit_average_topK_flatten[j * K + l] = (logit_average_topN[j] * (i - 1) + topk_values[l]) / i
                            smiles_ids_topK_flaten[j * K + l, :i] = decoder_inputs[j, :i]
                            smiles_ids_topK_flaten[j * K + l, i] = topk_indices[l]
                        
                    logit_average_topN, logit_average_topN_flatten_indices = logit_average_topK_flatten.topk(K)
                    Next_decoder_inputs_list = []
                    for j in range(K):
                        if smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], i] == eos_indice or i == smiles_max_length - 1:
                            save_smiles_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])
                        else:
                            Next_decoder_inputs_list.append(smiles_ids_topK_flaten[logit_average_topN_flatten_indices[j], :])
                if len(save_smiles_list) == N:
                    break
                    
                decoder_inputs = torch.stack(Next_decoder_inputs_list)
            
            #check
            try:
                predicts = tokenizer.decode_for_moses(torch.stack(save_smiles_list))
                predicts_ids = torch.stack(save_smiles_list)
            except Exception as e:
                print(save_smiles_list)
                raise Exception(e)
            answer = tokenizer.decode_for_moses([smiles_ids])[0]
            for p in predicts:
                if answer == p:
                    CORRECT_COUNT += 1
                    break
            
        return CORRECT_COUNT / DATA_NUM, predicts

    def eval_with_dataloader(self, model, params, data_loader, device):
        """
        あとで

        Return:
            model
            big_valid_loss: int
            big_valid_acc: torch.Tensor #[37, ]
        """
        epoch = None
        big_valid_loss = 0
        big_valid_acc = 0
        for val_step, batch in enumerate(tqdm(data_loader)):
            # print("batch & contents")
            # print(batch[0].shape)
            model, valid_loss, valid_acc = (
                self.batch_validation( 
                    model, params, batch, epoch, device
                )
            )
            big_valid_loss += valid_loss
            big_valid_acc += valid_acc

        big_valid_loss = big_valid_loss / len(data_loader)
        big_valid_acc = big_valid_acc / len(data_loader)
        return model, big_valid_loss, big_valid_acc

    def predicts_predictlogits_labels_return(self, model, data_loader, device):
        """
        あとで

        Return:
            model
            predicts: torch.Tensor
        """
        predicts_list = []
        predicts_logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in data_loader:
                freq = batch[0].to(device)
                Ir = batch[1].to(device)
                Raman = batch[2].to(device)
                attention_mask = batch[3].to(device)
                # [batch_size, max_length]
                labels = batch[4].to(device).to(torch.float32)
                # [batch_size, num_labels]
                predicts, predict_logits = model(freq, Ir, Raman, attention_mask)
                predicts_list.append(predicts)
                predicts_logits_list.append(predict_logits)
                labels_list.append(labels)

        predicts = torch.cat(predicts_list, dim=0)
        predicts_logits = torch.cat(predicts_logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        return model, predicts, predicts_logits, labels


# Backwards-compatible name used by the original scripts/checkpoints.
SmilesTrainer3dimFreqIrRaman = SmilesTrainer
