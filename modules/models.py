import logging

import torch
from torch import nn
from torch.autograd import Variable
import math

logging.getLogger("autoencoder")
logging.getLogger().setLevel(20)
logging.getLogger().addHandler(logging.StreamHandler())

class SmilesPredictor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = ThreeDimEncoder(params)
        self.decoder = Decoder(params)
        self.smiles_max_length = params["smiles_max_length"]
        self.smiles_emb: Embedding = Embedding(params)

    def forward(self, freq, ir, raman, spectrum_attention_mask, smiles_ids, smiles_attention_masks):
        z = self.encoder.encode(freq, ir, raman, spectrum_attention_mask)
        # [batch_size, spectrum_max_length, 3]
        smiles_ids = smiles_ids.to(torch.long)
        smiles_embed = self.smiles_emb(smiles_ids)
        # [batch_size, smiles_max_length, embedding_hidden_dimention]
        spectrum_attention_mask = spectrum_attention_mask.to(torch.bool)
        smiles_attention_masks = smiles_attention_masks.to(torch.bool)
        x = self.decoder(z, smiles_embed, spectrum_attention_mask, smiles_attention_masks)
        # [batch_size, smiles_max_length, vocab_size]
        return x
        

    def generate(self, freq, Ir, Raman, spectrum_attention_mask, smiles_attention_masks, bos_indice):
        z = self.encoder.encode(freq, Ir, Raman, spectrum_attention_mask)
        #[batch_size, spectrum_max_length, hidden_dimention]
        decoder_inputs = torch.zeros(
            (z.shape[0], self.smiles_max_length), device=z.device, dtype=torch.long
        )
        # [batch_size, smiles_max_length]
        decoder_inputs[:, 0] = bos_indice
        for i in range(1, self.smiles_max_length):
            decoder_embed = self.smiles_emb(decoder_inputs)
            if i == self.smiles_max_length - 1:
                a = decoder_inputs.detach().clone()
                decoder_embed_last = self.smiles_emb(a)
                # [batch_size, smiles_max_length, embed_dimmention] #今はemed_dimentionはhidden_dimentionと同じ
            logits = self.decoder(z, decoder_embed, spectrum_attention_mask, smiles_attention_masks)
            # [batch_size, smiles_max_length, smiles_vocab_size]
            if i == self.smiles_max_length - 1:
                logits_last = self.decoder(z, decoder_embed_last, spectrum_attention_mask, smiles_attention_masks)
                # [batch_size, smiles_max_length, hidden_dimmention]

            logits = logits.permute([0, 2, 1])
            # [batch_size, smiles_vocab_size, smiles_max_len]
            decoder_inputs[:, i] = logits.max(1)[1][:, i - 1]
            # [batch_size, smiles_max_length]
        
        generated_smiles_ids = decoder_inputs
        generated_logits = logits_last
        return generated_smiles_ids, generated_logits

class Decoder(nn.Module):
    """
    Smilespredictorのdecoder
    """
    def __init__(self, params):
        super().__init__()
        #check all parameter exists
        if not "smiles_max_length" in params:
            raise ValueError("smiles_max_length is required.")
        if not "decoder_hidden_dimention" in params:
            raise ValueError("decoder_hidden_dimention is required.")
        if not "decoder_n_heads" in params:
            raise ValueError("decoder_n_heads is required.")
        if not "decoder_dropout_rate" in params:
            raise ValueError("decoder_dropout_rate is required.")
        if not "decoder_num_layers" in params:
            raise ValueError("decoder_num_layers is required.")
        if not "smiles_vocab_size" in params:
            raise ValueError("smiles_vocab_size is required.")
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=params["decoder_hidden_dimention"],
            nhead=params["decoder_n_heads"],
            dropout=params["decoder_dropout_rate"],
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=params["decoder_num_layers"]
        )
        self.tgt_mask = torch.triu(
            torch.full((params["smiles_max_length"], params["smiles_max_length"]), True), diagonal=1
        )
        self.fc_out = nn.Linear(
            params["decoder_hidden_dimention"], params["smiles_vocab_size"]
        )
        # self.soft_max = nn.Softmax(dim=-1)

    def forward(self, z, smiles_embed, composition_attention_mask, smiles_attention_masks):
        """
        学習用。teacher forcing
        z: latent [batch_size, smiles_max_length, decoder_hidden_dimention] smilesじゃなくても良い。
        smiles_embed: not embed target [batch_size, smiles_max_length]
        """
        # [batch_size, smiles_max_length, decoder_hidden_dimention]
        self.tgt_mask = self.tgt_mask.to(smiles_embed.device)

        x = self.transformer_decoder(
            tgt=smiles_embed,
            memory=z,
            tgt_mask=self.tgt_mask,
            tgt_key_padding_mask=smiles_attention_masks, 
            memory_key_padding_mask=composition_attention_mask,
        )
        x = self.fc_out(x)
        return x

class ThreeDimEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Checkpoints created before modality selection was introduced use all
        # three inputs. Keeping this default preserves their compatibility.
        self.input_modalities = params.get(
            "input_modalities", ["freq", "ir", "raman"]
        )
        supported_modalities = {"freq", "ir", "raman"}
        if not self.input_modalities or not set(self.input_modalities) <= supported_modalities:
            raise ValueError(
                "input_modalities must be a non-empty subset of: freq, ir, raman"
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params["encoder_hidden_dimention"],
            nhead=params["encoder_n_heads"],
            dropout=params["encoder_dropout_rate"],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=params["encoder_num_layers"]
        )
        self.dim_convert_layer = nn.Sequential(
            nn.Linear(len(self.input_modalities), params["encoder_hidden_dimention"]),
            nn.ReLU(),
            nn.Linear(params["encoder_hidden_dimention"], params["encoder_hidden_dimention"]),
            nn.ReLU(),
            nn.Linear(params["encoder_hidden_dimention"], params["encoder_hidden_dimention"]),
            nn.ReLU(),
        )

    def forward(self, freq, ir, raman, attention_mask):
        """
        finetuning用
        freq, ir, raman, attention_mask: [batch_size, max_length]
        attention_maskは1, 0
        """
        attention_mask = attention_mask.to(torch.bool)

        spectra = {"freq": freq, "ir": ir, "raman": raman}
        x = torch.stack([spectra[name] for name in self.input_modalities], dim=1)
        # [batch_size, num_modalities, max_length]
        x = x.permute([0, 2, 1])
        # [batch_size, max_length, 3]
        x = self.dim_convert_layer(x)
        # [batch_size, max_length, hidden_dimention]
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        # [batch_size, max_length, hidden_dimention]
        return x
    
    def encode(self, freq, ir, raman, attention_mask):
        return self.forward(freq, ir, raman, attention_mask)

class Embedding(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        #check all parameter exists
        if not "smiles_max_length" in params:
            raise ValueError("smiles_max_length is required.")
        if not "smiles_vocab_size" in params:
            raise ValueError("smiles_vocab_size is required.")
        if not "embed_dimention" in params:
            raise ValueError("embed_dimention is required.")
        if not "embed_dropout_rate" in params:
            raise ValueError("embed_dropout_rate is required.")
        self.positions = torch.arange(params["smiles_max_length"])
        self.word_embedding = nn.Embedding(
            params["smiles_vocab_size"], params["embed_dimention"]
        )
        self.positional_embeddings = PositionalEncoder(params)

    def forward(self, x):
        x = self.word_embedding(x)
        x = self.positional_embeddings(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.d_model = params["embed_dimention"]
        self.dropout = nn.Dropout(params["embed_dropout_rate"])
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(params["smiles_max_length"], params["embed_dimention"])
        for pos in range(params["smiles_max_length"]):
            for i in range(0, params["embed_dimention"], 2):
                pe[pos, i] = math.sin(
                    pos / (10000 ** ((2 * i) / params["embed_dimention"]))
                )
                pe[pos, i + 1] = math.cos(
                    pos
                    / (10000 ** ((2 * (i + 1)) / params["embed_dimention"]))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.to(x.device)
        x = x + pe
        return self.dropout(x)


# Backwards-compatible names used by the original entry-point scripts and old
# full-object checkpoints.
SmilesPredictor3dimFreqIrRaman = SmilesPredictor
