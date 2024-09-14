import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from models.Embedding import WaveformEmbedding, MidiEmbedding    
from prettytable import PrettyTable

class TransformerModel(nn.Module):

    """
    Args:
        ntoken: Number of tokens in the vocabulary
        d_model: Dimension of Embedding (Dimension of single sample e.g. 480)
    """
    def __init__(self, output_depth: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.1, params: dict = {}):
        super().__init__()
        self.device = params.get('device', 'cpu')
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.output_depth = output_depth
        self.start_token = params.get('start_token', -1)
        self.end_token = params.get('end_token', 0) 
        self.src_embedding = WaveformEmbedding(params={"embedding_input_size": 480, "embedding_size": d_model, "embedding_hidden_size": d_model})
        self.tgt_embedding = MidiEmbedding(params={"embedding_input_size": 129, "embedding_size": d_model, "embedding_hidden_size": d_model})
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dim_feedforward=d_hid, dropout=dropout, batch_first=True, device=self.device)
        self.final_linear = nn.Linear(d_model, output_depth)
        
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.get('learning_rate', 0.001))


    def forward(self, src: Tensor, tgt: Tensor, src_pad_mask: Tensor = None, tgt_pad_mask: Tensor = None) -> Tensor:
        # src = (batch_size, seq_len, input_size)

        _, src_seq_len, _ = src.shape
        _, tgt_seq_len, _ = tgt.shape
        src = self.src_embedding(src)
        src = self.pos_encoder(src)
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoder(tgt)

        # src_mask = nn.Transformer.generate_square_subsequent_mask(src_seq_len).to(self.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        src_mask = None
        # tgt_mask = None

        output = self.transformer.forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        output = self.final_linear(output)
        return output
    

    def training_step(self, src, tgt, src_mask, tgt_mask):
        # TODO Might want to standardize the audio input somehow, e.g. same mean and std.
        # TODO Start and end tokens are negative, which is not possible after a sigmoid. How to encode start and end tokens alternatively? All ones?
        # switch to train mode
        self.train()
        # Reset gradients
        self.optimizer.zero_grad()
        src, tgt, src_mask, tgt_mask = src.to(self.device), tgt.to(self.device), src_mask.to(self.device), tgt_mask.to(self.device)

        # We need to shift the input target by one to the right and add a start token if this is the beginning of a song. Since we don't know here if this is the beginning of a song, we instead shift the output target by one to the left, the start_token will be added in the dataset itself. This also means that our sequence length will be one shorter.
        input_target = tgt[:, :-1]
        output_target = tgt[:, 1:]
        tgt_mask = tgt_mask[:, :-1]

        # Prediction
        pred_midi = self.forward(src, input_target, src_mask, tgt_mask)
        loss = self.loss_fn(pred_midi, output_target)
        # Backpropagation
        loss.backward()
        # Update parameters
        self.optimizer.step()

        return loss.item()

    def validation_step(self, src, tgt, src_mask, tgt_mask):
        # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            src, tgt, src_mask, tgt_mask = src.to(self.device), tgt.to(self.device), src_mask.to(self.device), tgt_mask.to(self.device)
                
            input_target = tgt[:, :-1]
            output_target = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :-1]
                    
            # Prediction
            pred_midi = self.forward(src, input_target, src_mask, tgt_mask)
            loss = self.loss_fn(pred_midi, output_target)

        return loss.item()
      
    def predict(self, src, src_pad_mask = None, threshold=0.5):
        self.eval()
        with torch.no_grad():
            src = src.to(self.device)
            # src = (song_length, input_size)
            song_length, input_size = src.shape
            src_pad_mask = torch.zeros(self.d_model, dtype=torch.bool, device=self.device)

            # Pad input if shorter than a single sequence
            if song_length < self.d_model:
                src = torch.cat((src, torch.zeros(song_length, self.d_model - input_size, device=self.device)), dim=1)
                src_pad_mask[song_length:] = True

            output = torch.zeros(song_length, self.output_depth, device=self.device)
            # output[0, :] = torch.ones(self.output_depth, device=self.device)
            tgt_mask = torch.ones(self.d_model, dtype=torch.bool, device=self.device)

            # First notes of the song (sliding window not filled yet)
            for i in range(self.d_model):
                tgt_mask[i] = False
                prediction = self.forward(src[0:self.d_model].unsqueeze(0), output[0:self.d_model].unsqueeze(0), src_pad_mask.unsqueeze(0), tgt_mask.unsqueeze(0)).squeeze()
                # new_notes = torch.where(prediction[0] > threshold, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
                new_notes = prediction[0]
                output[i] = new_notes
            
            # tgt_mask is filled with False now.
            # src_mask is filled with False as well, since otherwise the below code would not execute (only executes if song_length > self.d_model).
            # Remaining notes of the song (sliding window filled)
            for i in range(self.d_model, song_length):
                prediction = self.forward(src[i:i+self.d_model].unsqueeze(0), output[i:i+self.d_model].unsqueeze(0), src_pad_mask.unsqueeze(0), tgt_mask.unsqueeze(0)).squeeze()
                # new_notes = torch.where(prediction[0] > threshold, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
                new_notes = prediction[0]
                output[i] = new_notes

        return output
    
    def predict2(self, src: torch.tensor, max_length=512, threshold=0.5):
        self.eval()
        with torch.no_grad():

            # y_input = torch.zeros(1, self.output_depth, dtype=src.dtype, device=self.device)
            y_input = torch.full((1, self.output_depth), self.start_token, dtype=src.dtype, device=self.device)

            for i in range(max_length):
                if i % 100 == 0:
                    print(f"Predicting note {i}...")
                pred = self.forward(src.unsqueeze(0), y_input[i:i+1].unsqueeze(0)).squeeze(0)
                pred = pred[0].unsqueeze(0)
                # We use BCEWithLogitsLoss, so when training we don't need to apply sigmoid to the output, however, when predicting, we need to apply sigmoid to the output.
                pred = torch.sigmoid(pred)
                
                # pred = torch.where(pred[0].unsqueeze(0) > threshold, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))

                # Concatenate previous input with predicted best word
                y_input = torch.cat((y_input, pred))

            return y_input
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def save_state(self, path):
        """
        Save model state to the given path. Conventionally the
        path should end with "*.pt".

        Later to restore:
            model.load_state_dict(torch.load(filepath))
            model.eval()

        Inputs:
        - path: path string
        """
        torch.save(self.state_dict(), path)
    
    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MidiTransformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            d_model=dim_model, dropout=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)