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

class TransformerModel(nn.Module):

    """
    Args:
        ntoken: Number of tokens in the vocabulary
        d_model: Dimension of Embedding (Dimension of single sample e.g. 480)
    """
    def __init__(self, output_depth: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5, params: dict = {}):
        super().__init__()
        self.device = params.get('device', 'cpu')
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.src_embedding = WaveformEmbedding(params={"embedding_input_size": 480, "embedding_size": d_model, "embedding_hidden_size": d_model})
        self.tgt_embedding = MidiEmbedding(params={"embedding_input_size": 128, "embedding_size": d_model, "embedding_hidden_size": d_model})
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
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

        src_mask = nn.Transformer.generate_square_subsequent_mask(src_seq_len).to(self.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

        output = self.transformer.forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        output = self.final_linear(output)
        return output
    

    def training_step(self, src, tgt, src_pad_mask, tgt_pad_mask):
        # switch to train mode
        self.train()
        # Reset gradients
        self.optimizer.zero_grad()
        src, tgt, src_pad_mask, tgt_pad_mask = src.to(self.device), tgt.to(self.device), src_pad_mask.to(self.device), tgt_pad_mask.to(self.device)

        # Prediction
        pred_midi = self.forward(src, tgt, src_pad_mask, tgt_pad_mask)
        loss = self.loss_fn(pred_midi, tgt)
        # Backpropagation
        loss.backward()
        # Update parameters
        self.optimizer.step()

        return loss.item()

    def validation_step(self, src, tgt, src_pad_mask, tgt_pad_mask):
        # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            src, tgt, src_pad_mask, tgt_pad_mask = src.to(self.device), tgt.to(self.device), src_pad_mask.to(self.device), tgt_pad_mask.to(self.device)            
            # Prediction
            pred_midi = self.forward(src, tgt, src_pad_mask, tgt_pad_mask)
            loss = self.loss_fn(pred_midi, tgt)

        return loss.item()
    
    def predict_tut(self, model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
        model.eval()
        
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=self.device)

        num_tokens = len(input_sequence[0])

        for _ in range(max_length):
            # Get source mask
            tgt_mask = model.get_tgt_mask(y_input.size(1)).to(self.device)
            
            pred = model(input_sequence, y_input, tgt_mask)
            
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = torch.tensor([[next_item]], device=self.device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token:
                break

        return y_input.view(-1).tolist()
    
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