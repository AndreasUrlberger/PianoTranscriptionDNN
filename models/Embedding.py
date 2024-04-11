import torch
import torch.nn as nn

class WaveformEmbedding(nn.Module):
    def __init__(self, params = {}):
        super().__init__()
        self.input_size = params.get("embedding_input_size", 480)
        self.embedding_size = params.get("embedding_size", 512)
        self.hidden_size = params.get("embedding_hidden_size", 512)

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, x):
        # x = (batch_size, seq_len, input_size)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        # TODO It's debatable whether to normalize the final embedding or not.
        return x

class MidiEmbedding (nn.Module):
    def __init__(self, params = {}):
        super().__init__()
        self.input_size = params.get("embedding_input_size", 128)
        self.embedding_size = params.get("embedding_size", 512)
        self.hidden_size = params.get("embedding_hidden_size", 512)

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, x):
        # x = (batch_size, seq_len, input_size)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        # Embedding of padded positions should be zero
        return x