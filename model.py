import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    # seq_len is the max len of the sentence
    def __init(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        # Apply the sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unqueeze(0) # because we have a batch of sentences (1, seq_len, d_model)

        # the positional encodings does not change and get trained they are constants that's why we use register_buffers
        # the other paramteres that are being learned and modified by the model are saved as nn.Parameters
        # we access, save and load all of these parameters using state_dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Here I am slicing the seq_len to the actual dimension of the input sentence
        x = x + (self.pe[:, :x.shape[1], :])
        return self.dropout(x)


    class LayerNormalization(nn.Module):

        def __init__(self, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            # in this module we are normalizing each sentence, we will center the data around the mean of 0 and std of 1
            # but because this is too restrictive, we use two learneable paramters alpha and bias to this model
            self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
            self.bias = nn.Parameter(torch.zeros(0)) # Added

        def __format__(self, x):
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias

