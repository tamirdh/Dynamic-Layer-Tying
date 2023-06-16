import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, \
    TransformerEncoderLayer
from transformers import AutoModelForCausalLM


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
        self.self_feed = TransformerEncoder(encoder_layers, 1)
        self.embedding = gpt2.transformer.wte
        self.embedding.weight.requires_grad = True
        self.d_model = d_model
        self.layer_pos = nn.Embedding(nlayers, d_model)
        self.linear = nn.Linear(d_model, ntoken)
        self.iters = nlayers
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        og_shape = src.shape
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = src
        for index in range(self.iters):
            index_tensor = torch.full(og_shape, index, device=output.device, dtype=torch.long)
            layer_emb = self.layer_pos(index_tensor)
            output = output * layer_emb
            output = self.self_feed(output, src_mask, is_causal=True)
        output = self.linear(output)
        return output

    def generate(self, input_ids: Tensor, max_length: int = 20, temperature: float = 1.0) -> Tensor:
        """
        Generate text based on the provided input_ids.

        Arguments:
            input_ids: Tensor, shape ``[batch_size, seq_len]``
                The input sequence IDs.
            max_length: int, optional (default=20)
                The maximum length of the generated sequence.
            temperature: float, optional (default=1.0)
                The temperature value controlling the randomness of the generated text.
                Higher values (e.g., > 1.0) increase randomness, while lower values (e.g., < 1.0) make the output more focused.

        Returns:
            generated_ids: Tensor, shape ``[batch_size, seq_len]``
                The generated sequence IDs.
        """
        max_length += input_ids.size(1)
        with torch.no_grad():
            batch_size = input_ids.size(0)

            src_mask = torch.triu(torch.ones(max_length, max_length)).transpose(0, 1)
            src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            src_mask = src_mask.to(input_ids.device)
            generated_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=input_ids.device)
            for step in range(max_length):
                src = torch.cat([input_ids, generated_ids[:, :step]], dim=1)
                output = self.forward(src, src_mask)
                output = output[:, -1, :].unsqueeze(0)
    

                logits = output / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1).squeeze(-1)
                generated_ids[:, step] = next_token


        return generated_ids


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
