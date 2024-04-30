import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Absolute positional encoding
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# Multi-query attention layer
class MultiQueryAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = self.hid_dim // self.n_heads

        self.fc_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_k = nn.Linear(self.hid_dim, self.head_dim)
        self.fc_v = nn.Linear(self.hid_dim, self.head_dim)
        self.fc_o = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_iter_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_iter_k = nn.Linear(self.hid_dim, self.head_dim)
        self.fc_iter_v = nn.Linear(self.hid_dim, self.head_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device, dtype=torch.bfloat16)

    def forward(self, query, key, value, mask=None, iteration=None):
        batch_size = query.shape[0]

        Qbank = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Kbank = self.fc_k(key).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 3, 1)
        Vbank = self.fc_v(value).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 1, 3)
        QIterationbank = self.fc_iter_q(iteration).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        KIterationbank = self.fc_iter_k(iteration).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 3, 1)
        VIterationbank = self.fc_iter_v(value).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Qbank, Kbank) / self.scale
        energy_iteration = torch.matmul(QIterationbank, KIterationbank) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            energy_iteration = energy_iteration.masked_fill(mask == 0, -1e10)
        attention = F.softmax(energy, dim=-1)
        attention_iteration = F.softmax(energy_iteration, dim=-1)
        x = torch.matmul(self.dropout(attention), Vbank) + torch.matmul(self.dropout(attention_iteration),
                                                                        VIterationbank)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention


# Position-wise feed-forward layer
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, k):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiQueryAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.layer_indicator = nn.Sequential(nn.Embedding(k, hid_dim))
        self.dropout = nn.Dropout(dropout)

        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, self.k),
            nn.Softmax(dim=-1)
        )

    def forward(self, src, src_mask):
        src_og = src.clone()
        srcs = list()
        for iter_index in range(self.k):
            index_tensor = torch.full((src.shape[0], src.shape[1]), iter_index,
                                      dtype=torch.long, device=src.device)
            index_tensor = self.layer_indicator(index_tensor)
            _src, _ = self.self_attention(src, src, src, src_mask, index_tensor)
            src = self.self_attn_layer_norm(src_og + self.dropout(_src))
            _src = self.positionwise_feedforward(src)
            src = self.self_attn_layer_norm(src_og + self.dropout(_src))
            srcs.append(src)

        weights = self.mlp(src_og)

        src = sum(w * s for w, s in zip(weights.split(1, dim=-1), srcs))

        return src, weights


# Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, k, n_layers):
        super().__init__()
        assert n_layers > 0, f'Number of layers must be positive, {n_layers} were given'
        self.layers = nn.ModuleList(
            [TransformerBlock(hid_dim, n_heads, pf_dim, dropout, device, k) for _ in range(n_layers)])

    def forward(self, src, src_mask):
        for layer in self.layers:
            src, weights = layer(src, src_mask)
        return src, weights


# Transformer model
class Transformer(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_heads, pf_dim, dropout, device, k, n_layers, max_length=100, new=True):
        super().__init__()
        self.new = new
        self.tok_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = AbsolutePositionalEncoding(hid_dim, max_length)
        if new:
            self.encoder = TransformerEncoder(hid_dim, n_heads, pf_dim, dropout, device, k, n_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(hid_dim, n_heads, pf_dim, dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, src):
        src_mask = self.make_src_mask(src)

        # Create causal mask
        batch_size, src_len = src.shape
        causal_mask = torch.tril(torch.ones((src_len, src_len), dtype=torch.bool)).unsqueeze(0).unsqueeze(0).expand(
            batch_size, -1, -1, -1).to(src.device)

        src_mask = src_mask & causal_mask

        # Compute token embeddings
        src = self.tok_embedding(src)

        # Add positional encoding
        src = self.pos_embedding(src)
        if self.new:
            # Pass through encoder
            enc_src, weights = self.encoder(src, src_mask)
        else:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1], src.device)
            enc_src = self.encoder(src, src_mask, is_causal=True)

        # Output layer
        output = self.fc_out(enc_src)

        return output, src_mask


if __name__ == '__main__':
    # Initialize the necessary parameters
    INPUT_DIM = 256
    HID_DIM = 512
    N_HEADS = 8
    PF_DIM = 2048
    DROPOUT = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = 2
    N_LAYERS = 3
    MAX_LENGTH = 100
    BATCH_SIZE = 8
    SRC_LEN = 5

    # Initialize the transformer model
    model = Transformer(INPUT_DIM, HID_DIM, N_HEADS, PF_DIM, DROPOUT, DEVICE, K, N_LAYERS, MAX_LENGTH).to(DEVICE)

    # Generate a sample input
    src = torch.randint(1, INPUT_DIM, (BATCH_SIZE, SRC_LEN)).to(DEVICE)  # We start from 1 to avoid generating masks

    # Forward pass through the model and get the masks
    output, src_mask = model(src)

    # Print the attention masks
    print("Attention Masks:")
    print(src_mask[0, 0])  # Print the first mask for visualization

import math

# Define the VanillaTransformer class with the correct TransformerEncoder
class VanillaTransformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, pos_dropout, trans_dropout):
        super().__init__()

        self.d_model = d_model
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_enc = torch.nn.Embedding(max_seq_len, d_model)
        self.pos_dropout = torch.nn.Dropout(pos_dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, trans_dropout)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len, N = x.size()
        pos = torch.arange(0, seq_len).unsqueeze(1).repeat(1, N).to(x.device)  # (seq_len, N)

        embedding = self.embed(x) * math.sqrt(self.d_model)  # (seq_len, N, d_model)
        pos_encoding = self.pos_enc(pos)  # (seq_len, N, d_model)

        x = self.pos_dropout(embedding + pos_encoding)

        x = self.transformer(x)
        x = self.fc_out(x)

        return x, None