import copy
from typing import Optional, Tuple, Union

import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2LMHeadModel, GPT2Config


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SelfFeedConfig(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.niters = kwargs['niters']
        self.use_pretrained_model = 'gpt2-xl'


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super(Block, self).__init__(config, layer_idx=layer_idx)
        self.niters = config.niters
        assert self.niters >= 1, f'Must complete at least one iteration through each layer'
        if self.niters == 1:
            self.layer_pos = None
        else:

            self.layer_pos = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd * 4),
                nn.GELU(),
                nn.Linear(config.n_embd * 4, config.n_embd * 2),
                nn.GELU(),
                nn.Linear(config.n_embd * 2, config.n_embd),
                nn.LayerNorm(config.n_embd),  # Layer normalization
                nn.Dropout(0.1)  # Dropout
            )
        self.n_embed = config.n_embd

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        for index in range(self.niters):
            if (self.niters > 1):
                index_tensor = torch.full([hidden_states.shape[0], hidden_states.shape[1], self.n_embed], index,
                                          device=hidden_states.device, dtype=torch.float)
                layer_emb = self.layer_pos(index_tensor)
                hidden_states = hidden_states * layer_emb
            output = super().forward(hidden_states, layer_past, attention_mask, head_mask, encoder_hidden_states,
                                     encoder_attention_mask, use_cache, output_attentions)
            hidden_states = output[0]
        return output


class SelfFeedModel(nn.Module):
    def __init__(self, config):
        super(SelfFeedModel, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        gpt_model = AutoModelForCausalLM.from_pretrained(config.use_pretrained_model)
        emb = gpt_model.transformer.wte
        self.wte = emb
        self.wte.weight.requires_grad = False
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.set_weights_from_pretrained_gpt2(gpt_model)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def set_weights_from_pretrained_gpt2(self, gpt2):
        for i in range(min(self.n_layer, 48)):
            gpt2_block = gpt2.transformer.h[i]
            my_block = self.h[i]
            for sublayer in ['attn', 'ln_1', 'ln_2', 'mlp']:
                my_block_states = getattr(my_block, sublayer).state_dict()
                gpt2_block_states = getattr(gpt2_block, sublayer).state_dict()
                keys_to_remove = [key for key in gpt2_block_states.keys() if key not in my_block_states]
                for key in keys_to_remove:
                    gpt2_block_states.pop(key)
                getattr(my_block, sublayer).load_state_dict(gpt2_block_states)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class SelfFeedLMHead(nn.Module):
    def __init__(self, config):
        super(SelfFeedLMHead, self).__init__()
        self.n_embd = config.n_embd
        self.decoder = nn.Linear(self.n_embd, config.vocab_size, bias=False)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class SelfFeedLMHeadModel(nn.Module):
    def __init__(self, config):
        super(SelfFeedLMHeadModel, self).__init__()
        self.transformer = SelfFeedModel(config)
        self.lm_head = SelfFeedLMHead(config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents


class MyModel(GPT2LMHeadModel):
    def __init__(self, config: SelfFeedConfig, freeze_base=False):
        super().__init__(config)
        model = AutoModelForCausalLM.from_pretrained(config.use_pretrained_model)
        self.load_state_dict(model.state_dict())
        self.override_model(config)
        if freeze_base:
            self.freeze_base()

    def override_model(self, config: SelfFeedConfig):
        n_layers = config.max_layers
        new_h = list()
        for index in range(min(n_layers, len(self.transformer.h))):
            gpt2_block = self.transformer.h[index]
            my_block = Block(config, index)
            for sublayer in ['attn', 'ln_1', 'ln_2', 'mlp']:
                my_block_states = getattr(my_block, sublayer).state_dict()
                gpt2_block_states = getattr(gpt2_block, sublayer).state_dict()
                keys_to_remove = [key for key in gpt2_block_states.keys() if key not in my_block_states]
                for key in keys_to_remove:
                    gpt2_block_states.pop(key)
                getattr(my_block, sublayer).load_state_dict(gpt2_block_states)
            new_h.append(my_block)
        self.transformer.h = nn.ModuleList(new_h)

    def freeze_base(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if "layer_pos" in name:
                param.requires_grad = True
