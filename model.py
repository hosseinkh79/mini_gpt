import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math

from configs import get_gpt_configs, device
configs = get_gpt_configs()

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.token_embedding_table(x) # (batch, seq_len, d_model)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.positional_embedding_table = nn.Embedding(configs['seq_len'], d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, seq_len , _= x.shape
        # print(f'x shape : \n {x.shape}')
        pe = self.positional_embedding_table(torch.arange(seq_len, device=device))# (seq_len, d_model)
        x = x + pe
        return  self.dropout(x)
    

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, seq_len: int, dropout: float):
#         super().__init__()

#         # self.seq_len = seq_len
#         self.positional_embedding_table = nn.Embedding(seq_len, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         print(f'x shape : \n {x.shape}')
#         pe = self.positional_embedding_table(torch.arange(self.seq_len)) # (seq_len, d_model)
#         print(f'pe shape : \n {pe.shape}')
#         x = x + pe
#         return  self.dropout(x)

    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float=1e-9):
        super().__init__()

        self.eps = eps
        self.betta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.betta
     


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropuout: float):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropuout),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x):
        return self.ff(x)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        assert d_model % num_heads == 0, "d_model isn't divisible by num_heads"

        self.d_k = d_model // num_heads

        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):

        d_k = query.shape[-1]

        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

        # print(f'attention_scores befor: \n {attention_scores[2][0]} ')

        if mask is not None: 
            # print(f'mask: \n {mask} ')
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # print(f'attention_scores after: \n {attention_scores[2][:2]} ')
        # print(f'----------------------------------------------------')

        attention_scores = torch.softmax(attention_scores, dim=-1)
        # print(f'attention_scores: \n {attention_scores[0][0]} ')
        # print(f'values: \n {value[0][0]} ')
        # print(f'res before drop : \n {(attention_scores @ value)[0][0]} ')

        if dropout is not None: 
            attention_scores = dropout(attention_scores)

        res = attention_scores @ value
        # print(f'res after drop : \n {(attention_scores @ value)[0][0]} ')

        return (attention_scores @ value ), attention_scores


    def forward(self, q, k, v, mask):
        batch_size, seq_len, _ = q.shape
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, dropout=self.dropout)
        # print(f'att shape : {attention_scores.shape}')
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(x)



class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()

        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropuout=dropout)
        self.attention_block = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

    def forward(self, x, mask):
        x = self.layer_norm_1(x)
        x = x + self.attention_block(x, x, x, mask) 
        x = self.layer_norm_2(x)
        x = x + self.ff(x) 
        return x
    



class Encoder(nn.Module):
    def __init__(self, num_encoders: int, d_model: int, d_ff: int, num_heads: int, dropout: float):
        super().__init__()

        # self.encoders_list = nn.ModuleList([EncoderBlock() for _ in range(num_encoders)])
        self.encoders_list = nn.ModuleList(EncoderBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout)
                                            for _ in range(num_encoders))
        
        
    def forward(self, x, mask):
        for block in self.encoders_list:
            x = block(x, mask)
        return x
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # input to projection layer is : (batch, seq_len, d_model) and we want (batch, seq_len, vocab_size)
        self.projection = nn.Linear(d_model, vocab_size)
        self.layer_norm = LayerNormalization()

    def forward(self, x):
        return self.projection(self.layer_norm(x))
    

class GPT(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, seq_len: int, 
                 num_encoders: int, num_heads: int, d_ff: int, 
                 pos_drop: float, encoder_drop: float):
        
        super().__init__()
        self.input_embedding = InputEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=pos_drop)
        self.encoder = Encoder(num_encoders=num_encoders, d_model=d_model, num_heads=num_heads, d_ff=d_ff,  dropout=encoder_drop)
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
        
        # self.register_buffer('tril', torch.tril(torch.ones(1, 1, seq_len, seq_len)))

    def generate(self, promt_token_ids, num_max_generated_tokens):
        for _ in range(num_max_generated_tokens):
            
            # truncating seq, our model trained with seq_len and our input token should be model_configs['seq_len']
            promt_token_ids_cond = promt_token_ids[:, :configs['seq_len']] # (batch_size, model_configs['seq_len'])
            logits = self(promt_token_ids_cond) # logits : (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :] # we selece jsut last element in sequence (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            idx_next_tokens = torch.multinomial(probs, num_samples=1)
            promt_token_ids = torch.cat((promt_token_ids, idx_next_tokens), dim=1)
            
        return promt_token_ids

    def forward(self, x, mask=None):
        # print(f'self.tril : \n{self.tril}')
        # mask_att = self.tril
        mask_att = torch.tril(torch.ones(1, 1, x.shape[1], x.shape[1])).to(device)
        # input to model : (batch, seq_len)
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, mask_att)
        x = self.projection(x)
        # output of model : (batch, seq_len, vocab_size)
        return x