import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.Embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.Embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) 

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(position * denom)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
 
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim =-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self. dropout = dropout
        assert d_model % h == 0, 'd_model is not divisible by h'

        self.d_k = d_model // h

        # self.w_q = nn.Linear(d_model, d_model)
        # self.w_k = nn.Linear(d_model, d_model)
        # self.w_v = nn.Linear(d_model, d_model)

        # self.w_o = nn.Linear(d_model, d_model)


        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo

        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def Attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1) ) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) #Batch, Seqlen, d_model ---->Batch, Seqlen, d_model 
        key = self.w_k(k)
        value = self.w_v(v)

        

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, attention_scores = MultiheadAttention.Attention(query, key, value, mask, self.dropout)
        
        # Batch, h, Seq_len, d_k ----> Batch, Seq_len, h, d_k ----> Batcg, Seq_len, d_model
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiheadAttention, feedforward: FeedForward, dropout: float) -> None:
        super().__init__()

        self.self_attention = self_attention
        self.feedforward = feedforward
        self.dropout = nn.Dropout(dropout)

        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feedforward)

        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, cross_attention_block: MultiheadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward
        
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: InputEmbeddings, tgt_emb: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj: Projection):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj(x)
    

def BuildTransformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float=0.1, d_ff: int = 2048):
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_block = []

    for _ in range(N):
        encoder_self_attention = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_blo = EncoderBlock(encoder_self_attention, feed_forward_block, dropout)
        encoder_block.append(encoder_blo)

    decoder_block = []

    for _ in range(N):
        decoder_self_attention = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_blo = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward_block,dropout)

        decoder_block.append(decoder_blo)

    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))

    projection_layer = Projection(d_model, tgt_vocab_size)


    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            #Xavier
            nn.init.xavier_uniform(p)

    return transformer

