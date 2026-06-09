import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model.position import PositionalEncoding

class SAG(nn.Module):
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()

        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim

        self.hyper_nodes = nn.Parameter(torch.randn(1,sag_tokens,sag_dim))
        #self.pe = nn.Identity()
        self.pe = PositionalEncoding(num_hiddens=sag_dim,dropout=dropout,max_len=1024)

        self.emc_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout)
        self.dec_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout,vdim=emb_dim)

        self.enc_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)
        self.dec_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)

        self.x_fc = nn.Linear(in_features=emb_dim,out_features=sag_dim)


        self.en_ln = nn.LayerNorm(emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
    def encode(self,x):
        #x(B,N,D)
        B,N,H = x.shape
        # print(x.shape)

        kv = self.x_fc(x)

        q = self.pe(self.hyper_nodes)

        out,attn_weights = self.emc_mha(query=q.repeat(B,1,1),key=self.pe(kv),value=kv) #B,N',D

        out = self.enc_fc(out)

        out = self.en_ln(out)

        return out,attn_weights

    def decode(self,hidden_state,x):
        #hidden_state(B,N',D)
        B,_,_ = hidden_state.shape

        q = self.pe(self.x_fc(x))
        k = self.pe(self.hyper_nodes)
        v = hidden_state

        out,_ = self.dec_mha(query=q,key=k.repeat(B,1,1),value=v) #B,N,H

        out = self.dec_fc(out)

        out = self.de_ln(out)

        return out


class SetTransformerSAG(nn.Module):
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()
        
        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim
        
        # Inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, sag_tokens, sag_dim))
        self.pe = PositionalEncoding(num_hiddens=sag_dim, dropout=dropout, max_len=1024)
        
        self.x_fc = nn.Linear(emb_dim, sag_dim)
        
        # cross-attn -> self-attn -> cross-attn back
        self.enc_mha1 = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
        self.enc_self_attn = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
        self.enc_mha2 = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout)
        
        self.enc_norm1 = nn.LayerNorm(sag_dim)
        self.enc_norm2 = nn.LayerNorm(sag_dim)
        self.enc_norm3 = nn.LayerNorm(sag_dim)
        
        self.enc_fc = nn.Linear(sag_dim, emb_dim)
        self.en_ln = nn.LayerNorm(emb_dim)
        
        # Decoder
        self.dec_mha = nn.MultiheadAttention(sag_dim, self.num_heads, batch_first=True, dropout=dropout, vdim=emb_dim)
        self.dec_fc = nn.Linear(sag_dim, emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
    
    def encode(self, x):
        B, N, _ = x.shape
        
        h = self.pe(self.x_fc(x))  # (B, N, sag_dim)
        inducing = self.inducing_points.repeat(B, 1, 1)  # (B, K, sag_dim)
        
        h_ind, attn_weights = self.enc_mha1(query=self.pe(inducing), key=h, value=h)
        h_ind = self.enc_norm1(inducing + h_ind)
        
        h_ind_self, _ = self.enc_self_attn(query=h_ind, key=h_ind, value=h_ind)
        h_ind = self.enc_norm2(h_ind + h_ind_self)
        
        out = self.enc_fc(h_ind)
        out = self.en_ln(out)
        
        return out, attn_weights
    
    def decode(self, hidden_state, x):
        B = hidden_state.shape[0]
        
        q = self.pe(self.x_fc(x))
        k = self.pe(self.inducing_points)
        v = hidden_state
        
        out, _ = self.dec_mha(query=q, key=k.repeat(B, 1, 1), value=v)
        out = self.dec_fc(out)
        out = self.de_ln(out)
        
        return out
