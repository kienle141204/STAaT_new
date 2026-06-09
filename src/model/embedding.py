import torch 
from torch import nn
import numpy as np
from utils.utils import lap_eig, topological_sort

class TimeEmbedding(nn.Module):
    def __init__(self, t_dim, steps_per_day=288):
        super(TimeEmbedding, self).__init__()
        self.steps_per_day = steps_per_day
        self.day_embedding = nn.Embedding(num_embeddings=steps_per_day, embedding_dim=t_dim)
        self.week_embedding = nn.Embedding(num_embeddings=7, embedding_dim=t_dim)

    def forward(self, TE):
        B, T, _ = TE.shape

        week = (TE[..., 2].to(torch.long) % 7).view(B * T, -1)
        hour = (TE[..., 3].to(torch.long) % 24).view(B * T, -1)
        minute = (TE[..., 4].to(torch.long) % 60).view(B * T, -1)

        day_slot = ((hour * 60 + minute) * self.steps_per_day // (24 * 60)).clamp(0, self.steps_per_day - 1)
        WE = self.week_embedding(week).view(B, T, -1)
        DE = self.day_embedding(day_slot).view(B, T, -1)

        TE = torch.cat([WE, DE], dim=-1).view(B, T, -1)
        return TE, DE
    

class NodeEmbedding(nn.Module):
    def __init__(self, adj_mx, node_emb_dim, k=16, dropout=0):
        super(NodeEmbedding, self).__init__()
        N, _ = adj_mx.shape
        self.k = k
        self.set_adj(adj_mx)
        self.fc = nn.Linear(in_features=k, out_features=node_emb_dim)

    def forward(self):
        node_embedding = self.fc(self.lap_eigvec)

        return node_embedding
    
    def set_adj(self, adj_mx):
        N, _ = adj_mx.shape
        
        self.adj_mx = adj_mx
        eig_vec, eig_val = lap_eig(adj_mx)

        k = self.k
        if k > N:
            eig_vec = np.concatenate([eig_vec, np.zeros((N, k - N))], axis=-1)
            eig_val = np.concatenate([eig_val, np.zeros(k - N)], axis=-1)

        ind = np.abs(eig_val).argsort(axis=0)[:k]

        eig_vec = eig_vec[:, ind]

        if hasattr(self, 'lap_eigvec'):
            self.lap_eigvec = torch.tensor(eig_vec).float()
        else:
            self.register_buffer('lap_eigvec', torch.tensor(eig_vec).float())