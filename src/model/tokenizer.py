import torch
from torch import nn
import numpy as np

class Time2Token(nn.Module):
    def __init__(self, sample_len, features, emb_dim, tim_dim, num_tokens=6, dropout=0.1):
        super(Time2Token, self).__init__()
        self.sample_len = sample_len
        self.features = features
        self.emb_dim = emb_dim

        self.trend_conv_short = nn.Conv1d(features, emb_dim // 2, kernel_size=3, padding=1)
        self.trend_conv_long = nn.Conv1d(features, emb_dim // 2, kernel_size=7, padding=3)
        self.trend_fc = nn.Linear(emb_dim, emb_dim)

        self.season_conv1 = nn.Conv1d(features, emb_dim // 3, kernel_size=2, padding=1)
        self.season_conv2 = nn.Conv1d(features, emb_dim // 3, kernel_size=4, padding=2)
        self.season_conv3 = nn.Conv1d(features, emb_dim - 2*(emb_dim // 3), kernel_size=6, padding=3)
        self.season_fc = nn.Linear(emb_dim, emb_dim)

        self.residual_fc = nn.Sequential(
            nn.Linear(sample_len * features + tim_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        grad_input_dim = (sample_len - 1) * features + (sample_len - 2) * features + tim_dim
        self.grad_fc = nn.Sequential(
            nn.Linear(grad_input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        num_stats = 6
        self.stats_fc = nn.Sequential(
            nn.Linear(features * num_stats + tim_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.attn_query = nn.Linear(tim_dim, emb_dim)
        self.attn_key = nn.Linear(features, emb_dim)
        self.attn_value = nn.Linear(features, emb_dim)
        self.attn_fc = nn.Linear(emb_dim, emb_dim)

        self.ln = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, te, mask=None):
        B, N, TF = x.shape
        x = x.view(B, N, self.sample_len, self.features)
        x = x.mean(dim=1)  # (B, T, F)

        x_t = x.transpose(1, 2)

        trend_short = self.trend_conv_short(x_t)
        trend_long = self.trend_conv_long(x_t)
        trend_combined = torch.cat([trend_short.mean(dim=-1), trend_long.mean(dim=-1)], dim=-1)
        trend_token = self.trend_fc(trend_combined)

        season1 = self.season_conv1(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season2 = self.season_conv2(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season3 = self.season_conv3(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season_combined = torch.cat([season1, season2, season3], dim=-1)
        season_token = self.season_fc(season_combined)

        x_flat = x.reshape(B, -1)
        residual_input = torch.cat([x_flat, te[:, -1, :]], dim=1)
        residual_token = self.residual_fc(residual_input)

        grad1 = (x[:, 1:, :] - x[:, :-1, :]).reshape(B, -1)
        grad2 = (x[:, 2:, :] - 2*x[:, 1:-1, :] + x[:, :-2, :]).reshape(B, -1)
        grad_input = torch.cat([grad1, grad2, te[:, -1, :]], dim=1)
        grad_token = self.grad_fc(grad_input)

        x_min = x.min(dim=1)[0]
        x_max = x.max(dim=1)[0]
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1)
        x_centered = x - x_mean.unsqueeze(1)
        x_skew = (x_centered ** 3).mean(dim=1) / (x_std ** 3 + 1e-8)
        x_kurt = (x_centered ** 4).mean(dim=1) / (x_std ** 4 + 1e-8) - 3

        stats_combined = torch.cat([x_min, x_max, x_mean, x_std, x_skew, x_kurt, te[:, -1, :]], dim=1)
        stats_token = self.stats_fc(stats_combined)

        query = self.attn_query(te[:, -1, :]).unsqueeze(1)
        key = self.attn_key(x)
        value = self.attn_value(x)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.emb_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, value).squeeze(1)
        attn_token = self.attn_fc(attn_out)

        tokens = torch.stack([
            trend_token, season_token, residual_token,
            grad_token, stats_token, attn_token
        ], dim=1)  # (B, 6, emb_dim)

        tokens = self.dropout(tokens)
        tokens = self.ln(tokens)

        return tokens


class Time2TokenPerNode(nn.Module):
    """
    Time tokenizer that processes each node independently.
    Output shape: (B, N, 6, emb_dim) - 6 tokens per node.

    Bottleneck architecture: all 6 token branches compute in inner_dim (default emb_dim//4),
    then a single shared Linear projects to emb_dim. This reduces params ~8x vs the
    naive per-branch Linear(emb_dim, emb_dim) approach.
    """
    def __init__(self, sample_len, features, emb_dim, tim_dim, num_tokens=6, dropout=0.1,
                 drop_token_idx: int = -1, inner_dim: int = None):
        super().__init__()
        self.sample_len = sample_len
        self.features = features
        self.emb_dim = emb_dim
        self.drop_token_idx = drop_token_idx

        D = inner_dim if inner_dim is not None else emb_dim // 4
        self.inner_dim = D

        # Token 0: Trend
        self.trend_conv_short = nn.Conv1d(features, D // 2, kernel_size=3, padding=1)
        self.trend_conv_long  = nn.Conv1d(features, D // 2, kernel_size=7, padding=3)
        self.trend_fc = nn.Linear(D, D)

        # Token 1: Seasonal
        self.season_conv1 = nn.Conv1d(features, D // 3,         kernel_size=2, padding=1)
        self.season_conv2 = nn.Conv1d(features, D // 3,         kernel_size=4, padding=2)
        self.season_conv3 = nn.Conv1d(features, D - 2*(D // 3), kernel_size=6, padding=3)
        self.season_fc = nn.Linear(D, D)

        # Token 2: Residual / global context
        self.residual_fc = nn.Sequential(
            nn.Linear(sample_len * features + tim_dim, D * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * 2, D)
        )

        # Token 3: Gradient
        grad_input_dim = (sample_len - 1) * features + (sample_len - 2) * features + tim_dim
        self.grad_fc = nn.Sequential(
            nn.Linear(grad_input_dim, D),
            nn.GELU(),
            nn.Linear(D, D)
        )

        # Token 4: Stats
        self.stats_fc = nn.Sequential(
            nn.Linear(features * 6 + tim_dim, D),
            nn.GELU(),
            nn.Linear(D, D)
        )

        # Token 5: Attention-weighted
        self.attn_query = nn.Linear(tim_dim,  D)
        self.attn_key   = nn.Linear(features, D)
        self.attn_value = nn.Linear(features, D)
        self.attn_fc    = nn.Linear(D, D)

        # Shared projection: inner_dim → emb_dim (applied once to all 6 tokens)
        self.token_proj = nn.Linear(D, emb_dim)

        self.ln      = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, te, mask=None):
        B, N, TF = x.shape
        x = x.view(B, N, self.sample_len, self.features)
        x = x.view(B * N, self.sample_len, self.features)  # (B*N, T, F)

        te_expanded = te[:, -1, :].unsqueeze(1).expand(-1, N, -1)
        te_flat = te_expanded.reshape(B * N, -1)  # (B*N, tim_dim)

        x_t = x.transpose(1, 2)  # (B*N, F, T)

        # Token 0: Trend
        trend_combined = torch.cat([
            self.trend_conv_short(x_t).mean(dim=-1),
            self.trend_conv_long(x_t).mean(dim=-1)
        ], dim=-1)
        trend_token = self.trend_fc(trend_combined)

        # Token 1: Seasonal
        season_combined = torch.cat([
            self.season_conv1(x_t)[:, :, :self.sample_len].max(dim=-1)[0],
            self.season_conv2(x_t)[:, :, :self.sample_len].max(dim=-1)[0],
            self.season_conv3(x_t)[:, :, :self.sample_len].max(dim=-1)[0],
        ], dim=-1)
        season_token = self.season_fc(season_combined)

        # Token 2: Residual
        residual_token = self.residual_fc(
            torch.cat([x.reshape(B * N, -1), te_flat], dim=1)
        )

        # Token 3: Gradient
        grad1 = (x[:, 1:, :] - x[:, :-1, :]).reshape(B * N, -1)
        grad2 = (x[:, 2:, :] - 2*x[:, 1:-1, :] + x[:, :-2, :]).reshape(B * N, -1)
        grad_token = self.grad_fc(torch.cat([grad1, grad2, te_flat], dim=1))

        # Token 4: Stats
        x_mean = x.mean(dim=1)
        x_std  = x.std(dim=1)
        x_centered = x - x_mean.unsqueeze(1)
        stats_combined = torch.cat([
            x.min(dim=1)[0], x.max(dim=1)[0], x_mean, x_std,
            (x_centered ** 3).mean(dim=1) / (x_std ** 3 + 1e-8),
            (x_centered ** 4).mean(dim=1) / (x_std ** 4 + 1e-8) - 3,
            te_flat
        ], dim=1)
        stats_token = self.stats_fc(stats_combined)

        # Token 5: Attention-weighted
        query = self.attn_query(te_flat).unsqueeze(1)
        attn_scores = torch.matmul(query, self.attn_key(x).transpose(-2, -1)) / (self.inner_dim ** 0.5)
        attn_token  = self.attn_fc(
            torch.matmul(torch.softmax(attn_scores, dim=-1), self.attn_value(x)).squeeze(1)
        )

        # Stack → (B*N, 6, inner_dim)
        tokens = torch.stack([
            trend_token, season_token, residual_token,
            grad_token,  stats_token,  attn_token
        ], dim=1)

        if self.drop_token_idx >= 0:
            tokens[:, self.drop_token_idx, :] = 0.0

        # Project lên emb_dim một lần (shared across 6 tokens)
        tokens = self.token_proj(tokens)  # (B*N, 6, emb_dim)

        tokens = self.dropout(tokens)
        tokens = self.ln(tokens)

        return tokens.view(B, N, 6, self.emb_dim)


class Node2Token(nn.Module):
    def __init__(self, sample_len, features, node_emb_dim, emb_dim, tim_dim, dropout,
                 use_node_embedding, use_token_pooling=False):
        super().__init__()
        self.use_node_embedding = use_node_embedding
        self.use_token_pooling = use_token_pooling
        self.num_tokens = sample_len
        self.token_features = features
        self.emb_dim = emb_dim

        if use_token_pooling:
            # Attention-weighted pooling over the num_tokens token dimension.
            # ~769 params instead of sample_len*features*emb_dim.
            self.token_weight = nn.Linear(features, 1)
        else:
            in_features = sample_len * features
            self.fc1 = nn.Sequential(nn.Linear(in_features, emb_dim))

        state_features = tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        hidden_size = node_emb_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        self.ln = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, te, ne):
        # x: (B, N, T*F) or (B, N, num_tokens*token_features)
        # te: (B, T, tim_dim)   ne: (N, node_emb_dim)
        B, N, TF = x.shape

        if self.use_token_pooling:
            # Reshape → (B, N, num_tokens, token_features)
            x_tokens = x.view(B, N, self.num_tokens, self.token_features)
            # Learn a scalar weight per token, softmax over token dim
            weights = torch.softmax(self.token_weight(x_tokens), dim=2)  # (B, N, num_tokens, 1)
            x = (weights * x_tokens).sum(dim=2)  # (B, N, emb_dim)
        else:
            x = self.fc1(x)

        state = te[:, -1:, :].repeat(1, N, 1)

        if self.use_node_embedding:
            ne = ne.unsqueeze(0).repeat(B, 1, 1)
            state = torch.concat((state, ne), dim=-1)

        state = self.state_fc(state)

        out = x + state
        out = self.ln(out)
        out = self.dropout(out)

        return out
