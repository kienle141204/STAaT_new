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
        """
        Aggregate across nodes to extract global time tokens.
        
        Args:
            x: (B, N, T*F) - input time series for each node
            te: (B, T, tim_dim) - time embedding
            mask: optional mask
            
        Returns:
            tokens: (B, 6, emb_dim) - 6 global tokens
        """
        B, N, TF = x.shape
        x = x.view(B, N, self.sample_len, self.features)
        x = x.mean(dim=1)  # (B, T, F) - aggregate across nodes
        
        x_t = x.transpose(1, 2)  # (B, F, T) for Conv1d
        
        trend_short = self.trend_conv_short(x_t)  # (B, emb_dim//2, T)
        trend_long = self.trend_conv_long(x_t)    # (B, emb_dim//2, T)
        trend_combined = torch.cat([trend_short.mean(dim=-1), trend_long.mean(dim=-1)], dim=-1)
        trend_token = self.trend_fc(trend_combined)  # (B, emb_dim)
        
        season1 = self.season_conv1(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season2 = self.season_conv2(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season3 = self.season_conv3(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season_combined = torch.cat([season1, season2, season3], dim=-1)
        season_token = self.season_fc(season_combined)  # (B, emb_dim)
        
        x_flat = x.reshape(B, -1)  # (B, T*F)
        residual_input = torch.cat([x_flat, te[:, -1, :]], dim=1)
        residual_token = self.residual_fc(residual_input)  # (B, emb_dim)
        
        grad1 = (x[:, 1:, :] - x[:, :-1, :]).reshape(B, -1)  # (B, (T-1)*F)
        grad2 = (x[:, 2:, :] - 2*x[:, 1:-1, :] + x[:, :-2, :]).reshape(B, -1)  # (B, (T-2)*F)
        grad_input = torch.cat([grad1, grad2, te[:, -1, :]], dim=1)
        grad_token = self.grad_fc(grad_input)  # (B, emb_dim)
        
        x_min = x.min(dim=1)[0]      # (B, F)
        x_max = x.max(dim=1)[0]      # (B, F)
        x_mean = x.mean(dim=1)       # (B, F)
        x_std = x.std(dim=1)         # (B, F)
        x_centered = x - x_mean.unsqueeze(1)
        x_skew = (x_centered ** 3).mean(dim=1) / (x_std ** 3 + 1e-8)
        x_kurt = (x_centered ** 4).mean(dim=1) / (x_std ** 4 + 1e-8) - 3
        
        stats_combined = torch.cat([x_min, x_max, x_mean, x_std, x_skew, x_kurt, te[:, -1, :]], dim=1)
        stats_token = self.stats_fc(stats_combined)  # (B, emb_dim)
        
        query = self.attn_query(te[:, -1, :]).unsqueeze(1)  # (B, 1, emb_dim)
        key = self.attn_key(x)      # (B, T, emb_dim)
        value = self.attn_value(x)  # (B, T, emb_dim)
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.emb_dim ** 0.5)  # (B, 1, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, value).squeeze(1)  # (B, emb_dim)
        attn_token = self.attn_fc(attn_out)  # (B, emb_dim)
        
        tokens = torch.stack([
            trend_token,    # Long-term patterns
            season_token,   # Periodic patterns
            residual_token, # Global context
            grad_token,     # Rate of change
            stats_token,    # Statistical summary
            attn_token      # Attention-weighted summary
        ], dim=1)  # (B, 6, emb_dim)
        
        tokens = self.dropout(tokens)
        tokens = self.ln(tokens)
        
        return tokens


class Time2TokenPerNode(nn.Module):
    """
    Time tokenizer that processes each node independently.
    Output shape: (B, N, 6, emb_dim) - 6 tokens per node
    """
    def __init__(self, sample_len, features, emb_dim, tim_dim, num_tokens=6, dropout=0.1,
                 drop_token_idx: int = -1):
        super(Time2TokenPerNode, self).__init__()
        self.sample_len = sample_len
        self.features = features
        self.emb_dim = emb_dim

        self.drop_token_idx = drop_token_idx
        
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
        """
        Process each node individually to extract time tokens.
        
        Args:
            x: (B, N, T*F) - input time series for each node
            te: (B, T, tim_dim) - time embedding
            mask: optional mask
            
        Returns:
            tokens: (B, N, 6, emb_dim) - 6 tokens per node
        """
        B, N, TF = x.shape
        x = x.view(B, N, self.sample_len, self.features)  # (B, N, T, F)
        
        x = x.view(B * N, self.sample_len, self.features)  # (B*N, T, F)
        
        te_expanded = te[:, -1, :].unsqueeze(1).expand(-1, N, -1)  # (B, N, tim_dim)
        te_flat = te_expanded.reshape(B * N, -1)  # (B*N, tim_dim)
        
        x_t = x.transpose(1, 2)  # (B*N, F, T) for Conv1d
        
        trend_short = self.trend_conv_short(x_t)  # (B*N, emb_dim//2, T)
        trend_long = self.trend_conv_long(x_t)    # (B*N, emb_dim//2, T)
        trend_combined = torch.cat([trend_short.mean(dim=-1), trend_long.mean(dim=-1)], dim=-1)
        trend_token = self.trend_fc(trend_combined)  # (B*N, emb_dim)
        
        season1 = self.season_conv1(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season2 = self.season_conv2(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season3 = self.season_conv3(x_t)[:, :, :self.sample_len].max(dim=-1)[0]
        season_combined = torch.cat([season1, season2, season3], dim=-1)
        season_token = self.season_fc(season_combined)  # (B*N, emb_dim)
        
        x_flat = x.reshape(B * N, -1)  # (B*N, T*F)
        residual_input = torch.cat([x_flat, te_flat], dim=1)
        residual_token = self.residual_fc(residual_input)  # (B*N, emb_dim)
        
        grad1 = (x[:, 1:, :] - x[:, :-1, :]).reshape(B * N, -1)  # (B*N, (T-1)*F)
        grad2 = (x[:, 2:, :] - 2*x[:, 1:-1, :] + x[:, :-2, :]).reshape(B * N, -1)  # (B*N, (T-2)*F)
        grad_input = torch.cat([grad1, grad2, te_flat], dim=1)
        grad_token = self.grad_fc(grad_input)  # (B*N, emb_dim)
        
        x_min = x.min(dim=1)[0]      # (B*N, F)
        x_max = x.max(dim=1)[0]      # (B*N, F)
        x_mean = x.mean(dim=1)       # (B*N, F)
        x_std = x.std(dim=1)         # (B*N, F)
        x_centered = x - x_mean.unsqueeze(1)
        x_skew = (x_centered ** 3).mean(dim=1) / (x_std ** 3 + 1e-8)
        x_kurt = (x_centered ** 4).mean(dim=1) / (x_std ** 4 + 1e-8) - 3
        
        stats_combined = torch.cat([x_min, x_max, x_mean, x_std, x_skew, x_kurt, te_flat], dim=1)
        stats_token = self.stats_fc(stats_combined)  # (B*N, emb_dim)
        
        query = self.attn_query(te_flat).unsqueeze(1)  # (B*N, 1, emb_dim)
        key = self.attn_key(x)      # (B*N, T, emb_dim)
        value = self.attn_value(x)  # (B*N, T, emb_dim)
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.emb_dim ** 0.5)  # (B*N, 1, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, value).squeeze(1)  # (B*N, emb_dim)
        attn_token = self.attn_fc(attn_out)  # (B*N, emb_dim)
        
        tokens = torch.stack([
            trend_token,    # 0: Long-term patterns
            season_token,   # 1: Periodic patterns
            residual_token, # 2: Global context
            grad_token,     # 3: Rate of change
            stats_token,    # 4: Statistical summary
            attn_token      # 5: Attention-weighted summary
        ], dim=1)  # (B*N, 6, emb_dim)
        
        # Ablation: zero-out token bị drop (giữ nguyên shape)
        if self.drop_token_idx >= 0:
            tokens[:, self.drop_token_idx, :] = 0.0
        
        tokens = self.dropout(tokens)
        tokens = self.ln(tokens)
        
        # Reshape back to (B, N, 6, emb_dim)
        tokens = tokens.view(B, N, 6, self.emb_dim)
        
        return tokens

# class Time2TokenPerNode(nn.Module):
#     def __init__(self, sample_len, features, emb_dim, tim_dim, num_tokens=6, dropout=0.1,
#                  drop_token_idx: int = -1, inner_dim: int = None):
#         super().__init__()
#         self.sample_len = sample_len
#         self.features = features
#         self.emb_dim = emb_dim
#         self.drop_token_idx = drop_token_idx

#         # ── Bottleneck: mọi token được tính trong inner_dim ──────────────
#         self.inner_dim = inner_dim if inner_dim is not None else emb_dim // 4
#         D = self.inner_dim  # alias ngắn

#         # Token 0: Trend
#         self.trend_conv_short = nn.Conv1d(features, D // 2, kernel_size=3, padding=1)
#         self.trend_conv_long  = nn.Conv1d(features, D // 2, kernel_size=7, padding=3)
#         self.trend_fc = nn.Linear(D, D)

#         # Token 1: Seasonal
#         self.season_conv1 = nn.Conv1d(features, D // 3,          kernel_size=2, padding=1)
#         self.season_conv2 = nn.Conv1d(features, D // 3,          kernel_size=4, padding=2)
#         self.season_conv3 = nn.Conv1d(features, D - 2*(D // 3),  kernel_size=6, padding=3)
#         self.season_fc = nn.Linear(D, D)

#         # Token 2: Residual / global context
#         self.residual_fc = nn.Sequential(
#             nn.Linear(sample_len * features + tim_dim, D * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(D * 2, D)
#         )

#         # Token 3: Gradient
#         grad_input_dim = (sample_len - 1) * features + (sample_len - 2) * features + tim_dim
#         self.grad_fc = nn.Sequential(
#             nn.Linear(grad_input_dim, D),
#             nn.GELU(),
#             nn.Linear(D, D)
#         )

#         # Token 4: Stats
#         self.stats_fc = nn.Sequential(
#             nn.Linear(features * 6 + tim_dim, D),
#             nn.GELU(),
#             nn.Linear(D, D)
#         )

#         # Token 5: Attention-weighted
#         self.attn_query = nn.Linear(tim_dim,  D)
#         self.attn_key   = nn.Linear(features, D)
#         self.attn_value = nn.Linear(features, D)
#         self.attn_fc    = nn.Linear(D, D)

#         # ── Một lần project lên emb_dim (shared across 6 tokens) ─────────
#         self.token_proj = nn.Linear(D, emb_dim)   # << chỉ thêm dòng này

#         self.ln      = nn.LayerNorm(emb_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, te, mask=None):
#         B, N, TF = x.shape
#         x = x.view(B, N, self.sample_len, self.features)
#         x = x.view(B * N, self.sample_len, self.features)

#         te_expanded = te[:, -1, :].unsqueeze(1).expand(-1, N, -1)
#         te_flat = te_expanded.reshape(B * N, -1)

#         x_t = x.transpose(1, 2)

#         # Token 0: Trend
#         trend_combined = torch.cat([
#             self.trend_conv_short(x_t).mean(dim=-1),
#             self.trend_conv_long(x_t).mean(dim=-1)
#         ], dim=-1)
#         trend_token = self.trend_fc(trend_combined)

#         # Token 1: Seasonal
#         season_combined = torch.cat([
#             self.season_conv1(x_t)[:, :, :self.sample_len].max(dim=-1)[0],
#             self.season_conv2(x_t)[:, :, :self.sample_len].max(dim=-1)[0],
#             self.season_conv3(x_t)[:, :, :self.sample_len].max(dim=-1)[0],
#         ], dim=-1)
#         season_token = self.season_fc(season_combined)

#         # Token 2: Residual
#         residual_token = self.residual_fc(
#             torch.cat([x.reshape(B * N, -1), te_flat], dim=1)
#         )

#         # Token 3: Gradient
#         grad1 = (x[:, 1:, :] - x[:, :-1, :]).reshape(B * N, -1)
#         grad2 = (x[:, 2:, :] - 2*x[:, 1:-1, :] + x[:, :-2, :]).reshape(B * N, -1)
#         grad_token = self.grad_fc(torch.cat([grad1, grad2, te_flat], dim=1))

#         # Token 4: Stats
#         x_mean = x.mean(dim=1)
#         x_std  = x.std(dim=1)
#         x_centered = x - x_mean.unsqueeze(1)
#         stats_combined = torch.cat([
#             x.min(dim=1)[0], x.max(dim=1)[0], x_mean, x_std,
#             (x_centered ** 3).mean(dim=1) / (x_std ** 3 + 1e-8),
#             (x_centered ** 4).mean(dim=1) / (x_std ** 4 + 1e-8) - 3,
#             te_flat
#         ], dim=1)
#         stats_token = self.stats_fc(stats_combined)

#         # Token 5: Attention
#         query = self.attn_query(te_flat).unsqueeze(1)
#         attn_scores  = torch.matmul(query, self.attn_key(x).transpose(-2, -1)) / (self.inner_dim ** 0.5)
#         attn_token   = self.attn_fc(
#             torch.matmul(torch.softmax(attn_scores, dim=-1), self.attn_value(x)).squeeze(1)
#         )

#         # Stack → (B*N, 6, inner_dim)
#         tokens = torch.stack([
#             trend_token, season_token, residual_token,
#             grad_token,  stats_token,  attn_token
#         ], dim=1)

#         if self.drop_token_idx >= 0:
#             tokens[:, self.drop_token_idx, :] = 0.0

#         # Project lên emb_dim một lần → (B*N, 6, emb_dim)
#         tokens = self.token_proj(tokens)

#         tokens = self.dropout(tokens)
#         tokens = self.ln(tokens)

#         return tokens.view(B, N, 6, self.emb_dim)


class Node2Token(nn.Module):
    def __init__(self, sample_len, features, node_emb_dim, emb_dim, tim_dim, dropout, use_node_embedding):
        super().__init__()

        in_features = sample_len * features

        self.use_node_embedding = use_node_embedding

        state_features = tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        # Node feature embedding
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, emb_dim),
        )

        # State embedding (time + node_emb)
        hidden_size = node_emb_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x, te, ne):
        # x: (B, N, T*F)   te: (B, T, tim_dim)   ne: (N, node_emb_dim)
        B, N, TF = x.shape

        x = self.fc1(x) 

        state = te[:, -1:, :].repeat(1, N, 1)

        if self.use_node_embedding:
            ne = ne.unsqueeze(0).repeat(B, 1, 1)
            state = torch.concat((state, ne), dim=-1)

        state = self.state_fc(state)

        # Combine
        out = x + state
        out = self.ln(out)

        return out