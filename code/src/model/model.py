import torch 
from torch import nn
import numpy as np
from src.utils.utils import lap_eig, topological_sort
from typing import Optional
from src.model.sandglassAttn import SAG, SetTransformerSAG
from src.model.embedding import TimeEmbedding, NodeEmbedding
from src.model.tokenizer import Node2Token, Time2TokenPerNode

class DecodingLayer(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim):
        super(DecodingLayer, self).__init__()
        hidden_size = (emb_dim + output_dim) * 2 // 3

        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out

class AICLLM(nn.Module):
    def __init__(self, basemodel: nn.Module, sample_len: int, output_len: int,
                 input_dim: int, output_dim: int,
                 node_emb_dim: int,
                 sag_dim: int, sag_tokens: int,
                 dropout: float, adj_mx = None, dis_mx = None,
                 use_node_embedding: bool = True,
                 use_time_token: bool = True,
                 use_sandglassAttn: int = 0,
                 prompt_template: str = None,
                 task_type: str = 'prediction',
                 use_instruction: bool = True,
                 use_anchor_day: bool = False,
                 use_anchor_week: bool = True,
                 anchor_week_loss_weight: float = 0.05,
                 anchor_day_loss_weight: float = 0.05,
                 anchor_loss_type: str = 'huber',
                 t_dim: int = 64, trunc_k=16, wo_conloss=False, wo_conloss1=False, wo_conloss2=False,
                 ablation_drop_token: int = -1, steps_per_day: int = 288) :
        super(AICLLM, self).__init__()

        self.basemodel = basemodel
        self.sample_len = sample_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_node_embedding = use_node_embedding
        self.use_time_token = use_time_token
        self.sag_tokens = sag_tokens
        self.use_sandglassAttn = use_sandglassAttn
        self.adj_mx = adj_mx
        self.dis_mx = dis_mx
        self.task_type = task_type
        self.use_instruction = use_instruction
        self.use_anchor_day = use_anchor_day
        self.use_anchor_week = use_anchor_week
        self.anchor_week_loss_weight = anchor_week_loss_weight
        self.anchor_day_loss_weight = anchor_day_loss_weight
        self.anchor_loss_type = anchor_loss_type
        self.prompt_template = prompt_template
        self.tokenizer = basemodel.gettokenizer() if prompt_template else None
        self.wo_conloss1 = wo_conloss1
        self.wo_conloss2 = wo_conloss2

        self.topological_sort_node = True

        self.emb_dim = basemodel.dim
        tim_dim = t_dim*2     #day, week
        self.setadj(adj_mx,dis_mx)
        
        self.time_tokenizer = Time2TokenPerNode(
            sample_len=sample_len, 
            features=input_dim, 
            emb_dim=self.emb_dim, 
            tim_dim=tim_dim, 
            dropout=dropout,
            drop_token_idx=ablation_drop_token
        )

        if use_time_token:
            self.node_tokenizer = Node2Token(
                sample_len=6,  # 6 time tokens from Time2TokenPerNode
                features=self.emb_dim,  # each token has emb_dim features
                node_emb_dim=node_emb_dim,
                emb_dim=self.emb_dim,
                tim_dim=tim_dim,
                dropout=dropout,
                use_node_embedding=use_node_embedding,
                use_token_pooling=True  # weighted-sum over 6 tokens (~769 params vs 3.5M)
            )
        else:
            self.node_tokenizer = Node2Token(
                sample_len=sample_len,  # raw time steps
                features=input_dim,     # raw input features
                node_emb_dim=node_emb_dim,
                emb_dim=self.emb_dim,
                tim_dim=tim_dim,
                dropout=dropout,
                use_node_embedding=use_node_embedding,
                use_token_pooling=False
            )

        self.node_embedding = NodeEmbedding(adj_mx=adj_mx, node_emb_dim=node_emb_dim, k=trunc_k, dropout=dropout)
        self.time_embedding = TimeEmbedding(t_dim=t_dim, steps_per_day=steps_per_day)
        
        # Sandglass Attn
        if self.use_sandglassAttn == 1:
            self.sag = SAG(sag_dim=sag_dim, 
                           sag_tokens=sag_tokens, 
                           emb_dim=self.emb_dim, 
                           sample_len=sample_len, 
                           features=input_dim ,
                           dropout=dropout
                           ) 

        elif self.use_sandglassAttn == 2:
            self.sag = SetTransformerSAG(sag_dim=sag_dim, 
                                        sag_tokens=sag_tokens, 
                                        emb_dim=self.emb_dim, 
                                        sample_len=sample_len, 
                                        features=input_dim ,
                                        dropout=dropout
                                        )
            print("Using SetTransformer SAG")
            
        self.wo_conloss = wo_conloss
        
        self.out_mlp = DecodingLayer(
            input_dim=output_dim*sample_len,
            emb_dim=self.emb_dim,
            output_dim=output_dim * output_len
        )

        self.layer_norm = nn.LayerNorm(self.emb_dim)
    
    def forward(self, x: torch.FloatTensor, 
                day_anchor: torch.FloatTensor,
                week_anchor: torch.FloatTensor, 
                ya_week: torch.FloatTensor,
                ya_day: torch.FloatTensor, 
                timestamp: torch.Tensor, 
                prompt_prefix: Optional[torch.Tensor]):
        B, N, TF = x.shape
        other_loss = []
        
        timestamp = timestamp[:, :self.sample_len, :]
        te, de = self.time_embedding(timestamp)
        de_padded = torch.cat([torch.zeros_like(de), de], dim=-1)  # (B, T, tim_dim)
        ne = self.node_embedding()

        def process_anchor(anchor_data_input, time_emb, anchor_type="current"):
            if self.use_time_token:
                time_tokens_anchor = self.time_tokenizer(anchor_data_input, time_emb)  # (B, N, 6, emb_dim)
                spatio_input_anchor = time_tokens_anchor.view(B, N, -1)  # (B, N, 6*emb_dim)
            else:
                spatio_input_anchor = anchor_data_input  # (B, N, sample_len*input_dim)
            
            spatial_tokens_anchor = self.node_tokenizer(spatio_input_anchor, time_emb, ne)  # (B, N, emb_dim)
            
            if self.topological_sort_node:
                spatial_tokens_anchor = spatial_tokens_anchor[:, self.node_order, :]
            
            if self.use_sandglassAttn:
                encoded_anchor, anchor_attn = self.sag.encode(spatial_tokens_anchor)
            else:
                encoded_anchor, anchor_attn = self.precoder(spatial_tokens_anchor)
            
            return encoded_anchor, anchor_attn

        if self.use_time_token:
            time_tokens = self.time_tokenizer(x, te)  # (B, N, 6, emb_dim)
            spatio_input = time_tokens.view(B, N, -1)  # (B, N, 6*emb_dim)
        else:
            spatio_input = x  # (B, N, sample_len*input_dim) - bypass time tokenizer

        # spatial tokenizer
        spatial_tokens = self.node_tokenizer(spatio_input, te, ne)  # (B, N, emb_dim)
        
        if self.topological_sort_node:
            spatial_tokens = spatial_tokens[:, self.node_order, :]
        
        st_embedding_current = spatial_tokens
        s_num = self.sag_tokens
        
        # Encoding layer for current data
        if self.use_sandglassAttn:
            st_embedding_current, attn_weights = self.sag.encode(st_embedding_current)
        else:
            st_embedding_current, attn_weights = self.precoder(st_embedding_current)

        # Compute contrastive loss for current data
        if self.use_sandglassAttn and not self.wo_conloss:
            if attn_weights is not None:
                scale = attn_weights.sum(dim=1)    #(B,N)
                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],attn_weights)
                if not self.wo_conloss1:
                    other_loss.append(-((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)
                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                if not self.wo_conloss2:
                    other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())
        
        current_data = st_embedding_current

        anchor_embeddings = []
        
        if self.use_anchor_day:
            day_encoded, day_attn = process_anchor(day_anchor, de_padded, anchor_type="day")  # de_padded = day only
            anchor_embeddings.append(day_encoded)
            
            if self.use_sandglassAttn and not self.wo_conloss and day_attn is not None:
                scale = day_attn.sum(dim=1)
                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],day_attn)
                if not self.wo_conloss1:
                    other_loss.append(-((sag_score*day_attn-day_attn*day_attn)).sum(dim=2).mean()*10)
                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                if not self.wo_conloss2:
                    other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())
        
        if self.use_anchor_week:
            week_encoded, week_attn = process_anchor(week_anchor, te, anchor_type="week")  # te = week+day
            anchor_embeddings.append(week_encoded)
            
            if self.use_sandglassAttn and not self.wo_conloss and week_attn is not None:
                scale = week_attn.sum(dim=1)
                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],week_attn)
                if not self.wo_conloss1:
                    other_loss.append(-((sag_score*week_attn-week_attn*week_attn)).sum(dim=2).mean()*10)
                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                if not self.wo_conloss2:
                    other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())

        # ========== COMBINE ALL ==========
        if anchor_embeddings:
            # Concatenate: [day_anchor, week_anchor, current_data]
            st_embedding = torch.concat(anchor_embeddings + [current_data], dim=1)
        else:
            st_embedding = current_data


        # ========== LLM PROCESSING ==========
        hidden_state = self.basemodel(st_embedding)
        
        s_state = hidden_state[:, -s_num:, :]

        # Decoder
        if self.use_sandglassAttn:
            s_state = self.sag.decode(s_state, spatial_tokens)
        else:
            s_state = self.decoder(s_state, spatial_tokens)  
        s_state = s_state + spatial_tokens

        if self.topological_sort_node:
            s_state = s_state[:,self.node_order_rev,:]

        s_state = self.layer_norm(s_state)
        out = self.out_mlp(s_state)

        # ========== ANCHOR RELATIVE DEVIATION LOSS ==========
        def compute_anchor_loss(x_input, anchor_input, prediction, anchor_output, weight=0.05):
            input_deviation = anchor_input - x_input      
            output_deviation = anchor_output - prediction  
            
            deviation_diff = input_deviation - output_deviation
            
            if self.anchor_loss_type == 'mae':
                loss = torch.abs(deviation_diff).mean() * weight
            elif self.anchor_loss_type == 'mse':
                loss = (deviation_diff ** 2).mean() * weight
            else:  # huber
                deviation_mag = torch.abs(deviation_diff)
                loss = torch.where(
                    deviation_mag < 1.0,
                    0.5 * (deviation_diff ** 2),    
                    deviation_mag - 0.5              
                ).mean() * weight
            return loss
        
        x_for_loss = x.view(B, N, self.sample_len, self.input_dim)[:, :, -self.output_len:, :self.output_dim]
        out_reshaped = out.view(B, N, self.output_len, self.output_dim)
        
        # Week anchor loss
        if self.use_anchor_week and ya_week is not None:
            week_anchor_input = week_anchor.view(B, N, self.sample_len, self.input_dim)[:, :, -self.output_len:, :self.output_dim]
            ya_week_reshaped = ya_week.view(B, N, self.output_len, self.output_dim)
            week_anchor_loss = compute_anchor_loss(
                x_for_loss, week_anchor_input, 
                out_reshaped, ya_week_reshaped, 
                weight=self.anchor_week_loss_weight
            )
            other_loss.append(week_anchor_loss)
        
        # Day anchor loss
        if self.use_anchor_day and ya_day is not None:
            day_anchor_input = day_anchor.view(B, N, self.sample_len, self.input_dim)[:, :, -self.output_len:, :self.output_dim]
            ya_day_reshaped = ya_day.view(B, N, self.output_len, self.output_dim)
            day_anchor_loss = compute_anchor_loss(
                x_for_loss, day_anchor_input, 
                out_reshaped, ya_day_reshaped, 
                weight=self.anchor_day_loss_weight
            )
            other_loss.append(day_anchor_loss)

        return out, other_loss
            
    
    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params

    # def setadj(self,adj_mx,dis_mx):
    #     self.adj_mx = torch.tensor(adj_mx).cuda()
    #     self.dis_mx = torch.tensor(dis_mx).cuda()
    #     self.d_mx = self.adj_mx.sum(dim=1)
    #     N = self.adj_mx.shape[0]
    #     self.alpha = torch.tensor([1.05] * N).cuda() + torch.softmax(self.d_mx,dim=0)*5 
    #     self.node_order,self.node_order_rev = topological_sort(adj_mx)

    def setadj(self, adj_mx, dis_mx):
        self.adj_mx = torch.tensor(
            adj_mx,
            dtype=torch.float32
        ).cuda()

        self.dis_mx = torch.tensor(
            dis_mx,
            dtype=torch.float32
        ).cuda()

        self.d_mx = self.adj_mx.sum(dim=1)

        N = self.adj_mx.shape[0]

        self.alpha = (
            torch.tensor(
                [1.05] * N,
                dtype=torch.float32
            ).cuda()
            + torch.softmax(self.d_mx, dim=0) * 5
        )

        self.node_order, self.node_order_rev = topological_sort(adj_mx)