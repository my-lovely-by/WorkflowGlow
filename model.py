import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import numpy as np

import torch
from torch_geometric.nn import GCNConv, global_mean_pool,GATConv


class BatchMeanContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D]
            labels: [B]
        """
        device = embeddings.device
        B = embeddings.size(0)

        # Normalize embeddings for cosine distance
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # [B, B]
        dist_matrix = 1 - sim_matrix  # cosine distance

        # ------- Build masks based on new rules -------
        labels = labels.unsqueeze(1)  # [B, 1]

        # anchor 必须为 1
        is_anchor = (labels.squeeze(1) == 1)  # [B]

        # positive：其他 label=1
        label_matrix = labels == labels.T
        mask_positive = (label_matrix) & (labels == 1) & (~torch.eye(B, dtype=torch.bool, device=device))

        # negative：label=0
        mask_negative = (labels == 1) & (labels.T == 0)  # anchor=1, neg=0
        mask_negative = mask_negative.squeeze(1)  # [B, B]

        # 若没有 anchor 或正样本或负样本，则 loss = 0
        if is_anchor.sum() == 0 or mask_positive.sum() == 0 or mask_negative.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=embeddings.dtype)

        # ------- Compute positive & negative mean distances -------
        pos_dist_mean = (dist_matrix * mask_positive.float()).sum(dim=1) / \
                        mask_positive.sum(dim=1).clamp(min=1)

        neg_dist_mean = (dist_matrix * mask_negative.float()).sum(dim=1) / \
                        mask_negative.sum(dim=1).clamp(min=1)

        # Only keep anchors (label=1)
        pos_dist_mean = pos_dist_mean[is_anchor]
        neg_dist_mean = neg_dist_mean[is_anchor]

        # ------- Contrastive Margin Loss -------
        loss = F.relu(pos_dist_mean - neg_dist_mean + self.margin)

        # Prevent NaN/Inf
        valid_mask = torch.isfinite(loss)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=embeddings.dtype)

        return loss[valid_mask].mean()


class GNNEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=384, n_gnn_layers=2, dropout=0.5):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GATConv(input_dim if i == 0 else hidden_dim, hidden_dim//4, heads=4, concat=True)
            for i in range(n_gnn_layers)
        ])
        # self.batch_norms = nn.ModuleList([
        #     BatchNorm1d(hidden_dim)
        #     for i in range(n_gnn_layers)
        # ])
        self.batch_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for i in range(n_gnn_layers)
        ])
                
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(hidden_dim, hidden_dim)

        # pretraining heads
        self.node_decoder = nn.Linear(hidden_dim, input_dim)  # reconstruct node features
        self.adj_decoder = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, nodes_embedding, edge_index, batch, **kwargs):
        for i, (conv,batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            # nodes_embedding = conv(nodes_embedding, edge_index)
            nodes_embedding = batch_norm(conv(nodes_embedding, edge_index))

            if i != len(self.gnn_layers) - 1:
                nodes_embedding = F.leaky_relu(nodes_embedding)
                nodes_embedding = self.dropout(nodes_embedding)


        graph_embedding = global_mean_pool(nodes_embedding, batch)
        graph_embedding = self.lin(graph_embedding)

        return graph_embedding, nodes_embedding

def build_mlp(in_dim, out_dim, hidden_dim, n_layers, dropout=0.5):
    layers = []
    if n_layers == 1: 
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class Predictor(nn.Module):
    def __init__(self, llm_hidden_size, st_hidden_size,
                 hidden_dim=384, n_gnn_layers=2, n_mlplayers=2, dropout=0.5, contrastive_weight = 0.1, margin = 0):
        super(Predictor, self).__init__()

        self.gnn_encoder = GNNEncoder(st_hidden_size, hidden_dim, n_gnn_layers, dropout)

        self.task_projector = build_mlp(st_hidden_size, hidden_dim, hidden_dim, n_mlplayers, dropout)
        self.llm_projector = build_mlp(llm_hidden_size, hidden_dim, hidden_dim, n_mlplayers, dropout)

        self.final_mlp = build_mlp(hidden_dim, 1, hidden_dim, n_mlplayers, dropout)

        # self.task_embedding_batch_norm = BatchNorm1d(hidden_dim)
        # self.graph_embedding_batch_norm = BatchNorm1d(hidden_dim)
        # self.llm_embedding_batch_norm = BatchNorm1d(hidden_dim)
        self.task_embedding_batch_norm = nn.LayerNorm(hidden_dim)
        self.graph_embedding_batch_norm = nn.LayerNorm(hidden_dim)
        self.llm_embedding_batch_norm = nn.LayerNorm(hidden_dim)

        # === 可学习 [Pred] Token ===
        self.pred_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.pred_token, std=0.2)

        # === 类型编码 Type Embedding ===
        # 0 -> Pred, 1 -> LLM, 2 -> Graph, 3 -> Task
        self.type_embedding = nn.Embedding(4, hidden_dim)

        # self.fusion_encoder = nn.MultiheadAttention(
        #     embed_dim=hidden_dim,
        #     num_heads=4,
        #     dropout=dropout,
        #     batch_first=True
        # )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dropout=dropout, batch_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_mlplayers)
        
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.contrastive_loss_fn = BatchMeanContrastiveLoss(margin=margin)
        self.contrastive_weight = contrastive_weight  # 权重超参

    def forward(self, llm_embedding, nodes_embedding, edge_index, batch, task_embedding, label=None,**kwargs):
        graph_embedding, _ = self.gnn_encoder(nodes_embedding, edge_index, batch)
        graph_embedding = self.graph_embedding_batch_norm(graph_embedding)

        task_embedding = self.task_projector(task_embedding)
        task_embedding = self.task_embedding_batch_norm(task_embedding)

        llm_embedding = self.llm_projector(llm_embedding)
        llm_embedding = self.llm_embedding_batch_norm(llm_embedding)

        B = llm_embedding.size(0)
        pred_token = self.pred_token.expand(B, -1, -1)  # [B, 1, H]


        # [Pred], llm, graph, task
        embeddings = torch.stack([llm_embedding, graph_embedding, task_embedding], dim=1)  # [B, 3, H]
        embeddings = torch.cat([pred_token, embeddings], dim=1)  # [B, 4, H]

        # Pred=0, LLM=1, Graph=2, Task=3
        type_ids = torch.tensor([0, 1, 2, 3], device=llm_embedding.device).unsqueeze(0).repeat(B, 1)
        embeddings = embeddings + self.type_embedding(type_ids)

        # --- fusion ---
        # fused, _ = self.multihead_attn(
        #     query=embeddings, key=embeddings, value=embeddings, need_weights=False
        # )
        fused = self.fusion_encoder(embeddings)  # [B, 4, H]
        pred_output = fused[:, 0, :]  # [Pred] output

        logits = self.final_mlp(pred_output).squeeze(-1)

        # embedding = torch.cat([llm_embedding, graph_embedding], dim=1)
        #
        # logits = F.cosine_similarity(embedding, task_embedding, dim=-1)  # [-1,1]
        # logits = similarity * self.sim_scale  # 线性可学习缩放

        score = torch.sigmoid(logits)

        loss = None
        if label is not None:
            task_id = kwargs['task_id']

            label = label.float()
            pred_loss = self.loss_fn(logits, label)

            contrastive_loss_total = 0.0
            valid_tasks = 0

            # === 按 task_id 分组计算 contrastive Loss ===
            if task_id is not None and self.contrastive_weight > 0:            
                unique_tasks = torch.unique(task_id)
                for tid in unique_tasks:
                    mask = task_id == tid
                    if mask.sum() < 3:  # 至少3个样本才能形成contrastive
                        continue
                    sub_llm_emb = llm_embedding[mask]
                    sub_graph_emb = graph_embedding[mask]

                    sub_label = label[mask]
                    t_loss = self.contrastive_loss_fn(sub_llm_emb, sub_label)+self.contrastive_loss_fn(sub_graph_emb, sub_label)
                    contrastive_loss_total += t_loss
                    valid_tasks += 2

                if valid_tasks > 0:
                    contrastive_loss_total = contrastive_loss_total / valid_tasks
                else:
                    contrastive_loss_total = torch.tensor(0.0, device=llm_embedding.device)

                loss = pred_loss + self.contrastive_weight * contrastive_loss_total
            else:
                loss = pred_loss

        return  {
            'loss': loss,
            'score': score
        }

        
