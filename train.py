import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig
from custom_dataset import get_dataloader, get_pretrain_dataloader
from evaluate import eval_main
from model import Predictor
import numpy as np
import random
from itertools import chain

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, loader, optimizer, device):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    i =0

    if isinstance(loader, list):
        num_batches = len(loader[0])+len(loader[1])
        loader = chain(*loader) 

    else:
        num_batches = len(loader)

    model.train()
    for batch in tqdm(loader,leave=False):
        del batch['adj_true']
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch['label']

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs['loss']
        score = outputs['score']

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (score > 0.5).float()  
        correct_predictions += (preds == label).sum().item()
        total_predictions += label.size(0)

        i+=1

    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            del batch['adj_true']
            for k, v in batch.items():
                batch[k] = v.to(device)
            label = batch['label']
            outputs = model(**batch)
            loss = outputs['loss']
            score = outputs['score']
            total_loss += loss.item()
            preds = (score > 0.5).float()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct_predictions += (preds == label).sum().item()
            total_predictions += label.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    model.train()
    return avg_loss, accuracy


def pretrain_GNN(model, loader, optimizer, device, max_step=500, scheduler=None):
    model.train()
    total_loss = 0
    step = 0

    # 将 loader 转为无限迭代器
    from itertools import cycle
    loader_iter = cycle(loader)

    from tqdm import tqdm
    pbar = tqdm(range(max_step), desc="Pretraining Steps", leave=False)
    for _ in pbar:
        batch = next(loader_iter)
        adj_true_list = batch['adj_true']
        batch = {k: v for k, v in batch.items() if k != 'adj_true'}
        for k, v in batch.items():
            batch[k] = v.to(device)

        _, nodes_embedding = model.gnn_encoder(batch['nodes_embedding'], batch['edge_index'], batch['batch'])
        input_nodes_embedding = batch['nodes_embedding']

        # ===== 节点重建 =====
        node_recon = model.gnn_encoder.node_decoder(nodes_embedding)
        loss_node = F.mse_loss(node_recon, input_nodes_embedding)

        # ===== 邻接矩阵重建 =====
        b = batch['batch']

        loss_adj = 0
        num_graphs = 0
        for i, (num_nodes, adj_true) in enumerate(zip(b.bincount(), adj_true_list)):
            idx = (b == i).nonzero(as_tuple=True)[0]
            h_sub = nodes_embedding[idx]

            h_sub_i = h_sub.unsqueeze(1).repeat(1, num_nodes, 1)
            h_sub_j = h_sub.unsqueeze(0).repeat(num_nodes, 1, 1)
            adj_pred = torch.sigmoid(model.gnn_encoder.adj_decoder(h_sub_i, h_sub_j).squeeze(-1))

            # print(adj_pred, adj_true)
            adj_true = adj_true.to(adj_pred.device)
            loss_adj += F.binary_cross_entropy(adj_pred, adj_true)
            num_graphs += 1

        loss_adj /= num_graphs
        loss = loss_node + loss_adj

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # 每个 step 调度一次

        total_loss += loss.item()
        step += 1
        pbar.set_postfix({'current loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

    avg_loss = total_loss / max_step
    return avg_loss



def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    import os
    args.domain = os.path.basename(args.data_path)
    pretrain_loader = get_pretrain_dataloader(args, ['train'])[0]

    branches = ['train','val']
    train_loader, val_loader = get_dataloader(args, branches)

    st_hidden_size = AutoConfig.from_pretrained(args.st_model_path).hidden_size
    llm_hidden_size = AutoConfig.from_pretrained(args.llm_model_path).hidden_size

    model = Predictor(llm_hidden_size=llm_hidden_size, st_hidden_size=st_hidden_size, hidden_dim=args.hidden_dim,
                      n_gnn_layers=args.n_gnn_layers,n_mlplayers=args.n_mlplayers, dropout=args.dropout, contrastive_weight=args.contrastive_weight, margin= args.margin).to(device)

    ## pretraining
    pretrain_optimizer = torch.optim.AdamW(model.gnn_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 使用 StepLR，每100步学习率衰减0.95倍
    pretrain_scheduler = torch.optim.lr_scheduler.StepLR(pretrain_optimizer, step_size=100, gamma=0.95)

    pretrain_loss = pretrain_GNN(model, pretrain_loader, pretrain_optimizer, device, max_step=args.pretrain_steps,
                             scheduler=pretrain_scheduler)
    print(f'Pretraining finished. Avg Loss: {pretrain_loss:.4f}')

    # training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    model_save_dir = args.data_path + '/ckpt'
    import os 
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    best_model_path = os.path.join(model_save_dir, f'best_model.pth')

    best_val_acc = 0
    no_improve = 0
    best_epoch = 0  # 记录最佳模型的epoch
    for epoch in tqdm(range(1, args.epochs + 1), leave=False, desc='Epochs'):
        if no_improve > args.patience:
            break

        train_loss, train_acc = train(model, train_loader, optimizer, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % args.eval_steps == 0:
            val_loss, val_acc = validate(model, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch  # 保存当前epoch
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model with Val Acc: {val_acc:.4f} at epoch {epoch}\r')
                no_improve = 0
            else:
                no_improve += 1

            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr}')
        else:
            no_improve += 1
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {current_lr}')

    final_model_path = os.path.join(model_save_dir, f'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    # ===== 在训练结束后再打印一次结果 =====
    print(f'\nTraining finished. Best model saved at {best_model_path} with Val Acc: {best_val_acc:.4f} at epoch {best_epoch}')
    print(f'\nTraining finished. Final model saved at {final_model_path} with Val Acc: {val_acc:.4f} at epoch {epoch}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # === Required paths ===
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to the root data directory (e.g., ./data/Coding-AF).'
    )
    parser.add_argument(
        '--llm_model_path', type=str, required=True,
        help='Path to the pre-finetuned (merged) LLM model (e.g., GLOW/outputs/prefinetuning/graph_oriented_LLM).'
    )
    parser.add_argument(
        '--st_model_path', type=str, required=True,
        help='Path to the Sentence Transformer model (e.g., llmmodel/all-MiniLM-L6-v2).'
    )

    # === Model and training hyperparameters ===
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')  
    parser.add_argument('--n_gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation')
    parser.add_argument('--pretrain_batch_size', type=int, default=64, help='Batch size for pretraining')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  #1e-4
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_mlplayers', type=int, default=2, help='Number of MLP layers')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--contrastive_weight', type=float, default=1)
    parser.add_argument('--margin', type=float, default=0.2)  #The larger the margin, the farther the distance between pos and neg samples
    parser.add_argument('--pretrain_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=1)
    args = parser.parse_args()
    import time
    start = time.time()  # 记录开始时间
    main(args)
    end = time.time()  # 记录结束时间

    print(f"{args.data_path}; training time: {end - start:.4f} seconds")

    from pprint import pprint
    pprint(vars(args))

    args.cross_system = args.data_path
    print(f'For dataset: {args.cross_system}')
    eval_main(args)

    # # different generation methods
    # if args.cross_system.endswith('-GD'):
    #     args.data_path = args.cross_system[:-3] + '-AF'  # 去掉末尾的 -GD，换成 -AF
    #     print(f'{args.domain}-{os.path.basename(args.data_path)}')
    #     eval_main(args)
    # elif args.cross_system.endswith('-AF'):
    #     args.data_path = args.cross_system[:-3] + '-GD'  # 去掉末尾的 -AF，换成 -GD
    #     print(f'{args.domain}-{os.path.basename(args.data_path)}')
    #     eval_main(args)


    # # different task
    # cross_domains = {
    #     'Coding-AF': ['Math-AF', 'Reason-AF'],
    #     'Math-AF': ['Coding-AF', 'Reason-AF'],
    #     'Reason-AF': ['Math-AF', 'Coding-AF']
    # }

    # if args.domain in cross_domains:
    #     for target in cross_domains[args.domain]:
    #         args.data_path = os.path.join(os.path.dirname(args.cross_system), target)
    #         print(f'{args.domain}-{target}')
    #         eval_main(args)
