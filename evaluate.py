import os
from utils import calculate_utility
import argparse
import torch
from transformers import AutoConfig

import pandas as pd
from custom_dataset import get_dataloader
from model import Predictor
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def validate(args,model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    workflow_count_dict = {}    
    ground_workflow_dict = {}   
    predicted_workflow_dict = {}    
    with torch.no_grad():
        for batch in loader:
            del batch['adj_true']
            for k, v in batch.items():
                batch[k] = v.to(device)
            label = batch['label']
            del batch['label']
            outputs = model(**batch)
            score = outputs['score']

            preds = (score > 0.5).float()
            for i, workflow_id in enumerate(batch['workflow_id'].tolist()):
                if workflow_id not in workflow_count_dict.keys():
                    workflow_count_dict[workflow_id] = 1
                else:
                    workflow_count_dict[workflow_id] += 1
                if workflow_id not in ground_workflow_dict.keys():
                    ground_workflow_dict[workflow_id] = 0
                if workflow_id not in predicted_workflow_dict.keys():
                    predicted_workflow_dict[workflow_id] = 0
                ground_workflow_dict[workflow_id] += label[i].cpu().item()
                predicted_workflow_dict[workflow_id] += preds[i].cpu().item()   
            y_true.extend(label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    workflow_count_dict = {k: v for k, v in workflow_count_dict.items() if v >= 3}
    ground_workflow_score_dict = {k:ground_workflow_dict[k]/workflow_count_dict[k] for k in workflow_count_dict.keys()}
    predicted_workflow_score_dict = {k:predicted_workflow_dict[k]/workflow_count_dict[k] for k in workflow_count_dict.keys()}
    utility = calculate_utility(args,ground_workflow_score_dict,predicted_workflow_score_dict)
    return utility, y_true, y_pred


def save_metrics_to_csv(csv_path, metrics_dict):
    """Append metrics to CSV, create file if not exists"""
    df = pd.DataFrame([metrics_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

def eval_main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    branches = ['test']
    test_loader = get_dataloader(args, branches)[0]

    st_hidden_size = AutoConfig.from_pretrained(args.st_model_path).hidden_size
    llm_hidden_size = AutoConfig.from_pretrained(args.llm_model_path).hidden_size

    model = Predictor(
        llm_hidden_size=llm_hidden_size,
        st_hidden_size=st_hidden_size,
        hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        n_mlplayers=args.n_mlplayers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    model_save_dir = args.cross_system if args.cross_system else args.data_path
    model_save_dir = os.path.join(model_save_dir, 'ckpt')
    os.makedirs(model_save_dir, exist_ok=True)

    best_model_path = os.path.join(model_save_dir, f'best_model.pth')
    final_model_path = os.path.join(model_save_dir, f'final_model.pth')

    # CSV 路径
    csv_path = os.path.join(f'results.csv')

    def evaluate_and_save(model_path, tag):
        model.load_state_dict(torch.load(model_path))
        utility, y_true, y_pred = validate(args, model, test_loader, device)

        accuracy = round(accuracy_score(y_true, y_pred), 4)
        precision_1 = round(precision_score(y_true, y_pred, pos_label=1), 4)
        recall_1 = round(recall_score(y_true, y_pred, pos_label=1), 4)
        f1_1 = round(f1_score(y_true, y_pred, pos_label=1), 4)

        precision_0 = round(precision_score(y_true, y_pred, pos_label=0), 4)
        recall_0 = round(recall_score(y_true, y_pred, pos_label=0), 4)
        f1_0 = round(f1_score(y_true, y_pred, pos_label=0), 4)

        avg_f1 = round((f1_1 + f1_0) / 2, 4)

        print(f'-----{tag} model-----')
        print(f"Accuracy: {accuracy}, Utility: {round(utility,4)}")
        print(f"Precision_1: {precision_1}, Recall_1: {recall_1}, F1_1: {f1_1}")
        print(f"Precision_0: {precision_0}, Recall_0: {recall_0}, F1_0: {f1_0}")
        print(f"Avg F1: {avg_f1}")
        print('-' * 50)

        metrics_dict = {
            'data_path': os.path.basename(args.data_path),
            'cross_system': os.path.basename(args.cross_system) if args.cross_system else '',
            'tag': tag,
            'accuracy': accuracy,
            'utility': round(utility,4),
            'precision_1': precision_1,
            'recall_1': recall_1,
            'f1_1': f1_1,
            'precision_0': precision_0,
            'recall_0': recall_0,
            'f1_0': f1_0,
            'avg_f1': avg_f1,
        }
        save_metrics_to_csv(csv_path, metrics_dict)

    evaluate_and_save(best_model_path, 'Best Val')
    evaluate_and_save(final_model_path, 'Final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='GLOW/data/Coding-AF', help='Path to the root data directory')
    parser.add_argument('--llm_model_path', type=str,
                        default='GLOW/outputs/prefinetuning/graph_oriented_LLM',
                        help='Path to the pre-finetuned (merged) LLM model')
    parser.add_argument('--st_model_path', type=str, default='llmmodel/all-MiniLM-L6-v2',
                        help='Path to the sentence transformer')
    parser.add_argument('--cross_system', type=str, default='GLOW/data/Coding-AF', help='model path when cross system')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--n_gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_mlplayers', type=int, default=2, help='Number of MLP layers for task embedding')
    args = parser.parse_args()

    eval_main(args)
