"""
experiments/sensitivity_alpha.py — Dirichlet Alpha Sensitivity 
Tests robustness of FedPDG vs FedAvg across non-IID settings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import Config
from utils.data_loader import DatasetLoader
from utils.partitioner import DirichletPartitioner
from models.fedpdg import FedPDGServer
from baselines.fedavg import FedAvg
from experiments.main_experiment import run_baseline_federation

def run_alpha_sweep(dataset_name, alphas, seed, config):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    loader = DatasetLoader(config)
    data_path = config.get_dataset_path(dataset_name)
    X, y_multi, label_names = loader.load_dataset(dataset_name, data_path)
    
    # Filter rare classes
    unique_k, counts_k = np.unique(y_multi, return_counts=True)
    valid_k = unique_k[counts_k >= 5]
    if len(valid_k) < len(unique_k):
        mask = np.isin(y_multi, valid_k)
        X, y_multi = X[mask], y_multi[mask]

    # Stratified split 
    # Use smaller fraction for quick sweep if needed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, test_size=0.2, stratify=y_multi, random_state=seed)
    
    X_train, X_test = loader.normalize(X_train, X_test)
    
    unique_known = np.unique(y_train)
    label_map = {old: new for new, old in enumerate(sorted(unique_known))}
    y_train = np.array([label_map[y] for y in y_train])
    y_test = np.array([label_map[y] for y in y_test])
    num_classes = len(unique_known)
    input_dim = X.shape[1]
    config.INPUT_DIM = input_dim

    # Class weights for Focal Loss (Log scaling for stability)
    _, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / np.log(1.2 + counts)
    weights = weights / np.sum(weights) * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)

    results = {'alphas': alphas, 'FedPDG': [], 'FedAvg': []}

    for alpha in alphas:
        print(f"\n{'='*50}\n  Testing Alpha = {alpha}\n{'='*50}")
        partitioner = DirichletPartitioner(config.NUM_CLIENTS, alpha, seed)
        client_data = partitioner.partition(X_train, y_train)

        # 1. FedPDG
        print("\n--- FedPDG ---")
        server = FedPDGServer(config, num_classes, input_dim, class_weights=class_weights)
        history = server.run_federation(
            client_data, byzantine_ids=[], test_data=(X_test, y_test),
            eval_every=5
        )
        fedpdg_f1 = history['f1_macro'][-1] if history['f1_macro'] else 0
        results['FedPDG'].append(fedpdg_f1)
        print(f"  FedPDG F1 (α={alpha}): {fedpdg_f1:.4f}")

        # 2. FedAvg
        print("\n--- FedAvg ---")
        fedavg_metrics = run_baseline_federation(
            config, client_data, (X_test, y_test), FedAvg(),
            num_classes, input_dim, byzantine_ids=[],
            class_weights=class_weights
        )
        fedavg_f1 = fedavg_metrics['f1_macro']
        results['FedAvg'].append(fedavg_f1)
        print(f"  FedAvg F1 (α={alpha}): {fedavg_f1:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CICIDS2017')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=20)
    args = parser.parse_args()

    config = Config()
    config.NUM_ROUNDS = args.rounds
    
    alphas = [0.1, 0.3, 0.5, 1.0]
    results = run_alpha_sweep(args.dataset, alphas, args.seed, config)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(config.RESULTS_DIR, f'alpha_sweep_{args.dataset}.json')
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAlpha sweep results saved to {save_path}")

if __name__ == '__main__':
    main()
