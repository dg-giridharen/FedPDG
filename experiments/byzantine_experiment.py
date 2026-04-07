"""
experiments/byzantine_experiment.py — Byzantine Robustness Evaluation
Tests F1 degradation across increasing Byzantine ratios and attack types.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch

from config import Config
from utils.data_loader import DatasetLoader
from utils.partitioner import DirichletPartitioner, ByzantineInjector
from utils.metrics import compute_byzantine_metrics
from experiments.main_experiment import run_baseline_federation
from models.fedpdg import FedPDGServer
from baselines.fedavg import FedAvg
from baselines.krum import Krum
from baselines.flame import FLAME


def run_byzantine_sweep(dataset_name, seed, config):
    """Test all methods across increasing Byzantine ratios."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = DatasetLoader(config)
    data_path = config.get_dataset_path(dataset_name)
    X, y_multi, label_names = loader.load_dataset(dataset_name, data_path)
    input_dim = X.shape[1]
    config.INPUT_DIM = input_dim

    (X_known, y_known, _, _, _, _) = loader.zeroday_split(
        X, y_multi, label_names, holdout_n=config.HOLDOUT_CLASSES, seed=seed)

    # Filter rare classes
    unique_k, counts_k = np.unique(y_known, return_counts=True)
    valid_k = unique_k[counts_k >= 5]
    if len(valid_k) < len(unique_k):
        mask = np.isin(y_known, valid_k)
        X_known, y_known = X_known[mask], y_known[mask]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=0.2, stratify=y_known, random_state=seed)
    X_train, X_test = loader.normalize(X_train, X_test)

    unique_known = np.unique(y_train)
    label_map = {old: new for new, old in enumerate(sorted(unique_known))}
    y_train = np.array([label_map[y] for y in y_train])
    y_test = np.array([label_map[y] for y in y_test])
    num_classes = len(unique_known)

    partitioner = DirichletPartitioner(config.NUM_CLIENTS, 0.5, seed)
    client_data = partitioner.partition(X_train, y_train)

    # Class weights for Focal Loss (Log scaling for stability)
    _, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / np.log(1.2 + counts)
    weights = weights / np.sum(weights) * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)

    all_results = {}

    for byz_ratio in config.BYZANTINE_RATIOS:
        print(f"\n{'='*60}")
        print(f"Byzantine Ratio: {byz_ratio*100:.0f}%")
        print(f"{'='*60}")

        byz = ByzantineInjector('label_flip', byz_ratio, seed)
        byzantine_ids = byz.select_byzantine_clients(config.NUM_CLIENTS)
        n_byz = len(byzantine_ids)

        ratio_results = {}

        # FedPDG
        print(f"\n  --- FedPDG ---")
        server = FedPDGServer(config, num_classes, input_dim, class_weights=class_weights)
        history = server.run_federation(
            client_data, byzantine_ids=byzantine_ids,
            test_data=(X_test, y_test), eval_every=config.NUM_ROUNDS)

        if history['f1_macro']:
            detected = history['byzantine_detected'][-1] if history['byzantine_detected'] else 0
            ratio_results['FedPDG'] = {
                'f1_macro': history['f1_macro'][-1],
                'accuracy': history['accuracy'][-1],
                'byz_detected': detected,
            }
            byz_metrics = compute_byzantine_metrics(
                byzantine_ids,
                list(range(detected)),  # simplified
                config.NUM_CLIENTS)
            ratio_results['FedPDG'].update(byz_metrics)

        # Baselines
        for name, agg in [('FedAvg', FedAvg()),
                          ('Krum', Krum(n_byz)),
                          ('FLAME', FLAME())]:
            print(f"\n  --- {name} ---")
            metrics = run_baseline_federation(
                config, client_data, (X_test, y_test), agg,
                num_classes, input_dim, byzantine_ids,
                class_weights=class_weights)
            ratio_results[name] = metrics

        all_results[byz_ratio] = ratio_results

        for method, m in ratio_results.items():
            print(f"    {method:12s}: F1={m.get('f1_macro', 0):.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CICIDS2017')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=50)
    args = parser.parse_args()

    config = Config()
    config.NUM_ROUNDS = args.rounds

    results = run_byzantine_sweep(args.dataset, args.seed, config)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(
        config.RESULTS_DIR,
        f'byzantine_{args.dataset}_seed{args.seed}.json')

    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nByzantine results saved → {save_path}")


if __name__ == '__main__':
    main()
