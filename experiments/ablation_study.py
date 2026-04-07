"""
experiments/ablation_study.py — Component Ablation
Tests each FedPDG component by removing it and measuring degradation.
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
from experiments.main_experiment import run_baseline_federation
from models.fedpdg import FedPDGServer
from baselines.fedavg import FedAvg


def run_ablation(dataset_name, seed, config):
    """Run ablation study: remove PDS, DWA, APS one at a time."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load and prepare data (same preprocessing as main)
    loader = DatasetLoader(config)
    data_path = config.get_dataset_path(dataset_name)
    X, y_multi, label_names = loader.load_dataset(dataset_name, data_path)
    input_dim = X.shape[1]
    config.INPUT_DIM = input_dim

    (X_known, y_known, X_zd, y_zd,
     known_classes, holdout_classes) = loader.zeroday_split(
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
    X_zd_norm = np.clip(loader.scaler.transform(X_zd), -10, 10).astype(np.float32)

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
    byz = ByzantineInjector('label_flip', 0.2, seed)
    byzantine_ids = byz.select_byzantine_clients(config.NUM_CLIENTS)

    results = {}

    # Variant 1: Full FedPDG
    print("\n--- Variant 1: Full FedPDG ---")
    server = FedPDGServer(config, num_classes, input_dim, class_weights=class_weights)
    history = server.run_federation(
        client_data, byzantine_ids=byzantine_ids,
        test_data=(X_test, y_test), eval_every=config.NUM_ROUNDS)
    if history['accuracy']:
        results['Full_FedPDG'] = {
            'accuracy': history['accuracy'][-1],
            'f1_macro': history['f1_macro'][-1],
        }

    # Variant 2: FedPDG w/o PDS (use all clients equally)
    print("\n--- Variant 2: w/o PDS ---")
    cfg_no_pds = Config()
    cfg_no_pds.INPUT_DIM = input_dim
    cfg_no_pds.NUM_ROUNDS = config.NUM_ROUNDS
    cfg_no_pds.PDS_GAMMA = 999.0  # never triggers
    server2 = FedPDGServer(cfg_no_pds, num_classes, input_dim, class_weights=class_weights)
    h2 = server2.run_federation(
        client_data, byzantine_ids=byzantine_ids,
        test_data=(X_test, y_test), eval_every=config.NUM_ROUNDS)
    if h2['accuracy']:
        results['No_PDS'] = {
            'accuracy': h2['accuracy'][-1],
            'f1_macro': h2['f1_macro'][-1],
        }

    # Variant 3: FedPDG w/o DWA (use FedAvg aggregation)
    print("\n--- Variant 3: w/o DWA (FedAvg aggregation) ---")
    metrics_no_dwa = run_baseline_federation(
        config, client_data, (X_test, y_test), FedAvg(),
        num_classes, input_dim, byzantine_ids,
        class_weights=class_weights)
    results['No_DWA'] = metrics_no_dwa

    # Variant 4: FedPDG w/o APS (no prototype spawning)
    print("\n--- Variant 4: w/o APS ---")
    cfg_no_aps = Config()
    cfg_no_aps.INPUT_DIM = input_dim
    cfg_no_aps.NUM_ROUNDS = config.NUM_ROUNDS
    cfg_no_aps.APS_MIN_CLUSTER_SIZE = 999999  # never spawns
    server4 = FedPDGServer(cfg_no_aps, num_classes, input_dim)
    h4 = server4.run_federation(
        client_data, byzantine_ids=byzantine_ids,
        test_data=(X_test, y_test), eval_every=config.NUM_ROUNDS)
    if h4['accuracy']:
        results['No_APS'] = {
            'accuracy': h4['accuracy'][-1],
            'f1_macro': h4['f1_macro'][-1],
        }

    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    for variant, metrics in results.items():
        acc = metrics.get('accuracy', 0)
        f1 = metrics.get('f1_macro', 0)
        print(f"  {variant:20s}: Acc={acc:.4f} | F1={f1:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='FedPDG Ablation Study')
    parser.add_argument('--dataset', default='CICIDS2017')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=50)
    args = parser.parse_args()

    config = Config()
    config.NUM_ROUNDS = args.rounds

    results = run_ablation(args.dataset, args.seed, config)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(
        config.RESULTS_DIR,
        f'ablation_{args.dataset}_seed{args.seed}.json')

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nAblation results saved → {save_path}")


if __name__ == '__main__':
    main()
