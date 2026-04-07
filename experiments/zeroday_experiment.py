"""
experiments/zeroday_experiment.py — Zero-Day Detection Evaluation
Tests APS's ability to detect and spawn prototypes for unseen attacks.
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
from utils.partitioner import DirichletPartitioner
from models.fedpdg import FedPDGServer


def run_zeroday_experiment(dataset_name, seed, config, holdout_n=2):
    """
    Evaluate zero-day detection:
    1. Train on known classes only
    2. Inject zero-day traffic at specific round
    3. Measure if PDS detects it and APS spawns a prototype
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = DatasetLoader(config)
    data_path = config.get_dataset_path(dataset_name)
    X, y_multi, label_names = loader.load_dataset(dataset_name, data_path)
    input_dim = X.shape[1]
    config.INPUT_DIM = input_dim

    (X_known, y_known, X_zd, y_zd,
     known_classes, holdout_classes) = loader.zeroday_split(
        X, y_multi, label_names, holdout_n=holdout_n, seed=seed)

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

    # Partition known data into clients
    partitioner = DirichletPartitioner(config.NUM_CLIENTS, 0.5, seed)
    client_data = partitioner.partition(X_train, y_train)

    # Inject zero-day data into one client (simulate ZD appearing)
    inject_client = 2  # Client 2 will start seeing zero-day traffic
    if len(X_zd_norm) > 0:
        # Add zero-day samples to client 2's data
        X_c, y_c = client_data[inject_client]
        # Label zero-day as class 0 (they'll have wrong labels —
        # PDS should detect the prototype shift)
        y_zd_fake = np.zeros(len(X_zd_norm), dtype=y_c.dtype)
        X_combined = np.concatenate([X_c, X_zd_norm[:len(X_c)//2]], axis=0)
        y_combined = np.concatenate([y_c, y_zd_fake[:len(X_c)//2]], axis=0)
        client_data[inject_client] = (X_combined, y_combined)
        print(f"\n  Injected {len(X_zd_norm[:len(X_c)//2])} zero-day samples "
              f"into Client {inject_client}")

    # Run FedPDG
    server = FedPDGServer(config, num_classes, input_dim)
    history = server.run_federation(
        client_data,
        test_data=(X_test, y_test),
        zeroday_test_data=(X_zd_norm, y_zd),
        known_class_ids=np.arange(num_classes),
        eval_every=5,
    )

    results = {
        'dataset': dataset_name,
        'seed': seed,
        'holdout_classes': holdout_classes.tolist() if hasattr(
            holdout_classes, 'tolist') else list(holdout_classes),
        'holdout_names': [label_names[c] for c in holdout_classes],
        'classes_spawned': server.aps.get_num_spawned(),
        'spawn_history': server.aps.get_spawn_history(),
        'pds_history': {k: v for k, v in server.pds.get_pds_history().items()},
        'final_accuracy': history['accuracy'][-1] if history['accuracy'] else 0,
        'final_f1': history['f1_macro'][-1] if history['f1_macro'] else 0,
        'zd_detection_rates': history.get('zd_detection_rate', []),
        'inject_client': inject_client,
    }

    print(f"\n  ZERO-DAY RESULTS:")
    print(f"    Holdout classes: {results['holdout_names']}")
    print(f"    Classes spawned: {results['classes_spawned']}")
    print(f"    Final Acc: {results['final_accuracy']:.4f}")
    print(f"    Final F1:  {results['final_f1']:.4f}")
    if results['zd_detection_rates']:
        print(f"    ZD Detection Rate: {results['zd_detection_rates'][-1]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CICIDS2017')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--holdout', type=int, default=2)
    args = parser.parse_args()

    config = Config()
    config.NUM_ROUNDS = args.rounds

    results = run_zeroday_experiment(
        args.dataset, args.seed, config, args.holdout)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(
        config.RESULTS_DIR,
        f'zeroday_{args.dataset}_seed{args.seed}.json')

    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nZero-day results saved → {save_path}")


if __name__ == '__main__':
    main()
