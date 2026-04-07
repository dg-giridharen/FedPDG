"""
experiments/main_experiment.py — Main Experiment Runner
Runs FedPDG and all baselines across datasets with multiple seeds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import copy
from collections import defaultdict

from config import Config
from utils.data_loader import DatasetLoader
from utils.partitioner import DirichletPartitioner, ByzantineInjector
from utils.metrics import compute_all_metrics, compute_byzantine_metrics
from models.fedpdg import FedPDGServer
from models.transformer_encoder import TabularTransformerEncoder
from models.contrastive_loss import CombinedLoss
from baselines.fedavg import FedAvg
from baselines.fedprox import FedProx
from baselines.krum import Krum
from baselines.fltrust import FLTrust
from baselines.flame import FLAME


def run_baseline_federation(config, client_data, test_data, aggregator,
                            num_classes, input_dim, byzantine_ids=None,
                            byzantine_attack='label_flip', class_weights=None):
    """Run a baseline FL method and return final metrics."""
    device = config.DEVICE
    model = TabularTransformerEncoder(
        input_dim=input_dim, embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS,
    ).to(device)
    model.build_classifier(num_classes)
    criterion = CombinedLoss(config.TEMPERATURE, config.LAMBDA_CON, class_weights)
    global_state = copy.deepcopy(model.state_dict())

    if byzantine_ids is None:
        byzantine_ids = []

    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    for rnd in range(1, config.NUM_ROUNDS + 1):
        client_models = {}

        for cid, (X_c, y_c) in enumerate(client_data):
            if len(X_c) == 0:
                continue

            local_model = TabularTransformerEncoder(
                input_dim=input_dim, embed_dim=config.EMBED_DIM,
                num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS,
            ).to(device)
            local_model.build_classifier(num_classes)
            local_model.load_state_dict(global_state)
            local_model.train()

            optimizer = torch.optim.AdamW(
                local_model.parameters(), lr=config.LR,
                weight_decay=config.WEIGHT_DECAY)

            y_train = y_c.copy()
            if cid in byzantine_ids and byzantine_attack == 'label_flip':
                for i in range(len(y_train)):
                    y_train[i] = (y_train[i] + 1) % num_classes

            X_t = torch.tensor(X_c, dtype=torch.float32)
            y_t = torch.tensor(y_train, dtype=torch.long)
            loader = DataLoader(TensorDataset(X_t, y_t),
                                batch_size=config.BATCH_SIZE, shuffle=True,
                                drop_last=len(X_c) > config.BATCH_SIZE)

            for _ in range(config.LOCAL_EPOCHS):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits, features = local_model(xb)
                    loss, _, _ = criterion(logits, features, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()

            state = copy.deepcopy(local_model.state_dict())
            if cid in byzantine_ids and byzantine_attack == 'sign_flip':
                state = {k: -v for k, v in state.items()}
            elif cid in byzantine_ids and byzantine_attack == 'gaussian':
                state = {k: v + torch.randn_like(v) * 0.1
                         for k, v in state.items()}

            client_models[cid] = state

        # Aggregate
        if isinstance(aggregator, (FedAvg, FedProx, Krum, FLAME)):
            global_state = aggregator.aggregate(client_models)
        elif isinstance(aggregator, FLTrust):
            # FLTrust needs updates, not state params
            client_updates = {}
            for cid, state in client_models.items():
                client_updates[cid] = {k: state[k] - global_state[k] for k in state.keys()}
            # Server pristine update (use 1000 samples of test data as root data)
            root_data = (test_data[0][:1000], test_data[1][:1000])
            server_update = aggregator.compute_server_update(
                model, root_data, device, criterion, lr=config.LR, epochs=1)
            global_state = aggregator.aggregate(client_updates, server_update, global_state)
        else:
            global_state = FedAvg().aggregate(client_models)

        if rnd % 20 == 0:
            print(f"  Baseline round {rnd}/{config.NUM_ROUNDS}")

    # Final evaluation (batched to avoid CUDA OOM)
    model.load_state_dict(global_state)
    model.eval()
    X_test, y_test = test_data
    all_preds = []
    batch_sz = 4096
    with torch.no_grad():
        for i in range(0, len(X_test), batch_sz):
            X_batch = torch.tensor(
                X_test[i:i+batch_sz], dtype=torch.float32).to(device)
            logits, _ = model(X_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)

    return compute_all_metrics(y_test, preds)


def run_experiment(dataset_name, seed, config):
    """Run complete experiment for one dataset and seed."""
    print(f"\n{'#'*60}")
    print(f"  Dataset: {dataset_name} | Seed: {seed}")
    print(f"{'#'*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    loader = DatasetLoader(config)
    data_path = config.get_dataset_path(dataset_name)

    X, y_multi, label_names = loader.load_dataset(dataset_name, data_path)
    input_dim = X.shape[1]
    config.INPUT_DIM = input_dim

    # Zero-day split
    (X_known, y_known, X_zd, y_zd,
     known_classes, holdout_classes) = loader.zeroday_split(
        X, y_multi, label_names, holdout_n=config.HOLDOUT_CLASSES, seed=seed)

    # Filter out rare classes that break stratified splitting
    unique_k, counts_k = np.unique(y_known, return_counts=True)
    valid_k = unique_k[counts_k >= 5]
    if len(valid_k) < len(unique_k):
        mask = np.isin(y_known, valid_k)
        X_known, y_known = X_known[mask], y_known[mask]
        print(f"  Filtered {len(unique_k)-len(valid_k)} rare known classes "
              f"(<5 samples)")

    # Train/test split (from known data)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=0.2, stratify=y_known, random_state=seed)

    # Normalize
    X_train, X_test = loader.normalize(X_train, X_test)
    # Apply same scaler to zero-day data
    X_zd_norm = loader.scaler.transform(X_zd).astype(np.float32)
    X_zd_norm = np.clip(X_zd_norm, -10, 10)

    # Remap labels to 0..N-1
    unique_known = np.unique(y_train)
    label_map = {old: new for new, old in enumerate(sorted(unique_known))}
    y_train = np.array([label_map[y] for y in y_train])
    y_test = np.array([label_map[y] for y in y_test])
    num_classes = len(unique_known)

    # Compute class weights for Focal Loss (Log scaling for stability)
    _, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / np.log(1.2 + counts)
    weights = weights / np.sum(weights) * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)

    # Partition into clients
    alpha = 0.5
    partitioner = DirichletPartitioner(config.NUM_CLIENTS, alpha, seed)
    print(f"\nPartitioning (α={alpha}):")
    client_data = partitioner.partition(X_train, y_train)

    # Byzantine injection
    byz = ByzantineInjector('label_flip', 0.2, seed)
    byzantine_ids = byz.select_byzantine_clients(config.NUM_CLIENTS)

    results = {}

    # ── Run FedPDG ──────────────────────────────────────────
    print(f"\n--- FedPDG ---")
    server = FedPDGServer(config, num_classes, input_dim, class_weights=class_weights)
    history = server.run_federation(
        client_data, byzantine_ids=byzantine_ids,
        test_data=(X_test, y_test),
        zeroday_test_data=(X_zd_norm, y_zd),
        known_class_ids=np.arange(num_classes),
        eval_every=10,
    )
    if history['accuracy']:
        results['FedPDG'] = {
            'accuracy': history['accuracy'][-1],
            'f1_macro': history['f1_macro'][-1],
            'detection_rate': history['detection_rate'][-1],
            'false_alarm_rate': history['false_alarm_rate'][-1],
            'convergence': history,
        }

    # ── Run Baselines ───────────────────────────────────────
    baselines = {
        'FedAvg': FedAvg(),
        'FedProx': FedProx(mu=0.01),
        'Krum': Krum(num_byzantine=len(byzantine_ids)),
        'FLAME': FLAME(noise_multiplier=0.001),
        'FLTrust': FLTrust(),
    }

    for name, agg in baselines.items():
        print(f"\n--- {name} ---")
        metrics = run_baseline_federation(
            config, client_data, (X_test, y_test), agg,
            num_classes, input_dim, byzantine_ids,
            class_weights=class_weights)
        results[name] = metrics
        print(f"  {name}: Acc={metrics['accuracy']:.4f} | "
              f"F1={metrics['f1_macro']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='FedPDG Main Experiment')
    parser.add_argument('--dataset', default='CICIDS2017',
                        choices=['CICIDS2017', 'ToN-IoT', 'NbAIoT'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--clients', type=int, default=5)
    args = parser.parse_args()

    config = Config()
    config.NUM_ROUNDS = args.rounds
    config.NUM_CLIENTS = args.clients

    results = run_experiment(args.dataset, args.seed, config)

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(
        config.RESULTS_DIR,
        f'results_{args.dataset}_seed{args.seed}.json')

    # Convert numpy types for JSON
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
    print(f"\nResults saved → {save_path}")


if __name__ == '__main__':
    main()
