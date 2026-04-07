"""
models/fedpdg.py — Complete FedPDG Training Loop
Algorithm 1 from the paper: Federation with PDS → DWA → APS.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from collections import OrderedDict

from models.transformer_encoder import TabularTransformerEncoder
from models.prototype_layer import PrototypeLayer
from models.contrastive_loss import CombinedLoss
from components.pds import PrototypeDivergenceSignal
from components.dwa import DivergenceWeightedAggregation
from components.aps import AutomaticPrototypeSpawning


class FedPDGClient:
    """Local client for FedPDG federation."""

    def __init__(self, client_id, config, num_classes, class_weights=None):
        self.id = client_id
        self.cfg = config
        self.device = config.DEVICE
        self.num_classes = num_classes

        # Model
        self.model = TabularTransformerEncoder(
            input_dim=config.INPUT_DIM,
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
        ).to(self.device)
        self.model.build_classifier(num_classes)

        # Prototype layer
        self.prototype_layer = PrototypeLayer(
            config.EMBED_DIM, num_classes).to(self.device)

        # Loss
        self.criterion = CombinedLoss(
            temperature=config.TEMPERATURE,
            lambda_con=config.LAMBDA_CON,
            class_weights=class_weights)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_ROUNDS)

    def local_train(self, X_train, y_train, global_state_dict,
                    global_prototypes=None, byzantine=False,
                    byzantine_attack='label_flip'):
        """
        Local training for E epochs.

        Returns:
            state_dict: updated model parameters
            local_prototypes: dict {class_id: tensor}
            train_loss: float
        """
        # Load global model
        self.model.load_state_dict(global_state_dict)
        self.model.train()

        # Set global prototypes if provided
        if global_prototypes:
            self.prototype_layer.set_prototypes(global_prototypes)

        # Byzantine: poison labels
        if byzantine and byzantine_attack == 'label_flip':
            y_train = self._poison_labels(y_train)

        # Create dataloader
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE,
                            shuffle=True, drop_last=len(X_train) > self.cfg.BATCH_SIZE)

        total_loss = 0.0
        steps = 0

        for epoch in range(self.cfg.LOCAL_EPOCHS):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                logits, features = self.model(X_batch)
                loss, l_ce, l_con = self.criterion(logits, features, y_batch)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update prototypes
                with torch.no_grad():
                    self.prototype_layer.update_prototypes(
                        features.detach(), y_batch)

                total_loss += loss.item()
                steps += 1

        self.scheduler.step()

        # Get model state
        state_dict = copy.deepcopy(self.model.state_dict())

        # Byzantine: poison parameters
        if byzantine and byzantine_attack == 'sign_flip':
            state_dict = {k: -v for k, v in state_dict.items()}
        elif byzantine and byzantine_attack == 'gaussian':
            state_dict = {k: v + torch.randn_like(v) * 0.1
                          for k, v in state_dict.items()}
        elif byzantine and byzantine_attack == 'model_replace':
            state_dict = {k: torch.randn_like(v)
                          for k, v in state_dict.items()}

        local_prototypes = self.prototype_layer.get_prototypes()
        avg_loss = total_loss / max(steps, 1)
        return state_dict, local_prototypes, avg_loss

    def get_embeddings(self, X_data):
        """Get embeddings for APS outlier analysis."""
        self.model.eval()
        X_t = torch.tensor(X_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z_norm, h = self.model(X_t, return_embeddings=True)
        return z_norm

    def _poison_labels(self, y):
        y_p = y.copy()
        for i in range(len(y_p)):
            y_p[i] = (y_p[i] + 1) % self.num_classes
        return y_p


class FedPDGServer:
    """Central server orchestrating the FedPDG federation."""

    def __init__(self, config, num_classes, input_dim, class_weights=None):
        self.cfg = config
        self.cfg.INPUT_DIM = input_dim
        self.device = config.DEVICE
        self.num_classes = num_classes
        self.class_weights = class_weights

        # Global model
        self.global_model = TabularTransformerEncoder(
            input_dim=input_dim,
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
        ).to(self.device)
        self.global_model.build_classifier(num_classes)

        self.global_prototypes = {}

        # Components
        self.pds = PrototypeDivergenceSignal(config)
        self.dwa = DivergenceWeightedAggregation(config)
        self.aps = AutomaticPrototypeSpawning(config)
        self.aps.set_next_class_id(num_classes)

        # Logging
        self.round_logs = []

    def run_federation(self, client_data, byzantine_ids=None,
                       byzantine_attack='label_flip',
                       test_data=None, zeroday_test_data=None,
                       known_class_ids=None, eval_every=5):
        """
        Main FL training loop — Algorithm 1.

        Args:
            client_data:      list of (X_train, y_train) per client
            byzantine_ids:    list of Byzantine client indices
            byzantine_attack: type of Byzantine attack
            test_data:        (X_test, y_test) for closed-set eval
            zeroday_test_data: (X_zd, y_zd) for open-set eval
            known_class_ids:  array of known class IDs
            eval_every:       evaluate global model every N rounds

        Returns:
            metrics_history: dict of per-round metrics
        """
        if byzantine_ids is None:
            byzantine_ids = []

        num_clients = len(client_data)

        # Initialize clients
        clients = [
            FedPDGClient(i, self.cfg, self.num_classes, class_weights=self.class_weights)
            for i in range(num_clients)
        ]

        global_state = copy.deepcopy(self.global_model.state_dict())

        history = {
            'round': [], 'accuracy': [], 'f1_macro': [],
            'detection_rate': [], 'false_alarm_rate': [],
            'pds_scores': [], 'byzantine_detected': [],
            'zeroday_detected': [], 'classes_spawned': [],
            'zd_detection_rate': [],
        }

        for rnd in range(1, self.cfg.NUM_ROUNDS + 1):
            print(f"\n{'='*60}")
            print(f"  ROUND {rnd}/{self.cfg.NUM_ROUNDS}")
            print(f"{'='*60}")

            # ── 1. Local Training ───────────────────────────────────
            client_models = {}
            client_prototypes = {}

            for cid in range(num_clients):
                is_byz = cid in byzantine_ids
                X_c, y_c = client_data[cid]

                if len(X_c) == 0:
                    continue

                state, protos, loss = clients[cid].local_train(
                    X_c, y_c, global_state,
                    self.global_prototypes,
                    byzantine=is_byz,
                    byzantine_attack=byzantine_attack,
                )
                client_models[cid] = state
                client_prototypes[cid] = protos

                tag = ' [BYZ]' if is_byz else ''
                print(f"  Client {cid:2d}: loss={loss:.4f}{tag}")

            if not client_models:
                continue

            # ── 2. PDS Evaluation ───────────────────────────────────
            if self.global_prototypes:
                pds_results = self.pds.evaluate_all_clients(
                    client_prototypes, self.global_prototypes)
            else:
                pds_results = {
                    cid: {
                        'status': 'normal', 'trust_weight': 1.0,
                        'pds_score': 0.0, 'gini_index': 0.0,
                        'per_class_div': {},
                    }
                    for cid in client_models
                }

            # ── 3. DWA Aggregation ──────────────────────────────────
            new_global, new_protos, agg_stats = self.dwa.aggregate(
                client_models, client_prototypes,
                pds_results, global_state)

            global_state = new_global
            if new_protos:
                self.global_prototypes.update(new_protos)
            self.global_model.load_state_dict(global_state)

            # ── 4. APS — Zero-day Class Discovery ───────────────────
            zd_clients = agg_stats.get('zeroday_clients', [])
            zd_embs = []
            for cid in zd_clients:
                X_c, y_c = client_data[cid]
                if len(X_c) == 0:
                    continue
                embs = clients[cid].get_embeddings(X_c)
                zd_embs.append((embs, y_c))

            new_spawned = self.aps.run_aps(
                zd_embs, self.global_prototypes, rnd)
            if new_spawned:
                self.global_prototypes.update(new_spawned)
                print(f"  APS: Spawned {len(new_spawned)} new classes!")

            # ── 5. Evaluation ───────────────────────────────────────
            if test_data and (rnd % eval_every == 0 or rnd == self.cfg.NUM_ROUNDS):
                from utils.metrics import compute_all_metrics
                metrics = self._evaluate(
                    test_data, zeroday_test_data, known_class_ids)

                history['round'].append(rnd)
                history['accuracy'].append(metrics.get('accuracy', 0))
                history['f1_macro'].append(metrics.get('f1_macro', 0))
                history['detection_rate'].append(
                    metrics.get('detection_rate', 0))
                history['false_alarm_rate'].append(
                    metrics.get('false_alarm_rate', 0))
                history['byzantine_detected'].append(
                    len(agg_stats.get('excluded_byzantine', [])))
                history['zeroday_detected'].append(len(zd_clients))
                history['classes_spawned'].append(self.aps.get_num_spawned())
                history['zd_detection_rate'].append(
                    metrics.get('zd_detection_rate', 0))

                # Store PDS scores for plotting
                pds_vals = {cid: r['pds_score']
                            for cid, r in pds_results.items()}
                history['pds_scores'].append(pds_vals)

                print(f"\n  ── EVAL Round {rnd} ──")
                print(f"  Acc={metrics.get('accuracy', 0):.4f} | "
                      f"F1={metrics.get('f1_macro', 0):.4f} | "
                      f"DR={metrics.get('detection_rate', 0):.4f} | "
                      f"FAR={metrics.get('false_alarm_rate', 0):.4f}")
                if 'zd_detection_rate' in metrics:
                    print(f"  ZD-DetRate="
                          f"{metrics['zd_detection_rate']:.4f} | "
                          f"ZD-AUROC={metrics.get('zd_auc_roc', 0):.4f}")
                          
                if 'f1_per_class' in metrics:
                    f1s = metrics['f1_per_class']
                    # To track minority class collapse
                    lowest_classes = np.argsort(f1s)[:3]
                    print(f"  Lowest Class F1s: " + ", ".join([f"C{c}: {f1s[c]:.4f}" for c in lowest_classes]))

            # Log
            self.round_logs.append({
                'round': rnd,
                'agg_stats': agg_stats,
                'pds_results': {
                    cid: {k: v for k, v in r.items()
                          if k != 'per_class_div'}
                    for cid, r in pds_results.items()
                },
            })

        return history

    def _evaluate(self, test_data, zeroday_test_data=None,
                  known_class_ids=None):
        """Evaluate global model on test set (batched to avoid CUDA OOM)."""
        from utils.metrics import compute_all_metrics
        self.global_model.eval()
        batch_sz = 4096

        X_test, y_test = test_data

        # Batched closed-set evaluation
        all_preds = []
        all_features = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_sz):
                X_batch = torch.tensor(
                    X_test[i:i+batch_sz], dtype=torch.float32).to(self.device)
                logits, features = self.global_model(X_batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_features.append(features.cpu())
        preds = np.concatenate(all_preds)

        metrics = compute_all_metrics(y_test, preds)

        # Zero-day evaluation (if provided)
        if zeroday_test_data is not None and known_class_ids is not None:
            X_zd, y_zd = zeroday_test_data

            # Batched zero-day embedding extraction
            zd_features_list = []
            with torch.no_grad():
                for i in range(0, len(X_zd), batch_sz):
                    X_batch = torch.tensor(
                        X_zd[i:i+batch_sz], dtype=torch.float32).to(self.device)
                    _, zd_feat = self.global_model(X_batch, return_embeddings=True)
                    zd_features_list.append(zd_feat.cpu())
            zd_features = torch.cat(zd_features_list, dim=0).to(self.device)

            # Prototype-based open-set detection
            from models.prototype_layer import PrototypeLayer
            proto_layer = PrototypeLayer(
                self.cfg.EMBED_DIM, self.num_classes).to(self.device)
            proto_layer.set_prototypes(self.global_prototypes)

            _, distances = proto_layer.predict_by_prototype(zd_features)

            # Threshold = mean + 2*std of known-class distances
            # Threshold = 95th percentile of known-class distances (calibrated, Prompt 3)
            known_features = torch.cat(all_features, dim=0).to(self.device)
            _, known_dists = proto_layer.predict_by_prototype(known_features)

            known_dists_np = known_dists.cpu().numpy()
            zd_dists_np = distances.cpu().numpy()
            
            threshold = float(np.percentile(known_dists_np, 95))
            
            from utils.metrics import compute_zeroday_metrics
            all_dists = np.concatenate([known_dists_np, zd_dists_np])
            is_zeroday = np.concatenate([np.zeros(len(known_dists_np)), 
                                        np.ones(len(zd_dists_np))]).astype(bool)
            
            zd_metrics = compute_zeroday_metrics(all_dists, is_zeroday, threshold)
            for k, v in zd_metrics.items():
                metrics[k] = v
            metrics['zeroday_threshold'] = threshold

        return metrics

    def get_global_model(self):
        """Return the current global model."""
        return copy.deepcopy(self.global_model)

    def get_prototype_count(self):
        """Total prototypes including spawned."""
        return len(self.global_prototypes)
