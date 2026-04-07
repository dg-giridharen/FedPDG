"""
components/aps.py — Automatic Prototype Spawning
§III-F of the FedPDG paper.

When PDS detects a zero-day event, APS automatically:
1. Collects outlier embeddings from the flagged client
2. Clusters them with DBSCAN
3. Validates cluster coherence
4. Spawns new prototypes for genuine unknown classes
5. Propagates to the federation
"""
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


class AutomaticPrototypeSpawning:
    """
    5-step pipeline for autonomous class discovery in federated setting.
    """

    def __init__(self, config):
        self.cfg = config
        self.spawned_classes = {}   # new_class_id → prototype tensor
        self.spawn_history = []     # log of spawning events
        self.next_class_id = None   # set from known classes

    def set_next_class_id(self, num_known_classes):
        """Initialize the counter for new class IDs."""
        self.next_class_id = num_known_classes

    def run_aps(self, zeroday_client_data, global_prototypes, round_num):
        """
        Main APS pipeline.

        Args:
            zeroday_client_data: list of (embeddings_tensor, labels_array)
                                 from zero-day flagged clients
            global_prototypes:   dict {class_id: tensor(D,)}
            round_num:           current FL round

        Returns:
            new_prototypes: dict {new_class_id: tensor(D,)} or empty dict
        """
        if not zeroday_client_data:
            return {}

        # ── Step 1: Aggregate embeddings from zero-day clients ─────────
        all_embs = []
        for embs, _ in zeroday_client_data:
            if isinstance(embs, torch.Tensor):
                embs = embs.detach().cpu().numpy()
            if len(embs.shape) == 1:
                embs = embs.reshape(1, -1)
            all_embs.append(embs)

        if not all_embs:
            return {}

        all_embs = np.concatenate(all_embs, axis=0)
        all_embs = normalize(all_embs, norm='l2')

        print(f"  APS Round {round_num}: {len(all_embs)} candidate embeddings")

        # ── Step 2: Filter novel embeddings (far from known protos) ────
        if global_prototypes:
            proto_matrix = np.stack([
                p.detach().cpu().numpy() if isinstance(p, torch.Tensor)
                else np.array(p)
                for p in global_prototypes.values()
            ])
            proto_matrix = normalize(proto_matrix, norm='l2')

            # Cosine similarity to nearest known prototype
            sims = all_embs @ proto_matrix.T  # (N, C)
            max_sim = sims.max(axis=1)         # (N,)

            # Keep only embeddings dissimilar to ALL known classes
            threshold = 1.0 - self.cfg.APS_SPAWN_THRESHOLD
            novel_mask = max_sim < threshold
            novel_embs = all_embs[novel_mask]
            print(f"  APS: {novel_mask.sum()}/{len(all_embs)} "
                  f"passed novelty filter (threshold={threshold:.2f})")
        else:
            novel_embs = all_embs

        if len(novel_embs) < self.cfg.DBSCAN_MIN_SAMPLES:
            print("  APS: Not enough novel embeddings to spawn")
            return {}

        # ── Step 3: DBSCAN clustering ──────────────────────────────────
        dbscan = DBSCAN(
            eps=self.cfg.DBSCAN_EPS,
            min_samples=self.cfg.DBSCAN_MIN_SAMPLES,
            metric='cosine',
            n_jobs=-1,
        )
        cluster_labels = dbscan.fit_predict(novel_embs)
        unique_clusters = set(cluster_labels) - {-1}  # exclude noise

        n_noise = (cluster_labels == -1).sum()
        print(f"  APS: DBSCAN → {len(unique_clusters)} clusters, "
              f"{n_noise} noise points")

        # ── Steps 4 & 5: Validate and spawn ───────────────────────────
        new_prototypes = {}

        for cluster_id in sorted(unique_clusters):
            mask = (cluster_labels == cluster_id)
            cluster_embs = novel_embs[mask]

            # Minimum size check
            if len(cluster_embs) < self.cfg.APS_MIN_CLUSTER_SIZE:
                print(f"  APS: Cluster {cluster_id} too small "
                      f"({len(cluster_embs)} < {self.cfg.APS_MIN_CLUSTER_SIZE})")
                continue

            # Compute cluster centroid
            centroid = cluster_embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Separation check: must be far from existing prototypes
            is_novel = True
            if global_prototypes:
                for existing_proto in global_prototypes.values():
                    ep = existing_proto.detach().cpu().numpy() \
                        if isinstance(existing_proto, torch.Tensor) \
                        else np.array(existing_proto)
                    ep = ep / (np.linalg.norm(ep) + 1e-8)
                    cos_sim = float(np.dot(centroid, ep))
                    if cos_sim > (1.0 - self.cfg.DBSCAN_EPS):
                        is_novel = False
                        break

            # Also check against already-spawned prototypes
            for sp in self.spawned_classes.values():
                sp_np = sp.cpu().numpy() if isinstance(sp, torch.Tensor) \
                    else np.array(sp)
                sp_np = sp_np / (np.linalg.norm(sp_np) + 1e-8)
                if float(np.dot(centroid, sp_np)) > (1.0 - self.cfg.DBSCAN_EPS):
                    is_novel = False
                    break

            if not is_novel:
                print(f"  APS: Cluster {cluster_id} too similar to "
                      f"existing prototype — skipped")
                continue

            # ── SPAWN ──────────────────────────────────────────────────
            new_class_id = self.next_class_id
            self.next_class_id += 1

            new_proto = torch.tensor(centroid, dtype=torch.float32)
            new_prototypes[new_class_id] = new_proto
            self.spawned_classes[new_class_id] = new_proto

            coherence = self._compute_coherence(cluster_embs)
            event = {
                'round': round_num,
                'new_class_id': new_class_id,
                'cluster_size': int(mask.sum()),
                'coherence': coherence,
            }
            self.spawn_history.append(event)

            print(f"  APS: ✓ SPAWNED class {new_class_id} "
                  f"({mask.sum()} samples, coherence={coherence:.4f})")

        return new_prototypes

    def _compute_coherence(self, embeddings):
        """Intra-cluster coherence (avg pairwise cosine similarity)."""
        if len(embeddings) < 2:
            return 1.0
        norms = normalize(embeddings, norm='l2')
        sim_matrix = norms @ norms.T
        n = len(embeddings)
        return float((sim_matrix.sum() - n) / (n * (n - 1)))

    def get_spawn_history(self):
        """Return spawn event log for visualization."""
        return self.spawn_history

    def get_num_spawned(self):
        """Number of classes spawned so far."""
        return len(self.spawned_classes)
