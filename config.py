"""
FedPDG — Central Configuration
All hyperparameters for the complete experiment pipeline.
"""
import torch
import os

class Config:
    # ── Paths ─────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # ── Datasets ──────────────────────────────────────
    DATASETS = ['CICIDS2017', 'ToN-IoT', 'NbAIoT']
    MAX_SAMPLES = 100_000  

    # ── Default Dataset Paths (Override via local setup) ──
    DATASET_PATHS = {
        'CICIDS2017': os.path.join(DATA_ROOT, 'CICIDS2017'),
        'ToN-IoT':    os.path.join(DATA_ROOT, 'ToN-IoT'),
        'NbAIoT':     os.path.join(DATA_ROOT, 'NbAIoT'),
    }

    # ── Federated Learning ────────────────────────────
    NUM_CLIENTS = 5
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 3
    FRACTION_FIT = 1.0
    DIRICHLET_ALPHAS = [0.05, 0.1, 0.5, 1.0]

    # ── Byzantine Simulation ─────────────────────────
    BYZANTINE_RATIOS = [0.1, 0.2, 0.3, 0.4]
    ATTACK_TYPES = ['label_flip', 'sign_flip', 'gaussian', 'model_replace']

    # ── Zero-day Simulation ───────────────────────────
    HOLDOUT_CLASSES = 2

    # ── Model Architecture ───────────────────────────
    INPUT_DIM = 78              # CICIDS2017 default (adjust per dataset)
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    DROPOUT = 0.1
    PROTOTYPE_DIM = 128

    # ── Training ─────────────────────────────────────
    BATCH_SIZE = 256
    LR = 5e-4              # Reduced for Focal Loss stability
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.1           # contrastive loss τ
    LAMBDA_CON = 0.3            # Slightly reduced to prioritize CE

    # ── PDS ───────────────────────────────────────────
    PDS_GAMMA = 2.5             # z-score threshold for anomaly
    PDS_THRESHOLD_ZERODAY = 0.6
    PDS_THRESHOLD_BYZANTINE = 0.8
    GINI_SPLIT_THRESHOLD = 0.5

    # ── DWA ───────────────────────────────────────────
    DWA_LAMBDA = 0.5            # Reduced for smoother trust weighting
    DWA_EPSILON = 1e-6          # near-zero weight for Byzantine

    # ── APS (DBSCAN) ─────────────────────────────────
    DBSCAN_EPS_VALUES = [0.1, 0.3, 0.5, 0.8]
    DBSCAN_EPS = 0.3
    DBSCAN_MIN_SAMPLES = 5
    APS_SPAWN_THRESHOLD = 0.7
    APS_MIN_CLUSTER_SIZE = 20

    # ── Statistical Tests ─────────────────────────────
    RANDOM_SEEDS = [42, 123, 456, 789, 2024]
    SIGNIFICANCE_LEVEL = 0.05

    # ── Ablation ──────────────────────────────────────
    ABLATION_LAYERS = [1, 2, 4, 6]
    ABLATION_HEADS = [2, 4, 8]
    ABLATION_DIMS = [64, 128, 256]
    ABLATION_TEMPERATURES = [0.05, 0.1, 0.2, 0.5]

    # ── Device ────────────────────────────────────────
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Paths (Recursive) ─────────────────────────────
    TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
    PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
    LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

    @classmethod
    def get_dataset_path(cls, dataset_name):
        return cls.DATASET_PATHS.get(dataset_name, '')

    @classmethod
    def get_input_dim(cls, dataset_name):
        dims = {
            'CICIDS2017': 78,
            'ToN-IoT': 40,
            'NbAIoT': 115,
        }
        return dims.get(dataset_name, 78)
