"""
utils/data_loader.py — Multi-Dataset Loader
Loads CICIDS2017, ToN-IoT, and N-BaIoT datasets with cleaning and normalization.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for non-ASCII class names
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


class DatasetLoader:
    """Unified loader for all three benchmark datasets."""

    def __init__(self, config):
        self.cfg = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    # ─────────────────────────────────────────────────────────────────────
    # CICIDS2017
    # ─────────────────────────────────────────────────────────────────────
    def load_cicids2017(self, path):
        """Load and clean CICIDS2017 dataset from directory of CSVs."""
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        dfs = []
        for f in files:
            df = pd.read_csv(os.path.join(path, f), low_memory=False)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # Clean column names (trailing spaces)
        df.columns = df.columns.str.strip()

        # Drop infinite & NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # Label column
        label_col = 'Label' if 'Label' in df.columns else ' Label'

        # Encode labels
        df['label_multiclass'] = self.label_encoder.fit_transform(
            df[label_col].str.strip())

        # Feature columns = all numeric except labels
        feature_cols = [c for c in df.columns
                        if c not in [label_col, 'label_multiclass']]
        X = df[feature_cols].select_dtypes(include=[np.number])

        y_multi = df['label_multiclass'].values
        label_names = self.label_encoder.classes_

        print(f"CICIDS2017: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(label_names)} classes")
        print(f"  Classes: {dict(zip(label_names, np.bincount(y_multi)))}")
        return X.values.astype(np.float32), y_multi, label_names

    # ─────────────────────────────────────────────────────────────────────
    # ToN-IoT  (directory of Network_dataset_1..23.csv)
    # ─────────────────────────────────────────────────────────────────────
    def load_toniot(self, path):
        """Load and clean ToN-IoT Processed Network dataset."""
        # Load all Network_dataset_*.csv files from directory
        files = sorted([f for f in os.listdir(path)
                        if f.startswith('Network_dataset') and f.endswith('.csv')])
        if not files:
            raise FileNotFoundError(f"No Network_dataset_*.csv files in {path}")

        print(f"  ToN-IoT: Loading {len(files)} CSV files...")
        dfs = []
        for f in files:
            df_chunk = pd.read_csv(os.path.join(path, f), low_memory=False)
            dfs.append(df_chunk)
            print(f"    Loaded {f}: {len(df_chunk)} rows")

        df = pd.concat(dfs, ignore_index=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # ToN-IoT has 'label' (binary) and 'type' (multi-class)
        label_col = 'label' if 'label' in df.columns else 'Label'
        type_col = 'type' if 'type' in df.columns else None

        if type_col and type_col in df.columns:
            df['label_multiclass'] = self.label_encoder.fit_transform(df[type_col])
            label_names = self.label_encoder.classes_
        else:
            df['label_multiclass'] = df[label_col].astype(int)
            label_names = np.array(['benign', 'attack'])

        drop_cols = [label_col, 'label_multiclass']
        if type_col and type_col in df.columns:
            drop_cols.append(type_col)
        # Drop non-numeric identifier columns if present
        for c in ['ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port']:
            if c in df.columns:
                drop_cols.append(c)

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=[np.number])

        y_multi = df['label_multiclass'].values

        print(f"ToN-IoT: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(label_names)} classes")
        return X.values.astype(np.float32), y_multi, label_names

    # ─────────────────────────────────────────────────────────────────────
    # N-BaIoT
    # ─────────────────────────────────────────────────────────────────────
    def load_nbaiot(self, path):
        """Load and clean N-BaIoT dataset from directory tree."""
        dfs = []
        label_map = {}
        label_id = 0

        for root, dirs, files in os.walk(path):
            for f in files:
                if not f.endswith('.csv'):
                    continue
                fpath = os.path.join(root, f)
                try:
                    df_tmp = pd.read_csv(fpath)
                except Exception:
                    continue

                # Infer label from directory/filename
                combined = (root + '/' + f).lower()
                if 'benign' in combined:
                    attack_name = 'benign'
                elif 'gafgyt' in combined or 'bashlite' in combined:
                    # More specific gafgyt sub-labels
                    for sub in ['combo', 'junk', 'scan', 'tcp', 'udp']:
                        if sub in combined:
                            attack_name = f'gafgyt_{sub}'
                            break
                    else:
                        attack_name = 'gafgyt'
                elif 'mirai' in combined:
                    for sub in ['ack', 'scan', 'syn', 'udp', 'udpplain']:
                        if sub in combined:
                            attack_name = f'mirai_{sub}'
                            break
                    else:
                        attack_name = 'mirai'
                else:
                    attack_name = 'unknown'

                if attack_name not in label_map:
                    label_map[attack_name] = label_id
                    label_id += 1

                df_tmp['label'] = label_map[attack_name]
                dfs.append(df_tmp)

        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {path}")

        df = pd.concat(dfs, ignore_index=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        y_multi = df['label'].values
        X = df.drop(columns=['label']).select_dtypes(include=[np.number])
        label_names = np.array(
            [k for k, v in sorted(label_map.items(), key=lambda x: x[1])])

        print(f"N-BaIoT: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(label_names)} classes")
        return X.values.astype(np.float32), y_multi, label_names

    # ─────────────────────────────────────────────────────────────────────
    # Unified loader
    # ─────────────────────────────────────────────────────────────────────
    def load_dataset(self, dataset_name, path=None):
        """Load any supported dataset by name, with optional subsampling."""
        if path is None:
            from config import Config
            path = Config.get_dataset_path(dataset_name)
        loaders = {
            'CICIDS2017': self.load_cicids2017,
            'ToN-IoT': self.load_toniot,
            'NbAIoT': self.load_nbaiot,
        }
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                             f"Supported: {list(loaders.keys())}")
        X, y, label_names = loaders[dataset_name](path)

        # Stratified subsample if dataset exceeds MAX_SAMPLES
        max_samples = getattr(self.cfg, 'MAX_SAMPLES', None)
        if max_samples and len(X) > max_samples:
            # First, drop classes with too few samples for stratification
            unique, counts = np.unique(y, return_counts=True)
            min_count = 10  # need enough for train/test/stratify
            valid_classes = unique[counts >= min_count]
            if len(valid_classes) < len(unique):
                dropped = len(unique) - len(valid_classes)
                print(f"  Dropping {dropped} rare classes (< {min_count} samples)")
                mask = np.isin(y, valid_classes)
                X, y = X[mask], y[mask]
                # Re-encode labels to 0..N-1
                old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
                y = np.array([old_to_new[yi] for yi in y])
                label_names = np.array([label_names[c] for c in sorted(valid_classes)])

            print(f"  Subsampling {len(X)} -> {max_samples} (stratified)...")
            from sklearn.model_selection import train_test_split
            X, _, y, _ = train_test_split(
                X, y, train_size=max_samples,
                stratify=y, random_state=42)
            print(f"  After subsample: {X.shape[0]} samples, "
                  f"{len(np.unique(y))} classes")

        return X, y, label_names

    # ─────────────────────────────────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────────────────────────────────
    def normalize(self, X_train, X_test):
        """StandardScaler normalization with clipping."""
        X_train_norm = self.scaler.fit_transform(X_train)
        X_test_norm = self.scaler.transform(X_test)
        X_train_norm = np.clip(X_train_norm, -10, 10)
        X_test_norm = np.clip(X_test_norm, -10, 10)
        return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Zero-day holdout split
    # ─────────────────────────────────────────────────────────────────────
    def zeroday_split(self, X, y_multi, label_names, holdout_n=2, seed=42):
        """
        Split into known (training) and unknown (zero-day test) classes.
        Always keeps benign (class 0) in training.
        """
        np.random.seed(seed)
        unique_classes = np.unique(y_multi)

        # Benign = class 0 always known
        benign_id = 0
        for i, name in enumerate(label_names):
            if 'benign' in name.lower():
                benign_id = i
                break

        attack_classes = unique_classes[unique_classes != benign_id]
        np.random.shuffle(attack_classes)
        holdout_classes = attack_classes[:holdout_n]
        known_classes = np.append(
            attack_classes[holdout_n:], [benign_id])

        known_mask = np.isin(y_multi, known_classes)
        holdout_mask = np.isin(y_multi, holdout_classes)

        X_known = X[known_mask]
        y_known = y_multi[known_mask]
        X_zeroday = X[holdout_mask]
        y_zeroday = y_multi[holdout_mask]

        known_names = [label_names[c] for c in sorted(known_classes)]
        holdout_names = [label_names[c] for c in holdout_classes]

        print(f"\n  Zero-Day Split:")
        print(f"    Known classes ({len(known_classes)}): {known_names}")
        print(f"    Holdout classes ({len(holdout_classes)}): {holdout_names}")
        print(f"    Known samples: {X_known.shape[0]} | "
              f"Zero-day samples: {X_zeroday.shape[0]}")

        return (X_known, y_known, X_zeroday, y_zeroday,
                known_classes, holdout_classes)
